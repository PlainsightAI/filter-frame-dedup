"""SSIM computed with cv2.boxFilter instead of scikit-image.

The dedup is the stage that binds the CPD pipeline, at 43.8 ms per input frame on
a full RTX PRO 6000 against 19.8 ms for the detector, and SSIM is the bulk of it.
scikit-image averages each window with scipy.ndimage.uniform_filter, which is
single-threaded scalar code; cv2.boxFilter computes the same box mean with SIMD.

This only earns its place if it is not an approximation. The tests below pin the
equivalence rather than the speed: the same score, and more importantly the same
keep decision, on the shapes and window sizes the filter can actually produce.

Why an exact match is achievable at all: scikit-image crops the SSIM map by
pad = (win_size - 1) // 2 before averaging, so every pixel whose window touched
the border is discarded and the boundary mode is unobservable.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity as skimage_ssim

from filter_frame_dedup.ssim_processor import SSIMProcessor, _ssim_boxfilter

# Floating-point noise between two different summation orders. Measured worst case
# over every shape and window in this file is ~1e-13; a keep decision would have to
# land within that of the threshold to flip, which no real score does.
TOLERANCE = 1e-9


def gray_pair(seed: int, w: int, h: int, change: bool = True):
    """A blurred, noisy grey pair, optionally with a changed region.

    Uniform noise is a poor stand-in for video: SSIM saturates near 0 and every
    implementation agrees trivially. Blurred structure plus light sensor noise puts
    the score in the 0.85-0.95 band the threshold actually sits in.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 190, (max(1, h // 8), max(1, w // 8)), dtype=np.uint8)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    nxt = base.copy()
    if change and h > 40 and w > 40:
        nxt[h // 3:h // 3 + h // 8, w // 4:w // 4 + w // 6] = rng.integers(
            0, 255, (h // 8, w // 6), dtype=np.uint8)
    noise = rng.integers(-4, 5, (h, w), dtype=np.int16)
    return base, np.clip(nxt.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class TestMatchesScikitImage:
    """The claim this change rests on."""

    SHAPES = [
        (1228, 720),    # what video_in actually delivers (capped at 1280x720)
        (2592, 1520),   # the raw source, for the patch-grid path
        (480, 281),     # ssim_eval_width=480
        (64, 48),       # small frame, clamped window
        (9, 9),         # the smallest frame that still clears win_size=7
    ]

    @pytest.mark.parametrize("w,h", SHAPES)
    @pytest.mark.parametrize("win", [3, 5, 7])
    def test_score_matches(self, w, h, win):
        if min(w, h) < win:
            pytest.skip(f"{win}px window does not fit a {h}x{w} frame")
        a, b = gray_pair(1, w, h)
        assert _ssim_boxfilter(a, b, win) == pytest.approx(
            skimage_ssim(a, b, full=False, win_size=win), abs=TOLERANCE)

    @pytest.mark.parametrize("win", [3, 5, 7])
    def test_identical_frames_score_one(self, win):
        a, _ = gray_pair(2, 320, 240)
        assert _ssim_boxfilter(a, a, win) == pytest.approx(1.0, abs=1e-9)

    def test_the_score_lands_where_the_threshold_lives(self):
        """A guard on the fixtures: agreement at SSIM ~0 would prove nothing."""
        a, b = gray_pair(3, 1228, 720)
        assert 0.80 < _ssim_boxfilter(a, b, 7) < 0.99


class TestKeepDecisionIsUnchanged:
    """Scores matching is the mechanism; the keep set is what the pipeline cares about."""

    @staticmethod
    def _cfg(threshold):
        return SimpleNamespace(ssim_threshold=threshold, ssim_patch_grid_size=1,
                               ssim_eval_width=0)

    @pytest.mark.parametrize("threshold", [0.85, 0.90, 0.95])
    def test_same_verdict_over_a_sweep_of_pairs(self, threshold):
        """Every pair must fall on the same side of the threshold as scikit-image."""
        proc = SSIMProcessor(self._cfg(threshold))
        disagreements = []
        for seed in range(24):
            a, b = gray_pair(seed, 480, 270, change=seed % 3 == 0)
            reference = skimage_ssim(a, b, full=False, win_size=7) <= threshold
            ours = _ssim_boxfilter(a, b, 7) <= threshold
            if reference != ours:
                disagreements.append(seed)
        assert not disagreements, f"keep decision changed on seeds {disagreements}"

    def test_first_frame_is_always_kept(self):
        proc = SSIMProcessor(self._cfg(0.90))
        frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        assert proc.should_save_frame(frame) is True

    def test_an_identical_frame_is_dropped_and_a_different_one_kept(self):
        proc = SSIMProcessor(self._cfg(0.90))
        rng = np.random.default_rng(4)
        first = cv2.GaussianBlur(
            rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 3)
        assert proc.should_save_frame(first) is True
        proc.prev_frame = first
        assert proc.should_save_frame(first.copy()) is False
        other = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        assert proc.should_save_frame(other) is True


class TestPatchGridPath:
    def test_patch_grid_still_decides(self):
        """The patch path calls the same helper with a clamped window."""
        cfg = SimpleNamespace(ssim_threshold=0.90, ssim_patch_grid_size=2,
                              ssim_eval_width=0)
        proc = SSIMProcessor(cfg)
        rng = np.random.default_rng(5)
        first = cv2.GaussianBlur(
            rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 3)
        proc.prev_frame = first
        assert proc.should_save_frame(first.copy()) is False, \
            "an identical frame must be dropped on the patch path too"
