"""SSIM cost reduction: evaluate at a reduced width, and stop building the SSIM map.

The dedup was measured as the pipeline bottleneck (97.7 ms per input frame on a
2592x1520 source, against 53.4 ms for the whole RT-DETR detector). Both costs in here
are avoidable: SSIM was computed at full resolution while the motion gatekeeper in the
same filter already evaluates at 480 px wide, and it passed full=True, which allocates a
float SSIM map the size of the input that the caller immediately discards.

Downscaling changes the score, so it is opt-in: the default must stay byte-identical.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from filter_frame_dedup.ssim_processor import SSIMProcessor


def cfg(**over):
    base = dict(ssim_threshold=0.90, ssim_patch_grid_size=1, ssim_eval_width=0)
    base.update(over)
    return SimpleNamespace(**base)


def frame(seed: int, w: int = 2592, h: int = 1520) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestDefaultIsUnchanged:
    def test_default_config_does_not_resize(self):
        p = SSIMProcessor(cfg())
        assert p.eval_width == 0

        img = frame(1, w=64, h=48)
        assert p._to_gray(img).shape == (48, 64), "default must compare at full resolution"

    def test_missing_knob_falls_back_to_full_resolution(self):
        """An older config object with no ssim_eval_width must still work."""
        p = SSIMProcessor(SimpleNamespace(ssim_threshold=0.9, ssim_patch_grid_size=1))
        assert p.eval_width == 0
        assert p._to_gray(frame(2, w=64, h=48)).shape == (48, 64)

    def test_identical_frames_score_one(self):
        p = SSIMProcessor(cfg())
        img = frame(3, w=128, h=96)
        assert p.compute_ssim(img, img) == pytest.approx(1.0, abs=1e-6)


class TestDownscaling:
    def test_resizes_to_eval_width_preserving_aspect(self):
        p = SSIMProcessor(cfg(ssim_eval_width=480))
        gray = p._to_gray(frame(4, w=2592, h=1520))
        assert gray.shape[1] == 480
        # 1520 * (480/2592) = 281.5 -> 281 or 282 depending on rounding; assert the ratio.
        assert abs(gray.shape[0] / gray.shape[1] - 1520 / 2592) < 0.01

    def test_never_upscales_a_smaller_frame(self):
        """A frame already narrower than eval_width must be left alone."""
        p = SSIMProcessor(cfg(ssim_eval_width=480))
        gray = p._to_gray(frame(5, w=320, h=240))
        assert gray.shape == (240, 320)

    def test_identical_frames_still_score_one_when_downscaled(self):
        p = SSIMProcessor(cfg(ssim_eval_width=480))
        img = frame(6, w=2592, h=1520)
        assert p.compute_ssim(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_downscaled_comparison_is_much_cheaper(self):
        """The whole point. Not a wall-clock assertion, a pixel-count one.

        Timing in CI is flaky; the pixel count is the thing that actually drives SSIM
        cost and it is deterministic.
        """
        full = SSIMProcessor(cfg())
        small = SSIMProcessor(cfg(ssim_eval_width=480))
        img = frame(7, w=2592, h=1520)

        full_px = full._to_gray(img).size
        small_px = small._to_gray(img).size

        assert full_px / small_px > 25, f"expected >25x fewer pixels, got {full_px / small_px:.1f}x"


class TestKeepDecisionStillWorks:
    def test_first_frame_is_always_kept(self):
        p = SSIMProcessor(cfg(ssim_eval_width=480))
        assert p.should_save_frame(frame(8, w=640, h=480)) is True

    def test_identical_frame_is_dropped(self):
        p = SSIMProcessor(cfg(ssim_eval_width=480))
        img = frame(9, w=640, h=480)
        p.update_reference_frame(img)
        assert p.should_save_frame(img.copy()) is False, "an identical frame must not be kept"

    def test_completely_different_frame_is_kept(self):
        p = SSIMProcessor(cfg(ssim_eval_width=480))
        p.update_reference_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        assert p.should_save_frame(np.full((480, 640, 3), 255, dtype=np.uint8)) is True

    def test_patch_mode_also_honours_eval_width(self):
        p = SSIMProcessor(cfg(ssim_eval_width=480, ssim_patch_grid_size=2))
        img = frame(10, w=2592, h=1520)
        p.update_reference_frame(img)
        # Same frame, so every patch scores ~1.0 and none is below threshold: drop it.
        assert p.should_save_frame(img.copy()) is False
