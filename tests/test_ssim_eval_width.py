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


class TestTinyEvalWidthIsRejectedNotCrashed:
    """A width below the SSIM window used to raise from inside scikit-image.

    `compute_ssim` called ssim() with no win_size, so scikit-image used its default of
    7 and raised "win_size exceeds image extent" once the downscaled frame was smaller
    than that. The patch-grid path already clamped win_size; the whole-frame path did
    not. Config now rejects it up front, and the method clamps defensively for callers
    that bypass config.
    """

    @pytest.mark.parametrize("width", [1, 3, 6])
    def test_config_rejects_a_width_below_the_ssim_window(self, width):
        from filter_frame_dedup.filter import FilterFrameDedup, FilterFrameDedupConfig

        with pytest.raises(ValueError, match="at least 7"):
            FilterFrameDedup.normalize_config(
                FilterFrameDedupConfig({"ssim_eval_width": width}))

    @pytest.mark.parametrize("width", [0, 7, 480])
    def test_config_accepts_zero_and_anything_from_seven_up(self, width):
        from filter_frame_dedup.filter import FilterFrameDedup, FilterFrameDedupConfig

        conf = FilterFrameDedup.normalize_config(
            FilterFrameDedupConfig({"ssim_eval_width": width}))
        assert conf.get("ssim_eval_width") == width

    def test_compute_ssim_clamps_the_window_instead_of_raising(self):
        """Called directly with a tiny frame, it must not raise."""
        p = SSIMProcessor(cfg(ssim_eval_width=8))
        small = np.random.default_rng(0).integers(0, 255, (5, 8, 3), dtype=np.uint8)
        assert p.compute_ssim(small, small.copy()) == pytest.approx(1.0, abs=1e-5)

    def test_compute_ssim_fails_open_below_the_minimum(self):
        """Uncomputable means keep the frame, never drop it."""
        p = SSIMProcessor(cfg(ssim_eval_width=0))
        tiny = np.zeros((2, 2, 3), dtype=np.uint8)
        assert p.compute_ssim(tiny, tiny.copy()) == 0.0


class TestExtremeAspectRatioKeepsTheFilterWorking:
    """A valid width can still produce an unusable height.

    ``normalize_config`` bounds the configured width, but it never sees a frame. On a
    wide, short source the height scales independently and lands below the SSIM window,
    at which point ``compute_ssim`` fails open on every frame and the filter silently
    stops deduplicating. The downscale is skipped for those shapes instead.
    """

    WIDE_SHORT = (10, 2592)  # height, width

    def test_downscale_is_skipped_when_it_would_flatten_the_frame(self):
        p = SSIMProcessor(cfg(ssim_eval_width=7))
        img = frame(1, w=self.WIDE_SHORT[1], h=self.WIDE_SHORT[0])

        # Naively, 2592 -> 7 scales 10 -> 0.027, rounded up to a 1px tall frame.
        assert p._to_gray(img).shape == self.WIDE_SHORT, \
            "a downscale that crosses the SSIM window must be skipped, not applied"

    def test_the_decision_survives_instead_of_failing_open(self):
        """The regression this guards: identical frames must score as identical."""
        p = SSIMProcessor(cfg(ssim_eval_width=7))
        img = frame(2, w=self.WIDE_SHORT[1], h=self.WIDE_SHORT[0])

        score = p.compute_ssim(img, img.copy())
        assert score == pytest.approx(1.0, abs=1e-5), \
            f"expected a real comparison, got {score} (0.0 means it failed open)"

    def test_a_changed_frame_is_still_detected_as_different(self):
        p = SSIMProcessor(cfg(ssim_eval_width=7))
        a = frame(3, w=self.WIDE_SHORT[1], h=self.WIDE_SHORT[0])
        b = frame(4, w=self.WIDE_SHORT[1], h=self.WIDE_SHORT[0])

        assert p.compute_ssim(a, b) < 0.9, "the comparison must still discriminate"

    def test_the_warning_fires_once_per_shape_not_once_per_frame(self, caplog):
        p = SSIMProcessor(cfg(ssim_eval_width=7))
        img = frame(5, w=self.WIDE_SHORT[1], h=self.WIDE_SHORT[0])

        with caplog.at_level("WARNING"):
            for _ in range(5):
                p._to_gray(img)

        hits = [r for r in caplog.records if "below the" in r.getMessage()]
        assert len(hits) == 1, f"expected one warning per shape, got {len(hits)}"

    def test_normal_aspect_ratios_still_downscale(self):
        """The guard must not disable the optimisation for ordinary frames."""
        p = SSIMProcessor(cfg(ssim_eval_width=480))
        img = frame(6, w=2592, h=1520)

        assert p._to_gray(img).shape == (281, 480)
