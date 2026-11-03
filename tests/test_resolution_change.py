"""A frame of a different size must not kill the filter.

`is_motion_detected` subtracts the current frame from the stored reference, and
`cv2.absdiff` raises on mismatched shapes instead of returning a score. One
video_in reading a folder of clips in sequence hits that on every resolution
change: measured on an FDC batch, 1920x1080 followed by 1366x720 killed the
filter at the clip boundary and took the rest of the batch down with it, because
nothing downstream of a dead dedup ever sees end-of-stream.

A different size is a different source, not a changed scene, so it counts as
motion: the frame goes through and becomes the new reference.

The same subtraction is reached from three places, so all three are covered
here: `hash_processor.is_motion_detected`, `ssim_processor.should_save_frame`
(which runs right after it on the default `active_processors`), and
`motion_gate_processor.should_process_frame`. Guarding only the first moves the
traceback one stage later and leaves the hang in place.
"""
from types import SimpleNamespace

import numpy as np
from openfilter.filter_runtime.frame import Frame

from filter_frame_dedup.filter import FilterFrameDedup
from filter_frame_dedup.hash_processor import HashFrameProcessor
from filter_frame_dedup.motion_gate_processor import FastMotionGatekeeper
from filter_frame_dedup.ssim_processor import SSIMProcessor


def cfg(**over):
    base = dict(
        hash_threshold=5,
        motion_threshold=1200,
        min_time_between_frames=0.0,
        roi=None,
        use_hash_dedup=True,
        debug=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def frame(seed: int, w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def flat(value: int, w: int, h: int) -> np.ndarray:
    """A uniform frame: similar to any other flat frame, so nothing short-circuits."""
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestResolutionChange:
    def test_smaller_frame_does_not_raise(self):
        p = HashFrameProcessor(cfg())
        assert p.is_motion_detected(frame(1, 1920, 1080), frame(2, 1366, 720)) is True

    def test_larger_frame_does_not_raise(self):
        p = HashFrameProcessor(cfg())
        assert p.is_motion_detected(frame(1, 640, 360), frame(2, 1920, 1080)) is True

    def test_same_size_still_compares(self):
        """The guard must not short-circuit the normal path."""
        p = HashFrameProcessor(cfg())
        still = frame(7, 320, 240)
        assert p.is_motion_detected(still, still.copy()) is False
        assert p.is_motion_detected(still, frame(8, 320, 240)) is True

    def test_batch_survives_a_clip_boundary(self):
        """The real sequence: frames of clip A, then frames of clip B."""
        p = HashFrameProcessor(cfg())
        clip_a = [frame(i, 1920, 1080) for i in range(3)]
        clip_b = [frame(100 + i, 1366, 720) for i in range(3)]

        kept = 0
        for img in clip_a + clip_b:
            if p.should_process_frame(img):
                kept += 1
                p.update_reference_frame(img)

        assert kept == 6, "every frame of this noise is new; none may be lost to a crash"


class TestMotionDetectedReturnsPythonBool:
    """The annotation says ``bool``, so both branches must return one.

    ``np.count_nonzero`` is ``np.int64`` on numpy >= 2.4, which makes the
    comparison an ``np.bool_`` and breaks every ``is True`` / ``is False``
    assertion against this method. CI caught it as ``assert np.False_ is False``
    on the 3.14 leg, which resolves numpy 2.5.4 while 3.10-3.13 resolve 2.2.6.
    """

    def test_both_branches_return_bool(self):
        p = HashFrameProcessor(cfg())
        still = frame(7, 320, 240)
        assert type(p.is_motion_detected(still, still.copy())) is bool
        assert type(p.is_motion_detected(still, frame(8, 320, 240))) is bool
        assert type(p.is_motion_detected(still, frame(9, 640, 480))) is bool


class TestSSIMResolutionChange:
    """The hash guard alone does not fix the hang: ssim_dedup runs next by default.

    ``normalize_config`` sets ``active_processors = ["hash_dedup", "ssim_dedup"]``,
    and a frame the hash guard accepts reaches ``should_save_frame`` against a
    reference of the old size, where the box filters broadcast and raise.
    """

    def ssim(self, **over):
        base = dict(ssim_threshold=0.90, ssim_patch_grid_size=1, ssim_eval_width=0)
        base.update(over)
        return SSIMProcessor(cfg(**base))

    def test_smaller_frame_is_kept(self):
        p = self.ssim()
        p.update_reference_frame(frame(1, 1920, 1080))
        assert p.should_save_frame(frame(2, 1366, 720)) is True

    def test_larger_frame_is_kept(self):
        p = self.ssim()
        p.update_reference_frame(frame(1, 1366, 720))
        assert p.should_save_frame(frame(2, 1920, 1080)) is True

    def test_patch_grid_smaller_frame_is_kept(self):
        """The grid path slices both frames with indices taken from the reference.

        A leading patch that fits in both shapes compares fine and can
        short-circuit to True, which is why noise hides this. A trailing patch
        runs to the reference's edge, so it is clipped on the smaller frame and
        the shapes diverge: (540,960) against (540,406). Flat frames are similar
        enough that no earlier patch short-circuits, so the comparison gets
        there.
        """
        p = self.ssim(ssim_patch_grid_size=2)
        p.update_reference_frame(flat(100, 1920, 1080))
        assert p.should_save_frame(flat(100, 1366, 720)) is True

    def test_patch_grid_larger_frame_is_kept(self):
        p = self.ssim(ssim_patch_grid_size=2)
        p.update_reference_frame(flat(100, 1366, 720))
        assert p.should_save_frame(flat(100, 1920, 1080)) is True

    def test_same_size_still_compares(self):
        p = self.ssim()
        still = frame(7, 320, 240)
        p.update_reference_frame(still)
        assert p.should_save_frame(still.copy()) is False
        assert p.should_save_frame(frame(8, 320, 240)) is True


class TestMotionGateResolutionChange:
    """``motion_gate`` has the same ``cv2.absdiff`` crash, one processor earlier.

    ``_to_gray`` scales the height by the aspect ratio, so the pair that motivated
    this PR reaches the subtraction as 480x270 against 480x253.
    """

    def test_shapes_really_do_differ_after_downscale(self):
        g = FastMotionGatekeeper()
        assert g._to_gray(frame(1, 1920, 1080)).shape == (270, 480)
        assert g._to_gray(frame(2, 1366, 720)).shape == (253, 480)

    def test_smaller_frame_is_kept(self):
        g = FastMotionGatekeeper()
        g.update_reference_frame(frame(1, 1920, 1080))
        assert g.should_process_frame(frame(2, 1366, 720)) is True

    def test_larger_frame_is_kept(self):
        g = FastMotionGatekeeper()
        g.update_reference_frame(frame(1, 1366, 720))
        assert g.should_process_frame(frame(2, 1920, 1080)) is True

    def test_patch_grid_frame_is_kept(self):
        g = FastMotionGatekeeper(patch_grid_size=2)
        g.update_reference_frame(frame(1, 1920, 1080))
        assert g.should_process_frame(frame(2, 1366, 720)) is True

    def test_same_size_still_compares(self):
        g = FastMotionGatekeeper()
        still = frame(7, 640, 480)
        g.update_reference_frame(still)
        assert g.should_process_frame(still.copy()) is False
        assert g.should_process_frame(frame(8, 640, 480)) is True


class TestDefaultPipelineSurvivesAClipBoundary:
    """End to end, through the pipeline a deployment actually gets.

    This is the case the PR is for: one video_in reading a folder of clips, the
    resolution changing at a clip boundary. Guarding only hash_dedup moved the
    traceback from hash_processor to ssim_processor and left the hang in place.
    """

    def build(self, tmp_path, processors):
        config = {
            'config': {
                'id': 'dedup',
                'active_processors': processors,
                'min_time_between_frames': 0.0,
                'save_images': False,
                'output_folder': str(tmp_path / 'frames'),
            }
        }
        f = FilterFrameDedup(config)
        f.config = f.normalize_config(f.config)
        f.setup(f.config)
        return f

    def run_boundary(self, f):
        clip_a = [frame(i, 1920, 1080) for i in range(3)]
        clip_b = [frame(100 + i, 1366, 720) for i in range(3)]
        for img in clip_a + clip_b:
            f.process({'main': Frame(img, {}, 'BGR')})
        return f.unique_frame_count

    def test_default_pipeline(self, tmp_path):
        """The default: hash_dedup then ssim_dedup, with no active_processors set."""
        f = self.build(tmp_path, None)
        assert f.config.active_processors == ['hash_dedup', 'ssim_dedup']
        assert self.run_boundary(f) == 6

    def test_pipeline_with_motion_gate(self, tmp_path):
        f = self.build(tmp_path, ['motion_gate', 'hash_dedup', 'ssim_dedup'])
        assert self.run_boundary(f) == 6
