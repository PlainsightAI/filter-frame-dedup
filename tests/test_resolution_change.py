"""A frame of a different size must not kill the filter.

`is_motion_detected` subtracts the current frame from the stored reference, and
`cv2.absdiff` raises on mismatched shapes instead of returning a score. One
video_in reading a folder of clips in sequence hits that on every resolution
change: measured on an FDC batch, 1920x1080 followed by 1366x720 killed the
filter at the clip boundary and took the rest of the batch down with it, because
nothing downstream of a dead dedup ever sees end-of-stream.

A different size is a different source, not a changed scene, so it counts as
motion: the frame goes through and becomes the new reference.
"""
from types import SimpleNamespace

import numpy as np

from filter_frame_dedup.hash_processor import HashFrameProcessor


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
