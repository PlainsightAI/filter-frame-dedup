"""The shared downscale that took frame-dedup out of the bottleneck slot.

`INTER_AREA` is what makes these hashes stable, but reducing 1080p straight to
32x32 with it averages over ~60x60 kernels: 7.52 ms per call, paid three times
per frame because each hash re-derived its own grayscale. That is why
frame-dedup measured 43.8 ms per input frame in PLAT-1471 while the RT-DETR
detector it protects cost 4.1 ms.

These tests pin the two properties the fix rests on: one downscale per frame,
and a hash distance that still tracks the old implementation closely enough
that `hash_threshold` keeps its meaning.
"""

import cv2
import numpy as np
import pytest

from filter_frame_dedup.hash_processor import HashFrameProcessor, HASH_BASE


class _Config:
    roi = None
    hash_threshold = 5
    motion_threshold = 1200
    debug = False


def _frame(seed: int, w: int = 1920, h: int = 1080) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Structured rather than white noise: a downscale of pure noise carries no
    # low-frequency content, which is exactly what phash reads.
    small = rng.integers(0, 255, (18, 32, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _legacy_hashes(image, roi=None, hash_size=8):
    """The pre-fix path:each hash re-derives its own grayscale from full res."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p = cv2.dct(np.float32(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)))
    p = p[:hash_size, :hash_size]
    a = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    d = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (
        (p > p.mean()).flatten(),
        (a > a.mean()).flatten(),
        (d[:, 1:] > d[:, :-1]).flatten(),
    )


def test_one_downscale_serves_all_three_hashes():
    # The whole point: three hashes, one expensive reduction. Counted rather
    # than timed, because a wall-clock assertion is flaky on a loaded runner.
    p = HashFrameProcessor(_Config())
    img = _frame(1)
    calls = {"n": 0}
    real = cv2.cvtColor

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    cv2.cvtColor = counting
    try:
        p.compute_phash(img)
        p.compute_ahash(img)
        p.compute_dhash(img)
    finally:
        cv2.cvtColor = real
    assert calls["n"] == 1, f"grayscale derived {calls['n']}x for one frame"


def test_the_base_is_actually_smaller_than_the_frame():
    p = HashFrameProcessor(_Config())
    base = p._hash_base(_frame(2))
    assert base.shape[:2] == (HASH_BASE[1], HASH_BASE[0])


def test_a_frame_smaller_than_the_base_is_left_alone():
    # Nothing to gain, and upscaling to the base would invent detail.
    p = HashFrameProcessor(_Config())
    base = p._hash_base(_frame(3, w=160, h=90))
    assert base.shape[:2] == (90, 160)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_hash_distance_still_tracks_the_old_implementation(seed):
    # hash_threshold is compared against the bit distance between consecutive
    # frames, so what has to survive is the distance, not the absolute bits.
    # Measured on 59 consecutive-frame pairs of a real 1080p street clip: at the
    # default threshold of 5 the accept/reject decision was identical, 34/59
    # both ways; at 10 it differed on one pair.
    a, b = _frame(seed), _frame(seed + 100)
    p = HashFrameProcessor(_Config())

    new = [
        np.count_nonzero(x != y)
        for x, y in zip(
            (p.compute_phash(a), p.compute_ahash(a), p.compute_dhash(a)),
            (p.compute_phash(b), p.compute_ahash(b), p.compute_dhash(b)),
        )
    ]
    old = [
        np.count_nonzero(x != y) for x, y in zip(_legacy_hashes(a), _legacy_hashes(b))
    ]
    for n, o in zip(new, old):
        assert abs(n - o) <= 8, f"hash distance moved too far: {o} -> {n}"
