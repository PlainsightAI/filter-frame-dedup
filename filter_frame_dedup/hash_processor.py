import cv2
import numpy as np
import time
from openfilter.filter_runtime.filter import FilterConfig

# Intermediate size the three hashes are derived from. Large enough that the
# INTER_AREA step below it still averages meaningfully, small enough that the
# expensive reduction happens once and on a small image.
HASH_BASE = (240, 135)


class HashFrameProcessor:
    """
    A class that handles hash-based frame processing and motion detection.
    """
    def __init__(self, config: FilterConfig):
        self.config = config
        self.prev_phash = None
        self.prev_ahash = None
        self.prev_dhash = None
        self.prev_frame = None
        self.last_saved_time = 0  # Initialize to 0 instead of current time
        # Hashes computed by the most recent should_process_frame call, promoted
        # by update_reference_frame so the accepted frame is not hashed twice.
        self._pending_phash = None
        self._pending_ahash = None
        self._pending_dhash = None
        self._pending_frame = None
        # One downscale per frame, shared by phash/ahash/dhash. See _hash_base.
        self._base = None
        self._base_for = None

    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the region of interest (ROI) from the image.
        If ROI is None, returns the entire image.

        Args:
            image: Input image in BGR format

        Returns:
            Extracted ROI from the image or entire image if ROI is None
        """
        if self.config.roi is None:
            return image
        x, y, w, h = self.config.roi
        return image[y:y+h, x:x+w]

    # Cheap intermediate the three hashes share. INTER_AREA is what makes the
    # hashes stable, but reducing 1920x1080 straight to 32x32 with it averages
    # over ~60x60 kernels and measured 7.52 ms per call on a 1080p frame. Three
    # hashes meant paying that three times, 15.17 ms per frame, which is why
    # frame-dedup showed up as the pipeline bottleneck at 43.8 ms/frame while
    # the detector it protects cost 4.1 ms (PLAT-1471).
    #
    # A cheap NEAREST step down to HASH_BASE first, then INTER_AREA from there,
    # costs 0.39 ms for all three: 39x less, and the second stage still does the
    # averaging the hash relies on.
    #
    # Cached per frame by identity, so should_process_frame's three calls and a
    # later re-hash of the same array reuse one downscale.
    def _hash_base(self, image: np.ndarray) -> np.ndarray:
        if self._base_for is image and self._base is not None:
            return self._base
        roi = self.extract_roi(image)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w > HASH_BASE[0] and h > HASH_BASE[1]:
            gray = cv2.resize(gray, HASH_BASE, interpolation=cv2.INTER_NEAREST)
        self._base_for, self._base = image, gray
        return gray

    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> np.ndarray:
        """
        Compute the perceptual hash (phash) of the image.

        Args:
            image: Input image in BGR format
            hash_size: Size of the hash

        Returns:
            Computed phash of the image
        """
        base = self._hash_base(image)
        resized_image = cv2.resize(base, (32, 32), interpolation=cv2.INTER_AREA)
        dct_image = cv2.dct(np.float32(resized_image))
        dct_low_freq = dct_image[:hash_size, :hash_size]
        dct_mean = np.mean(dct_low_freq)
        return (dct_low_freq > dct_mean).flatten()

    def compute_ahash(self, image: np.ndarray, hash_size: int = 8) -> np.ndarray:
        """
        Compute the average hash (ahash) of the image.

        Args:
            image: Input image in BGR format
            hash_size: Size of the hash

        Returns:
            Computed ahash of the image
        """
        base = self._hash_base(image)
        resized_image = cv2.resize(base, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = resized_image.mean()
        return (resized_image > avg).flatten()

    def compute_dhash(self, image: np.ndarray, hash_size: int = 8) -> np.ndarray:
        """
        Compute the difference hash (dhash) of the image.

        Args:
            image: Input image in BGR format
            hash_size: Size of the hash

        Returns:
            Computed dhash of the image
        """
        base = self._hash_base(image)
        resized_image = cv2.resize(base, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized_image[:, 1:] > resized_image[:, :-1]
        return diff.flatten()

    def is_motion_detected(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """
        Detect motion between two frames by calculating their absolute differences.

        Args:
            prev_frame: Previous frame in BGR format
            curr_frame: Current frame in BGR format

        Returns:
            True if motion is detected, False otherwise
        """
        prev_roi = self.extract_roi(prev_frame)
        curr_roi = self.extract_roi(curr_frame)
        gray_prev = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_prev, gray_curr)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        non_zero_count = np.count_nonzero(thresh)
        return non_zero_count > self.config.motion_threshold

    def should_process_frame(self, image: np.ndarray) -> bool:
        """
        Determine if a frame should be processed based on hash changes and motion detection.

        Args:
            image: Current frame in BGR format

        Returns:
            True if frame should be processed, False otherwise
        """
        # Calculating hash values
        phash = self.compute_phash(image)
        ahash = self.compute_ahash(image)
        dhash = self.compute_dhash(image)

        # Stash so update_reference_frame can reuse them instead of recomputing
        self._pending_phash = phash
        self._pending_ahash = ahash
        self._pending_dhash = dhash
        self._pending_frame = image

        # Check motion detection
        motion_detected = self.prev_frame is None or self.is_motion_detected(self.prev_frame, image)

        # Check if there are significant changes in hash values
        hash_changed = (
            self.prev_phash is None or
            np.count_nonzero(self.prev_phash != phash) > self.config.hash_threshold or
            np.count_nonzero(self.prev_ahash != ahash) > self.config.hash_threshold or
            np.count_nonzero(self.prev_dhash != dhash) > self.config.hash_threshold
        )

        current_time = time.time()
        time_elapsed = current_time - self.last_saved_time

        # Debug logging
        if self.config.debug:
            print(f"Hash differences - pHash: {np.count_nonzero(self.prev_phash != phash) if self.prev_phash is not None else 'None'}, "
                  f"aHash: {np.count_nonzero(self.prev_ahash != ahash) if self.prev_ahash is not None else 'None'}, "
                  f"dHash: {np.count_nonzero(self.prev_dhash != dhash) if self.prev_dhash is not None else 'None'}")
            print(f"Motion detected: {motion_detected}")
            print(f"Hash changed: {hash_changed}")
            print(f"Time elapsed since last save: {time_elapsed:.2f}s")
            print(f"Should process: {(hash_changed or motion_detected) and (time_elapsed >= self.config.min_time_between_frames)}")

        # For the first frame (when last_saved_time is 0), always process if there are changes
        if self.last_saved_time == 0:
            return hash_changed or motion_detected

        return (hash_changed or motion_detected) and (time_elapsed >= self.config.min_time_between_frames)

    def update_reference_frame(self, image: np.ndarray):
        """
        Update the reference frame, hashes, and last saved time to the newly saved/key frame.

        Reuses the hashes computed by the preceding should_process_frame call for the
        same image; only recomputes if this image was not the one just evaluated.
        """
        if self._pending_frame is image:
            self.prev_phash = self._pending_phash
            self.prev_ahash = self._pending_ahash
            self.prev_dhash = self._pending_dhash
        else:
            self.prev_phash = self.compute_phash(image)
            self.prev_ahash = self.compute_ahash(image)
            self.prev_dhash = self.compute_dhash(image)
        self.prev_frame = image
        self.last_saved_time = time.time() 