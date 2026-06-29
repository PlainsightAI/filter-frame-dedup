import cv2
import numpy as np
import time
from openfilter.filter_runtime.filter import FilterConfig


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

    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> np.ndarray:
        """
        Compute the perceptual hash (phash) of the image.

        Args:
            image: Input image in BGR format
            hash_size: Size of the hash

        Returns:
            Computed phash of the image
        """
        roi = self.extract_roi(image)
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (32, 32), interpolation=cv2.INTER_AREA)
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
        roi = self.extract_roi(image)
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
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
        roi = self.extract_roi(image)
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
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

        # Update previous values
        self.prev_phash = phash
        self.prev_ahash = ahash
        self.prev_dhash = dhash
        self.prev_frame = image

        # For the first frame (when last_saved_time is 0), always process if there are changes
        if self.last_saved_time == 0:
            return hash_changed or motion_detected

        return (hash_changed or motion_detected) and (time_elapsed >= self.config.min_time_between_frames)

    def update_last_saved_time(self):
        """
        Update the last saved time when a frame is actually saved.
        This should be called by the filter after successfully saving a frame.
        """
        self.last_saved_time = time.time() 