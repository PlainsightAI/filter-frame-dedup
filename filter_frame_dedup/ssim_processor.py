import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from openfilter.filter_runtime.filter import FilterConfig


class SSIMProcessor:
    """
    A class that handles SSIM-based frame processing.
    """
    def __init__(self, config: FilterConfig):
        self.config = config
        self.prev_frame = None

    def compute_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute the Structural Similarity Index (SSIM) between two frames.

        Args:
            frame1: First frame in BGR format
            frame2: Second frame in BGR format

        Returns:
            SSIM score between the two frames
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        return score

    def should_save_frame(self, image: np.ndarray) -> bool:
        """
        Determine if a frame should be saved based on SSIM comparison.

        Args:
            image: Current frame in BGR format

        Returns:
            True if frame should be saved, False otherwise
        """
        if self.prev_frame is None:
            self.prev_frame = image
            return True

        ssim_score = self.compute_ssim(self.prev_frame, image)
        self.prev_frame = image
        return ssim_score <= self.config.ssim_threshold 