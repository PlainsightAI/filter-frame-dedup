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
        self.patch_grid_size = getattr(config, "ssim_patch_grid_size", 1)

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
            return True

        if self.patch_grid_size > 1:
            L = self.patch_grid_size
            gray1 = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            sh, sw = gray1.shape[:2]
            patch_h = sh // L
            patch_w = sw // L
            
            # Check patch by patch
            for i in range(L):
                for j in range(L):
                    y_start = i * patch_h
                    y_end = sh if i == L - 1 else (i + 1) * patch_h
                    x_start = j * patch_w
                    x_end = sw if j == L - 1 else (j + 1) * patch_w
                    
                    gray1_patch = gray1[y_start:y_end, x_start:x_end]
                    gray2_patch = gray2[y_start:y_end, x_start:x_end]
                    
                    # Compute SSIM for this patch
                    # Use a smaller win_size if the patch size is small
                    min_dim = min(gray1_patch.shape[:2])
                    win_size = min(7, min_dim)
                    # win_size must be odd and >= 3 for ssim
                    if win_size % 2 == 0:
                        win_size = max(3, win_size - 1)
                    
                    if min_dim >= 3:
                        score, _ = ssim(gray1_patch, gray2_patch, full=True, win_size=win_size)
                    else:
                        # Fallback for extremely small patches where SSIM cannot be computed.
                        # An uncomputable comparison must KEEP the frame (fail-open for a dedup
                        # filter), so force the keep path with a "definitely different" score.
                        score = 0.0

                    # If ANY patch passes (SSIM is <= ssim_threshold), the entire frame passes
                    if score <= self.config.ssim_threshold:
                        return True
            return False
        else:
            ssim_score = self.compute_ssim(self.prev_frame, image)
            return ssim_score <= self.config.ssim_threshold 

    def update_reference_frame(self, image: np.ndarray):
        """
        Update the reference frame to the newly saved/key frame.
        """
        self.prev_frame = image