import cv2
import numpy as np

class FastMotionGatekeeper:
    """
    Ultra-fast Stage 1 Gatekeeper. 
    Drops 100% frozen/static frames in <0.1ms using raw pixel deltas.
    """
    def __init__(self, pixel_delta_threshold: float = 1.5, eval_width: int = 480, patch_grid_size: int = 1):
        self.pixel_delta_threshold = pixel_delta_threshold
        self.eval_width = eval_width
        self.patch_grid_size = patch_grid_size
        self.prev_gray = None

    def should_process_frame(self, frame: np.ndarray) -> bool:
        # Downsample slightly to 480p for ultra-fast CPU evaluation (preserves big motion, kills 0.05ms)
        h, w = frame.shape[:2]
        scale = self.eval_width / float(w)
        small_frame = cv2.resize(frame, (self.eval_width, int(h * scale)), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True

        if self.patch_grid_size > 1:
            L = self.patch_grid_size
            sh, sw = gray.shape[:2]
            patch_h = sh // L
            patch_w = sw // L
            
            # Check patch by patch
            for i in range(L):
                for j in range(L):
                    y_start = i * patch_h
                    y_end = sh if i == L - 1 else (i + 1) * patch_h
                    x_start = j * patch_w
                    x_end = sw if j == L - 1 else (j + 1) * patch_w
                    
                    gray_patch = gray[y_start:y_end, x_start:x_end]
                    prev_gray_patch = self.prev_gray[y_start:y_end, x_start:x_end]
                    
                    mean_diff = cv2.mean(cv2.absdiff(gray_patch, prev_gray_patch))[0]
                    if mean_diff >= self.pixel_delta_threshold:
                        return True
            return False
        else:
            # Calculate average per-pixel absolute change across the entire image
            mean_diff = cv2.mean(cv2.absdiff(gray, self.prev_gray))[0]

            # If pixel fluctuation is higher than sensor noise, pass to Stage 2 (Patchified SSIM)
            if mean_diff >= self.pixel_delta_threshold:
                return True

            return False

    def update_reference_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = self.eval_width / float(w)
        small_frame = cv2.resize(frame, (self.eval_width, int(h * scale)), interpolation=cv2.INTER_NEAREST)
        self.prev_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)