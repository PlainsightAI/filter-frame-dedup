import cv2
import numpy as np

class FastMotionGatekeeper:
    """
    Ultra-fast motion gatekeeper.
    Drops 100% frozen/static frames in <0.1ms using raw pixel deltas.

    This is one optional processor in the pipeline; its position and the
    processor that runs after it depend on active_processors, so no fixed
    stage number is assumed.
    """
    def __init__(self, pixel_delta_threshold: float = 1.5, eval_width: int = 480, patch_grid_size: int = 1, roi: tuple | None = None):
        self.pixel_delta_threshold = pixel_delta_threshold
        self.eval_width = eval_width
        self.patch_grid_size = patch_grid_size
        self.roi = roi
        self.prev_gray = None

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        # Crop to ROI (if configured) before evaluating, matching the other processors
        if self.roi is not None:
            x, y, w, h = self.roi
            frame = frame[y:y+h, x:x+w]
        # Downsample slightly to eval_width for ultra-fast CPU evaluation (preserves big motion)
        h, w = frame.shape[:2]
        scale = self.eval_width / float(w)
        # Clamp target height to >=1 so extreme aspect ratios (e.g. line-scan frames,
        # or eval_width=1) don't round to 0 and make cv2.resize raise.
        target_h = max(1, int(h * scale))
        small_frame = cv2.resize(frame, (self.eval_width, target_h), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    def should_process_frame(self, frame: np.ndarray) -> bool:
        gray = self._to_gray(frame)

        # Pure check: on the very first frame there is no reference to compare
        # against, so treat it as motion. The reference is set by
        # update_reference_frame only after the frame is accepted.
        if self.prev_gray is None:
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

            # If pixel fluctuation is higher than sensor noise, let the frame pass to the next processor
            if mean_diff >= self.pixel_delta_threshold:
                return True

            return False

    def update_reference_frame(self, frame: np.ndarray):
        self.prev_gray = self._to_gray(frame)
