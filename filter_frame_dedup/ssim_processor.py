import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from openfilter.filter_runtime.filter import FilterConfig
import logging

logger = logging.getLogger(__name__)

# scikit-image's default SSIM window. A frame smaller than this in either dimension
# cannot be compared at all, so it is also the floor a downscale must not cross.
SSIM_WIN_SIZE = 7


class SSIMProcessor:
    """
    A class that handles SSIM-based frame processing.
    """
    def __init__(self, config: FilterConfig):
        self.config = config
        self.prev_frame = None
        self.patch_grid_size = getattr(config, "ssim_patch_grid_size", 1)
        self.eval_width = int(getattr(config, "ssim_eval_width", 0) or 0)
        self._skipped_downscale_shapes: set[tuple[int, int]] = set()

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Grayscale, optionally downscaled to ``ssim_eval_width`` first.

        SSIM cost is linear in pixel count, and a dedup decision does not need full
        resolution: the motion gatekeeper in this same filter already evaluates at
        ``motion_gate_eval_width`` (480 by default) for exactly this reason. Resizing
        before the colour conversion rather than after keeps the conversion cheap too.

        Default is 0, meaning full resolution and byte-identical behaviour to before,
        because downscaling shifts SSIM scores upward (it smooths sensor noise) and the
        same ``ssim_threshold`` therefore keeps fewer frames. That is a recall change,
        so it is opt-in and must be measured per deployment rather than assumed.

        The downscale is skipped when it would drive either dimension below the SSIM
        window. ``normalize_config`` bounds the configured *width*, but it cannot see
        the frame: an extreme aspect ratio scales the height independently, so a wide,
        short frame reaches a valid width with a height of 1 or 2. Comparing that is
        impossible, ``compute_ssim`` would fail open on every frame, and the filter
        would silently stop deduplicating instead of failing. Comparing at full
        resolution is slower and correct, which is the right trade for an optimisation
        that cannot apply to this shape.
        """
        if self.eval_width > 0 and frame.shape[1] > self.eval_width:
            scale = self.eval_width / float(frame.shape[1])
            height = max(1, int(round(frame.shape[0] * scale)))
            if min(height, self.eval_width) >= SSIM_WIN_SIZE:
                frame = cv2.resize(frame, (self.eval_width, height), interpolation=cv2.INTER_AREA)
            else:
                shape = (frame.shape[0], frame.shape[1])
                if shape not in self._skipped_downscale_shapes:
                    # Once per shape, not once per frame: a stream holds its shape, and
                    # this would otherwise log on every frame at full frame rate.
                    self._skipped_downscale_shapes.add(shape)
                    logger.warning(
                        "ssim_eval_width=%d would reduce a %dx%d frame to %dx%d, below the "
                        "%dpx SSIM window; comparing at full resolution for this shape.",
                        self.eval_width, shape[1], shape[0], self.eval_width, height, SSIM_WIN_SIZE,
                    )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def compute_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute the Structural Similarity Index (SSIM) between two frames.

        Args:
            frame1: First frame in BGR format
            frame2: Second frame in BGR format

        Returns:
            SSIM score between the two frames
        """
        gray1 = self._to_gray(frame1)
        gray2 = self._to_gray(frame2)
        # scikit-image defaults to win_size=7 and raises "win_size exceeds image
        # extent" on anything smaller, so clamp it the way the patch-grid path below
        # already does. normalize_config rejects an eval width under the window, and
        # _to_gray skips a downscale that would cross it, but a caller can still hand
        # this method a tiny frame directly and a dedup filter should not crash the
        # pipeline over a comparison it could simply decline.
        min_dim = min(gray1.shape[0], gray1.shape[1], gray2.shape[0], gray2.shape[1])
        if min_dim < 3:
            # Too small to compare at all. Fail open: an uncomputable comparison must
            # KEEP the frame, and 0.0 is "definitely different" against any threshold.
            logger.warning("SSIM could not be computed for a %dpx frame; forcing keep frame.", min_dim)
            return 0.0
        win_size = min(SSIM_WIN_SIZE, min_dim)
        if win_size % 2 == 0:
            win_size = max(3, win_size - 1)
        # full=False: only the scalar score is used. full=True additionally builds a
        # float SSIM map the size of the input, which was allocated and discarded on
        # every frame.
        return ssim(gray1, gray2, full=False, win_size=win_size)

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
            gray1 = self._to_gray(self.prev_frame)
            gray2 = self._to_gray(image)

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
                    win_size = min(SSIM_WIN_SIZE, min_dim)
                    # win_size must be odd and >= 3 for ssim
                    if win_size % 2 == 0:
                        win_size = max(3, win_size - 1)
                    
                    if min_dim >= 3:
                        score = ssim(gray1_patch, gray2_patch, full=False, win_size=win_size)
                    else:
                        # Fallback for extremely small patches where SSIM cannot be computed.
                        # An uncomputable comparison must KEEP the frame (fail-open for a dedup
                        # filter), so force the keep path with a "definitely different" score.
                        score = 0.0
                        logger.warning("SSIM could not be computed for a very small patch; forcing keep frame.")

                    # If ANY patch passes (SSIM is <= ssim_threshold), the entire frame passes
                    if score <= self.config.ssim_threshold:
                        return True
            return False
        else:
            ssim_score = self.compute_ssim(self.prev_frame, image)
            # bool(): the comparison yields np.bool_, and the annotation says bool.
            return bool(ssim_score <= self.config.ssim_threshold)

    def update_reference_frame(self, image: np.ndarray):
        """
        Update the reference frame to the newly saved/key frame.
        """
        self.prev_frame = image