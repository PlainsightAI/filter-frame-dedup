import logging, os, cv2, time
from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame
from filter_frame_dedup.hash_processor import HashFrameProcessor
from filter_frame_dedup.ssim_processor import SSIMProcessor

__all__ = ["FilterFrameDedupConfig", "FilterFrameDedup"]

logger = logging.getLogger(__name__)


class FilterFrameDedupConfig(FilterConfig):
    """
    Configuration for the Frame Deduplication filter, loaded typically from environment variables.
    """
    hash_threshold:                     int = 5                             # Threshold for hash difference
    motion_threshold:                   int = 1200                          # Threshold for motion detection
    min_time_between_frames:            float = 1.0                         # Minimum time between saved frames
    ssim_threshold:                     float = 0.90                        # Threshold for SSIM comparison
    roi:                                tuple | None                        # Region of interest (x, y, width, height) or None for full image
    output_folder:                      str = "/output"                     # Output folder for saved frames
    save_images:                        bool = True                         # Whether to save images to disk
    debug:                              bool = False                        # Enable debug logging
    forward_deduped_frames:             bool = False                       # Forward deduplicated frames in a side channel
    forward_upstream_data:              bool = True                         # Forward data from upstream filters


class FilterFrameDedup(Filter):
    """
    A filter that:
    1) Detects duplicate frames using multiple methods (hash-based and SSIM)
    2) Saves only unique frames based on configurable thresholds
    3) Supports ROI-based processing (or full image when ROI is None)
    4) Maintains minimum time between saved frames
    """
    
    @classmethod
    def normalize_config(cls, config: FilterFrameDedupConfig):
        """
        Convert environment variables to correct data types, apply checks, etc.
        """
        # Call parent class normalize_config first to handle sources/outputs parsing
        config = super().normalize_config(config)
        
        # Handle nested config structure
        if isinstance(config, dict) and 'config' in config:
            config = config['config']

        # Convert string values to proper types before creating FilterFrameDedupConfig
        if isinstance(config, dict):
            # Convert numeric strings to proper types
            if 'hash_threshold' in config and isinstance(config['hash_threshold'], str):
                config['hash_threshold'] = int(config['hash_threshold'])
            if 'motion_threshold' in config and isinstance(config['motion_threshold'], str):
                config['motion_threshold'] = int(config['motion_threshold'])
            if 'min_time_between_frames' in config and isinstance(config['min_time_between_frames'], str):
                config['min_time_between_frames'] = float(config['min_time_between_frames'])
            if 'ssim_threshold' in config and isinstance(config['ssim_threshold'], str):
                config['ssim_threshold'] = float(config['ssim_threshold'])
            if 'roi' in config and isinstance(config['roi'], str):
                # Parse tuple string like "(100, 100, 200, 200)"
                roi_str = config['roi'].strip('()')
                config['roi'] = tuple(map(int, roi_str.split(', ')))

        # Convert to FilterFrameDedupConfig
        config = FilterFrameDedupConfig(**config)
        
        # Validate debug mode
        if isinstance(config.debug, str):   
            debug_str = config.debug.lower()
            if debug_str in ['true', 'false']:
                config.debug = debug_str == 'true'
            else:
                raise ValueError(f"Invalid debug mode: {config.debug}. It should be True or False.")
        elif not isinstance(config.debug, bool):
            raise ValueError(f"Invalid debug mode: {config.debug}. It should be True or False.")
        
        # Validate forward_deduped_frames mode
        if isinstance(config.forward_deduped_frames, str):   
            config.forward_deduped_frames = config.forward_deduped_frames.lower() == 'true'
        elif not isinstance(config.forward_deduped_frames, bool):
            raise ValueError(f"Invalid forward_deduped_frames mode: {config.forward_deduped_frames}. It should be True or False.")
        
        # Validate forward_upstream_data mode
        if isinstance(config.forward_upstream_data, str):   
            config.forward_upstream_data = config.forward_upstream_data.lower() == 'true'
        elif not isinstance(config.forward_upstream_data, bool):
            raise ValueError(f"Invalid forward_upstream_data mode: {config.forward_upstream_data}. It should be True or False.")
        
        # Validate save_images mode
        if isinstance(config.save_images, str):   
            config.save_images = config.save_images.lower() == 'true'
        elif not isinstance(config.save_images, bool):
            raise ValueError(f"Invalid save_images mode: {config.save_images}. It should be True or False.")

        # Validate thresholds
        if config.hash_threshold < 0:
            raise ValueError("Hash threshold must be non-negative")
        if config.motion_threshold < 0:
            raise ValueError("Motion threshold must be non-negative")
        if config.min_time_between_frames < 0:
            raise ValueError("Minimum time between frames must be non-negative")
        if not 0 <= config.ssim_threshold <= 1:
            raise ValueError("SSIM threshold must be between 0 and 1")

        # Validate ROI if provided
        if config.roi is not None:
            if len(config.roi) != 4:
                raise ValueError("ROI must be a tuple of 4 values (x, y, width, height)")
            x, y, w, h = config.roi
            if w <= 0 or h <= 0:
                raise ValueError("ROI width and height must be positive")

        return config

    def setup(self, config: FilterFrameDedupConfig):
        """
        Called once at the start of the filter's lifecycle.
        """
        logger.info("========= Setting up FilterFrameDedup =========")
        self.config = config
        
        # Initialize processors
        self.hash_processor = HashFrameProcessor(config)
        self.ssim_processor = SSIMProcessor(config)
        
        # Initialize counters
        self.processed_frame_count = 0
        self.frame_count = 1
        
        # Create output folder if it doesn't exist and save_images is enabled
        if config.save_images and not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)
            
        logger.info(f"FilterFrameDedup setup completed. Config: {config.__dict__}")

    def process(self, frames: dict[str, Frame]) -> dict[str, Frame]:
        """
        Process frames and determine if they should be saved based on motion detection and hash changes.

        Args:
            frames: Dictionary containing frames

        Returns:
            Updated frames dictionary with selected frame and optional side channels
        """
        # Get the main frame from the frames dictionary
        main_frame = frames.get('main')
        if main_frame is None or not main_frame.has_image:
            if self.config.debug:
                logger.info("No valid frame received")
            return frames

        # Increment the frame counter
        self.processed_frame_count += 1
        if self.config.debug:
            logger.info(f"Processing frame {self.processed_frame_count}")

        # Access the raw BGR image from the frame and create a copy for processing
        processed_image = main_frame.rw_bgr.image.copy()

        # Initialize output frames dictionary
        output_frames = {}
        
        # Always forward the main frame with the processed image first
        # Create a new frame with the processed image to maintain the original frame data
        processed_main_frame = Frame(
            image=processed_image,
            data=main_frame.data,
            format='BGR'
        )
        self.frame_count += 1
        output_frames['main'] = processed_main_frame
        
        # Forward upstream data if enabled
        if self.config.forward_upstream_data:
            # Copy all non-main frames from upstream
            for key, frame in frames.items():
                if key != 'main':
                    output_frames[key] = frame

        # First check if frame should be processed based on hash and motion
        if self.hash_processor.should_process_frame(processed_image):
            if self.config.debug:
                logger.info("Frame passed hash/motion check")
            # Then check if frame should be saved based on SSIM
            if self.ssim_processor.should_save_frame(processed_image):
                frame_path = None
                
                # Save frame to disk only if save_images is enabled
                if self.config.save_images:
                    frame_path = os.path.join(self.config.output_folder, f"frame_{self.frame_count:06d}.jpg")
                    lock_path = frame_path + '.lock'

                    try:
                        # Create lock file
                        with open(lock_path, 'x') as _:
                            # Write the processed image
                            cv2.imwrite(frame_path, processed_image)
                            # time.sleep(30) # for testing
                    finally:
                        # Always remove the lock file, even if writing fails
                        try:
                            os.remove(lock_path)
                        except:
                            pass
                    
                    if self.config.debug:
                        logger.info(f"Saved frame to {frame_path}")
                else:
                    if self.config.debug:
                        logger.info("Frame passed deduplication criteria but not saved (save_images=False)")
                
                # Update the last saved time only when frame is actually saved
                self.hash_processor.update_last_saved_time()
                
                # Forward deduplicated frame in side channel if enabled
                if self.config.forward_deduped_frames:
                    # Create a frame with the actual deduplicated image (the one that was saved)
                    # This creates an asynchronous channel that only contains frames that passed deduplication
                    deduped_frame = Frame(
                        image=processed_image,  # Use the processed image that was saved
                        data=main_frame.data.copy() if main_frame.data else {},
                        format='BGR'
                    )
                    # Add metadata about the deduplication
                    if deduped_frame.data is None:
                        deduped_frame.data = {}
                    deduped_frame.data['deduped'] = True
                    deduped_frame.data['frame_number'] = self.frame_count - 1
                    if frame_path:
                        deduped_frame.data['saved_path'] = frame_path
                    else:
                        deduped_frame.data['saved_path'] = None
                    deduped_frame.data['original_frame_id'] = getattr(main_frame.data, 'id', None) if main_frame.data else None
                    
                    output_frames['deduped'] = deduped_frame
                    
                    if self.config.debug:
                        logger.info("Forwarded deduplicated frame in side channel")
            else:
                if self.config.debug:
                    logger.info("Skipping frame due to high SSIM score")
        else:
            if self.config.debug:
                logger.info("Frame did not pass hash/motion check")

        # Ensure main topic comes first in the output dictionary
        if 'main' in output_frames:
            main_frame = output_frames.pop('main')
            return {'main': main_frame, **output_frames}
        
        return output_frames

    def shutdown(self):
        """
        Called once when the filter is shutting down.
        """
        logger.info("========= Shutting down FilterFrameDedup =========")
        logger.info(f"Total frames processed: {self.processed_frame_count}")
        logger.info(f"Total frames saved: {self.frame_count - 1}")
        logger.info("FilterFrameDedup shutdown complete.")


if __name__ == "__main__":
    FilterFrameDedup.run()
