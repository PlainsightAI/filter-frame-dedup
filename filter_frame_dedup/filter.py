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
    
    active_processors:                  list[str] | None = None             # List of strings naming the filters to use and their execution order
    use_hash_dedup:                     bool = True                       # Whether to use a hash-based deduplication method
    use_model_dedup:                    bool = False                       # Whether to use a model-based deduplication method
    
    # although cosine similarity ranges from -1 to 1, we only care about positive similarity for deduplication, so we set the threshold between 0 and 1
    model_dedup_threshold:              float = 0.9                         # Threshold for model-based deduplication
    model_hf_id:                        str = "facebook/dinov3-vits16-pretrain-lvd1689m"                         # Hugging Face model path for deduplication model
    
    # Motion gate parameters
    motion_gate_pixel_delta_threshold:  float = 1.5                         # Pixel delta threshold for motion gatekeeper
    motion_gate_eval_width:             int = 480                           # Evaluation width for motion gatekeeper
    motion_gate_patch_grid_size:        int = 1                             # Grid dimension L for LxL patch-based motion gating
    ssim_patch_grid_size:               int = 1                             # Grid dimension L for LxL patch-based SSIM comparison
    ssim_eval_width:                    int = 0                             # Downscale width for SSIM comparison; 0 = full resolution
    
class FilterFrameDedup(Filter):
    """
    A sequential, configurable frame deduplication filter that filters out duplicate frames 
    from video streams using multiple pluggable processors in a specified order.

    Processors:
    1) Motion Gatekeeper ('motion_gate'): An ultra-fast frame differencing stage that acts 
       as a first-pass gatekeeper. Supports patch-based grids.
    2) Hash Processor ('hash_dedup'): Combines perceptual (pHash), average (aHash), and 
       difference (dHash) hashing with motion thresholding to detect significant changes.
    3) SSIM Processor ('ssim_dedup'): Compares Structural Similarity Index (SSIM) values 
       globally or across a patch-based grid to identify visually distinct frames.
    4) Model Processor ('model_dedup'): Extracts high-dimensional features using a 
       Hugging Face vision/CNN model and measures cosine similarity to find duplicates.

    Features:
    - Pipeline Configuration: Define execution order via `active_processors` (e.g. 
      `['motion_gate', 'hash_dedup', 'ssim_dedup', 'model_dedup']`).
    - Deferred Reference Updates: Reference frames and features are updated only after 
      a frame is fully accepted by all stages in the active pipeline, ensuring robust 
      comparison against the last saved frame.
    - ROI Support: Crop images to a Region of Interest (ROI) before checking.
    - Time-Based Filtering: Enforces a minimum interval (`min_time_between_frames`) between saved frames.
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
                config['roi'] = tuple(map(int, [part.strip() for part in roi_str.split(',')]))
            if 'model_dedup_threshold' in config and isinstance(config['model_dedup_threshold'], str):
                config['model_dedup_threshold'] = float(config['model_dedup_threshold'])
            if 'motion_gate_pixel_delta_threshold' in config and isinstance(config['motion_gate_pixel_delta_threshold'], str):
                config['motion_gate_pixel_delta_threshold'] = float(config['motion_gate_pixel_delta_threshold'])
            if 'motion_gate_eval_width' in config and isinstance(config['motion_gate_eval_width'], str):
                config['motion_gate_eval_width'] = int(config['motion_gate_eval_width'])
            if 'use_hash_dedup' in config and isinstance(config['use_hash_dedup'], str):
                config['use_hash_dedup'] = config['use_hash_dedup'].lower() == 'true'
            if 'motion_gate_patch_grid_size' in config and isinstance(config['motion_gate_patch_grid_size'], str):
                config['motion_gate_patch_grid_size'] = int(config['motion_gate_patch_grid_size'])
            if 'ssim_patch_grid_size' in config and isinstance(config['ssim_patch_grid_size'], str):
                config['ssim_patch_grid_size'] = int(config['ssim_patch_grid_size'])
            if 'ssim_eval_width' in config and isinstance(config['ssim_eval_width'], str):
                config['ssim_eval_width'] = int(config['ssim_eval_width'])
            if 'active_processors' in config and isinstance(config['active_processors'], str):
                val = config['active_processors'].strip()
                if val.startswith('[') and val.endswith(']'):
                    val = val[1:-1]
                if val:
                    config['active_processors'] = [p.strip().strip("'\"") for p in val.split(',')]
                else:
                    config['active_processors'] = []
           
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

        # Validate use_hash_dedup mode
        if isinstance(config.use_hash_dedup, str):   
            config.use_hash_dedup = config.use_hash_dedup.lower() == 'true'
        elif not isinstance(config.use_hash_dedup, bool):
            raise ValueError(f"Invalid use_hash_dedup mode: {config.use_hash_dedup}. It should be True or False.")

        # Validate use_model_dedup mode
        if isinstance(config.use_model_dedup, str):   
            config.use_model_dedup = config.use_model_dedup.lower() == 'true'
        elif not isinstance(config.use_model_dedup, bool):
            raise ValueError(f"Invalid use_model_dedup mode: {config.use_model_dedup}. It should be True or False.")

        # Validate thresholds
        if config.hash_threshold < 0:
            raise ValueError("Hash threshold must be non-negative")
        if config.motion_threshold < 0:
            raise ValueError("Motion threshold must be non-negative")
        if config.min_time_between_frames < 0:
            raise ValueError("Minimum time between frames must be non-negative")
        if not 0 <= config.ssim_threshold <= 1:
            raise ValueError("SSIM threshold must be between 0 and 1")
        if not 0 <= config.model_dedup_threshold <= 1:
            raise ValueError("Model deduplication threshold must be between 0 and 1")
        # Delta is a mean absolute grayscale difference (0..255); a threshold above 255
        # can never be reached and would silently reject every frame, disabling dedup.
        if not 0 <= config.motion_gate_pixel_delta_threshold <= 255:
            raise ValueError("Motion gate pixel delta threshold must be between 0 and 255")
        if config.motion_gate_eval_width < 1:
            raise ValueError("Motion gate evaluation width must be at least 1")
        if config.motion_gate_patch_grid_size <= 0:
            raise ValueError("Motion gate patch grid size must be positive")
        if config.ssim_patch_grid_size <= 0:
            raise ValueError("SSIM patch grid size must be positive")
        # 0 means "full resolution", any other negative value is a mistake rather than
        # a smaller image, and would otherwise be silently ignored by the resize guard.
        # Below 7 there is nothing to compare: scikit-image's default SSIM window is
        # 7x7, and a dedup decision taken on a 6px-wide image is meaningless anyway.
        if config.ssim_eval_width < 0:
            raise ValueError("SSIM evaluation width must be 0 (full resolution) or positive")
        if 0 < config.ssim_eval_width < 7:
            raise ValueError(
                "SSIM evaluation width must be 0 (full resolution) or at least 7; "
                f"got {config.ssim_eval_width}, which is smaller than the SSIM window"
            )
        
        # Validate ROI if provided
        if config.roi is not None:
            if len(config.roi) != 4:
                raise ValueError("ROI must be a tuple of 4 values (x, y, width, height)")
            x, y, w, h = config.roi
            if w <= 0 or h <= 0:
                raise ValueError("ROI width and height must be positive")

        # Determine active processors from the legacy use_hash_dedup / use_model_dedup
        # flags when active_processors is None. This preserves v1.1.6 behavior: the
        # motion gate is NOT added here and stays opt-in via explicit active_processors.
        if config.active_processors is None:
            processors = []
            if config.use_hash_dedup:
                processors.append("hash_dedup")
            processors.append("ssim_dedup")
            if config.use_model_dedup:
                processors.append("model_dedup")
            config.active_processors = processors

        # Validate active_processors contains only valid options
        valid_processor_names = {"motion_gate", "motion_gatekeeper", "hash_dedup", "hash_processor", "ssim_dedup", "ssim_processor", "model_dedup", "model_processor"}
        for p in config.active_processors:
            if p not in valid_processor_names:
                raise ValueError(f"Invalid processor name in active_processors: '{p}'. Must be one of {sorted(list(valid_processor_names))}")

        # Canonicalize and deduplicate processor names to avoid redundant execution of same stages
        canonical_map = {
            "motion_gatekeeper": "motion_gate",
            "hash_processor": "hash_dedup",
            "ssim_processor": "ssim_dedup",
            "model_processor": "model_dedup"
        }
        canonicalized = [canonical_map.get(p, p) for p in config.active_processors]
        config.active_processors = list(dict.fromkeys(canonicalized))

        # An empty pipeline dedups nothing; reject it explicitly (e.g. [] or the env string "[]")
        if len(config.active_processors) == 0:
            raise ValueError("active_processors must contain at least one processor")

        # active_processors is authoritative: keep the legacy use_* flags in sync with it
        # so each processor's own guard (e.g. ModelProcessor.__init__, which raises when
        # use_model_dedup is False) agrees with the pipeline that was actually requested.
        config.use_hash_dedup = any(p == "hash_dedup" for p in config.active_processors)
        config.use_model_dedup = any(p == "model_dedup" for p in config.active_processors)

        return config

    def setup(self, config: FilterFrameDedupConfig):
        """
        Called once at the start of the filter's lifecycle.
        """
        logger.info("========= Setting up FilterFrameDedup =========")
        self.config = config
        
        # Initialize processors conditionally based on active_processors
        active_set = set(config.active_processors)
        
        # 1. Motion Gatekeeper
        if "motion_gate" in active_set or "motion_gatekeeper" in active_set:
            from filter_frame_dedup.motion_gate_processor import FastMotionGatekeeper
            self.motion_gatekeeper = FastMotionGatekeeper(
                pixel_delta_threshold=config.motion_gate_pixel_delta_threshold,
                eval_width=config.motion_gate_eval_width,
                patch_grid_size=config.motion_gate_patch_grid_size,
                roi=config.roi
            )
        else:
            self.motion_gatekeeper = None
            
        # 2. Hash Processor
        if "hash_dedup" in active_set or "hash_processor" in active_set:
            self.hash_processor = HashFrameProcessor(config)
        else:
            self.hash_processor = None
            
        # 3. SSIM Processor
        if "ssim_dedup" in active_set or "ssim_processor" in active_set:
            self.ssim_processor = SSIMProcessor(config)
        else:
            self.ssim_processor = None
            
        # 4. Model Processor
        if "model_dedup" in active_set or "model_processor" in active_set:
            logger.info("Model-based deduplication is enabled. Initializing ModelProcessor...")
            from filter_frame_dedup.model_processor import ModelProcessor
            try:
                self.model_processor = ModelProcessor(config)
            except Exception as e:
                logger.error(f"Failed to initialize ModelProcessor: {e}")
                raise
        else:
            self.model_processor = None
            logger.info("Model-based deduplication is disabled. Skipping ModelProcessor initialization.")
            
        # Initialize counters
        self.processed_frame_count = 0
        self.unique_frame_count = 0
        
        if config.save_images and not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)
            
        # Build the sequential processing pipeline in the requested order
        self.pipeline = []
        for processor_name in config.active_processors:
            if processor_name in ("motion_gate", "motion_gatekeeper"):
                self.pipeline.append({
                    "name": "Motion Gatekeeper",
                    "check": self.motion_gatekeeper.should_process_frame,
                    "update": self.motion_gatekeeper.update_reference_frame,
                    "image_type": "bgr"
                })
            elif processor_name in ("hash_dedup", "hash_processor"):
                self.pipeline.append({
                    "name": "Hash Processor",
                    "check": self.hash_processor.should_process_frame,
                    "update": self.hash_processor.update_reference_frame,
                    "image_type": "bgr"
                })
            elif processor_name in ("ssim_dedup", "ssim_processor"):
                self.pipeline.append({
                    "name": "SSIM Processor",
                    "check": self.ssim_processor.should_save_frame,
                    "update": self.ssim_processor.update_reference_frame,
                    "image_type": "bgr"
                })
            elif processor_name in ("model_dedup", "model_processor"):
                self.pipeline.append({
                    "name": "Model Processor",
                    "check": self.model_processor.frame_is_unique,
                    "update": self.model_processor.update_reference_frame,
                    "image_type": "rgb"
                })
                
        logger.info(f"FilterFrameDedup setup completed. Config: {config.__dict__}")
        logger.info(f"Using filters in order: {', '.join([step['name'] for step in self.pipeline])}")

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
        output_frames['main'] = processed_main_frame
        
        # Forward upstream data if enabled
        if self.config.forward_upstream_data:
            # Copy all non-main frames from upstream
            for key, frame in frames.items():
                if key != 'main':
                    output_frames[key] = frame

        # Run the sequential processing pipeline
        processed_image_rgb = None
        is_unique = True
        
        for step in self.pipeline:
            # Lazily convert to RGB only if a step requires it
            if step["image_type"] == "rgb":
                if processed_image_rgb is None:
                    processed_image_rgb = main_frame.rw_rgb.image.copy()
                img = processed_image_rgb
            else:
                img = processed_image
                
            if not step["check"](img):
                is_unique = False
                if self.config.debug:
                    logger.info(f"Skipping frame: failed {step['name']} check")
                break
            else:
                if self.config.debug:
                    logger.info(f"Frame passed {step['name']} check")

        if is_unique:
            self.unique_frame_count += 1
            frame_path = None
            if self.config.save_images:
                frame_path = os.path.join(self.config.output_folder, f"frame_{self.processed_frame_count:06d}.jpg")
                lock_path = frame_path + '.lock'

                try:
                    # Create lock file
                    with open(lock_path, 'x') as _:
                        # Write the processed image
                        cv2.imwrite(frame_path, processed_image)
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
            
            # Update reference frames for all steps in the pipeline that support it
            for step in self.pipeline:
                if step["update"] is not None:
                    img = processed_image_rgb if step["image_type"] == "rgb" else processed_image
                    step["update"](img)
            
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
                deduped_frame.data['frame_number'] = self.processed_frame_count
                if frame_path:
                    deduped_frame.data['saved_path'] = frame_path
                else:
                    deduped_frame.data['saved_path'] = None
                deduped_frame.data['original_frame_id'] = main_frame.data.get('meta', {}).get('id') if main_frame.data else None                        
                output_frames['deduped'] = deduped_frame
                
                if self.config.debug:
                    logger.info("Forwarded deduplicated frame in side channel")

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
        logger.info(f"Total unique frames: {self.unique_frame_count}")
        logger.info("FilterFrameDedup shutdown complete.")


if __name__ == "__main__":
    FilterFrameDedup.run()
