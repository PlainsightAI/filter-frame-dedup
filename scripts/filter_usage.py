#!/usr/bin/env python3
"""
Filter Frame Deduplication Usage Example

This script demonstrates how to use FilterFrameDedup in a pipeline:
VideoIn → FilterFrameDedup → VideoOut + Webvis

Prerequisites:
- Sample video file (specify via VIDEO_INPUT environment variable)
- Output directory will be created automatically

Environment Variables:
- VIDEO_INPUT: Input video file path (default: ./data/sample-video.mp4)
- OUTPUT_VIDEO_PATH: Output video file path (default: ./output/output.mp4)
- OUTPUT_FPS: Output video frames per second (default: 30)
- WEBVIS_PORT: Port for Webvis visualization (default: 8000)
- FILTER_HASH_THRESHOLD: Hash difference threshold (default: 5)
- FILTER_MOTION_THRESHOLD: Motion detection threshold (default: 1200)
- FILTER_MIN_TIME_BETWEEN_FRAMES: Minimum time between saved frames in seconds (default: 1.0)
- FILTER_SSIM_THRESHOLD: SSIM similarity threshold (default: 0.90)
- FILTER_ROI: Region of interest as tuple (x, y, width, height) or None (default: None)
- FILTER_OUTPUT_FOLDER: Directory to save deduplicated frames (default: ./output)
- FILTER_DEBUG: Enable debug logging (default: False)
- FILTER_FORWARD_DEDUPED_FRAMES: Forward deduplicated frames in side channel (default: False)
- FILTER_FORWARD_UPSTREAM_DATA: Forward data from upstream filters (default: True)
"""

import logging
import os
import sys

# Add the filter module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import OpenFilter components
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.video_out import VideoOut
from openfilter.filter_runtime.filters.webvis import Webvis

# Import our frame deduplication filter
from filter_frame_dedup import FilterFrameDedup, FilterFrameDedupConfig


def load_config():
    """
    Load configuration from environment variables for the Frame Deduplication filter.
    """
    # Parse ROI from string if provided
    roi_str = os.getenv("FILTER_ROI")
    roi = None
    if roi_str and roi_str.lower() != "none":
        try:
            # Safely evaluate the tuple string
            roi = eval(roi_str)
            if not isinstance(roi, tuple) or len(roi) != 4:
                raise ValueError("ROI must be a tuple of 4 values (x, y, width, height)")
        except Exception as e:
            print(f"Warning: Invalid ROI format '{roi_str}': {e}. Using None.")
            roi = None
    
    config = {  
        "hash_threshold": int(os.getenv("FILTER_HASH_THRESHOLD", "5")),
        "motion_threshold": int(os.getenv("FILTER_MOTION_THRESHOLD", "1200")),
        "min_time_between_frames": float(os.getenv("FILTER_MIN_TIME_BETWEEN_FRAMES", "1.0")),
        "ssim_threshold": float(os.getenv("FILTER_SSIM_THRESHOLD", "0.90")),
        "roi": roi,
        "output_folder": os.getenv("FILTER_OUTPUT_FOLDER", "./output"),
        "debug": os.getenv("FILTER_DEBUG", "False").lower() == "true",
        "forward_deduped_frames": os.getenv("FILTER_FORWARD_DEDUPED_FRAMES", "False").lower() == "true",
        "forward_upstream_data": os.getenv("FILTER_FORWARD_UPSTREAM_DATA", "True").lower() == "true",
        "save_images": os.getenv("FILTER_SAVE_IMAGES", "False").lower() == "true",
    }
    
    return config


def main():
    """Run the FilterFrameDedup pipeline."""
    
    # Configuration from environment variables
    VIDEO_INPUT = os.getenv("VIDEO_INPUT", "./data/sample-video.mp4")
    OUTPUT_VIDEO_PATH = os.getenv("OUTPUT_VIDEO_PATH", "./output/output.mp4")
    FPS = int(os.getenv("OUTPUT_FPS", "30"))
    WEBVIS_PORT = int(os.getenv("WEBVIS_PORT", "8000"))
    
    # Load filter configuration
    config_values = load_config()
    dedup_config = FilterFrameDedupConfig(**config_values)
    
    print("=" * 60)
    print("Frame Deduplication Filter Pipeline")
    print("=" * 60)
    print(f"Input Video: {VIDEO_INPUT}")
    print(f"Output Video: {OUTPUT_VIDEO_PATH}")
    print(f"Output FPS: {FPS}")
    print(f"Webvis Port: {WEBVIS_PORT}")
    print(f"Output Folder: {dedup_config.output_folder}")
    print(f"Hash Threshold: {dedup_config.hash_threshold}")
    print(f"Motion Threshold: {dedup_config.motion_threshold}")
    print(f"Min Time Between Frames: {dedup_config.min_time_between_frames}s")
    print(f"SSIM Threshold: {dedup_config.ssim_threshold}")
    print(f"ROI: {dedup_config.roi}")
    print(f"Debug Mode: {dedup_config.debug}")
    print(f"Forward Deduped Frames: {dedup_config.forward_deduped_frames}")
    print(f"Forward Upstream Data: {dedup_config.forward_upstream_data}")
    print("=" * 60)

    # Configure output path with FPS if specified
    output_path = f"file://{OUTPUT_VIDEO_PATH}"
    if FPS:
        output_path += f"!fps={FPS}"

    # Create output directory if it doesn't exist
    os.makedirs(dedup_config.output_folder, exist_ok=True)

    # Define the filter pipeline
    pipeline = [
        # Input video source
        (
            VideoIn,
            {
                "id": "video_in",
                "sources": f"file://{VIDEO_INPUT}!loop",
                "outputs": "tcp://*:6000",
            }
        ),
        
        # Frame deduplication filter
        (
            FilterFrameDedup,
            FilterFrameDedupConfig(
                id="filter_frame_dedup",
                sources="tcp://127.0.0.1:6000",
                outputs="tcp://*:6002",
                mq_log="pretty",
                hash_threshold=dedup_config.hash_threshold,
                motion_threshold=dedup_config.motion_threshold,
                min_time_between_frames=dedup_config.min_time_between_frames,
                ssim_threshold=dedup_config.ssim_threshold,
                roi=dedup_config.roi,
                output_folder=dedup_config.output_folder,
                debug=dedup_config.debug,
                forward_deduped_frames=dedup_config.forward_deduped_frames,
                forward_upstream_data=dedup_config.forward_upstream_data,
                save_images=dedup_config.save_images,
            )
        ),
        
        # Web visualization
        (
            Webvis, 
            {
                "id": "webvis", 
                "sources": "tcp://127.0.0.1:6002", 
                "port": WEBVIS_PORT
            }
        )
    ]
    
    print("Starting pipeline...")
    print("Press Ctrl+C to stop")
    print(f"Webvis available at: http://localhost:{WEBVIS_PORT}")
    print(f"Deduplicated frames will be saved to: {dedup_config.output_folder}")
    print("=" * 60)
    
    try:
        # Run the pipeline
        Filter.run_multi(pipeline)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
