---
title: FrameSelect
sidebar_label: Overview
sidebar_position: 1
---

import Admonition from '@theme/Admonition';

# FrameSelect

The `FilterFrameDedup` is a sophisticated filter designed to intelligently reduce redundant frames in video streams. It uses multiple detection methods (hashing, motion analysis, and SSIM comparison) to identify and save only frames that represent significant visual changes, making it ideal for keyframe extraction, storage optimization, and intelligent video sampling.

## Features

- **Multi-Method Detection**:
  - Perceptual hashing (pHash, aHash, dHash) for structural change detection
  - Motion analysis for pixel-level change detection
  - SSIM (Structural Similarity Index) for detailed visual comparison

- **Intelligent Filtering**:
  - Configurable thresholds for fine-tuning sensitivity
  - Minimum time intervals between saved frames
  - Region of Interest (ROI) support for focused processing

- **Advanced Output Options**:
  - Forward deduplicated frames in side channels (accessible via `localhost:8000/deduped`)
  - Forward upstream data from other filters
  - Configurable output channels and metadata
  - Side channel only emits when frames are actually saved (asynchronous)

- **Performance Optimized**:
  - Lightweight processing with minimal overhead
  - Debug mode for parameter tuning
  - Support for both real-time and batch processing

## Example Configuration

```python
# Basic frame deduplication
{
    "hash_threshold": 5,
    "motion_threshold": 1200,
    "min_time_between_frames": 1.0,
    "ssim_threshold": 0.90,
    "output_folder": "/output",
    "debug": False
}

# High sensitivity for detailed keyframes
{
    "hash_threshold": 3,
    "motion_threshold": 800,
    "min_time_between_frames": 0.5,
    "ssim_threshold": 0.85,
    "roi": (100, 100, 400, 300),
    "forward_deduped_frames": True
}

# Low sensitivity for major scene changes
{
    "hash_threshold": 10,
    "motion_threshold": 2000,
    "min_time_between_frames": 5.0,
    "ssim_threshold": 0.95,
    "forward_upstream_data": True
}
```

## Sample Pipelines

### 1. Security Camera Keyframe Extraction

**Use Case**: Extract keyframes from security camera footage for efficient storage and review

```python
# Pipeline: VideoIn → FilterFrameDedup → Webvis
from openfilter import Filter

# Video source configuration
video_config = {
    "sources": "rtsp://security-camera.company.com:554/stream",
    "outputs": "tcp://127.0.0.1:5550"
}

# Frame deduplication for keyframes
dedup_config = {
    "sources": "tcp://127.0.0.1:5550",
    "outputs": "tcp://127.0.0.1:5551",
    "hash_threshold": 5,
    "motion_threshold": 1200,
    "min_time_between_frames": 2.0,
    "ssim_threshold": 0.90,
    "output_folder": "/security_keyframes",
    "forward_deduped_frames": True,
    "debug": True
}

# Webvis for monitoring
webvis_config = {
    "sources": "tcp://127.0.0.1:5551",
    "outputs": "tcp://127.0.0.1:8080"
}

# Run the pipeline
filters = [
    Filter("VideoIn", video_config),
    Filter("FilterFrameDedup", dedup_config),
    Filter("Webvis", webvis_config)
]

Filter.run_multi(filters, exit_time=3600.0)  # 1 hour

# View results in Webvis at: 
# - http://localhost:8080/main (all processed frames)
# - http://localhost:8080/deduped (only saved keyframes)
# Deduplicated frames saved to: /security_keyframes/
```

## How it Works

The filter uses a multi-stage approach to detect and save unique frames:

1. **Frame Input Stage**
   - Receives video frames from the input source
   - Supports both file-based and stream-based input
   - Can process frames in real-time or from a video file

2. **Hash-based Detection Stage**
   - Computes three types of image hashes:
     - Perceptual Hash (pHash): Detects structural changes using DCT
     - Average Hash (aHash): Detects overall image changes
     - Difference Hash (dHash): Detects edge and gradient changes
   - Compares current frame hashes with previous frame hashes
   - Detects significant changes based on hash differences

3. **Motion Detection Stage**
   - Analyzes pixel-level differences between consecutive frames
   - Uses absolute difference and thresholding to detect motion
   - Helps identify frames with significant movement

4. **SSIM-based Refinement Stage**
   - Uses Structural Similarity Index (SSIM) for detailed comparison
   - Provides a more nuanced similarity score between frames
   - Helps prevent saving frames that are too similar

5. **Frame Selection Criteria**
   A frame is saved if it meets ALL of these conditions:
   - Hash differences exceed the threshold OR motion is detected
   - SSIM score is below the threshold (frames are different enough)
   - Minimum time has elapsed since the last saved frame

6. **Output Stage**
   - Saves selected frames to the specified output directory (if `save_images=True`)
   - Maintains a counter for frame numbering
   - Updates timing information for frame selection
   - Can operate in "detection-only" mode when `save_images=False`

## Structure
The filtering pipeline is composed of multiple stages:

- Video Input (VideoIn): Reads the input video file
- HashFrameProcessor: Processes frames to detect motion or significant hash changes
- SSIMProcessor: Further refines the frame selection by comparing SSIM scores
- Output: Saves selected frames to the specified directory

## Example Output

### When `save_images=True` (default)
Saved frames are written to disk using sequential names:

```
/output/
├── frame_000001.jpg
├── frame_000002.jpg
└── ...
```

Only frames that pass all deduplication filters are saved.

### When `save_images=False`
The filter operates in "detection-only" mode:
- No files are written to disk
- Deduplication logic still runs and updates timing
- Side channels (`deduped`) still work and contain frames that would have been saved
- Useful for real-time processing without storage overhead

## Side Channel: Deduplicated Frames

When `forward_deduped_frames` is enabled, the filter creates a special side channel called `deduped` that contains only the frames that were actually saved. This channel is **asynchronous** - it only emits data when a frame meets all the deduplication criteria and gets saved to disk.

### Key Features of the Deduped Channel:

- **Asynchronous Operation**: Only emits when frames are actually saved, not for every input frame
- **Rich Metadata**: Each deduped frame includes:
  - `deduped`: Boolean flag indicating the frame was saved
  - `frame_number`: Sequential number of the saved frame
  - `saved_path`: Full path to the saved file on disk
  - `original_frame_id`: Original frame identifier for tracking

- **Webvis Visualization**: Access the deduped channel at `http://localhost:8000/deduped`
- **Real-time Monitoring**: Perfect for monitoring keyframe extraction in real-time

### Example Usage:

```python
# Enable side channel forwarding
dedup_config = {
    "sources": "tcp://127.0.0.1:5550",
    "outputs": "tcp://127.0.0.1:5551",
    "forward_deduped_frames": True,  # Enable side channel
    "output_folder": "/keyframes"
}

# In Webvis, you'll see:
# - http://localhost:8000/main (all processed frames)
# - http://localhost:8000/deduped (only saved keyframes)
```

### Channel Behavior:

| Channel | Content | Frequency | Use Case |
|---------|---------|-----------|----------|
| `main` | All processed frames | Every input frame | General processing pipeline |
| `deduped` | Only saved frames | Only when frame is saved | Keyframe monitoring, storage verification |

## When to Use

Use this filter when:

- You need to extract keyframes or snapshots from a long video
- You want to avoid duplicate-looking frames in downstream storage or processing
- You want a low-overhead way to sample frames from video streams

## Configuration Reference

### Required Configuration

| Key | Type | Description |
|-----|------|-------------|
| `sources` | `string[]` | Input sources (e.g., `tcp://127.0.0.1:5550`) |

### Optional Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `id` | `string` | _auto_ | Filter instance identifier |
| `outputs` | `string[]` | _required_ | Output destinations |
| `hash_threshold` | `int` | `5` | Minimum hash difference to consider a frame unique |
| `motion_threshold` | `int` | `1200` | Minimum motion intensity to consider for processing |
| `min_time_between_frames` | `float` | `1.0` | Minimum time (in seconds) between saved frames |
| `ssim_threshold` | `float` | `0.90` | SSIM score threshold (lower = more dissimilar) |
| `roi` | `tuple` \| `null` | `None` | ROI as `(x, y, width, height)` or `None` for full frame |
| `output_folder` | `string` | `"/output"` | Directory to save selected frames |
| `save_images` | `boolean` | `true` | Whether to save images to disk |
| `debug` | `boolean` | `false` | Enable detailed logging |
| `forward_deduped_frames` | `boolean` | `false` | Forward deduplicated frames in a side channel |
| `forward_upstream_data` | `boolean` | `true` | Forward data from upstream filters |

### Threshold Guidelines

| Use Case | Hash Threshold | Motion Threshold | SSIM Threshold | Time Between |
|----------|----------------|------------------|----------------|--------------|
| High Detail Keyframes | 3-4 | 800-1000 | 0.85-0.88 | 0.5-1.0s |
| Security Surveillance | 5-6 | 1200-1500 | 0.90-0.92 | 2.0-3.0s |
| Content Analysis | 4-5 | 1000-1200 | 0.88-0.90 | 1.0-2.0s |
| Storage Optimization | 8-10 | 2000+ | 0.95+ | 5.0s+ |

<Admonition type="tip" title="Tip">
For optimal performance:
- Use ROI to focus on important areas and reduce processing time
- Lower thresholds for detailed analysis, higher for storage optimization
- Enable `forward_deduped_frames` for side channel access to keyframes
- Use `debug` mode to tune parameters for your specific use case
</Admonition>

<Admonition type="warning" title="Performance Considerations">
- Higher sensitivity (lower thresholds) increases processing time
- ROI processing is faster than full-frame analysis
- `forward_deduped_frames` creates additional output channels
- Debug mode adds logging overhead but helps with parameter tuning
</Admonition>