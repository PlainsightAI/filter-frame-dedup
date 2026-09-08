---
title: FrameSelect
sidebar_label: Overview
sidebar_position: 1
slug: /filters/frame-select/overview
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

The filter uses a sequential, multi-stage processing pipeline built from a configurable list of active processors to detect and save unique frames:

1. **Pipeline Ordering and Execution**
   - Active stages and execution order are defined via `active_processors` (e.g., `["motion_gate", "hash_dedup", "ssim_dedup", "model_dedup"]`).
   - If a frame fails any active check, pipeline execution stops immediately, bypassing more expensive downstream checks (such as the model forward pass).

2. **Ultra-fast Motion Gatekeeper Stage (`motion_gate`)**
   - An extremely fast first-pass stage that computes average pixel intensity differences (deltas) between consecutive frames.
   - Can optionally run on a grid patchify mode (defined by `motion_gate_patch_grid_size`).
   - Acts as a coarse pre-filter to reject static or near-static frames with zero GPU overhead.

3. **Hash-based Detection Stage (`hash_dedup`)**
   - Computes three robust image hashes (Perceptual, Average, and Difference Hashing).
   - Compares current hashes to the last saved frame to identify structural alterations.
   - Integrates a pixel-level `motion_threshold` change count to capture local motion.

4. **SSIM-based Stage (`ssim_dedup`)**
   - Uses Structural Similarity Index (SSIM) to evaluate pixel structure and luminance similarity.
   - Supports grid-based patch partitioning (`ssim_patch_grid_size`) for local similarity analysis.
   - Prevents saving visually redundant frames.

5. **Model-based Feature Cosine Stage (`model_dedup`)**
   - Extracts deep features from vision models (e.g., ResNet or DINOv3) hosted on Hugging Face.
   - Computes cosine similarity of features against the last saved frame.
   - Catches complex, non-local semantic visual changes.

6. **Deferred Reference Frame Architecture**
   - All check operations are pure. No processor mutates its reference frame, hashes, or stashed features unless a frame is fully accepted by all active steps.
   - This ensures that slowly-changing footage is correctly compared against the **last saved frame** rather than the previous frame (which would otherwise suffer from undetected drift).

7. **Output Stage**
   - Saves accepted unique frames sequentially to `/output` (if `save_images=True`).
   - Forwards frames asynchronously in a special `deduped` side-channel (if `forward_deduped_frames=True`).

## Structure
The filtering pipeline is composed of multiple configurable, sequential processor stages:

- **Video Input (VideoIn)**: Reads the input video frames.
- **FastMotionGatekeeper (`motion_gate`)**: An ultra-fast, lightweight frame differencing stage that acts as a first-pass gatekeeper. Supports optional patch-based grids.
- **HashFrameProcessor (`hash_dedup`)**: Computes and compares three types of image hashes (pHash, aHash, dHash) combined with motion detection.
- **SSIMProcessor (`ssim_dedup`)**: Refines frame selection by comparing SSIM scores globally or across a patch-based grid.
- **ModelProcessor (`model_dedup`)**: Uses high-dimensional features extracted from Hugging Face models (e.g. `facebook/dinov3` or `microsoft/resnet-18`) and cosine similarity thresholds.
- **Output**: Saves accepted unique frames to the specified directory.

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
| `active_processors` | `string[]` \| `null` | `None` | Ordered list of pipeline stages to execute. Available: `["motion_gate", "hash_dedup", "ssim_dedup", "model_dedup"]`. If `None`, legacy defaults are used. Empty list raises. |
| `use_hash_dedup` | `boolean` | `true` | Use hash-based deduplication (when `active_processors` is `None`) |
| `use_model_dedup` | `boolean` | `false` | Use model-based deduplication (when `active_processors` is `None`) |
| `hash_threshold` | `int` | `5` | Minimum hash difference to consider a frame unique |
| `motion_threshold` | `int` | `1200` | Minimum motion intensity to consider for processing |
| `min_time_between_frames` | `float` | `1.0` | Minimum time (in seconds) between saved frames |
| `ssim_threshold` | `float` | `0.90` | SSIM score threshold (lower = more dissimilar) |
| `ssim_patch_grid_size` | `int` | `1` | Grid dimension L for LxL patch-based SSIM |
| `model_dedup_threshold` | `float` | `0.90` | Cosine similarity threshold for model features |
| `model_hf_id` | `string` | `"facebook/dinov3-vits16-pretrain-lvd1689m"` | Hugging Face model path/ID for feature extraction |
| `motion_gate_pixel_delta_threshold` | `float` | `1.5` | Pixel delta threshold for motion gatekeeper |
| `motion_gate_eval_width` | `int` | `480` | Resized evaluation width for motion gatekeeper |
| `motion_gate_patch_grid_size` | `int` | `1` | Grid dimension L for LxL patch-based motion gating |
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