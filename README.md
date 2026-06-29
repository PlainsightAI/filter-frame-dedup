# FrameSelect

[![PyPI version](https://img.shields.io/pypi/v/filter-frame-dedup.svg?style=flat-square)](https://pypi.org/project/filter-frame-dedup/)
[![Docker Version](https://img.shields.io/docker/v/plainsightai/openfilter-frame-dedup?sort=semver)](https://hub.docker.com/r/plainsightai/openfilter-frame-dedup)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PlainsightAI/filter-frame-dedup/blob/main/LICENSE)

FrameSelect is a sophisticated OpenFilter component that intelligently reduces redundant frames in video streams. It uses multiple detection methods (hashing, motion analysis, and SSIM comparison) to identify and save only frames that represent significant visual changes, making it ideal for keyframe extraction, storage optimization, and intelligent video sampling.

## Features

- **Multi-Method Detection**: Uses perceptual hashing (pHash, aHash, dHash), motion analysis, and SSIM comparison
- **Intelligent Filtering**: Configurable thresholds for fine-tuning sensitivity
- **Side Channel Support**: Forward deduplicated frames in separate channels (accessible via `localhost:8000/deduped`)
- **Upstream Data Forwarding**: Preserve metadata from upstream filters
- **ROI Processing**: Focus on specific regions of interest
- **Performance Optimized**: Lightweight processing with minimal overhead

## Quick Start

### Basic Usage

```bash
# Set environment variables
export VIDEO_INPUT="../data/sample-video.mp4"
export OUTPUT_FOLDER="./output"
export HASH_THRESHOLD="5"
export MOTION_THRESHOLD="1200"

# Run the filter
python scripts/filter_usage.py
```

### Docker Usage

```bash
# Build and run with Docker Compose
docker-compose up
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_INPUT` | `../data/sample-video.mp4` | Input video file path |
| `OUTPUT_FOLDER` | `./output` | Directory to save deduplicated frames |
| `SAVE_IMAGES` | `true` | Whether to save images to disk |
| `HASH_THRESHOLD` | `5` | Minimum hash difference to consider unique |
| `MOTION_THRESHOLD` | `1200` | Minimum motion intensity threshold |
| `MIN_TIME_BETWEEN_FRAMES` | `1.0` | Minimum time between saved frames (seconds) |
| `SSIM_THRESHOLD` | `0.90` | SSIM score threshold (lower = more dissimilar) |
| `ROI` | `None` | Region of interest as `(x, y, width, height)` |
| `DEBUG` | `false` | Enable debug logging |
| `FORWARD_DEDUPED_FRAMES` | `false` | Forward deduplicated frames in side channel |
| `FORWARD_UPSTREAM_DATA` | `true` | Forward data from upstream filters |

### Configuration Examples

#### High Sensitivity (Detailed Keyframes)
```python
{
    "hash_threshold": 3,
    "motion_threshold": 800,
    "min_time_between_frames": 0.5,
    "ssim_threshold": 0.85,
    "forward_deduped_frames": True
}
```

#### Security Surveillance
```python
{
    "hash_threshold": 5,
    "motion_threshold": 1200,
    "min_time_between_frames": 2.0,
    "ssim_threshold": 0.90,
    "debug": True
}
```

#### Storage Optimization
```python
{
    "hash_threshold": 10,
    "motion_threshold": 2000,
    "min_time_between_frames": 5.0,
    "ssim_threshold": 0.95
}
```

#### Detection-Only Mode (No Disk Saving)
```python
{
    "hash_threshold": 5,
    "motion_threshold": 1200,
    "min_time_between_frames": 1.0,
    "ssim_threshold": 0.90,
    "save_images": False,
    "forward_deduped_frames": True
}
```

## Sample Pipelines

### 1. Security Camera Keyframe Extraction

```python
from openfilter import Filter

# Pipeline: VideoIn → FilterFrameDedup → Webvis
filters = [
    Filter("VideoIn", {
        "sources": "rtsp://security-camera.company.com:554/stream",
        "outputs": "tcp://127.0.0.1:5550"
    }),
    Filter("FilterFrameDedup", {
        "sources": "tcp://127.0.0.1:5550",
        "outputs": "tcp://127.0.0.1:5551",
        "hash_threshold": 5,
        "motion_threshold": 1200,
        "min_time_between_frames": 2.0,
        "ssim_threshold": 0.90,
        "output_folder": "/security_keyframes",
        "forward_deduped_frames": True,
        "debug": True
    }),
    Filter("Webvis", {
        "sources": "tcp://127.0.0.1:5551",
        "outputs": "tcp://127.0.0.1:8080"
    })
]

Filter.run_multi(filters, exit_time=3600.0)  # 1 hour

# View results in Webvis:
# - http://localhost:8080/main (all processed frames)
# - http://localhost:8080/deduped (only saved keyframes)
```

### 2. Content Analysis with ROI

```python
# Pipeline: VideoIn → FilterFrameDedup → FilterCrop → Webvis
filters = [
    Filter("VideoIn", {
        "sources": "file://content_video.mp4",
        "outputs": "tcp://127.0.0.1:5550"
    }),
    Filter("FilterFrameDedup", {
        "sources": "tcp://127.0.0.1:5550",
        "outputs": "tcp://127.0.0.1:5551",
        "hash_threshold": 3,
        "motion_threshold": 800,
        "min_time_between_frames": 0.5,
        "ssim_threshold": 0.85,
        "roi": (100, 100, 800, 600),
        "output_folder": "/content_keyframes",
        "forward_deduped_frames": True
    }),
    Filter("FilterCrop", {
        "sources": "tcp://127.0.0.1:5551",
        "outputs": "tcp://127.0.0.1:5552",
        "polygon_points": "[[(100, 100), (700, 100), (700, 500), (100, 500)]]",
        "output_prefix": "thumbnail_",
        "topic_mode": "main_only"
    }),
    Filter("Webvis", {
        "sources": "tcp://127.0.0.1:5552",
        "outputs": "tcp://127.0.0.1:8080"
    })
]

Filter.run_multi(filters, exit_time=1800.0)  # 30 minutes

# View results in Webvis:
# - http://localhost:8080/thumbnail_main (cropped frames)
# - http://localhost:8080/deduped (keyframes before cropping)
```

## Use Cases

### 1. Security Surveillance
Extract meaningful keyframes from 24/7 security camera footage for efficient storage and review.

**Configuration:**
```bash
export HASH_THRESHOLD="5"
export MOTION_THRESHOLD="1200"
export MIN_TIME_BETWEEN_FRAMES="2.0"
export FORWARD_DEDUPED_FRAMES="true"
```

### 2. Content Analysis
Extract keyframes from video content for automated thumbnail generation and scene analysis.

**Configuration:**
```bash
export HASH_THRESHOLD="3"
export MOTION_THRESHOLD="800"
export MIN_TIME_BETWEEN_FRAMES="0.5"
export ROI="(100, 100, 800, 600)"
```

### 3. Live Stream Processing
Process live video streams with real-time deduplication and analytics.

**Configuration:**
```bash
export HASH_THRESHOLD="4"
export MOTION_THRESHOLD="1000"
export MIN_TIME_BETWEEN_FRAMES="1.0"
export FORWARD_UPSTREAM_DATA="true"
export DEBUG="true"
```

### 4. Storage Optimization
Process large video files to extract only unique frames for storage optimization.

**Configuration:**
```bash
export HASH_THRESHOLD="10"
export MOTION_THRESHOLD="2000"
export MIN_TIME_BETWEEN_FRAMES="5.0"
export SSIM_THRESHOLD="0.95"
```

## Side Channel: Deduplicated Frames

The filter supports a special **side channel** called `deduped` that contains only the frames that were actually saved. This channel is **asynchronous** - it only emits data when a frame meets all deduplication criteria and gets saved to disk.

### Key Features:

- **Asynchronous Operation**: Only emits when frames are actually saved, not for every input frame
- **Webvis Visualization**: Access at `http://localhost:8000/deduped`
- **Rich Metadata**: Each frame includes deduplication status, frame number, and saved path
- **Real-time Monitoring**: Perfect for monitoring keyframe extraction

### Channel Comparison:

| Channel | Content | Frequency | Webvis URL |
|---------|---------|-----------|------------|
| `main` | All processed frames | Every input frame | `localhost:8000/main` |
| `deduped` | Only saved frames | Only when saved | `localhost:8000/deduped` |

### Enable Side Channel:

```python
{
    "forward_deduped_frames": True,  # Enable side channel
    "output_folder": "/keyframes"
}
```

## How It Works

The filter uses a multi-stage approach:

1. **Hash Analysis**: Computes perceptual, average, and difference hashes
2. **Motion Detection**: Analyzes pixel-level differences between frames
3. **SSIM Comparison**: Uses Structural Similarity Index for detailed comparison
4. **Frame Selection**: Saves frames that meet all criteria:
   - Hash differences exceed threshold OR motion is detected
   - SSIM score is below threshold
   - Minimum time has elapsed since last save
5. **Side Channel Output**: If enabled, forwards saved frames to `deduped` channel

## Output

### Saved Frames (when `save_images=True`)
Frames are saved to the specified directory with sequential naming:
```
/output/
├── frame_000001.jpg
├── frame_000002.jpg
└── ...
```

### Detection-Only Mode (when `save_images=False`)
When `save_images=False`, the filter operates in detection-only mode:
- No files are written to disk
- Deduplication logic still runs and updates timing
- Side channels (`deduped`) still work and contain frames that would have been saved
- Useful for real-time processing without storage overhead

### Side Channels
When `forward_deduped_frames` is enabled:
- **Main channel**: All processed frames (accessible via `localhost:8000/main`)
- **Deduped channel**: Only saved frames with metadata (accessible via `localhost:8000/deduped`)
  - **Asynchronous**: Only emits when frames are actually saved
  - **Rich Metadata**: Includes frame number, saved path, and deduplication status
  - **Real-time Monitoring**: Perfect for monitoring keyframe extraction

### Metadata
Deduplicated frames include:
- `deduped`: Boolean flag indicating frame was saved
- `frame_number`: Sequential frame number
- `saved_path`: Path to saved file
- `original_frame_id`: Original frame identifier

## Debug Mode

Enable debug mode for detailed processing information:
```bash
export DEBUG="true"
```

Debug output shows:
- Hash differences between frames
- Motion detection results
- SSIM scores
- Frame acceptance/rejection decisions
- Timing information

## Performance Tuning

### Threshold Guidelines

| Use Case | Hash Threshold | Motion Threshold | SSIM Threshold | Time Between |
|----------|----------------|------------------|----------------|--------------|
| High Detail Keyframes | 3-4 | 800-1000 | 0.85-0.88 | 0.5-1.0s |
| Security Surveillance | 5-6 | 1200-1500 | 0.90-0.92 | 2.0-3.0s |
| Content Analysis | 4-5 | 1000-1200 | 0.88-0.90 | 1.0-2.0s |
| Storage Optimization | 8-10 | 2000+ | 0.95+ | 5.0s+ |

### Performance Tips

- Use ROI to focus on important areas and reduce processing time
- Lower thresholds for detailed analysis, higher for storage optimization
- Enable `forward_deduped_frames` for side channel access to keyframes
- Use `debug` mode to tune parameters for your specific use case

## Troubleshooting

### Common Issues

**Too many saved frames:**
- Increase `ssim_threshold` (closer to 1.0)
- Increase `min_time_between_frames`
- Increase `hash_threshold`

**Too few saved frames:**
- Decrease `ssim_threshold` (closer to 0.0)
- Decrease `hash_threshold` and `motion_threshold`
- Decrease `min_time_between_frames`

**High CPU usage:**
- Increase `hash_threshold` and `motion_threshold`
- Use ROI to reduce processing area
- Increase `min_time_between_frames`

## Requirements

Install dependencies:
```bash
make run
```

Or install manually:
```bash
pip install -r requirements.txt
```

## Documentation

For more detailed information, configuration examples, and advanced usage scenarios, see the [comprehensive documentation](docs/overview.md).

## License

See LICENSE file for details.