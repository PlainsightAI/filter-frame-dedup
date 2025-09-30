# Changelog
FrameSelect release notes

## [Unreleased]

## v1.1.1 - 2025-09-27

### Changed
- **Updated Documentation**

## v1.1.0 - 2025-09-16

### Added
- **Side Channel Support**
  - Added `forward_deduped_frames` option to forward deduplicated frames in a separate channel
  - Deduplicated frames are available on the 'deduped' topic with metadata including frame number and saved path
  - Enables asynchronous processing where deduped channel only emits when frames are actually saved

- **Upstream Data Forwarding**
  - Added `forward_upstream_data` option to forward data from upstream filters
  - Preserves metadata and additional channels from previous filters in the pipeline
  - Defaults to `true` to maintain backward compatibility

- **Enhanced Configuration Validation**
  - Added comprehensive configuration validation with type conversion for string inputs
  - Validates boolean flags (`debug`, `forward_deduped_frames`, `forward_upstream_data`) with helpful error messages
  - Converts string values to proper types (int, float, tuple) automatically
  - Improved error handling for invalid configuration values

- **Comprehensive Testing Suite**
  - Added integration tests for configuration normalization and validation
  - Added smoke tests for basic filter functionality and new features
  - Added unit tests for side channel and upstream data forwarding
  - Tests cover all new configuration options and edge cases

- **Enhanced Documentation**
  - Updated `docs/overview.md` with comprehensive examples and use cases
  - Added sample pipelines for security surveillance, content analysis, and live streaming
  - Included configuration guidelines and troubleshooting information
  - Added detailed configuration reference with threshold guidelines

- **Improved Usage Script**
  - Enhanced `scripts/filter_usage.py` with better examples and environment variable support
  - Added VS Code launch configuration for debugging
  - Simplified configuration management with environment variables

### Changed
- **Configuration Processing**
  - Moved `main` channel to be the first element in output dictionary for consistency
  - Enhanced `normalize_config` method to call parent class first for proper sources/outputs parsing
  - Improved type conversion and validation for all configuration parameters

- **Frame Processing**
  - Modified `process` method to ensure main channel returns processed image, not original
  - Deduplicated channel now uses actual processed image, not a copy
  - Improved frame metadata handling and channel synchronization

### Fixed
- **Configuration Parsing**
  - Fixed `invalid source 't'` error by properly calling parent class `normalize_config`
  - Resolved string-to-type conversion issues in configuration validation
  - Fixed boolean flag validation to handle string inputs correctly


## v1.0.13 - 2025-07-15
- Migrated from filter_runtime to openfilter
  
## v1.0.12 - 2024-04-23
### Added
- Internal improvements

## v1.0.8 - 2024-04-08
- Added locking mechanism during file writing in disk.

## v1.0.7 - 2024-03-25
- Initial Release: new filter for saving only unique frames from video streams

- **Multi-Stage Deduplication**
  - Uses hash-based difference detection to identify significant frame changes
  - Applies SSIM (Structural Similarity Index) to avoid saving visually similar frames

- **Motion Thresholding**
  - Detects motion intensity to suppress noise and low-impact changes

- **Time-Based Filtering**
  - Enforces a minimum time interval between saved frames via `min_time_between_frames`

- **ROI Support**
  - Optional support for region-of-interest (ROI) processing to focus on specific areas of the frame

- **Debug Logging**
  - Verbose logs available via `debug: true` to help visualize and tune deduplication behavior

- **Structured Output**
  - Saves frames to disk under a user-defined directory, with sequential naming
