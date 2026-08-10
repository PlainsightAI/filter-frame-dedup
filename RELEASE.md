# Changelog
FrameSelect release notes

## [Unreleased]

### Changed

- Bump the openfilter dependency to 1.2.2

## v1.2.1 - 2026-08-04

### Changed
- Update `openfilter[all]` to `>=1.2.1`
- Grant `id-token: write` for keyless (cosign) SBOM
- Fix RELEASE.md header (stray H1 + duplicated block)
- Pin Docker base to `python:3.11.12-slim`
- Point compose utility images at `openfilter-{video-in,webvis}:1.2.1` and pin the filter's own image to the release version (`openfilter-frame-dedup:1.2.1`)
- Update dev-tooling floors and switch to ranges

## v1.2.0 - 2026-07-31

### Added
- Configurable `active_processors`: name the processors and their execution order; also accepts the env-var string form (e.g. `'["motion_gate", "hash_dedup"]'`).
- ROI-aware motion gate: `FastMotionGatekeeper` now applies the same ROI crop as the other processors before computing the pixel delta.

### Changed
- Motion gating and patchify mode (SSIM + motion gate) are now first-class, reorderable pipeline steps.
- Reference-frame updates are strictly deferred: each processor's check is pure and its reference is updated only after a frame is accepted by every step. NOTE: This changes deduplication behavior on slowly-changing footage from frame-to-frame (v1.1.6) to compare-to-last-saved. This corrects gradual change detection but can result in increased saved-frame counts on very slow video streams.
- Legacy fallback (when `active_processors` is unset) preserves v1.1.6 processor order and no longer injects `motion_gate` (which remains opt-in via explicit `active_processors`). Note that the fallback path also inherits the new deferred reference-frame compare-to-last-saved behavior.
- Renamed the internal `saved_frame_count` counter to `unique_frame_count` and clarified the shutdown log ("Total unique frames").

### Fixed
- Model-dedup no longer mutates its reference frame inside the uniqueness check; `frame_is_unique` is a pure check and `update_reference_frame` is wired into the pipeline.
- An empty `active_processors` (e.g. `[]` / `"[]"`) now raises instead of silently building a pipeline that dedups nothing.

## v1.1.6 - 2026-07-30

### Changed
- Added handling of different model types (e.g. resnet)


## v1.1.5 - 2026-06-23

### Changed
- Add option for users to use HF model to deduplicate frames. 

## v1.1.4 - 2026-04-23

### Changed
- Update the openfilter dependency to `>=0.1.30`, and align the CI workflow with the shared release gate (source-paths).
- Fix release workflow secret names: `PYPI_API_TOKEN` → `PLAINSIGHT_PYPI_TOKEN`, `DOCKERHUB_TOKEN` → `DOCKERHUB_ACCESS_TOKEN` (org-level secret names). Without this the PyPI / Docker Hub tokens resolved to empty and no package has been published since the migration.

## v1.1.3 - 2026-04-20

### Changed
- Remove redundant ci.yaml (shared workflow handles PR testing)
- Add push + pull_request triggers to create-release.yaml


## v1.1.2 - 2026-04-15

### Changed
- Add CI/CD workflows: create-release.yaml (Docker Hub publishing), ci.yaml (PR testing), security-scan.yaml
- Update openfilter dependency to >=0.1.27


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
