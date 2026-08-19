# Changelog
FrameSelect release notes

## [Unreleased]

## v1.3.4 - 2026-08-18

### Changed

- Update the openfilter dependency to 1.3.0
- Add Python 3.14 support: raise the `requires-python` ceiling to `<3.15` and add 3.14 to the CI
  test matrix (3.10–3.13 unchanged). `scikit-image` is split by interpreter (`0.25.2` for
  3.10–3.13, `>=0.26.0` for 3.14, which is the first release with cp314 wheels). The image base
  moves to `py3.14` in a follow-up, once a 3.14-supporting wheel is published.

## v1.3.3 - 2026-08-12

### Changed

- SSIM is computed with `cv2.boxFilter` instead of `skimage.metrics.structural_similarity`. Same estimator, same window, same crop: scikit-image averages each window with `scipy.ndimage.uniform_filter`, which is single-threaded scalar code, while OpenCV does it with SIMD.

  This is a cost change only. The worst disagreement with the previous result, measured across every frame shape and window size the filter can produce, is **2.9e-14** (`tests/test_ssim_boxfilter.py` documents ~1e-13 as the ceiling it asserts against). No threshold in the 0.85-0.95 band the filter operates in can resolve a difference that small, so no keep decision changes.

  It composes with `ssim_eval_width` rather than replacing it: the downscale decides how many pixels are compared, this decides how fast each comparison runs. It matters because the dedup is the stage that binds the CPD pipeline, at 43.8 ms per input frame against 19.8 ms for the detector, and SSIM is the bulk of that cost.

## v1.3.2 - 2026-08-10

### Fixed
- `ssim_eval_width` could silently disable deduplication on extreme aspect ratios. Config validation bounds the configured *width*, but the height scales independently and never reaches it: a wide, short frame (2592x10 at `ssim_eval_width=7`) reduces to 7x1, below the SSIM window, and `compute_ssim` then failed open on every frame. The filter kept everything instead of erroring, so dedup stopped working with no signal that it had. The downscale is now skipped for shapes that would cross the window and the comparison runs at full resolution: slower, and correct. The warning is logged once per frame shape rather than once per frame, since a stream holds its shape.

## v1.3.1 - 2026-08-07

### Fixed
- An `ssim_eval_width` below 7 crashed the whole-frame SSIM path. scikit-image defaults to `win_size=7` and raises `win_size exceeds image extent` once the downscaled frame is smaller than that; the patch-grid path already clamped the window but the whole-frame path did not. Config now rejects `0 < ssim_eval_width < 7` with an explicit message, and `compute_ssim` clamps the window and fails open (score 0.0, so the frame is kept) for frames too small to compare at all.

## v1.3.0 - 2026-08-07

### Added
- `ssim_eval_width`: compare frames for SSIM at a reduced width instead of full resolution. Default `0` keeps the current full-resolution behaviour. At `480` (the width the motion gatekeeper already uses) SSIM is **7.76x cheaper** on a 2592x1520 source, 213.5 ms -> 27.0 ms.

  **Measure recall before turning this on.** The speedup is real, but it changes which frames survive, not only what they cost. In a full-pipeline A/B over the same 3600 frames, full resolution kept 783 frames (21.8%) and `480` kept 26 (0.7%) at the same threshold. Retuning the threshold to restore the keep *rate* does not restore the *decision*: at the threshold that matches the count, recall against the full-resolution keep set is **0.265**, so the two configurations keep a nearly disjoint set of frames. For a dedup feeding downstream detection, a frame this drops is one nothing else ever sees. The default is `0` for exactly this reason.

### Changed
- SSIM is computed with `full=False`. The previous `full=True` built a float SSIM map the size of the input on every frame and the caller discarded it; dropping it is **1.27x** on its own, 213.5 ms -> 168.5 ms.
- `SSIMProcessor.should_save_frame` returns a real `bool` rather than `numpy.bool_`, matching its annotation.

## v1.2.2 - 2026-08-10

### Changed

- Build the image on `openfilter-base` (weekly apt-upgraded python-slim) instead of a stale `python:X.Y.Z-slim` pin, clearing the OS-package CVEs the pin carried.
- Update the openfilter dependency to 1.2.2

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
