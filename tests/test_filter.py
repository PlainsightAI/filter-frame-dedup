import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from filter_frame_dedup.filter import FilterFrameDedup, FilterFrameDedupConfig
from unittest.mock import MagicMock, patch

import numpy as np
import time

class TestFilterFrameDedup(unittest.TestCase):
    def setUp(self):
        # Setting up test configuration for FilterFrameDedup and initializing it
        config = {
            'config': {
                # motion_gate is opt-in; request it explicitly so tests that rely
                # on self.filter.motion_gatekeeper keep the v1.1.6 default pipeline
                'active_processors': ["motion_gate", "hash_dedup", "ssim_dedup"],
                'hash_threshold': 5,
                'motion_threshold': 1200,  # Match the default in filter
                'min_time_between_frames': 0.1,  # Reduced from 1.0 to 0.1 for testing
                'roi': (0,0,150,300),
                'ssim_threshold': 0.90,
                'output_folder': 'test_frames',
                'debug': True,
                'forward_deduped_frames': True,
                'forward_upstream_data': True
            }
        }
        self.filter = FilterFrameDedup(config)
        # Normalize config first
        self.filter.config = self.filter.normalize_config(self.filter.config)
        # Call setup to initialize the filter
        self.filter.setup(self.filter.config)
        
        if not os.path.exists('test_frames'):
            os.makedirs('test_frames')
        else:
            # Clear the directory if it exists
            for file in os.listdir('test_frames'):
                os.remove(os.path.join('test_frames', file))

    def tearDown(self):
        # Clean up test files
        for file in os.listdir('test_frames'):
            os.remove(os.path.join('test_frames', file))
        os.rmdir('test_frames')

    def test_init(self):
        # Test initialization of FilterFrameDedup parameters
        self.assertEqual(self.filter.config.hash_threshold, 5)
        self.assertEqual(self.filter.config.motion_threshold, 1200)  # Match the default
        self.assertEqual(self.filter.config.min_time_between_frames, 0.1)
        self.assertEqual(self.filter.config.roi, (0,0,150,300))
        self.assertEqual(self.filter.config.ssim_threshold, 0.90)
        self.assertEqual(self.filter.config.output_folder, 'test_frames')
        self.assertTrue(self.filter.config.debug)
        self.assertTrue(self.filter.config.forward_deduped_frames)
        self.assertTrue(self.filter.config.forward_upstream_data)

    def test_extract_roi(self):
        # Testing extraction of ROI from frame
        frame = np.full((1280,720,3), (255,45,70), dtype=np.uint8)
        roi_frame = self.filter.hash_processor.extract_roi(frame)
        self.assertEqual(roi_frame.shape, (300,150,3))

    def test_extract_roi_none(self):
        # Testing extraction of ROI when ROI is None
        self.filter.config.roi = None
        frame = np.full((1280,720,3), (255,45,70), dtype=np.uint8)
        roi_frame = self.filter.hash_processor.extract_roi(frame)
        self.assertEqual(roi_frame.shape, frame.shape)

    def generate_mock_frame(self, w, h, color):
        # Helper method to generate mock frames
        image = np.full((h,w,3), color, dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.has_image = True
        mock_frame.rw_bgr.image = image
        mock_frame.data = MagicMock()
        mock_frame.data.copy.return_value = {}
        return mock_frame

    def test_process_first_frame(self):
        # Testing the first frame processing
        frame = self.generate_mock_frame(1280, 720, (255, 0, 0))
        frames = {'main': frame}
        processed = self.filter.process(frames)
        
        # Test that main channel is first in output
        self.assertIn('main', processed)
        self.assertEqual(list(processed.keys())[0], 'main')
        
        # Test that deduped channel is present when enabled
        self.assertIn('deduped', processed)
        self.assertTrue(processed['deduped'].data['deduped'])
        
        # Check if frame was saved (this depends on the actual processors)
        # The frame should be saved if it passes the hash and SSIM checks
        saved_files = os.listdir('test_frames')
        if saved_files:  # If any files were saved
            self.assertTrue(any(f.startswith('frame_') for f in saved_files))

    def test_process_same_frame(self):
        # Testing processing of identical frames
        frame = self.generate_mock_frame(1280, 720, (255, 0, 0))
        frames = {'main': frame}
        # Process first frame
        self.filter.process(frames)
        # Process same frame again
        self.filter.process(frames)
        # Should only have one frame saved
        self.assertEqual(len(os.listdir('test_frames')), 1)

    # def test_process_different_frames(self):
    #     # Testing processing of different frames
    #     frame1 = self.generate_mock_frame(1280, 720, (255, 0, 0))
    #     frame2 = self.generate_mock_frame(1280, 720, (0, 255, 0))
    #     frames1 = {'main': frame1}
    #     frames2 = {'main': frame2}
    #     # Process first frame
    #     self.filter.process(frames1)
    #     # Process different frame
    #     self.filter.process(frames2)
    #     # Should have two frames saved
    #     self.assertEqual(len(os.listdir('test_frames')), 2)

    def test_process_time_threshold(self):
        # Testing time threshold between frames
        frame1 = self.generate_mock_frame(1280, 720, (255, 0, 0))
        frame2 = self.generate_mock_frame(1280, 720, (0, 255, 0))
        frames1 = {'main': frame1}
        frames2 = {'main': frame2}
        # Process first frame
        self.filter.process(frames1)
        # Set last_saved_time to current time
        self.filter.hash_processor.last_saved_time = time.time()
        # Process different frame immediately
        self.filter.process(frames2)
        # Should not save second frame due to time threshold
        self.assertEqual(len(os.listdir('test_frames')), 1)

    def test_process_empty_frame(self):
        # Testing processing with an empty frame
        frame = {}
        processed = self.filter.process(frame)
        self.assertEqual(len(os.listdir('test_frames')), 0)

    def test_upstream_data_forwarding(self):
        # Testing that upstream data is forwarded when enabled
        frame = self.generate_mock_frame(1280, 720, (255, 0, 0))
        upstream_frame = self.generate_mock_frame(640, 480, (0, 255, 0))
        frames = {
            'main': frame,
            'upstream_data': upstream_frame
        }
        processed = self.filter.process(frames)
        
        # Test that upstream data is forwarded
        self.assertIn('upstream_data', processed)
        self.assertEqual(processed['upstream_data'], upstream_frame)
        
        # Test that main is still first
        self.assertEqual(list(processed.keys())[0], 'main')

    def test_deduped_channel_metadata(self):
        # Testing that deduped channel has correct metadata
        frame = self.generate_mock_frame(1280, 720, (255, 0, 0))
        frames = {'main': frame}
        processed = self.filter.process(frames)
        
        deduped_frame = processed['deduped']
        # Check that deduped frame has the correct metadata
        self.assertIsInstance(deduped_frame.data, dict)
        self.assertTrue(deduped_frame.data['deduped'])
        self.assertEqual(deduped_frame.data['frame_number'], 1)  # First frame is 1 (incremented before check)
        self.assertIn('saved_path', deduped_frame.data)
        # Check that saved_path contains the expected filename pattern
        saved_path = deduped_frame.data['saved_path']
        self.assertTrue('frame_' in saved_path and saved_path.endswith('.jpg'))

    def test_no_deduped_channel_when_disabled(self):
        # Testing that deduped channel is not created when disabled
        config = {
            'config': {
                'hash_threshold': 5,
                'motion_threshold': 1200,
                'min_time_between_frames': 0.1,
                'roi': (0,0,150,300),
                'ssim_threshold': 0.90,
                'output_folder': 'test_frames',
                'debug': True,
                'forward_deduped_frames': False,  # Disabled
                'forward_upstream_data': True
            }
        }
        filter_no_dedup = FilterFrameDedup(config)
        filter_no_dedup.config = filter_no_dedup.normalize_config(filter_no_dedup.config)
        filter_no_dedup.setup(filter_no_dedup.config)
        
        frame = self.generate_mock_frame(1280, 720, (255, 0, 0))
        frames = {'main': frame}
        processed = filter_no_dedup.process(frames)
        
        # Test that deduped channel is not present
        self.assertNotIn('deduped', processed)
        # Test that main channel is still present
        self.assertIn('main', processed)

    def test_no_upstream_forwarding_when_disabled(self):
        # Testing that upstream data is not forwarded when disabled
        config = {
            'config': {
                'hash_threshold': 5,
                'motion_threshold': 1200,
                'min_time_between_frames': 0.1,
                'roi': (0,0,150,300),
                'ssim_threshold': 0.90,
                'output_folder': 'test_frames',
                'debug': True,
                'forward_deduped_frames': True,
                'forward_upstream_data': False  # Disabled
            }
        }
        filter_no_upstream = FilterFrameDedup(config)
        filter_no_upstream.config = filter_no_upstream.normalize_config(filter_no_upstream.config)
        filter_no_upstream.setup(filter_no_upstream.config)
        
        frame = self.generate_mock_frame(1280, 720, (255, 0, 0))
        upstream_frame = self.generate_mock_frame(640, 480, (0, 255, 0))
        frames = {
            'main': frame,
            'upstream_data': upstream_frame
        }
        processed = filter_no_upstream.process(frames)
        
        # Test that upstream data is not forwarded
        self.assertNotIn('upstream_data', processed)
        # Test that main and deduped channels are still present
        self.assertIn('main', processed)
        self.assertIn('deduped', processed)

    def generate_gradual_mock_frame(self, w, h, square_intensity):
        # Generate a base black image
        image = np.zeros((h, w, 3), dtype=np.uint8)
        # Create a 40x40 square with the given intensity in the ROI area
        image[10:50, 10:50, :] = square_intensity
        mock_frame = MagicMock()
        mock_frame.has_image = True
        mock_frame.rw_bgr.image = image
        mock_frame.data = MagicMock()
        mock_frame.data.copy.return_value = {}
        return mock_frame

    def test_gradual_change_detection(self):
        # Set min_time_between_frames to 0 to focus on image changes
        self.filter.config.min_time_between_frames = 0.0
        # Set motion gatekeeper threshold to 0 to let tiny test changes pass
        self.filter.motion_gatekeeper.pixel_delta_threshold = 0.0
        # Set high hash threshold so we isolate motion detection
        self.filter.config.hash_threshold = 100
        # Set high SSIM threshold because our changing region is very small compared to the whole image
        self.filter.config.ssim_threshold = 0.9999
        
        # Frame 1: Base image with square at intensity 0
        frame1 = self.generate_gradual_mock_frame(1280, 720, 0)
        self.filter.process({'main': frame1})
        self.assertEqual(len(os.listdir('test_frames')), 1)

        # Process a sequence of frames where the square gradually increases in intensity by 5 each step.
        # Step-to-step difference is always 5, which is below the motion threshold of 25,
        # so frame-by-frame comparison would never register any motion.
        # But comparing to the last saved frame (intensity 0) will eventually exceed 25.
        for i in range(1, 10):
            intensity = i * 5
            frame = self.generate_gradual_mock_frame(1280, 720, intensity)
            self.filter.process({'main': frame})

        # Under the old implementation (frame-to-frame comparison), only 1 frame (the first one) is saved
        # because the difference between adjacent frames is always 5 (which is below the threshold of 25).
        # Under the new implementation (comparing to the last saved frame), once the intensity reaches 30 (step 6),
        # the difference from the last saved frame (0) is 30, which is above 25.
        # Since the square is 40x40 = 1600 pixels and 1600 > motion_threshold (1200), it registers motion and saves!
        saved_count = len(os.listdir('test_frames'))
        self.assertGreater(saved_count, 1)

    def test_hash_processor_optional_legacy_fallback(self):
        # Legacy fallback (no active_processors) with use_hash_dedup=False preserves
        # v1.1.6 behavior: only ssim_dedup, no auto-injected motion_gate, no hash.
        from filter_frame_dedup.filter import FilterFrameDedupConfig, FilterFrameDedup
        config_dict = {
            'config': {
                'use_hash_dedup': False,
                'output_folder': 'test_frames',
                'save_images': True,
                'roi': None,
                'forward_deduped_frames': False
            }
        }
        custom_filter = FilterFrameDedup(config_dict)
        custom_filter.config = custom_filter.normalize_config(custom_filter.config)
        custom_filter.setup(custom_filter.config)

        # Verify hash processor is disabled/None
        self.assertIsNone(custom_filter.hash_processor)
        # motion_gate is opt-in and must NOT be auto-added by the legacy fallback
        self.assertIsNone(custom_filter.motion_gatekeeper)

        # Verify pipeline structure
        pipeline_names = [step["name"] for step in custom_filter.pipeline]
        self.assertNotIn("Motion Gatekeeper", pipeline_names)
        self.assertNotIn("Hash Processor", pipeline_names)
        self.assertIn("SSIM Processor", pipeline_names)

        # Test processing frames
        frame1 = self.generate_mock_frame(1280, 720, (255, 0, 0))
        processed1 = custom_filter.process({'main': frame1})
        self.assertIn('main', processed1)

    def test_patchified_motion_gate(self):
        # Frame 1: Base black image
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Frame 2: Only a small region (50x50) has a high change in intensity (+100)
        # On the full image, average delta is ~0.27 (below default threshold of 1.5)
        # But in a 4x4 grid, the patch containing it has a high average delta (~4.3)
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame2[50:100, 50:100, :] = 100

        from filter_frame_dedup.filter import FilterFrameDedup
        # 1. With patch_grid_size = 1 (default): should return False (motion ignored)
        config_dict_non_patch = {
            'config': {
                'active_processors': ["motion_gate"],
                'motion_gate_patch_grid_size': 1,
                'motion_gate_pixel_delta_threshold': 1.5,
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        filter_non_patch = FilterFrameDedup(config_dict_non_patch)
        filter_non_patch.config = filter_non_patch.normalize_config(filter_non_patch.config)
        filter_non_patch.setup(filter_non_patch.config)

        filter_non_patch.motion_gatekeeper.should_process_frame(frame1)
        filter_non_patch.motion_gatekeeper.update_reference_frame(frame1)
        self.assertFalse(filter_non_patch.motion_gatekeeper.should_process_frame(frame2))

        # 2. With patch_grid_size = 4: should return True (motion caught)
        config_dict_patch = {
            'config': {
                'active_processors': ["motion_gate"],
                'motion_gate_patch_grid_size': 4,
                'motion_gate_pixel_delta_threshold': 1.5,
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        filter_patch = FilterFrameDedup(config_dict_patch)
        filter_patch.config = filter_patch.normalize_config(filter_patch.config)
        filter_patch.setup(filter_patch.config)

        filter_patch.motion_gatekeeper.should_process_frame(frame1)
        filter_patch.motion_gatekeeper.update_reference_frame(frame1)
        self.assertTrue(filter_patch.motion_gatekeeper.should_process_frame(frame2))

    def test_patchified_ssim(self):
        # Frame 1: Base white image
        frame1 = np.full((720, 1280, 3), 255, dtype=np.uint8)
        # Frame 2: Only a small region (50x50) is black.
        # Across the whole image, SSIM is extremely high (~0.999), above threshold of 0.99
        # But in a 4x4 grid, the patch containing this change will have low SSIM score
        frame2 = np.full((720, 1280, 3), 255, dtype=np.uint8)
        frame2[50:100, 50:100, :] = 0

        from filter_frame_dedup.filter import FilterFrameDedup
        # 1. With ssim_patch_grid_size = 1 (default): should return False (ssim threshold not reached, ignored)
        config_dict_non_patch = {
            'config': {
                'ssim_patch_grid_size': 1,
                'ssim_threshold': 0.99,
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        filter_non_patch = FilterFrameDedup(config_dict_non_patch)
        filter_non_patch.config = filter_non_patch.normalize_config(filter_non_patch.config)
        filter_non_patch.setup(filter_non_patch.config)

        filter_non_patch.ssim_processor.should_save_frame(frame1)
        filter_non_patch.ssim_processor.update_reference_frame(frame1)
        self.assertFalse(filter_non_patch.ssim_processor.should_save_frame(frame2))

        # 2. With ssim_patch_grid_size = 4: should return True (local change caught)
        config_dict_patch = {
            'config': {
                'ssim_patch_grid_size': 4,
                'ssim_threshold': 0.99,
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        filter_patch = FilterFrameDedup(config_dict_patch)
        filter_patch.config = filter_patch.normalize_config(filter_patch.config)
        filter_patch.setup(filter_patch.config)

        filter_patch.ssim_processor.should_save_frame(frame1)
        filter_patch.ssim_processor.update_reference_frame(frame1)
        self.assertTrue(filter_patch.ssim_processor.should_save_frame(frame2))

    def test_active_processors_ordering(self):
        # Configure custom active processors list
        config_dict = {
            'config': {
                'active_processors': ["ssim_dedup", "motion_gate"],
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        from filter_frame_dedup.filter import FilterFrameDedup
        custom_filter = FilterFrameDedup(config_dict)
        custom_filter.config = custom_filter.normalize_config(custom_filter.config)
        custom_filter.setup(custom_filter.config)

        # Assert only ssim and motion_gate are initialized
        self.assertIsNotNone(custom_filter.ssim_processor)
        self.assertIsNotNone(custom_filter.motion_gatekeeper)
        self.assertIsNone(custom_filter.hash_processor)
        self.assertIsNone(custom_filter.model_processor)

        # Assert that pipeline step execution order is exactly as requested
        pipeline_names = [step["name"] for step in custom_filter.pipeline]
        self.assertEqual(pipeline_names, ["SSIM Processor", "Motion Gatekeeper"])

    def test_invalid_roi(self):
        # Testing ROI extraction with invalid ROI
        frame = np.full((1280,720,3), (255,0,0), dtype=np.uint8)
        self.filter.config.roi = (2000,2000,100)
        with self.assertRaises(ValueError):
            self.filter.hash_processor.extract_roi(frame)

    def test_active_processors_env_string_form(self):
        # active_processors can arrive as an env-var STRING (not a Python list)
        from filter_frame_dedup.filter import FilterFrameDedup
        config_dict = {
            'config': {
                'active_processors': '["motion_gate", "hash_dedup"]',
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        custom_filter = FilterFrameDedup(config_dict)
        custom_filter.config = custom_filter.normalize_config(custom_filter.config)
        custom_filter.setup(custom_filter.config)

        # The string is parsed into a real list, preserving order
        self.assertEqual(custom_filter.config.active_processors, ["motion_gate", "hash_dedup"])
        # Legacy flags are kept in sync with the resolved list
        self.assertTrue(custom_filter.config.use_hash_dedup)
        self.assertFalse(custom_filter.config.use_model_dedup)

        pipeline_names = [step["name"] for step in custom_filter.pipeline]
        self.assertEqual(pipeline_names, ["Motion Gatekeeper", "Hash Processor"])

    def test_model_dedup_ordering_reference_not_mutated_on_rejection(self):
        # When model_dedup runs BEFORE a later rejecting step, the model reference
        # must only be updated after the frame is accepted by every step.
        from filter_frame_dedup.filter import FilterFrameDedup
        config_dict = {
            'config': {
                'active_processors': ["model_dedup", "ssim_dedup"],
                'roi': None,
                'ssim_threshold': 0.90,
                'output_folder': 'test_frames',
                'save_images': False,
                'forward_deduped_frames': False
            }
        }
        # Patch ModelProcessor so no real HF model is loaded; the mock always
        # reports the frame as unique so the later SSIM step is the rejecter.
        with patch('filter_frame_dedup.model_processor.ModelProcessor') as MockMP:
            mock_instance = MockMP.return_value
            mock_instance.frame_is_unique.return_value = True

            custom_filter = FilterFrameDedup(config_dict)
            custom_filter.config = custom_filter.normalize_config(custom_filter.config)
            custom_filter.setup(custom_filter.config)

            # model_dedup must be first in the pipeline, ssim second
            pipeline_names = [step["name"] for step in custom_filter.pipeline]
            self.assertEqual(pipeline_names, ["Model Processor", "SSIM Processor"])

            frame1 = self.generate_mock_frame(1280, 720, (255, 0, 0))
            frame2 = self.generate_mock_frame(1280, 720, (255, 0, 0))  # identical -> SSIM rejects

            custom_filter.process({'main': frame1})  # accepted -> model reference updated once
            custom_filter.process({'main': frame2})  # rejected by SSIM -> model reference NOT updated

            self.assertEqual(mock_instance.update_reference_frame.call_count, 1)

    def test_empty_active_processors_raises(self):
        # An empty active_processors (list or env string "[]") must be rejected
        from filter_frame_dedup.filter import FilterFrameDedup
        for empty_value in ([], "[]"):
            config_dict = {
                'config': {
                    'active_processors': empty_value,
                    'output_folder': 'test_frames',
                    'save_images': False
                }
            }
            with self.assertRaises(ValueError):
                f = FilterFrameDedup(config_dict)
                f.config = f.normalize_config(f.config)

    def test_motion_gate_pixel_delta_threshold_upper_bound(self):
        # A threshold above the 0..255 mean-delta range would reject every frame
        # forever, silently disabling dedup; it must be rejected at config time.
        from filter_frame_dedup.filter import FilterFrameDedup
        config_dict = {
            'config': {
                'active_processors': ["motion_gate"],
                'motion_gate_pixel_delta_threshold': 300,
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        with self.assertRaises(ValueError):
            f = FilterFrameDedup(config_dict)
            f.config = f.normalize_config(f.config)

    def test_motion_gate_tiny_frame_resize_no_crash(self):
        # A wide, 1px-tall frame rounds the resized height to 0 without clamping,
        # which makes cv2.resize raise. It must be handled gracefully.
        from filter_frame_dedup.filter import FilterFrameDedup
        config_dict = {
            'config': {
                'active_processors': ["motion_gate"],
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        f = FilterFrameDedup(config_dict)
        f.config = f.normalize_config(f.config)
        f.setup(f.config)

        frame = np.zeros((1, 1920, 3), dtype=np.uint8)  # h=1, w=1920, eval_width=480 -> h*scale rounds to 0
        # Neither the check nor the reference update should raise
        self.assertTrue(f.motion_gatekeeper.should_process_frame(frame))
        f.motion_gatekeeper.update_reference_frame(frame)

    def test_ssim_uncomputable_patch_keeps_frame(self):
        # When patches are too small to compute SSIM, the comparison is uncomputable
        # and a dedup filter must fail open (KEEP the frame), not treat it as duplicate.
        from filter_frame_dedup.filter import FilterFrameDedup
        config_dict = {
            'config': {
                'active_processors': ["ssim_dedup"],
                'ssim_patch_grid_size': 4,   # 8x8 image -> 2x2 patches -> SSIM uncomputable
                'ssim_threshold': 0.90,
                'output_folder': 'test_frames',
                'save_images': False
            }
        }
        f = FilterFrameDedup(config_dict)
        f.config = f.normalize_config(f.config)
        f.setup(f.config)

        frame1 = np.zeros((8, 8, 3), dtype=np.uint8)
        frame2 = np.zeros((8, 8, 3), dtype=np.uint8)  # identical, but patches are uncomputable
        f.ssim_processor.should_save_frame(frame1)
        f.ssim_processor.update_reference_frame(frame1)
        # Even for identical frames, an uncomputable patch comparison keeps the frame
        self.assertTrue(f.ssim_processor.should_save_frame(frame2))

if __name__ == "__main__":
    unittest.main()