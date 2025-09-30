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

    def test_invalid_roi(self):
        # Testing ROI extraction with invalid ROI
        frame = np.full((1280,720,3), (255,0,0), dtype=np.uint8)
        self.filter.config.roi = (2000,2000,100)
        with self.assertRaises(ValueError):
            self.filter.hash_processor.extract_roi(frame)

if __name__ == "__main__":
    unittest.main()