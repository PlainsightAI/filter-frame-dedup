"""
Smoke tests for FilterFrameDedup basic functionality.

These tests verify that the filter can be initialized, configured, and perform
basic operations without complex pipeline orchestration.
"""

import pytest
import numpy as np
import tempfile
import os
import time
from unittest.mock import patch, MagicMock
from openfilter.filter_runtime.filter import Frame
from filter_frame_dedup.filter import FilterFrameDedup, FilterFrameDedupConfig


class TestSmokeSimple:
    """Test basic filter functionality and lifecycle."""

    @pytest.fixture
    def temp_workdir(self):
        """Create a temporary working directory for tests."""
        with tempfile.TemporaryDirectory(prefix='filter_frame_dedup_smoke_') as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        frame_data = {"meta": {"id": 1, "topic": "test"}}
        return Frame(image, frame_data, 'BGR')

    def test_filter_initialization(self, temp_workdir):
        """Test that the filter can be initialized with valid config."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': False,
            'forward_deduped_frames': False,
            'forward_upstream_data': True
        }
        
        # Test config normalization
        config = FilterFrameDedup.normalize_config(config_data)
        assert config.hash_threshold == 5
        assert config.motion_threshold == 1200
        assert config.min_time_between_frames == 1.0
        assert config.ssim_threshold == 0.90
        assert config.output_folder == temp_workdir
        assert config.debug is False
        
        # Test filter initialization
        filter_instance = FilterFrameDedup(config=config)
        assert filter_instance is not None
        # The config is stored internally, so we just verify the filter was created

    def test_setup_and_shutdown(self, temp_workdir):
        """Test that setup() and shutdown() work correctly."""
        config_data = {
            'hash_threshold': 3,
            'motion_threshold': 1000,
            'min_time_between_frames': 0.5,
            'ssim_threshold': 0.85,
            'output_folder': temp_workdir,
            'roi': (100, 100, 200, 200),
            'debug': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        
        # Test setup
        filter_instance.setup(config)
        assert filter_instance.config is not None
        assert filter_instance.hash_processor is not None
        assert filter_instance.ssim_processor is not None
        assert filter_instance.processed_frame_count == 0
        assert filter_instance.frame_count == 1
        assert os.path.exists(temp_workdir)
        
        # Test shutdown
        filter_instance.shutdown()  # Should not raise any exceptions

    def test_config_validation(self):
        """Test that configuration validation works correctly."""
        # Test valid configuration
        valid_config = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test'
        }
        
        config = FilterFrameDedup.normalize_config(valid_config)
        assert config.hash_threshold == 5
        assert config.motion_threshold == 1200
        
        # Test configuration with typo - should not raise error anymore
        config_with_typo = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'hash_threshhold': 3  # Typo
        }
        
        # Should not raise an error - unknown keys are passed through
        config = FilterFrameDedup.normalize_config(config_with_typo)
        assert config.hash_threshold == 5
        assert config.output_folder == '/tmp/test'

    def test_first_frame_processing(self, sample_frame, temp_workdir):
        """Test processing of the first frame (should always be saved)."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Process first frame
        frames = {"main": sample_frame}
        output_frames = filter_instance.process(frames)
        
        # Verify output
        assert "main" in output_frames
        assert "deduped" in output_frames  # Deduped channel should be present
        assert len(output_frames) == 2  # Main and deduped frames should be returned
        
        # Verify frame was saved to disk
        saved_files = os.listdir(temp_workdir)
        assert len(saved_files) == 1
        assert saved_files[0].startswith('frame_')
        assert saved_files[0].endswith('.jpg')
        
        # Verify counters
        assert filter_instance.processed_frame_count == 1
        assert filter_instance.frame_count == 2

    def test_duplicate_frame_processing(self, sample_frame, temp_workdir):
        """Test processing of duplicate frames (should not be saved)."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,  # Short time for testing
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Process first frame
        frames = {"main": sample_frame}
        filter_instance.process(frames)
        
        # Process same frame again (should be considered duplicate)
        output_frames = filter_instance.process(frames)
        
        # Verify output
        assert "main" in output_frames
        # Deduped channel is only present when frame is actually saved
        if "deduped" in output_frames:
            assert len(output_frames) == 2  # Main and deduped frames
        else:
            assert len(output_frames) == 1  # Only main frame
        
        # Verify only one frame was saved (duplicate should not be saved)
        saved_files = os.listdir(temp_workdir)
        assert len(saved_files) == 1
        
        # Verify counters
        assert filter_instance.processed_frame_count == 2
        assert filter_instance.frame_count == 3

    def test_different_frame_processing(self, temp_workdir):
        """Test processing of different frames (should be saved)."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,  # Short time for testing
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Create two different frames
        image1 = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        frame1 = Frame(image1, {"meta": {"id": 1}}, 'BGR')
        frame2 = Frame(image2, {"meta": {"id": 2}}, 'BGR')
        
        # Process first frame
        frames1 = {"main": frame1}
        filter_instance.process(frames1)
        
        # Wait a bit to ensure time threshold is met
        time.sleep(0.2)
        
        # Process different frame
        frames2 = {"main": frame2}
        output_frames = filter_instance.process(frames2)
        
        # Verify output
        assert "main" in output_frames
        assert "deduped" in output_frames  # Deduped channel should be present
        assert len(output_frames) == 2  # Main and deduped frames should be returned
        
        # Verify both frames were saved
        saved_files = os.listdir(temp_workdir)
        assert len(saved_files) == 2
        
        # Verify counters
        assert filter_instance.processed_frame_count == 2
        assert filter_instance.frame_count == 3

    def test_roi_processing(self, temp_workdir):
        """Test processing with ROI configuration."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'roi': (100, 100, 200, 200),  # ROI configuration
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Process frame with ROI
        image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        frame = Frame(image, {"meta": {"id": 1}}, 'BGR')
        frames = {"main": frame}
        output_frames = filter_instance.process(frames)
        
        # Verify output
        assert "main" in output_frames
        assert "deduped" in output_frames  # Deduped channel should be present
        assert len(output_frames) == 2  # Main and deduped frames should be returned
        
        # Verify frame was saved
        saved_files = os.listdir(temp_workdir)
        assert len(saved_files) == 1

    def test_time_threshold_processing(self, sample_frame, temp_workdir):
        """Test time threshold between frame saves."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,  # 1 second threshold
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Process first frame
        frames = {"main": sample_frame}
        filter_instance.process(frames)
        
        # Process different frame immediately (should not be saved due to time threshold)
        image2 = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        frame2 = Frame(image2, {"meta": {"id": 2}}, 'BGR')
        frames2 = {"main": frame2}
        output_frames = filter_instance.process(frames2)
        
        # Verify output
        assert "main" in output_frames
        # Deduped channel is only present when frame is actually saved
        if "deduped" in output_frames:
            assert len(output_frames) == 2  # Main and deduped frames
        else:
            assert len(output_frames) == 1  # Only main frame
        
        # Verify only first frame was saved (second should be skipped due to time)
        saved_files = os.listdir(temp_workdir)
        assert len(saved_files) == 1

    def test_empty_frame_processing(self, temp_workdir):
        """Test processing with empty frame dictionary."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Process empty frame
        frames = {}
        output_frames = filter_instance.process(frames)
        
        # Verify output
        assert len(output_frames) == 0
        
        # Verify no frames were saved
        saved_files = os.listdir(temp_workdir)
        assert len(saved_files) == 0

    def test_debug_mode_processing(self, sample_frame, temp_workdir):
        """Test processing with debug mode enabled."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Process frame with debug enabled
        frames = {"main": sample_frame}
        
        with patch('filter_frame_dedup.filter.logger') as mock_logger:
            output_frames = filter_instance.process(frames)
            
            # Verify debug logging was called
            assert mock_logger.debug.called or mock_logger.info.called

    def test_string_config_conversion(self):
        """Test that string configs are properly converted to types."""
        # Test with string values that should be converted
        config_data = {
            'hash_threshold': '3',
            'motion_threshold': '1000',
            'min_time_between_frames': '0.5',
            'ssim_threshold': '0.85',
            'output_folder': '/tmp/string_test',
            'debug': 'true'
        }
        
        normalized = FilterFrameDedup.normalize_config(config_data)
        
        # Check that string values are converted to correct types
        assert normalized.hash_threshold == 3
        assert normalized.motion_threshold == 1000
        assert normalized.min_time_between_frames == 0.5
        assert normalized.ssim_threshold == 0.85
        assert normalized.output_folder == '/tmp/string_test'
        assert normalized.debug is True

    def test_error_handling_invalid_config(self):
        """Test error handling for invalid configuration values."""
        # Test negative hash threshold
        config_invalid = {
            'hash_threshold': -5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test'
        }
        
        with pytest.raises(ValueError, match="Hash threshold must be non-negative"):
            FilterFrameDedup.normalize_config(config_invalid)
        
        # Test invalid SSIM threshold
        config_invalid_ssim = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 1.5,  # > 1.0
            'output_folder': '/tmp/test'
        }
        
        with pytest.raises(ValueError, match="SSIM threshold must be between 0 and 1"):
            FilterFrameDedup.normalize_config(config_invalid_ssim)

    def test_environment_variable_loading(self):
        """Test environment variable configuration loading."""
        # Set environment variables
        os.environ['FILTER_HASH_THRESHOLD'] = '3'
        os.environ['FILTER_MOTION_THRESHOLD'] = '1000'
        os.environ['FILTER_MIN_TIME_BETWEEN_FRAMES'] = '0.5'
        os.environ['FILTER_SSIM_THRESHOLD'] = '0.85'
        os.environ['FILTER_OUTPUT_FOLDER'] = '/tmp/env_test'
        os.environ['FILTER_DEBUG'] = 'true'
        
        try:
            # Create config from environment variables
            config = {
                'hash_threshold': os.environ.get('FILTER_HASH_THRESHOLD', '5'),
                'motion_threshold': os.environ.get('FILTER_MOTION_THRESHOLD', '1200'),
                'min_time_between_frames': os.environ.get('FILTER_MIN_TIME_BETWEEN_FRAMES', '1.0'),
                'ssim_threshold': os.environ.get('FILTER_SSIM_THRESHOLD', '0.90'),
                'output_folder': os.environ.get('FILTER_OUTPUT_FOLDER', '/tmp/output'),
                'debug': os.environ.get('FILTER_DEBUG', 'false')
            }
            normalized = FilterFrameDedup.normalize_config(config)
            
            assert normalized.hash_threshold == 3
            assert normalized.motion_threshold == 1000
            assert normalized.min_time_between_frames == 0.5
            assert normalized.ssim_threshold == 0.85
            assert normalized.output_folder == '/tmp/env_test'
            assert normalized.debug is True
            
        finally:
            # Clean up environment variables
            for key in ['FILTER_HASH_THRESHOLD', 'FILTER_MOTION_THRESHOLD', 'FILTER_MIN_TIME_BETWEEN_FRAMES', 
                       'FILTER_SSIM_THRESHOLD', 'FILTER_OUTPUT_FOLDER', 'FILTER_DEBUG']:
                if key in os.environ:
                    del os.environ[key]

    def test_deduped_channel_forwarding(self, sample_frame, temp_workdir):
        """Test that deduplicated frames are forwarded in side channel when enabled."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,  # Short time for testing
            'ssim_threshold': 0.9,
            'roi': None,
            'output_folder': temp_workdir,
            'debug': False,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        
        # Mock the processors
        with patch('filter_frame_dedup.filter.HashFrameProcessor') as mock_hash_class, \
             patch('filter_frame_dedup.filter.SSIMProcessor') as mock_ssim_class:
            
            mock_hash_processor = MagicMock()
            mock_hash_processor.should_process_frame.return_value = True
            mock_hash_class.return_value = mock_hash_processor
            
            mock_ssim_processor = MagicMock()
            mock_ssim_processor.should_save_frame.return_value = True
            mock_ssim_class.return_value = mock_ssim_processor
            
            filter_instance.setup(config)
            
            # Process frame
            frames = {"main": sample_frame}
            output_frames = filter_instance.process(frames)
            
            # Verify output
            assert "main" in output_frames
            assert "deduped" in output_frames
            assert len(output_frames) == 2  # Only main and deduped
            
            # Verify deduped frame metadata
            deduped_frame = output_frames["deduped"]
            assert deduped_frame.data['deduped'] is True
            assert 'frame_number' in deduped_frame.data
            assert 'saved_path' in deduped_frame.data

    def test_upstream_data_forwarding(self, sample_frame, temp_workdir):
        """Test that upstream data is forwarded when enabled."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,
            'ssim_threshold': 0.9,
            'roi': None,
            'output_folder': temp_workdir,
            'debug': False,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        
        # Mock the processors
        with patch('filter_frame_dedup.filter.HashFrameProcessor') as mock_hash_class, \
             patch('filter_frame_dedup.filter.SSIMProcessor') as mock_ssim_class:
            
            mock_hash_processor = MagicMock()
            mock_hash_processor.should_process_frame.return_value = True
            mock_hash_class.return_value = mock_hash_processor
            
            mock_ssim_processor = MagicMock()
            mock_ssim_processor.should_save_frame.return_value = True
            mock_ssim_class.return_value = mock_ssim_processor
            
            filter_instance.setup(config)
            
            # Create upstream frame
            upstream_frame = Frame(
                image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                data={"upstream": "data"},
                format='BGR'
            )
            
            # Process frame with upstream data
            frames = {
                "main": sample_frame,
                "upstream_data": upstream_frame
            }
            output_frames = filter_instance.process(frames)
            
            # Verify output
            assert "main" in output_frames
            assert "upstream_data" in output_frames
            assert "deduped" in output_frames
            
            # Verify main is first
            assert list(output_frames.keys())[0] == "main"
            
            # Verify upstream data is forwarded
            assert output_frames["upstream_data"] == upstream_frame

    def test_no_deduped_channel_when_disabled(self, sample_frame, temp_workdir):
        """Test that deduped channel is not created when disabled."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,
            'ssim_threshold': 0.9,
            'roi': None,
            'output_folder': temp_workdir,
            'debug': False,
            'forward_deduped_frames': False,  # Disabled
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        
        # Mock the processors
        with patch('filter_frame_dedup.filter.HashFrameProcessor') as mock_hash_class, \
             patch('filter_frame_dedup.filter.SSIMProcessor') as mock_ssim_class:
            
            mock_hash_processor = MagicMock()
            mock_hash_processor.should_process_frame.return_value = True
            mock_hash_class.return_value = mock_hash_processor
            
            mock_ssim_processor = MagicMock()
            mock_ssim_processor.should_save_frame.return_value = True
            mock_ssim_class.return_value = mock_ssim_processor
            
            filter_instance.setup(config)
            
            # Process frame
            frames = {"main": sample_frame}
            output_frames = filter_instance.process(frames)
            
            # Verify output
            assert "main" in output_frames
            assert "deduped" not in output_frames
            assert len(output_frames) == 1  # Only main

    def test_no_upstream_forwarding_when_disabled(self, sample_frame, temp_workdir):
        """Test that upstream data is not forwarded when disabled."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 0.1,
            'ssim_threshold': 0.9,
            'roi': None,
            'output_folder': temp_workdir,
            'debug': False,
            'forward_deduped_frames': True,
            'forward_upstream_data': False  # Disabled
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        
        # Mock the processors
        with patch('filter_frame_dedup.filter.HashFrameProcessor') as mock_hash_class, \
             patch('filter_frame_dedup.filter.SSIMProcessor') as mock_ssim_class:
            
            mock_hash_processor = MagicMock()
            mock_hash_processor.should_process_frame.return_value = True
            mock_hash_class.return_value = mock_hash_processor
            
            mock_ssim_processor = MagicMock()
            mock_ssim_processor.should_save_frame.return_value = True
            mock_ssim_class.return_value = mock_ssim_processor
            
            filter_instance.setup(config)
            
            # Create upstream frame
            upstream_frame = Frame(
                image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                data={"upstream": "data"},
                format='BGR'
            )
            
            # Process frame with upstream data
            frames = {
                "main": sample_frame,
                "upstream_data": upstream_frame
            }
            output_frames = filter_instance.process(frames)
            
            # Verify output
            assert "main" in output_frames
            assert "upstream_data" not in output_frames
            assert "deduped" in output_frames
            assert len(output_frames) == 2  # Only main and deduped

    def test_main_topic_ordering(self, sample_frame, temp_workdir):
        """Test that main topic always comes first in output dictionary."""
        config_data = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': temp_workdir,
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': True
        }
        
        config = FilterFrameDedup.normalize_config(config_data)
        filter_instance = FilterFrameDedup(config=config)
        filter_instance.setup(config)
        
        # Process frames with multiple topics in different order
        frames = {
            "stream2": sample_frame,
            "main": sample_frame,
            "other": sample_frame
        }
        output_frames = filter_instance.process(frames)
        
        # Verify main topic comes first regardless of input order
        output_keys = list(output_frames.keys())
        assert output_keys[0] == "main"
        assert len(output_frames) == 4  # main, stream2, other, and potentially deduped


if __name__ == '__main__':
    pytest.main([__file__])
