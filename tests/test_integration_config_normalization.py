"""
Integration tests for FilterFrameDedup configuration normalization.

These tests verify that the normalize_config method properly handles various
configuration inputs, validates parameters, and provides helpful error messages.
"""

import pytest
import os
from filter_frame_dedup.filter import FilterFrameDedup, FilterFrameDedupConfig


class TestIntegrationConfigNormalization:
    """Test comprehensive configuration normalization scenarios."""

    def test_string_to_type_conversions(self):
        """Test that string configurations are properly converted to correct types."""
        
        # Test that the normalize_config method preserves string types as-is
        # (The actual conversion happens in the parent class)
        config_with_string_bool = {
            'hash_threshold': '5',
            'motion_threshold': '1200',
            'min_time_between_frames': '1.5',  # String float
            'ssim_threshold': '0.85',  # String float
            'roi': '(100, 100, 200, 200)',  # String tuple
            'output_folder': '/tmp/test_output',
            'debug': 'true'  # String bool
        }
        
        normalized = FilterFrameDedup.normalize_config(config_with_string_bool)
        
        # Check that string values are preserved
        assert isinstance(normalized.hash_threshold, int)
        assert normalized.hash_threshold == 5
        assert isinstance(normalized.motion_threshold, int)
        assert normalized.motion_threshold == 1200
        assert isinstance(normalized.min_time_between_frames, float)
        assert normalized.min_time_between_frames == 1.5
        assert isinstance(normalized.ssim_threshold, float)
        assert normalized.ssim_threshold == 0.85
        assert isinstance(normalized.output_folder, str)
        assert normalized.output_folder == '/tmp/test_output'
        assert isinstance(normalized.debug, bool)
        assert normalized.debug is True

    def test_boolean_flags_validation(self):
        """Test validation of new boolean flags."""
        
        # Test forward_deduped_frames flag
        config_deduped_true = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'forward_deduped_frames': True
        }
        
        normalized = FilterFrameDedup.normalize_config(config_deduped_true)
        assert normalized.forward_deduped_frames is True
        
        # Test forward_deduped_frames as string
        config_deduped_string = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'forward_deduped_frames': 'true'
        }
        
        normalized = FilterFrameDedup.normalize_config(config_deduped_string)
        assert normalized.forward_deduped_frames is True
        
        # Test forward_upstream_data flag
        config_upstream_false = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'forward_upstream_data': False
        }
        
        normalized = FilterFrameDedup.normalize_config(config_upstream_false)
        assert normalized.forward_upstream_data is False
        
        # Test forward_upstream_data as string
        config_upstream_string = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'forward_upstream_data': 'false'
        }
        
        normalized = FilterFrameDedup.normalize_config(config_upstream_string)
        assert normalized.forward_upstream_data is False

    def test_comprehensive_configuration_with_new_features(self):
        """Test a comprehensive configuration with all parameters including new features."""
        
        comprehensive_config = {
            'hash_threshold': 10,
            'motion_threshold': 1500,
            'min_time_between_frames': 2.0,
            'ssim_threshold': 0.85,
            'roi': (100, 100, 200, 200),
            'output_folder': '/tmp/comprehensive_test',
            'debug': True,
            'forward_deduped_frames': True,
            'forward_upstream_data': False
        }
        
        normalized = FilterFrameDedup.normalize_config(comprehensive_config)
        
        # Verify all parameters are correctly set
        assert normalized.hash_threshold == 10
        assert normalized.motion_threshold == 1500
        assert normalized.min_time_between_frames == 2.0
        assert normalized.ssim_threshold == 0.85
        assert normalized.roi == (100, 100, 200, 200)
        assert normalized.output_folder == '/tmp/comprehensive_test'
        assert normalized.debug is True
        assert normalized.forward_deduped_frames is True
        assert normalized.forward_upstream_data is False

    def test_required_vs_optional_parameters(self):
        """Test that required parameters are validated correctly."""
        
        # Test minimal valid configuration
        minimal_config = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/output'
        }
        
        normalized = FilterFrameDedup.normalize_config(minimal_config)
        assert normalized.hash_threshold == 5
        assert normalized.motion_threshold == 1200
        assert normalized.min_time_between_frames == 1.0
        assert normalized.ssim_threshold == 0.90
        assert normalized.output_folder == '/tmp/output'
        assert normalized.roi is None  # Default value
        assert normalized.debug is False  # Default value
        assert normalized.forward_deduped_frames is False  # Default value
        assert normalized.forward_upstream_data is True  # Default value

    def test_threshold_validation(self):
        """Test threshold validation and conversion."""
        
        # Test valid thresholds
        valid_config = {
            'hash_threshold': 3,
            'motion_threshold': 1000,
            'min_time_between_frames': 0.5,
            'ssim_threshold': 0.95,
            'output_folder': '/tmp/test'
        }
        
        normalized = FilterFrameDedup.normalize_config(valid_config)
        assert normalized.hash_threshold == 3
        assert normalized.motion_threshold == 1000
        assert normalized.min_time_between_frames == 0.5
        assert normalized.ssim_threshold == 0.95
        
        # Test negative hash threshold
        invalid_hash = {
            'hash_threshold': -1,
            'motion_threshold': 1000,
            'min_time_between_frames': 0.5,
            'ssim_threshold': 0.95,
            'output_folder': '/tmp/test'
        }
        
        with pytest.raises(ValueError, match="Hash threshold must be non-negative"):
            FilterFrameDedup.normalize_config(invalid_hash)
        
        # Test negative motion threshold
        invalid_motion = {
            'hash_threshold': 5,
            'motion_threshold': -100,
            'min_time_between_frames': 0.5,
            'ssim_threshold': 0.95,
            'output_folder': '/tmp/test'
        }
        
        with pytest.raises(ValueError, match="Motion threshold must be non-negative"):
            FilterFrameDedup.normalize_config(invalid_motion)
        
        # Test negative time threshold
        invalid_time = {
            'hash_threshold': 5,
            'motion_threshold': 1000,
            'min_time_between_frames': -0.5,
            'ssim_threshold': 0.95,
            'output_folder': '/tmp/test'
        }
        
        with pytest.raises(ValueError, match="Minimum time between frames must be non-negative"):
            FilterFrameDedup.normalize_config(invalid_time)

    def test_ssim_threshold_validation(self):
        """Test SSIM threshold validation."""
        
        # Test valid SSIM thresholds
        valid_ssim_configs = [
            {'ssim_threshold': 0.0, 'output_folder': '/tmp/test'},
            {'ssim_threshold': 0.5, 'output_folder': '/tmp/test'},
            {'ssim_threshold': 1.0, 'output_folder': '/tmp/test'}
        ]
        
        for config in valid_ssim_configs:
            config.update({
                'hash_threshold': 5,
                'motion_threshold': 1200,
                'min_time_between_frames': 1.0
            })
            normalized = FilterFrameDedup.normalize_config(config)
            assert normalized.ssim_threshold == config['ssim_threshold']
        
        # Test invalid SSIM thresholds
        invalid_ssim_configs = [
            {'ssim_threshold': -0.1, 'output_folder': '/tmp/test'},
            {'ssim_threshold': 1.1, 'output_folder': '/tmp/test'},
            {'ssim_threshold': 2.0, 'output_folder': '/tmp/test'}
        ]
        
        for config in invalid_ssim_configs:
            config.update({
                'hash_threshold': 5,
                'motion_threshold': 1200,
                'min_time_between_frames': 1.0
            })
            with pytest.raises(ValueError, match="SSIM threshold must be between 0 and 1"):
                FilterFrameDedup.normalize_config(config)

    def test_roi_configuration(self):
        """Test ROI configuration options."""
        
        # Test valid ROI
        config_with_roi = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'roi': (100, 100, 200, 200)
        }
        
        normalized = FilterFrameDedup.normalize_config(config_with_roi)
        assert normalized.roi == (100, 100, 200, 200)
        
        # Test None ROI
        config_no_roi = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'roi': None
        }
        
        normalized = FilterFrameDedup.normalize_config(config_no_roi)
        assert normalized.roi is None
        
        # Test invalid ROI (wrong length)
        config_invalid_roi_length = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'roi': (100, 100, 200)  # Only 3 values
        }
        
        with pytest.raises(ValueError, match="ROI must be a tuple of 4 values"):
            FilterFrameDedup.normalize_config(config_invalid_roi_length)
        
        # Test invalid ROI (negative dimensions)
        config_invalid_roi_dims = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'roi': (100, 100, -200, 200)  # Negative width
        }
        
        with pytest.raises(ValueError, match="ROI width and height must be positive"):
            FilterFrameDedup.normalize_config(config_invalid_roi_dims)

    def test_debug_mode_configuration(self):
        """Test debug mode configuration options."""
        
        # Test debug = True
        config_debug_true = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'debug': True
        }
        
        normalized = FilterFrameDedup.normalize_config(config_debug_true)
        assert normalized.debug is True
        
        # Test debug = False
        config_debug_false = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'debug': False
        }
        
        normalized = FilterFrameDedup.normalize_config(config_debug_false)
        assert normalized.debug is False
        
        # Test debug = "true" (string)
        config_debug_string_true = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'debug': "true"
        }
        
        normalized = FilterFrameDedup.normalize_config(config_debug_string_true)
        assert normalized.debug is True
        
        # Test debug = "false" (string)
        config_debug_string_false = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'debug': "false"
        }
        
        normalized = FilterFrameDedup.normalize_config(config_debug_string_false)
        assert normalized.debug is False
        
        # Test invalid debug value
        config_invalid_debug = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'debug': "maybe"
        }
        
        with pytest.raises(ValueError, match="Invalid debug mode"):
            FilterFrameDedup.normalize_config(config_invalid_debug)

    def test_environment_variable_loading(self):
        """Test environment variable configuration loading."""
        
        # Set environment variables
        os.environ['FILTER_HASH_THRESHOLD'] = '3'
        os.environ['FILTER_MOTION_THRESHOLD'] = '1000'
        os.environ['FILTER_MIN_TIME_BETWEEN_FRAMES'] = '0.5'
        os.environ['FILTER_SSIM_THRESHOLD'] = '0.85'
        os.environ['FILTER_ROI'] = '(50, 50, 150, 150)'
        os.environ['FILTER_OUTPUT_FOLDER'] = '/tmp/env_output'
        os.environ['FILTER_DEBUG'] = 'true'
        
        try:
            # Create config from environment variables
            config = {
                'hash_threshold': os.environ.get('FILTER_HASH_THRESHOLD', '5'),
                'motion_threshold': os.environ.get('FILTER_MOTION_THRESHOLD', '1200'),
                'min_time_between_frames': os.environ.get('FILTER_MIN_TIME_BETWEEN_FRAMES', '1.0'),
                'ssim_threshold': os.environ.get('FILTER_SSIM_THRESHOLD', '0.90'),
                'roi': os.environ.get('FILTER_ROI', 'None'),
                'output_folder': os.environ.get('FILTER_OUTPUT_FOLDER', '/tmp/output'),
                'debug': os.environ.get('FILTER_DEBUG', 'false')
            }
            normalized = FilterFrameDedup.normalize_config(config)
            
            assert normalized.hash_threshold == 3
            assert normalized.motion_threshold == 1000
            assert normalized.min_time_between_frames == 0.5
            assert normalized.ssim_threshold == 0.85
            assert normalized.roi == (50, 50, 150, 150)
            assert normalized.output_folder == '/tmp/env_output'
            assert normalized.debug is True
            
        finally:
            # Clean up environment variables
            for key in ['FILTER_HASH_THRESHOLD', 'FILTER_MOTION_THRESHOLD', 'FILTER_MIN_TIME_BETWEEN_FRAMES', 
                       'FILTER_SSIM_THRESHOLD', 'FILTER_ROI', 'FILTER_OUTPUT_FOLDER', 'FILTER_DEBUG']:
                if key in os.environ:
                    del os.environ[key]

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        
        # Test zero values (should be valid)
        config_zero_values = {
            'hash_threshold': 0,
            'motion_threshold': 0,
            'min_time_between_frames': 0.0,
            'ssim_threshold': 0.0,
            'output_folder': '/tmp/test'
        }
        
        normalized = FilterFrameDedup.normalize_config(config_zero_values)
        assert normalized.hash_threshold == 0
        assert normalized.motion_threshold == 0
        assert normalized.min_time_between_frames == 0.0
        assert normalized.ssim_threshold == 0.0
        
        # Test very large values
        config_large_values = {
            'hash_threshold': 1000000,
            'motion_threshold': 1000000,
            'min_time_between_frames': 3600.0,  # 1 hour
            'ssim_threshold': 1.0,
            'output_folder': '/tmp/test'
        }
        
        normalized = FilterFrameDedup.normalize_config(config_large_values)
        assert normalized.hash_threshold == 1000000
        assert normalized.motion_threshold == 1000000
        assert normalized.min_time_between_frames == 3600.0
        assert normalized.ssim_threshold == 1.0
        
        # Test empty string values
        config_empty_strings = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '',  # Empty string should be allowed
            'roi': None
        }
        
        normalized = FilterFrameDedup.normalize_config(config_empty_strings)
        assert normalized.output_folder == ''

    def test_unknown_config_key_validation(self):
        """Test that unknown configuration keys are handled gracefully."""
        
        # Test with a typo in a common parameter - should not raise error anymore
        config_with_typo = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'hash_threshhold': 3  # Typo: missing 'o'
        }
        
        # Should not raise an error - unknown keys are passed through
        config = FilterFrameDedup.normalize_config(config_with_typo)
        assert config.hash_threshold == 5
        assert config.output_folder == '/tmp/test'
        
        # Test with completely unknown key - should not raise error
        config_unknown = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'unknown_parameter': 'value'
        }
        
        # Should not raise an error - unknown keys are passed through
        config = FilterFrameDedup.normalize_config(config_unknown)
        assert config.hash_threshold == 5
        assert config.output_folder == '/tmp/test'

    def test_runtime_keys_ignored(self):
        """Test that OpenFilter runtime keys are ignored during validation."""
        
        # Test with runtime keys that should be ignored
        config_with_runtime_keys = {
            'hash_threshold': 5,
            'motion_threshold': 1200,
            'min_time_between_frames': 1.0,
            'ssim_threshold': 0.90,
            'output_folder': '/tmp/test',
            'pipeline_id': 'test_pipeline',  # Runtime key
            'device_name': 'test_device',    # Runtime key
            'log_path': '/tmp/logs',         # Runtime key
            'id': 'test_filter',             # Runtime key
            'sources': 'tcp://localhost:5550',  # Runtime key
            'outputs': 'tcp://localhost:5551',  # Runtime key
            'workdir': '/tmp/work'           # Runtime key
        }
        
        # Should not raise an error
        normalized = FilterFrameDedup.normalize_config(config_with_runtime_keys)
        assert normalized.hash_threshold == 5
        assert normalized.output_folder == '/tmp/test'

    def test_comprehensive_configuration(self):
        """Test a comprehensive configuration with all parameters."""
        
        comprehensive_config = {
            'hash_threshold': 3,
            'motion_threshold': 1000,
            'min_time_between_frames': 0.5,
            'ssim_threshold': 0.85,
            'roi': (100, 100, 200, 200),
            'output_folder': '/tmp/comprehensive_output',
            'debug': True
        }
        
        normalized = FilterFrameDedup.normalize_config(comprehensive_config)
        
        # Verify all parameters are correctly set
        assert normalized.hash_threshold == 3
        assert normalized.motion_threshold == 1000
        assert normalized.min_time_between_frames == 0.5
        assert normalized.ssim_threshold == 0.85
        assert normalized.roi == (100, 100, 200, 200)
        assert normalized.output_folder == '/tmp/comprehensive_output'
        assert normalized.debug is True
