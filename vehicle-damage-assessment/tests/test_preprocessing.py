"""Unit tests for the preprocessing module."""

import numpy as np
import pytest

from src.preprocessing import Preprocessor


@pytest.fixture
def default_config():
    """Minimal preprocessing config for testing."""
    return {
        "target_size": [320, 320],
        "grayscale": True,
        "blur_kernel": 5,
        "blur_sigma": 0,
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    }


@pytest.fixture
def sample_bgr_image():
    """Create a synthetic BGR test image (480×640×3)."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


class TestPreprocessor:
    """Tests for Preprocessor."""

    def test_init(self, default_config):
        """Preprocessor initializes without error."""
        # TODO: instantiate and assert attributes are set correctly
        preprocessor = Preprocessor(default_config)
        assert isinstance(preprocessor, Preprocessor)

    def test_process_returns_correct_shape(self, default_config, sample_bgr_image):
        """process() should return an image of target_size."""
        # TODO: call process, assert output shape matches target_size
        preprocessor = Preprocessor(default_config)
        output_image = preprocessor.process(sample_bgr_image)
        assert tuple(default_config["target_size"]) == output_image.shape

    def test_process_grayscale_single_channel(self, default_config, sample_bgr_image):
        """When grayscale=True, output should be single-channel."""
        # TODO: assert output.ndim == 2
        preprocessor = Preprocessor(default_config)
        output_image = preprocessor._to_grayscale(sample_bgr_image)
        assert output_image.ndim == 2

    def test_process_pair_returns_two_images(self, default_config, sample_bgr_image):
        """process_pair should return a tuple of two processed images."""
        # TODO: call process_pair with two images, check tuple length
        preprocessor = Preprocessor(default_config)
        output_image = preprocessor.process_pair(sample_bgr_image, sample_bgr_image)
        assert len(output_image) == 2

    def test_process_no_grayscale(self, default_config, sample_bgr_image):
        """When grayscale=False, output should remain 3-channel."""
        # TODO: modify config, assert output.ndim == 3
        copy = default_config
        copy["grayscale"] = False
        preprocessor = Preprocessor(copy)
        output_image = preprocessor.process(sample_bgr_image)
        assert output_image.ndim == 3
        
    def test_blur_reduces_noise(self, default_config, sample_bgr_image):
        """Blurring a noisy image should reduce pixel variance."""
        # TODO: create noisy image, process, compare variance before/after
        original = sample_bgr_image
        preprocessor = Preprocessor(default_config)
        output_image = preprocessor._blur(sample_bgr_image)
        original_variance = np.var(original)
        new_variance = np.var(output_image)
        assert original_variance > new_variance
