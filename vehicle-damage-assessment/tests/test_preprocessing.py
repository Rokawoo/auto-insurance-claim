"""Unit tests for the preprocessing module."""

import numpy as np
import pytest

# Assuming the file above is named preprocessor.py
from preprocessor import Preprocessor


@pytest.fixture
def default_config():
    """Minimal preprocessing config for testing."""
    return {
        "target_size": [320, 320], # [width, height]
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
        """Preprocessor initializes without error and assigns attributes."""
        processor = Preprocessor(default_config)
        assert processor.target_size == (320, 320)
        assert processor.grayscale is True
        assert processor.blur_kernel == 5

    def test_process_returns_correct_shape(self, default_config, sample_bgr_image):
        """process() should return an image of target_size."""
        processor = Preprocessor(default_config)
        processed = processor.process(sample_bgr_image)
        # Note: OpenCV resizes to (width, height), but shape is (height, width)
        assert processed.shape[:2] == (320, 320)

    def test_process_grayscale_single_channel(self, default_config, sample_bgr_image):
        """When grayscale=True, output should be single-channel."""
        processor = Preprocessor(default_config)
        processed = processor.process(sample_bgr_image)
        assert processed.ndim == 2

    def test_process_pair_returns_two_images(self, default_config, sample_bgr_image):
        """process_pair should return a tuple of two processed images."""
        processor = Preprocessor(default_config)
        before, after = processor.process_pair(sample_bgr_image, sample_bgr_image)
        assert before.shape == (320, 320)
        assert after.shape == (320, 320)

    def test_process_no_grayscale(self, default_config, sample_bgr_image):
        """When grayscale=False, output should remain 3-channel (and CLAHE should still work)."""
        default_config["grayscale"] = False
        processor = Preprocessor(default_config)
        processed = processor.process(sample_bgr_image)
        assert processed.ndim == 3
        assert processed.shape == (320, 320, 3)

    def test_blur_reduces_noise(self, default_config):
        """Blurring a noisy image should reduce pixel variance."""
        # Create a flat gray image with high noise
        base_gray = np.full((480, 640, 3), 128, dtype=np.uint8)
        noise = np.random.normal(0, 25, (480, 640, 3)).astype(np.uint8)
        noisy_image = cv2.add(base_gray, noise)

        # Process without blur
        config_no_blur = default_config.copy()
        config_no_blur["blur_kernel"] = 0
        processor_no_blur = Preprocessor(config_no_blur)
        out_no_blur = processor_no_blur.process(noisy_image)

        # Process with blur
        config_blur = default_config.copy()
        config_blur["blur_kernel"] = 9 # high blur
        processor_blur = Preprocessor(config_blur)
        out_blur = processor_blur.process(noisy_image)

        # The variance (spread of pixel values) should be lower in the blurred image
        assert np.var(out_blur) < np.var(out_no_blur)