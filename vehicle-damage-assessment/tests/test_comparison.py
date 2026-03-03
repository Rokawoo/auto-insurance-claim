"""Unit tests for the comparison / diff engine module."""

import cv2
import numpy as np
import pytest

from src.comparison import DiffEngine
from src.comparison.diff_engine import DiffResult


@pytest.fixture
def default_config():
    return {
        "diff_method": "absolute",
        "threshold": 30,
        "morph_kernel": 5,
        "morph_iterations": 2,
        "min_contour_area": 100,
        "max_contour_area": 50000,
    }


@pytest.fixture
def identical_pair():
    """Two identical grayscale images — should produce zero diff."""
    img = np.full((320, 320), 128, dtype=np.uint8)
    return img, img.copy()


@pytest.fixture
def pair_with_known_damage():
    """Before image + after image with a bright rectangle (simulated damage)."""
    before = np.full((320, 320), 100, dtype=np.uint8)
    after = before.copy()
    # draw a "damage" rectangle
    after[100:150, 100:200] = 200
    return before, after


class TestDiffEngine:

    def test_identical_images_no_damage(self, default_config, identical_pair):
        """Identical images should produce no contours."""
        # TODO: run compare, assert len(result.contours) == 0
        pass

    def test_known_damage_detected(self, default_config, pair_with_known_damage):
        """A known bright patch should be detected as a damage region."""
        # TODO: run compare, assert at least one contour found
        pass

    def test_vehicle_mask_constrains_diff(self, default_config, pair_with_known_damage):
        """Damage outside the vehicle mask should be ignored."""
        # TODO:
        #   create a mask that does NOT cover the damage rectangle
        #   run compare with that mask
        #   assert no contours found
        pass

    def test_small_contours_filtered(self, default_config):
        """Contours below min_contour_area should be discarded."""
        # TODO: create pair with tiny diff spot, assert filtered out
        pass

    def test_diff_result_fields(self, default_config, pair_with_known_damage):
        """DiffResult should have all expected fields populated."""
        # TODO: assert raw_diff, thresh_diff, cleaned_mask are non-empty
        pass
