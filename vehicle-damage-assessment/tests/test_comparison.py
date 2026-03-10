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
        engine = DiffEngine(default_config)
        before, after = identical_pair

        result = engine.compare(before, after)

        assert isinstance(result, DiffResult)
        assert len(result.contours) == 0
        assert len(result.bounding_boxes) == 0
        assert result.damage_area_px == 0
        assert np.count_nonzero(result.cleaned_mask) == 0

    def test_known_damage_detected(self, default_config, pair_with_known_damage):
        """A known bright patch should be detected as a damage region."""
        engine = DiffEngine(default_config)
        before, after = pair_with_known_damage

        result = engine.compare(before, after)

        assert isinstance(result, DiffResult)
        assert len(result.contours) >= 1
        assert len(result.bounding_boxes) >= 1
        assert result.damage_area_px > 0

    def test_vehicle_mask_constrains_diff(self, default_config, pair_with_known_damage):
        """Damage outside the vehicle mask should be ignored."""
        engine = DiffEngine(default_config)
        before, after = pair_with_known_damage

        mask = np.zeros_like(before, dtype=np.uint8)
        mask[0:50, 0:50] = 255

        result = engine.compare(before, after, vehicle_mask=mask)

        assert len(result.contours) == 0
        assert len(result.bounding_boxes) == 0
        assert result.damage_area_px == 0
        assert np.count_nonzero(result.cleaned_mask) == 0


    def test_small_contours_filtered(self, default_config):
        """Contours below min_contour_area should be discarded."""
        engine = DiffEngine(default_config)

        before = np.full((320, 320), 100, dtype=np.uint8)
        after = before.copy()
        after[10:15, 10:15] = 200

        result = engine.compare(before, after)

        assert len(result.contours) == 0
        assert len(result.bounding_boxes) == 0

    def test_diff_result_fields(self, default_config, pair_with_known_damage):
        """DiffResult should have all expected fields populated."""
        engine = DiffEngine(default_config)
        before, after = pair_with_known_damage

        result = engine.compare(before, after)

        assert isinstance(result, DiffResult)
        assert result.raw_diff is not None
        assert result.thresh_diff is not None
        assert result.cleaned_mask is not None

        assert result.raw_diff.shape == before.shape
        assert result.thresh_diff.shape == before.shape
        assert result.cleaned_mask.shape == before.shape
