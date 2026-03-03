"""Unit tests for the alignment module."""

import cv2
import numpy as np
import pytest

from src.alignment import ImageAligner
from src.alignment.aligner import AlignmentResult


@pytest.fixture
def default_config():
    return {
        "feature_method": "orb",
        "max_features": 5000,
        "match_method": "bf",
        "ratio_threshold": 0.75,
        "ransac_reproj_threshold": 5.0,
        "min_match_count": 10,
    }


@pytest.fixture
def textured_pair():
    """Create a synthetic pair where 'after' is a known warp of 'before'.

    This lets us verify alignment recovers the correct homography.
    """
    # TODO:
    #   1. create a richly textured image (e.g., random patches)
    #   2. apply a known small perspective warp to create 'after'
    #   3. return (before, after, known_homography)
    pass


class TestImageAligner:

    def test_invalid_feature_method_raises(self):
        """Should raise ValueError for unsupported detector."""
        # TODO: pass feature_method="foobar", expect ValueError
        pass

    def test_align_returns_alignment_result(self, default_config, textured_pair):
        """align() should return an AlignmentResult dataclass."""
        # TODO: run align, assert isinstance(result, AlignmentResult)
        pass

    def test_warped_shape_matches_before(self, default_config, textured_pair):
        """Warped after image should have the same shape as before."""
        # TODO: assert result.warped_after.shape == before.shape
        pass

    def test_too_few_matches_raises(self, default_config):
        """Should raise RuntimeError when images have no common features."""
        # TODO: use two completely different images, expect RuntimeError
        pass
