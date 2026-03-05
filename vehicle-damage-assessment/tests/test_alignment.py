"""Unit tests for the image aligner (synthetic images).

Uses synthetic textured images with known ground-truth transforms
so we can verify the aligner recovers the correct warp.

Run:
    cd vda/
    python -m pytest tests/test_alignment.py -v -s
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alignment.aligner import ImageAligner, AlignmentResult, WarpMethod


# =====================================================================
# helpers
# =====================================================================

def make_textured_image(h=480, w=640, seed=42):
    """grayscale image with plenty of corners and edges for feature detection."""
    rng = np.random.RandomState(seed)
    img = rng.randint(60, 180, (h, w), dtype=np.uint8)

    for _ in range(80):
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        x2, y2 = rng.randint(0, w), rng.randint(0, h)
        color = int(rng.randint(0, 256))
        thickness = int(rng.randint(1, 4))
        shape = rng.randint(0, 3)
        if shape == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        elif shape == 1:
            cv2.circle(img, (x1, y1), int(rng.randint(5, 50)), color, thickness)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    cv2.putText(img, "REFERENCE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
    return img


def apply_known_warp(image, tx=15, ty=10, angle_deg=2.0):
    """apply a known rotation+translation and return (warped, 3x3 H)."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)

    H = np.array([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + tx],
        [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + ty],
        [0, 0, 1],
    ], dtype=np.float64)

    warped = cv2.warpPerspective(image, H, (w, h), borderValue=0)
    return warped, H


# =====================================================================
# fixtures
# =====================================================================

@pytest.fixture
def orb_config():
    return {
        "feature_method": "orb",
        "max_features": 5000,
        "match_method": "bf",
        "ratio_threshold": 0.75,
        "ransac_reproj_threshold": 5.0,
        "min_match_count": 10,
        "warp_method": "homography",
    }


@pytest.fixture
def sift_config():
    return {
        "feature_method": "sift",
        "max_features": 5000,
        "match_method": "bf",
        "ratio_threshold": 0.75,
        "ransac_reproj_threshold": 5.0,
        "min_match_count": 10,
        "warp_method": "homography",
    }


@pytest.fixture
def affine_config():
    return {
        "feature_method": "orb",
        "max_features": 5000,
        "match_method": "bf",
        "ratio_threshold": 0.75,
        "ransac_reproj_threshold": 5.0,
        "min_match_count": 10,
        "warp_method": "affine",
    }


@pytest.fixture
def textured_image():
    return make_textured_image()


@pytest.fixture
def textured_pair():
    before = make_textured_image(seed=42)
    after, H_true = apply_known_warp(before, tx=12, ty=8, angle_deg=1.5)
    return before, after, H_true


# =====================================================================
# init / validation
# =====================================================================

class TestInit:

    def test_orb_config(self, orb_config):
        a = ImageAligner(orb_config)
        assert a.feature_method == "orb"
        assert a.warp_method == WarpMethod.HOMOGRAPHY

    def test_sift_config(self, sift_config):
        a = ImageAligner(sift_config)
        assert a.feature_method == "sift"

    def test_affine_config(self, affine_config):
        a = ImageAligner(affine_config)
        assert a.warp_method == WarpMethod.AFFINE

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown feature method"):
            ImageAligner({"feature_method": "superglue"})

    def test_defaults(self):
        a = ImageAligner({})
        assert a.feature_method == "orb"
        assert a.warp_method == WarpMethod.HOMOGRAPHY
        assert a.ratio_threshold == 0.75


# =====================================================================
# homography alignment
# =====================================================================

class TestHomographyAlign:

    def test_returns_result(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        assert isinstance(result, AlignmentResult)

    def test_warped_shape(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        assert result.warped_after.shape == before.shape

    def test_transform_is_3x3(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        assert result.transform.shape == (3, 3)

    def test_method_is_homography(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        assert result.warp_method == WarpMethod.HOMOGRAPHY

    def test_inlier_count(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        print(f"\n  inliers: {result.num_inliers}")
        assert result.num_inliers >= 10

    def test_inlier_ratio(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        print(f"\n  inlier ratio: {result.inlier_ratio:.2%}")
        assert result.inlier_ratio > 0.3

    def test_reprojection_error(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        print(f"\n  reprojection error: {result.reprojection_error:.3f} px")
        assert result.reprojection_error < 5.0

    def test_is_reliable(self, orb_config, textured_pair):
        """mild warp on textured image should be flagged reliable."""
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        print(f"\n  reliable: {result.is_reliable} "
              f"(inliers={result.num_inliers}, ratio={result.inlier_ratio:.2%}, "
              f"reproj={result.reprojection_error:.2f}px)")
        assert result.is_reliable

    def test_recovered_transform_accuracy(self, orb_config, textured_pair):
        """recovered transform should closely match the inverse of the known warp."""
        before, after, H_true = textured_pair
        result = ImageAligner(orb_config).align(before, after)

        h, w = before.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        H_true_inv = np.linalg.inv(H_true)
        corners_expected = cv2.perspectiveTransform(corners, H_true_inv)
        corners_recovered = cv2.perspectiveTransform(corners, result.transform)

        errors = np.linalg.norm(
            corners_recovered.reshape(-1, 2) - corners_expected.reshape(-1, 2), axis=1
        )
        max_err = errors.max()
        print(f"\n  max corner error: {max_err:.2f} px")
        assert max_err < 10.0

    def test_identity_case(self, orb_config, textured_image):
        """aligning an image to itself should give near-identity transform."""
        result = ImageAligner(orb_config).align(textured_image, textured_image.copy())
        identity = np.eye(3)
        diff = np.abs(result.transform - identity).max()
        print(f"\n  max deviation from identity: {diff:.6f}")
        assert diff < 0.1

    def test_sift(self, sift_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(sift_config).align(before, after)
        print(f"\n  SIFT: {result.num_inliers} inliers, "
              f"reproj={result.reprojection_error:.3f}px")
        assert result.is_reliable

    def test_bgr_images(self, orb_config):
        gray = make_textured_image()
        bgr_before = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        warped, _ = apply_known_warp(gray, tx=10, ty=5)
        bgr_after = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        result = ImageAligner(orb_config).align(bgr_before, bgr_after)
        assert result.warped_after.shape == bgr_before.shape


# =====================================================================
# affine alignment
# =====================================================================

class TestAffineAlign:

    def test_returns_result(self, affine_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(affine_config).align(before, after)
        assert isinstance(result, AlignmentResult)

    def test_transform_is_2x3(self, affine_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(affine_config).align(before, after)
        assert result.transform.shape == (2, 3)

    def test_method_is_affine(self, affine_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(affine_config).align(before, after)
        assert result.warp_method == WarpMethod.AFFINE

    def test_has_quality_metrics(self, affine_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(affine_config).align(before, after)
        print(f"\n  affine: inliers={result.num_inliers}, "
              f"ratio={result.inlier_ratio:.2%}, "
              f"reproj={result.reprojection_error:.3f}px, "
              f"reliable={result.is_reliable}")
        assert result.num_inliers >= 4
        assert result.reprojection_error < 10.0


# =====================================================================
# error cases
# =====================================================================

class TestErrors:

    def test_blank_images(self, orb_config):
        blank = np.full((480, 640), 128, dtype=np.uint8)
        with pytest.raises(RuntimeError):
            ImageAligner(orb_config).align(blank, blank)

    def test_unrelated_images_strict(self):
        """completely different images with strict settings should fail."""
        a = make_textured_image(seed=1)
        b = make_textured_image(seed=9999)
        cfg = {
            "feature_method": "orb",
            "ratio_threshold": 0.5,
            "min_match_count": 50,
        }
        with pytest.raises(RuntimeError):
            ImageAligner(cfg).align(a, b)

    def test_small_images(self, orb_config):
        """tiny images — should either work or raise cleanly."""
        small = make_textured_image(h=30, w=40, seed=42)
        warped, _ = apply_known_warp(small, tx=1, ty=1, angle_deg=0.5)
        try:
            result = ImageAligner(orb_config).align(small, warped)
            assert isinstance(result, AlignmentResult)
        except RuntimeError:
            pass  # acceptable


# =====================================================================
# reliability flag
# =====================================================================

class TestReliability:

    def test_good_pair_is_reliable(self, orb_config, textured_pair):
        before, after, _ = textured_pair
        result = ImageAligner(orb_config).align(before, after)
        assert result.is_reliable is True

    def test_unreliable_with_strict_thresholds(self, textured_pair):
        """very strict thresholds should flag otherwise-ok alignment as unreliable."""
        before, after, _ = textured_pair
        cfg = {
            "feature_method": "orb",
            "max_features": 5000,
            "match_method": "bf",
            "ratio_threshold": 0.75,
            "ransac_reproj_threshold": 5.0,
            "min_match_count": 10,
            "warp_method": "homography",
            # absurdly strict quality requirements
            "min_inlier_ratio": 0.99,
            "max_reprojection_error": 0.001,
        }
        result = ImageAligner(cfg).align(before, after)
        # alignment should succeed but be flagged unreliable
        assert isinstance(result, AlignmentResult)
        assert result.is_reliable is False
        print(f"\n  correctly flagged unreliable: "
              f"ratio={result.inlier_ratio:.2%} (need 99%), "
              f"reproj={result.reprojection_error:.3f}px (need <0.001)")