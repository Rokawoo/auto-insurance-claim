"""Unit tests for the image aligner (synthetic images).

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


def make_textured(h=480, w=640, seed=42):
    rng = np.random.RandomState(seed)
    img = rng.randint(60, 180, (h, w), dtype=np.uint8)
    for _ in range(80):
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        x2, y2 = rng.randint(0, w), rng.randint(0, h)
        c = int(rng.randint(0, 256))
        t = int(rng.randint(1, 4))
        s = rng.randint(0, 3)
        if s == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), c, t)
        elif s == 1:
            cv2.circle(img, (x1, y1), int(rng.randint(5, 50)), c, t)
        else:
            cv2.line(img, (x1, y1), (x2, y2), c, t)
    cv2.putText(img, "REFERENCE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
    return img


def apply_warp(image, tx=15, ty=10, angle_deg=2.0):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    H = np.array([[c, -s, (1-c)*cx + s*cy + tx],
                   [s,  c, -s*cx + (1-c)*cy + ty],
                   [0,  0, 1]], dtype=np.float64)
    return cv2.warpPerspective(image, H, (w, h), borderValue=0), H


@pytest.fixture
def cfg():
    return {"feature_method": "orb", "warp_method": "homography", "fallback": False}

@pytest.fixture
def cfg_sift():
    return {"feature_method": "sift", "warp_method": "homography", "fallback": False}

@pytest.fixture
def cfg_affine():
    return {"feature_method": "orb", "warp_method": "affine", "fallback": False}

@pytest.fixture
def cfg_fallback():
    return {"feature_method": "orb", "warp_method": "homography", "fallback": True}

@pytest.fixture
def img():
    return make_textured()

@pytest.fixture
def pair():
    before = make_textured(seed=42)
    after, H = apply_warp(before, tx=12, ty=8, angle_deg=1.5)
    return before, after, H


class TestInit:
    def test_defaults(self):
        a = ImageAligner({})
        assert a.feature_method == "orb"
        assert a.warp_method == WarpMethod.HOMOGRAPHY
        assert a.fallback is True
        assert a.normalize_exposure is True

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ImageAligner({"feature_method": "bad"})


class TestHomography:
    def test_result_type(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        assert isinstance(r, AlignmentResult)

    def test_warped_shape(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        assert r.warped_after.shape == b.shape

    def test_valid_mask_shape(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        assert r.valid_mask.shape == b.shape[:2]
        assert r.valid_mask.dtype == np.uint8

    def test_valid_mask_not_empty(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        pct = cv2.countNonZero(r.valid_mask) / r.valid_mask.size * 100
        print(f"\n  valid region: {pct:.1f}%")
        assert pct > 50

    def test_transform_3x3(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        assert r.transform.shape == (3, 3)

    def test_inliers(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        print(f"\n  inliers: {r.num_inliers}, ratio: {r.inlier_ratio:.2%}")
        assert r.num_inliers >= 10

    def test_reprojection_error(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        print(f"\n  reproj: {r.reprojection_error:.3f} px")
        assert r.reprojection_error < 5.0

    def test_reliable(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        assert r.is_reliable

    def test_feature_method_reported(self, cfg, pair):
        b, a, _ = pair
        r = ImageAligner(cfg).align(b, a)
        assert r.feature_method == "orb"

    def test_recovered_transform(self, cfg, pair):
        b, a, H_true = pair
        r = ImageAligner(cfg).align(b, a)
        h, w = b.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        H_inv = np.linalg.inv(H_true)
        expected = cv2.perspectiveTransform(corners, H_inv)
        recovered = cv2.perspectiveTransform(corners, r.transform)
        err = np.linalg.norm(recovered.reshape(-1,2) - expected.reshape(-1,2), axis=1).max()
        print(f"\n  max corner error: {err:.2f} px")
        assert err < 10.0

    def test_identity(self, cfg, img):
        r = ImageAligner(cfg).align(img, img.copy())
        diff = np.abs(r.transform - np.eye(3)).max()
        print(f"\n  identity deviation: {diff:.6f}")
        assert diff < 0.1

    def test_sift(self, cfg_sift, pair):
        b, a, _ = pair
        r = ImageAligner(cfg_sift).align(b, a)
        assert r.is_reliable
        assert r.feature_method == "sift"

    def test_bgr(self, cfg):
        gray = make_textured()
        bgr_b = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        warped, _ = apply_warp(gray)
        bgr_a = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        r = ImageAligner(cfg).align(bgr_b, bgr_a)
        assert r.warped_after.shape == bgr_b.shape


class TestAffine:
    def test_transform_2x3(self, cfg_affine, pair):
        b, a, _ = pair
        r = ImageAligner(cfg_affine).align(b, a)
        assert r.transform.shape == (2, 3)

    def test_has_valid_mask(self, cfg_affine, pair):
        b, a, _ = pair
        r = ImageAligner(cfg_affine).align(b, a)
        assert r.valid_mask.shape == b.shape[:2]

    def test_method_affine(self, cfg_affine, pair):
        b, a, _ = pair
        r = ImageAligner(cfg_affine).align(b, a)
        assert r.warp_method == WarpMethod.AFFINE


class TestFallback:
    def test_fallback_on_blank(self, cfg_fallback):
        """blank images with fallback should still fail (no detector can help)."""
        blank = np.full((480, 640), 128, dtype=np.uint8)
        with pytest.raises(RuntimeError, match="all methods"):
            ImageAligner(cfg_fallback).align(blank, blank)

    def test_fallback_reports_method(self, cfg_fallback, pair):
        """with a good pair, primary method should succeed (no fallback needed)."""
        b, a, _ = pair
        r = ImageAligner(cfg_fallback).align(b, a)
        assert isinstance(r, AlignmentResult)


class TestExposureNorm:
    def test_bright_shift(self, cfg, pair):
        """alignment should handle brightness differences."""
        b, a, _ = pair
        bright = cv2.convertScaleAbs(a, alpha=1.4, beta=40)
        r = ImageAligner(cfg).align(b, bright)
        print(f"\n  bright-shifted: inliers={r.num_inliers}, reproj={r.reprojection_error:.2f}")
        assert r.num_inliers >= 4

    def test_dark_shift(self, cfg, pair):
        """alignment should handle darker after image."""
        b, a, _ = pair
        dark = cv2.convertScaleAbs(a, alpha=0.6, beta=-20)
        r = ImageAligner(cfg).align(b, dark)
        print(f"\n  dark-shifted: inliers={r.num_inliers}, reproj={r.reprojection_error:.2f}")
        assert r.num_inliers >= 4


class TestErrors:
    def test_blank(self, cfg):
        blank = np.full((480, 640), 128, dtype=np.uint8)
        with pytest.raises(RuntimeError):
            ImageAligner(cfg).align(blank, blank)

    def test_unrelated_strict(self):
        a = make_textured(seed=1)
        b = make_textured(seed=9999)
        cfg = {"feature_method": "orb", "ratio_threshold": 0.5,
               "min_match_count": 50, "fallback": False}
        with pytest.raises(RuntimeError):
            ImageAligner(cfg).align(a, b)


class TestReliability:
    def test_good_pair_reliable(self, cfg, pair):
        b, a, _ = pair
        assert ImageAligner(cfg).align(b, a).is_reliable

    def test_strict_thresholds_unreliable(self, pair):
        b, a, _ = pair
        cfg = {"feature_method": "orb", "fallback": False,
               "min_inlier_ratio": 0.99, "max_reprojection_error": 0.001}
        r = ImageAligner(cfg).align(b, a)
        assert r.is_reliable is False
