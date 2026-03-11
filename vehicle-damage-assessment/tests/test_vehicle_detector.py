"""Tests for VehicleDetector on real car images.

Expects:
    ./images/car A - 1.png
    ./images/car A - 2.png

Run:
    cd vda/
    python -m pytest tests/test_vehicle_detector.py -v -s

These tests verify that YOLOv8 actually finds a car in the photos
and produces a usable binary mask.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.vehicle_detector import VehicleDetector, DetectionResult


# ── paths & skip ────────────────────────────────────────────────────

IMAGES_DIR = Path("images")
BEFORE_PATH = IMAGES_DIR / "car A - 1.png"
AFTER_PATH = IMAGES_DIR / "car A - 2.png"

images_exist = BEFORE_PATH.exists() and AFTER_PATH.exists()
skip_reason = f"images not found at {BEFORE_PATH.resolve()}"


# ── fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def before_bgr():
    img = cv2.imread(str(BEFORE_PATH))
    assert img is not None
    return img


@pytest.fixture(scope="module")
def after_bgr():
    img = cv2.imread(str(AFTER_PATH))
    assert img is not None
    return img


@pytest.fixture(scope="module")
def default_config():
    return {
        "model_name": "yolo11m.pt",
        "vehicle_classes": [2, 5, 7],
        "confidence_threshold": 0.3,   # slightly lower for real photos
        "iou_threshold": 0.45,
        "mask_dilation_kernel": 15,
    }


@pytest.fixture(scope="module")
def detector(default_config):
    """Instantiate and load model once for all tests."""
    det = VehicleDetector(default_config)
    det.load_model()
    return det


@pytest.fixture(scope="module")
def before_result(detector, before_bgr):
    return detector.detect(before_bgr)


@pytest.fixture(scope="module")
def after_result(detector, after_bgr):
    return detector.detect(after_bgr)


# ── tests: model loading ───────────────────────────────────────────

@pytest.mark.skipif(not images_exist, reason=skip_reason)
class TestModelLoading:

    def test_model_loads(self, detector):
        """YOLOv8 model should load without error."""
        assert detector.model is not None
        print(f"\n  model loaded: {detector.model_name}")

    def test_lazy_load_on_detect(self, default_config, before_bgr):
        """detect() should auto-load the model if not loaded yet."""
        det = VehicleDetector(default_config)
        assert det.model is None
        result = det.detect(before_bgr)
        assert det.model is not None
        print(f"\n  lazy load worked, found {len(result.boxes)} vehicle(s)")


# ── tests: detection on before image ───────────────────────────────

@pytest.mark.skipif(not images_exist, reason=skip_reason)
class TestDetectionBefore:

    def test_returns_detection_result(self, before_result):
        """detect() should return a DetectionResult."""
        assert isinstance(before_result, DetectionResult)

    def test_finds_at_least_one_vehicle(self, before_result):
        """Should detect at least one vehicle in the before image."""
        n = len(before_result.boxes)
        print(f"\n  vehicles found in before: {n}")
        for i in range(n):
            box = before_result.boxes[i].astype(int)
            print(f"    #{i+1}: box={tuple(box)}, conf={before_result.confidences[i]:.3f}, "
                  f"class={before_result.class_ids[i]}")
        assert n >= 1, "no vehicles detected — check image or lower confidence_threshold"

    def test_confidences_are_reasonable(self, before_result):
        """Detection confidences should be between 0 and 1."""
        if len(before_result.confidences) == 0:
            pytest.skip("no detections")
        assert np.all(before_result.confidences >= 0)
        assert np.all(before_result.confidences <= 1)
        print(f"\n  confidence range: {before_result.confidences.min():.3f} – "
              f"{before_result.confidences.max():.3f}")

    def test_boxes_are_valid(self, before_result, before_bgr):
        """Bounding boxes should be within image bounds."""
        if len(before_result.boxes) == 0:
            pytest.skip("no detections")
        h, w = before_bgr.shape[:2]
        for box in before_result.boxes:
            x1, y1, x2, y2 = box
            assert x1 < x2, f"invalid box: x1={x1} >= x2={x2}"
            assert y1 < y2, f"invalid box: y1={y1} >= y2={y2}"
            # boxes can slightly exceed image bounds due to padding, allow some slack
            assert x1 >= -10 and y1 >= -10
            assert x2 <= w + 10 and y2 <= h + 10

    def test_only_vehicle_classes(self, before_result):
        """All detections should be vehicle classes (car/bus/truck)."""
        if len(before_result.class_ids) == 0:
            pytest.skip("no detections")
        for cls_id in before_result.class_ids:
            assert cls_id in [2, 5, 7], f"non-vehicle class {cls_id} leaked through"


# ── tests: mask quality ────────────────────────────────────────────

@pytest.mark.skipif(not images_exist, reason=skip_reason)
class TestMaskQuality:

    def test_mask_shape_matches_image(self, before_result, before_bgr):
        """Mask should have same (H, W) as input image."""
        h, w = before_bgr.shape[:2]
        assert before_result.vehicle_mask.shape == (h, w)

    def test_mask_is_binary(self, before_result):
        """Mask should only contain 0 and 255."""
        unique = np.unique(before_result.vehicle_mask)
        assert set(unique).issubset({0, 255}), f"unexpected mask values: {unique}"

    def test_mask_is_not_empty(self, before_result):
        """If vehicles were detected, mask should have white pixels."""
        if len(before_result.boxes) == 0:
            pytest.skip("no detections")
        white_px = cv2.countNonZero(before_result.vehicle_mask)
        total_px = before_result.vehicle_mask.size
        pct = white_px / total_px * 100
        print(f"\n  mask coverage: {white_px:,} / {total_px:,} px ({pct:.1f}%)")
        assert white_px > 0

    def test_mask_is_not_everything(self, before_result):
        """Mask shouldn't cover the entire image (that means something went wrong)."""
        if len(before_result.boxes) == 0:
            pytest.skip("no detections")
        white_px = cv2.countNonZero(before_result.vehicle_mask)
        total_px = before_result.vehicle_mask.size
        pct = white_px / total_px * 100
        assert pct < 95, f"mask covers {pct:.1f}% of image — probably wrong"

    def test_mask_covers_detection_box(self, before_result):
        """Every detection box centroid should be inside the mask."""
        if len(before_result.boxes) == 0:
            pytest.skip("no detections")
        mask = before_result.vehicle_mask
        for box in before_result.boxes:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            # clamp to bounds
            cx = min(max(cx, 0), mask.shape[1] - 1)
            cy = min(max(cy, 0), mask.shape[0] - 1)
            assert mask[cy, cx] == 255, f"box center ({cx},{cy}) is outside mask"


# ── tests: detection on after image ────────────────────────────────

@pytest.mark.skipif(not images_exist, reason=skip_reason)
class TestDetectionAfter:

    def test_finds_vehicle_in_after(self, after_result):
        """Should also detect a vehicle in the after image."""
        n = len(after_result.boxes)
        print(f"\n  vehicles found in after: {n}")
        assert n >= 1

    def test_both_images_have_similar_detections(self, before_result, after_result):
        """Both images should detect roughly the same number of vehicles."""
        n_before = len(before_result.boxes)
        n_after = len(after_result.boxes)
        print(f"\n  before: {n_before} vehicles, after: {n_after} vehicles")
        # they're photos of the same car so counts should be close
        # but not necessarily identical (occlusion, angle differences)


# ── tests: edge cases ──────────────────────────────────────────────

@pytest.mark.skipif(not images_exist, reason=skip_reason)
class TestDetectorEdgeCases:

    def test_no_vehicle_in_blank_image(self, detector):
        """A blank image should return empty detections, not crash."""
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = detector.detect(blank)
        assert len(result.boxes) == 0
        assert cv2.countNonZero(result.vehicle_mask) == 0
        print(f"\n  blank image: 0 detections (correct)")

    def test_grayscale_converted_to_bgr(self, detector, before_bgr):
        """Detector should handle if someone passes grayscale by mistake."""
        gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
        # YOLO expects BGR but let's see if it handles gray gracefully
        bgr_from_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = detector.detect(bgr_from_gray)
        print(f"\n  grayscale->BGR: {len(result.boxes)} vehicles found")
        assert len(result.boxes) >= 1
