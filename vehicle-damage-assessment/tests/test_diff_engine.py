"""Tests for DiffEngine — the masked pixel comparison module.

These tests demonstrate WHY the vehicle mask is critical.
They run the diff both with and without a mask and compare results.

Expects:
    ./images/car A - 1.png
    ./images/car A - 2.png

Run:
    cd vda/
    python -m pytest tests/test_diff_engine.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.comparison.diff_engine import DiffEngine, DiffResult


# ── paths ───────────────────────────────────────────────────────────

IMAGES_DIR = Path("images")
BEFORE_PATH = IMAGES_DIR / "car A - 1.png"
AFTER_PATH = IMAGES_DIR / "car A - 2.png"

images_exist = BEFORE_PATH.exists() and AFTER_PATH.exists()
skip_reason = f"images not found at {BEFORE_PATH.resolve()}"


# ── fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gray_pair():
    """Load both images as grayscale, resize after to match before."""
    before = cv2.imread(str(BEFORE_PATH), cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(str(AFTER_PATH), cv2.IMREAD_GRAYSCALE)
    assert before is not None and after is not None
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    return before, after


@pytest.fixture(scope="module")
def vehicle_mask(gray_pair):
    """Run YOLOv8 to get a real vehicle mask."""
    before_bgr = cv2.imread(str(BEFORE_PATH))
    from src.detection.vehicle_detector import VehicleDetector
    det = VehicleDetector({"confidence_threshold": 0.3})
    det.load_model()
    result = det.detect(before_bgr)
    return result.vehicle_mask


@pytest.fixture
def default_config():
    return {
        "diff_method": "absolute",
        "threshold": 30,
        "morph_kernel": 5,
        "morph_iterations": 2,
        "min_contour_area": 300,
        "max_contour_area": 100000,
    }


# ── tests: basic diff engine functionality (synthetic) ─────────────

class TestDiffEngineSynthetic:
    """Tests with synthetic images — no real photos needed."""

    def test_identical_images_no_damage(self):
        """Two identical images should produce zero contours."""
        img = np.full((200, 200), 128, dtype=np.uint8)
        engine = DiffEngine({"threshold": 30, "min_contour_area": 10})
        result = engine.compare(img, img.copy())
        assert len(result.contours) == 0
        assert result.damage_area_px == 0
        print(f"\n  identical images: 0 contours (correct)")

    def test_known_damage_with_mask(self):
        """Damage inside the mask should be found."""
        before = np.full((200, 300), 100, dtype=np.uint8)
        after = before.copy()
        # add a bright patch (simulated damage)
        after[60:100, 100:180] = 220

        # mask covers the damage region
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[40:120, 80:200] = 255

        engine = DiffEngine({"threshold": 30, "min_contour_area": 50})
        result = engine.compare(before, after, vehicle_mask=mask)
        assert len(result.contours) >= 1
        print(f"\n  damage inside mask: {len(result.contours)} contour(s) found")

    def test_damage_outside_mask_is_ignored(self):
        """Damage OUTSIDE the vehicle mask should be invisible."""
        before = np.full((200, 300), 100, dtype=np.uint8)
        after = before.copy()
        # damage at top-left
        after[10:30, 10:50] = 220

        # mask only covers bottom-right (away from damage)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[150:200, 200:300] = 255

        engine = DiffEngine({"threshold": 30, "min_contour_area": 10})
        result = engine.compare(before, after, vehicle_mask=mask)
        assert len(result.contours) == 0
        print(f"\n  damage outside mask: 0 contours (correctly ignored)")

    def test_no_mask_picks_up_everything(self):
        """Without a mask, even background changes become 'damage'."""
        before = np.full((200, 300), 100, dtype=np.uint8)
        after = before.copy()
        # "damage" on the car (center)
        after[80:120, 120:180] = 220
        # "background change" (top-left corner — not the car)
        after[5:25, 5:45] = 200

        engine = DiffEngine({"threshold": 30, "min_contour_area": 50})
        result_no_mask = engine.compare(before, after, vehicle_mask=None)

        # with mask only covering center
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[60:140, 100:200] = 255
        result_masked = engine.compare(before, after, vehicle_mask=mask)

        print(f"\n  no mask: {len(result_no_mask.contours)} regions "
              f"({result_no_mask.damage_area_px}px)")
        print(f"  masked:  {len(result_masked.contours)} regions "
              f"({result_masked.damage_area_px}px)")

        # unmasked should find both regions, masked should find fewer
        assert result_no_mask.damage_area_px >= result_masked.damage_area_px

    def test_small_contours_filtered(self):
        """Contours below min_contour_area should be discarded."""
        before = np.full((200, 300), 100, dtype=np.uint8)
        after = before.copy()
        # tiny 3x3 "damage" spot
        after[100:103, 100:103] = 250

        engine = DiffEngine({"threshold": 30, "min_contour_area": 500})
        result = engine.compare(before, after)
        assert len(result.contours) == 0
        print(f"\n  tiny spot with min_area=500: filtered out (correct)")

    def test_result_fields_populated(self):
        """DiffResult should have all fields properly set."""
        before = np.full((200, 300), 100, dtype=np.uint8)
        after = before.copy()
        after[80:120, 120:180] = 220

        engine = DiffEngine({"threshold": 30, "min_contour_area": 50})
        result = engine.compare(before, after)

        assert result.raw_diff.shape == before.shape
        assert result.thresh_diff.shape == before.shape
        assert result.cleaned_mask.shape == before.shape
        assert isinstance(result.bounding_boxes, list)
        assert isinstance(result.damage_area_px, int)


# ── tests: real images — the money tests ───────────────────────────

@pytest.mark.skipif(not images_exist, reason=skip_reason)
class TestDiffOnRealImages:
    """The critical tests: does masking actually help on real photos?"""

    def test_unmasked_diff_is_noisy(self, default_config, gray_pair):
        """Without a vehicle mask, the diff should be full of noise."""
        before, after = gray_pair
        engine = DiffEngine(default_config)
        result = engine.compare(before, after, vehicle_mask=None)
        total = before.size
        noise_pct = result.damage_area_px / total * 100
        print(f"\n  unmasked diff: {len(result.contours)} regions, "
              f"{result.damage_area_px:,}px ({noise_pct:.1f}%)")
        # this number will be high — that's the point

    def test_masked_diff_has_fewer_regions(self, default_config, gray_pair, vehicle_mask):
        """With the vehicle mask, diff should be dramatically cleaner."""
        before, after = gray_pair
        engine = DiffEngine(default_config)

        result_unmasked = engine.compare(before, after, vehicle_mask=None)
        result_masked = engine.compare(before, after, vehicle_mask=vehicle_mask)

        total = before.size
        unmask_pct = result_unmasked.damage_area_px / total * 100
        mask_pct = result_masked.damage_area_px / total * 100

        print(f"\n  unmasked: {result_unmasked.damage_area_px:,}px ({unmask_pct:.1f}%) "
              f"in {len(result_unmasked.contours)} regions")
        print(f"  masked:   {result_masked.damage_area_px:,}px ({mask_pct:.1f}%) "
              f"in {len(result_masked.contours)} regions")

        if result_unmasked.damage_area_px > 0:
            reduction = (1 - result_masked.damage_area_px / result_unmasked.damage_area_px) * 100
            print(f"  reduction: {reduction:.1f}%")

        # the masked version should have fewer or equal damage pixels
        assert result_masked.damage_area_px <= result_unmasked.damage_area_px

    def test_mask_not_blank(self, vehicle_mask):
        """Sanity check: the vehicle mask should actually have content."""
        white = cv2.countNonZero(vehicle_mask)
        print(f"\n  vehicle mask: {white:,} white pixels "
              f"({white / vehicle_mask.size * 100:.1f}%)")
        assert white > 0, "vehicle mask is empty — YOLO didn't find a car"

    def test_contours_have_bounding_boxes(self, default_config, gray_pair, vehicle_mask):
        """Every contour should have a corresponding bounding box."""
        before, after = gray_pair
        engine = DiffEngine(default_config)
        result = engine.compare(before, after, vehicle_mask=vehicle_mask)
        assert len(result.contours) == len(result.bounding_boxes)
        for bbox in result.bounding_boxes:
            x, y, w, h = bbox
            assert w > 0 and h > 0
        print(f"\n  {len(result.bounding_boxes)} bounding boxes, all valid")
