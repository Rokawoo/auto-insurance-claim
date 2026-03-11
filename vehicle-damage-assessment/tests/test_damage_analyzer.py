"""Unit tests for the heuristic DamageAnalyzer."""

import cv2
import numpy as np
import pytest

# Assuming the file is in the same directory for tests, adjust import as needed
from damage_analyzer import (
    DamageAnalyzer,
    DamageRegion,
    DamageType,
    AnalysisResult,
)

@pytest.fixture
def analyzer():
    return DamageAnalyzer()

@pytest.fixture
def blank_diff():
    return np.zeros((320, 320), dtype=np.uint8)

@pytest.fixture
def vehicle_mask():
    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[40:280, 40:280] = 255
    return mask

def make_contour_from_rect(x, y, w, h):
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)

def make_contour_circle(cx, cy, r, n_points=64):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
    return pts.astype(np.int32).reshape(-1, 1, 2)


class TestDamageAnalyzerNoContours:
    def test_empty_contours_returns_none_severity(self, analyzer, blank_diff):
        result = analyzer.analyze([], blank_diff)
        assert result.overall_severity == "none"
        assert result.overall_severity_score == 0.0
        assert result.total_damage_area_px == 0
        assert len(result.regions) == 0

    def test_empty_type_summary(self, analyzer, blank_diff):
        result = analyzer.analyze([], blank_diff)
        assert result.damage_type_summary == {}


class TestDamageAnalyzerClassification:
    def test_compact_region_classified_as_dent(self, analyzer, vehicle_mask):
        diff = np.zeros((320, 320), dtype=np.uint8)
        contour = make_contour_circle(160, 160, 30)
        cv2.drawContours(diff, [contour], -1, 120, thickness=-1)
        result = analyzer.analyze([contour], diff, vehicle_mask)
        assert len(result.regions) == 1
        assert result.regions[0].damage_type == DamageType.DENT

    def test_elongated_region_classified_as_scratch(self, analyzer, vehicle_mask):
        diff = np.zeros((320, 320), dtype=np.uint8)
        # Jagged scratch: elongated with concavities to drop solidity < 0.6
        contour = np.array([
            [[50, 150]], [[100, 148]], [[130, 155]], [[170, 147]],
            [[210, 152]], [[250, 150]], [[250, 156]], [[210, 158]],
            [[170, 153]], [[130, 161]], [[100, 154]], [[50, 156]],
        ], dtype=np.int32)
        cv2.drawContours(diff, [contour], -1, 150, thickness=-1)
        result = analyzer.analyze([contour], diff, vehicle_mask)
        assert len(result.regions) == 1
        assert result.regions[0].damage_type == DamageType.SCRATCH

    def test_faint_small_region_classified_as_scuff(self, analyzer, vehicle_mask):
        diff = np.zeros((320, 320), dtype=np.uint8)
        contour = make_contour_from_rect(100, 100, 30, 30)
        cv2.drawContours(diff, [contour], -1, 25, thickness=-1)
        result = analyzer.analyze([contour], diff, vehicle_mask)
        assert len(result.regions) == 1
        assert result.regions[0].damage_type == DamageType.SCUFF


class TestDamageAnalyzerSeverity:
    def test_larger_damage_higher_severity(self, analyzer, vehicle_mask):
        diff1, diff2 = np.zeros((320, 320), dtype=np.uint8), np.zeros((320, 320), dtype=np.uint8)
        small, large = make_contour_from_rect(100, 100, 20, 20), make_contour_from_rect(50, 50, 100, 100)
        cv2.drawContours(diff1, [small], -1, 120, thickness=-1)
        cv2.drawContours(diff2, [large], -1, 120, thickness=-1)
        
        small_result = analyzer.analyze([small], diff1, vehicle_mask)
        large_result = analyzer.analyze([large], diff2, vehicle_mask)
        assert large_result.overall_severity_score > small_result.overall_severity_score

    def test_more_regions_higher_severity(self, analyzer, vehicle_mask):
        diff_one = np.zeros((320, 320), dtype=np.uint8)
        c1 = make_contour_from_rect(100, 100, 40, 40)
        cv2.drawContours(diff_one, [c1], -1, 120, thickness=-1)
        result_one = analyzer.analyze([c1], diff_one, vehicle_mask)

        diff_many = np.zeros((320, 320), dtype=np.uint8)
        contours = [make_contour_from_rect(50 + i * 45, 100, 35, 35) for i in range(5)]
        for c in contours:
            cv2.drawContours(diff_many, [c], -1, 120, thickness=-1)
        result_many = analyzer.analyze(contours, diff_many, vehicle_mask)

        assert result_many.overall_severity_score > result_one.overall_severity_score

    def test_severity_labels_ordered(self, analyzer):
        labels = [analyzer._score_to_label(s) for s in [0.0, 0.08, 0.15, 0.35, 0.6, 0.95]]
        expected = ["none", "minor", "moderate", "severe", "critical", "critical"]
        assert labels == expected