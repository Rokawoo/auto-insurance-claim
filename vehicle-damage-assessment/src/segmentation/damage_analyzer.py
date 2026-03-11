"""Heuristic damage analysis from diff contours — no trained model required.

Instead of a fine-tuned segmentation model, this module inspects the
geometric and intensity properties of each contour the DiffEngine found
and makes a best-effort classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import cv2
import numpy as np


class DamageType(str, Enum):
    """Supported damage type labels."""
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    SHATTER = "shatter"
    SCUFF = "scuff"
    UNKNOWN = "unknown"


@dataclass
class DamageRegion:
    """A single analyzed damage region."""
    contour: np.ndarray = field(repr=False)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    area_px: int = 0
    perimeter: float = 0.0
    damage_type: DamageType = DamageType.UNKNOWN
    severity_score: float = 0.0
    mean_intensity: float = 0.0
    circularity: float = 0.0
    aspect_ratio: float = 1.0
    solidity: float = 1.0


@dataclass
class AnalysisResult:
    """Complete damage analysis output."""
    regions: list[DamageRegion] = field(default_factory=list)
    total_damage_area_px: int = 0
    overall_severity: str = "none"
    overall_severity_score: float = 0.0
    damage_type_summary: dict[str, int] = field(default_factory=dict)


class DamageAnalyzer:
    """Classifies and scores damage regions using contour geometry heuristics."""

    SEVERITY_LABELS = [
        (0.05, "none"),
        (0.10, "minor"),
        (0.30, "moderate"),
        (0.55, "severe"),
        (1.01, "critical"),
    ]

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}

        # Severity scoring weights
        self.intensity_weight: float = cfg.get("intensity_weight", 0.40)
        self.area_weight: float = cfg.get("area_weight", 0.35)
        self.count_weight: float = cfg.get("count_weight", 0.25)

        # Ensure weights sum to exactly 1.0 to prevent score inflation
        total_weight = self.intensity_weight + self.area_weight + self.count_weight
        if not math.isclose(total_weight, 1.0, abs_tol=1e-5):
            self.intensity_weight /= total_weight
            self.area_weight /= total_weight
            self.count_weight /= total_weight

        # Classification thresholds
        self.scratch_min_aspect: float = cfg.get("scratch_min_aspect", 3.0)
        self.scratch_max_solidity: float = cfg.get("scratch_max_solidity", 0.6)
        self.dent_min_circularity: float = cfg.get("dent_min_circularity", 0.4)
        self.shatter_min_area_frac: float = cfg.get("shatter_min_area_frac", 0.05)
        self.scuff_max_intensity: float = cfg.get("scuff_max_intensity", 40.0)
        self.scuff_max_area: int = cfg.get("scuff_max_area", 2000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        contours: Sequence[np.ndarray],
        diff_image: np.ndarray,
        vehicle_mask: np.ndarray | None = None,
    ) -> AnalysisResult:
        """Analyze all damage contours from a DiffResult."""
        if not contours:
            return AnalysisResult()

        # Compute vehicle area safely
        if vehicle_mask is not None:
            vehicle_area = int(np.count_nonzero(vehicle_mask))
        else:
            vehicle_area = diff_image.shape[0] * diff_image.shape[1]
        
        vehicle_area = max(vehicle_area, 1)

        # Process regions
        regions = [self._analyze_region(c, diff_image, vehicle_area) for c in contours]
        
        # Aggregate stats
        total_area = sum(r.area_px for r in regions)
        type_summary = {}
        for r in regions:
            name = r.damage_type.value
            type_summary[name] = type_summary.get(name, 0) + 1

        overall_score = self._compute_overall_severity(regions, total_area, vehicle_area)
        
        return AnalysisResult(
            regions=regions,
            total_damage_area_px=total_area,
            overall_severity=self._score_to_label(overall_score),
            overall_severity_score=round(overall_score, 4),
            damage_type_summary=type_summary,
        )

    # ------------------------------------------------------------------
    # Internal: Per-region analysis
    # ------------------------------------------------------------------

    def _analyze_region(
        self, contour: np.ndarray, diff_image: np.ndarray, vehicle_area: int
    ) -> DamageRegion:
        """Compute geometric features and classify a single contour."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        x, y, w, h = cv2.boundingRect(contour)

        # Geometric features with division-by-zero protection
        circularity = (4.0 * np.pi * area) / max(perimeter * perimeter, 1e-5)
        aspect_ratio = max(w, h) / max(min(w, h), 1e-5)
        
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1e-5)

        # OPTIMIZATION: Localized Intensity Measurement
        # Shift contour to local ROI coordinates to avoid drawing massive masks
        roi_diff = diff_image[y:y+h, x:x+w]
        local_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=-1)
        
        mean_intensity = float(cv2.mean(roi_diff, mask=local_mask)[0])

        # Classify & Score
        area_frac = area / vehicle_area
        damage_type = self._classify_type(
            area, area_frac, circularity, aspect_ratio, solidity, mean_intensity
        )
        severity = self._region_severity(area_frac, mean_intensity, damage_type)

        return DamageRegion(
            contour=contour,
            bbox=(x, y, w, h),
            area_px=int(area),
            perimeter=round(perimeter, 2),
            damage_type=damage_type,
            severity_score=round(severity, 4),
            mean_intensity=round(mean_intensity, 2),
            circularity=round(circularity, 4),
            aspect_ratio=round(aspect_ratio, 4),
            solidity=round(solidity, 4),
        )

    def _classify_type(
        self, area: float, area_frac: float, circularity: float, 
        aspect_ratio: float, solidity: float, mean_intensity: float
    ) -> DamageType:
        """Heuristic damage type classification from geometric features."""
        if mean_intensity < self.scuff_max_intensity and area < self.scuff_max_area:
            return DamageType.SCUFF

        if aspect_ratio >= self.scratch_min_aspect and solidity <= self.scratch_max_solidity:
            return DamageType.SCRATCH

        if area_frac >= self.shatter_min_area_frac and solidity < 0.5:
            return DamageType.SHATTER

        if circularity >= self.dent_min_circularity and solidity > 0.7:
            return DamageType.DENT

        if circularity < 0.3 and solidity < 0.7:
            return DamageType.CRACK

        return DamageType.UNKNOWN

    def _region_severity(
        self, area_frac: float, mean_intensity: float, damage_type: DamageType
    ) -> float:
        """Score a single region's severity from 0.0 to 1.0."""
        area_score = min(area_frac / 0.15, 1.0)
        intensity_score = min(mean_intensity / 200.0, 1.0)

        type_multipliers = {
            DamageType.SCUFF: 0.3,
            DamageType.SCRATCH: 0.5,
            DamageType.DENT: 0.7,
            DamageType.CRACK: 0.85,
            DamageType.SHATTER: 1.0,
            DamageType.UNKNOWN: 0.5,
        }
        type_mult = type_multipliers.get(damage_type, 0.5)

        raw = 0.4 * area_score + 0.3 * intensity_score + 0.3 * type_mult
        return min(max(raw, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Internal: Overall severity
    # ------------------------------------------------------------------

    def _compute_overall_severity(
        self, regions: list[DamageRegion], total_damage_area: int, vehicle_area: int
    ) -> float:
        """Weighted composite severity score for the entire vehicle."""
        if not regions:
            return 0.0

        sorted_regions = sorted(regions, key=lambda r: r.severity_score, reverse=True)
        top_n = sorted_regions[:min(3, len(sorted_regions))]
        intensity_component = sum(r.severity_score for r in top_n) / len(top_n)

        area_component = min((total_damage_area / vehicle_area) / 0.25, 1.0)
        count_component = min(len(regions) / 10.0, 1.0)

        score = (
            self.intensity_weight * intensity_component
            + self.area_weight * area_component
            + self.count_weight * count_component
        )
        return min(max(score, 0.0), 1.0)

    def _score_to_label(self, score: float) -> str:
        """Convert a 0-1 score to a severity label."""
        for threshold, label in self.SEVERITY_LABELS:
            if score < threshold:
                return label
        return "critical"