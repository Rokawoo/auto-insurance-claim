"""Advanced heuristic damage scoring with certainty classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import cv2
import numpy as np


class DamageType(str, Enum):
    TRUE = "damage"
    UNCERTAIN = "uncertain"


@dataclass
class DamageRegion:
    contour: np.ndarray = field(repr=False)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    area_px: int = 0
    perimeter: float = 0.0
    damage_type: DamageType = DamageType.UNCERTAIN
    severity_score: float = 0.0
    confidence: float = 0.0
    mean_intensity: float = 0.0
    gradient_strength: float = 0.0
    texture_variance: float = 0.0
    circularity: float = 0.0
    aspect_ratio: float = 1.0
    solidity: float = 1.0


@dataclass
class AnalysisResult:
    regions: list[DamageRegion] = field(default_factory=list)
    total_damage_area_px: int = 0
    overall_severity: str = "none"
    overall_severity_score: float = 0.0
    damage_type_summary: dict[str, int] = field(default_factory=dict)


class DamageAnalyzer:

    SEVERITY_LABELS = [
        (0.05, "none"),
        (0.10, "minor"),
        (0.30, "moderate"),
        (0.55, "severe"),
        (1.01, "critical"),
    ]

    def __init__(self, config: dict | None = None) -> None:

        cfg = config or {}

        self.intensity_weight = cfg.get("intensity_weight", 0.35)
        self.area_weight = cfg.get("area_weight", 0.35)
        self.gradient_weight = cfg.get("gradient_weight", 0.20)
        self.texture_weight = cfg.get("texture_weight", 0.10)

        total = (
            self.intensity_weight
            + self.area_weight
            + self.gradient_weight
            + self.texture_weight
        )

        if not math.isclose(total, 1.0, abs_tol=1e-5):
            self.intensity_weight /= total
            self.area_weight /= total
            self.gradient_weight /= total
            self.texture_weight /= total

        self.min_area = cfg.get("min_region_area", 120)
        self.max_aspect_ratio = cfg.get("max_aspect_ratio", 6.0)
        self.min_solidity = cfg.get("min_solidity", 0.55)
        self.edge_margin = cfg.get("edge_margin", 8)

        self.true_damage_threshold = cfg.get("true_damage_threshold", 0.55)

    # ------------------------------------------------------------------

    def analyze(
        self,
        contours: Sequence[np.ndarray],
        diff_image: np.ndarray,
        vehicle_mask: np.ndarray | None = None,
    ) -> AnalysisResult:

        if not contours:
            return AnalysisResult()

        if vehicle_mask is not None:
            vehicle_area = int(np.count_nonzero(vehicle_mask))
        else:
            vehicle_area = diff_image.shape[0] * diff_image.shape[1]

        vehicle_area = max(vehicle_area, 1)

        regions: list[DamageRegion] = []

        grad = cv2.Sobel(diff_image, cv2.CV_32F, 1, 1, ksize=3)
        grad = cv2.convertScaleAbs(grad)

        for contour in contours:

            region = self._analyze_region(
                contour,
                diff_image,
                grad,
                vehicle_area,
                vehicle_mask,
            )

            if region:
                regions.append(region)

        total_area = sum(r.area_px for r in regions)

        overall_score = self._compute_overall_severity(
            regions,
            total_area,
            vehicle_area,
        )

        summary = {"damage": 0, "uncertain": 0}

        for r in regions:
            summary[r.damage_type.value] += 1

        return AnalysisResult(
            regions=regions,
            total_damage_area_px=total_area,
            overall_severity=self._score_to_label(overall_score),
            overall_severity_score=round(overall_score, 4),
            damage_type_summary=summary,
        )

    # ------------------------------------------------------------------

    def _analyze_region(
        self,
        contour,
        diff_image,
        grad,
        vehicle_area,
        vehicle_mask,
    ) -> DamageRegion | None:

        area = cv2.contourArea(contour)

        if area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)

        perimeter = cv2.arcLength(contour, True)

        circularity = (4 * np.pi * area) / max(perimeter * perimeter, 1e-5)

        aspect_ratio = max(w, h) / max(min(w, h), 1e-5)

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1e-5)

        if aspect_ratio > self.max_aspect_ratio:
            return None

        if solidity < self.min_solidity:
            return None

        if vehicle_mask is not None:
            if (
                x < self.edge_margin
                or y < self.edge_margin
                or x + w >= vehicle_mask.shape[1] - self.edge_margin
                or y + h >= vehicle_mask.shape[0] - self.edge_margin
            ):
                return None

        roi_diff = diff_image[y:y+h, x:x+w]
        roi_grad = grad[y:y+h, x:x+w]

        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - [x, y]

        cv2.drawContours(mask, [shifted], -1, 255, -1)

        mean_intensity = float(cv2.mean(roi_diff, mask=mask)[0])
        gradient_strength = float(cv2.mean(roi_grad, mask=mask)[0])

        pixels = roi_diff[mask > 0]
        texture_variance = float(np.var(pixels)) if pixels.size else 0.0

        area_frac = area / vehicle_area

        severity = self._region_severity(
            area_frac,
            mean_intensity,
            gradient_strength,
            texture_variance,
        )

        confidence = self._confidence_score(
            area_frac,
            mean_intensity,
            gradient_strength,
            texture_variance,
        )

        label = (
            DamageType.TRUE
            if confidence >= self.true_damage_threshold
            else DamageType.UNCERTAIN
        )

        return DamageRegion(
            contour=contour,
            bbox=(x, y, w, h),
            area_px=int(area),
            perimeter=round(perimeter, 2),
            damage_type=label,
            severity_score=round(severity, 4),
            confidence=round(confidence, 4),
            mean_intensity=round(mean_intensity, 2),
            gradient_strength=round(gradient_strength, 2),
            texture_variance=round(texture_variance, 2),
            circularity=round(circularity, 4),
            aspect_ratio=round(aspect_ratio, 4),
            solidity=round(solidity, 4),
        )

    # ------------------------------------------------------------------

    def _confidence_score(self, area, intensity, gradient, texture):

        area_score = min(area / 0.12, 1.0)
        intensity_score = min(intensity / 180.0, 1.0)
        gradient_score = min(gradient / 120.0, 1.0)
        texture_score = min(texture / 2000.0, 1.0)

        return float(
            0.35 * area_score
            + 0.30 * intensity_score
            + 0.20 * gradient_score
            + 0.15 * texture_score
        )

    # ------------------------------------------------------------------

    def _region_severity(self, area, intensity, gradient, texture):

        area_score = min(area / 0.12, 1.0)
        intensity_score = min(intensity / 180.0, 1.0)
        gradient_score = min(gradient / 120.0, 1.0)
        texture_score = min(texture / 2000.0, 1.0)

        score = (
            self.area_weight * area_score
            + self.intensity_weight * intensity_score
            + self.gradient_weight * gradient_score
            + self.texture_weight * texture_score
        )

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------

    def _compute_overall_severity(self, regions, total_area, vehicle_area):

        if not regions:
            return 0.0

        top = sorted(regions, key=lambda r: r.severity_score, reverse=True)[:3]

        intensity_component = sum(r.severity_score for r in top) / len(top)

        area_component = min((total_area / vehicle_area) / 0.25, 1.0)

        count_component = min(len(regions) / 8.0, 1.0)

        score = 0.5 * intensity_component + 0.35 * area_component + 0.15 * count_component

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------

    def _score_to_label(self, score):

        for threshold, label in self.SEVERITY_LABELS:
            if score < threshold:
                return label

        return "critical"
