"""Damage scoring with size filtering and confidence thresholding."""

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

    def __init__(self, config: dict | None = None):

        cfg = config or {}

        self.intensity_weight = 0.35
        self.area_weight = 0.35
        self.gradient_weight = 0.20
        self.texture_weight = 0.10

        self.min_area = cfg.get("min_damage_area", 1500)
        self.confidence_threshold = cfg.get("confidence_threshold", 0.4)

        # Only show results above this threshold
        self.display_threshold = cfg.get("display_threshold", 0.3)

    # -------------------------------------------------------------

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

            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            region = self._analyze_region(
                contour,
                diff_image,
                grad,
                vehicle_area,
            )

            if region:
                regions.append(region)

        # -------------------------------------------------------------
        # FINAL DISPLAY FILTER
        # Only keep damage above display threshold
        # -------------------------------------------------------------

        regions = [r for r in regions if r.confidence >= self.display_threshold]

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

    # -------------------------------------------------------------

    def _analyze_region(
        self,
        contour,
        diff_image,
        grad,
        vehicle_area,
    ) -> DamageRegion:

        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        perimeter = cv2.arcLength(contour, True)

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
            if confidence >= self.confidence_threshold
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
        )

    # -------------------------------------------------------------

    def _confidence_score(self, area, intensity, gradient, texture):

        area_score = min(area / 0.10, 1.0)
        intensity_score = min(intensity / 180.0, 1.0)
        gradient_score = min(gradient / 120.0, 1.0)
        texture_score = min(texture / 2000.0, 1.0)

        return float(
            0.35 * area_score
            + 0.30 * intensity_score
            + 0.20 * gradient_score
            + 0.15 * texture_score
        )

    # -------------------------------------------------------------

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

    # -------------------------------------------------------------

    def _compute_overall_severity(self, regions, total_area, vehicle_area):

        if not regions:
            return 0.0

        top = sorted(regions, key=lambda r: r.severity_score, reverse=True)[:3]

        intensity_component = sum(r.severity_score for r in top) / len(top)

        area_component = min((total_area / vehicle_area) / 0.25, 1.0)

        count_component = min(len(regions) / 8.0, 1.0)

        score = 0.5 * intensity_component + 0.35 * area_component + 0.15 * count_component

        return float(np.clip(score, 0.0, 1.0))

    # -------------------------------------------------------------

    def _score_to_label(self, score):

        for threshold, label in self.SEVERITY_LABELS:
            if score < threshold:
                return label

        return "critical"
