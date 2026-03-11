"""Pixel-wise comparison between aligned before/after images.

After alignment and vehicle masking, this module computes the actual
difference map, applies thresholding and morphological cleanup, and
extracts contours that represent candidate damage regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np


@dataclass
class DiffResult:
    """Container for comparison outputs.

    Attributes
    ----------
    raw_diff : np.ndarray
        Raw absolute difference image (grayscale).
    thresh_diff : np.ndarray
        Binary thresholded difference mask.
    cleaned_mask : np.ndarray
        Morphologically cleaned binary mask.
    contours : list
        List of cv2 contours representing candidate damage regions.
    bounding_boxes : list[tuple[int, int, int, int]]
        (x, y, w, h) bounding rects for each contour.
    damage_area_px : int
        Total white-pixel area in the cleaned mask (within vehicle ROI).
    """
    raw_diff: np.ndarray = field(repr=False)
    thresh_diff: np.ndarray = field(repr=False)
    cleaned_mask: np.ndarray = field(repr=False)
    contours: list = field(default_factory=list, repr=False)
    bounding_boxes: list = field(default_factory=list)
    damage_area_px: int = 0


class DiffEngine:
    """Computes and post-processes the pixel difference between two images.

    Parameters
    ----------
    config : dict
        The ``comparison`` section of the pipeline config.
    """

    def __init__(self, config: dict) -> None:
        self.diff_method: str = config.get("diff_method", "absolute")
        self.threshold: int = config.get("threshold", 30)
        self.morph_kernel: int = config.get("morph_kernel", 5)
        self.morph_iterations: int = config.get("morph_iterations", 2)
        self.min_contour_area: int = config.get("min_contour_area", 500)
        self.max_contour_area: int = config.get("max_contour_area", 100000)

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def compare(
        self,
        before: np.ndarray,
        after: np.ndarray,
        vehicle_mask: np.ndarray | None = None,
    ) -> DiffResult:
        """Compute the difference between aligned before/after images."""
        if before.shape != after.shape:
            raise ValueError("before and after images must have the same shape")

        if self.diff_method == "absolute":
            raw_diff = self._absolute_diff(before, after)
        elif self.diff_method == "structural_similarity":
            raw_diff = self._ssim_diff(before, after)
        else:
            raise ValueError(f"Unsupported diff_method: {self.diff_method}")

        if vehicle_mask is not None:
            if vehicle_mask.shape != raw_diff.shape:
                raise ValueError("vehicle_mask must have the same shape as the input images")
            raw_diff = cv2.bitwise_and(raw_diff, vehicle_mask)

        thresh_diff = self._threshold(raw_diff)
        cleaned_mask = self._morphological_cleanup(thresh_diff)
        contours, bounding_boxes = self._extract_contours(cleaned_mask)
        damage_area_px = int(np.count_nonzero(cleaned_mask))

        return DiffResult(
            raw_diff=raw_diff,
            thresh_diff=thresh_diff,
            cleaned_mask=cleaned_mask,
            contours=contours,
            bounding_boxes=bounding_boxes,
            damage_area_px=damage_area_px,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _absolute_diff(
        self, before: np.ndarray, after: np.ndarray
    ) -> np.ndarray:
        """Simple per-pixel absolute difference."""
        return cv2.absdiff(before, after)

    def _ssim_diff(
        self, before: np.ndarray, after: np.ndarray
    ) -> np.ndarray:
        """Structural Similarity Index–based difference."""
        from skimage.metrics import structural_similarity as ssim

        _, diff_map = ssim(before, after, full=True)
        diff_map = (1.0 - diff_map) * 255.0
        diff_map = np.clip(diff_map, 0, 255).astype(np.uint8)
        return diff_map

    def _threshold(self, diff: np.ndarray) -> np.ndarray:
        """Apply binary threshold to the diff image."""
        _, binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return binary

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Remove noise and fill small holes in the binary mask."""
        kernel = np.ones((self.morph_kernel, self.morph_kernel), dtype=np.uint8)
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.morph_iterations,
        )
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.morph_iterations,
        )
        return closed




    def _extract_contours(
        self, mask: np.ndarray
    ) -> tuple[list, list[tuple[int, int, int, int]]]:
        """Find contours and their bounding rectangles."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        bounding_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_contour_area <= area <= self.max_contour_area:
                filtered_contours.append(contour)
                bounding_boxes.append(cv2.boundingRect(contour))

        return filtered_contours, bounding_boxes
