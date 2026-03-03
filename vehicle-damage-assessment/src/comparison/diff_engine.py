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
        """Compute the difference between aligned before/after images.

        Parameters
        ----------
        before : np.ndarray
            Preprocessed + aligned "before" image (grayscale).
        after : np.ndarray
            Preprocessed + aligned "after" image (grayscale, warped).
        vehicle_mask : np.ndarray | None
            Binary mask from VehicleDetector.  If provided, diff is
            computed only within the masked region.

        Returns
        -------
        DiffResult
            Raw diff, thresholded mask, cleaned contours, bounding boxes.
        """
        # TODO:
        #   1. compute raw diff (absolute or SSIM-based)
        #   2. if vehicle_mask provided, bitwise_and to zero out non-vehicle
        #   3. threshold the diff
        #   4. morphological cleanup (open then close)
        #   5. find contours, filter by area
        #   6. compute bounding rects
        #   7. return DiffResult
        raise NotImplementedError

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _absolute_diff(
        self, before: np.ndarray, after: np.ndarray
    ) -> np.ndarray:
        """Simple per-pixel absolute difference.

        Parameters
        ----------
        before, after : np.ndarray
            Same-size grayscale images.

        Returns
        -------
        np.ndarray
            Absolute difference image (uint8).
        """
        # TODO: cv2.absdiff
        raise NotImplementedError

    def _ssim_diff(
        self, before: np.ndarray, after: np.ndarray
    ) -> np.ndarray:
        """Structural Similarity Index–based difference.

        Uses skimage.metrics.structural_similarity to get a per-pixel
        SSIM map, then inverts it so that *low* similarity = *high* diff.

        Parameters
        ----------
        before, after : np.ndarray
            Same-size grayscale images.

        Returns
        -------
        np.ndarray
            Inverted SSIM difference map, scaled to uint8.
        """
        # TODO:
        #   from skimage.metrics import structural_similarity as ssim
        #   score, diff_map = ssim(before, after, full=True)
        #   convert diff_map to uint8 (invert so damage = bright)
        raise NotImplementedError

    def _threshold(self, diff: np.ndarray) -> np.ndarray:
        """Apply binary threshold to the diff image.

        Parameters
        ----------
        diff : np.ndarray
            Grayscale difference image.

        Returns
        -------
        np.ndarray
            Binary mask (0 or 255).
        """
        # TODO: cv2.threshold with THRESH_BINARY
        raise NotImplementedError

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Remove noise and fill small holes in the binary mask.

        Applies morphological opening (remove small specks) followed by
        closing (fill small gaps).

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.

        Returns
        -------
        np.ndarray
            Cleaned binary mask.
        """
        # TODO: create kernel, morphologyEx MORPH_OPEN then MORPH_CLOSE
        raise NotImplementedError

    def _extract_contours(
        self, mask: np.ndarray
    ) -> tuple[list, list[tuple[int, int, int, int]]]:
        """Find contours and their bounding rectangles.

        Filters by ``min_contour_area`` and ``max_contour_area``.

        Parameters
        ----------
        mask : np.ndarray
            Cleaned binary mask.

        Returns
        -------
        tuple[list, list[tuple]]
            (filtered_contours, bounding_boxes)
        """
        # TODO:
        #   1. cv2.findContours
        #   2. filter by area
        #   3. cv2.boundingRect for each surviving contour
        raise NotImplementedError
