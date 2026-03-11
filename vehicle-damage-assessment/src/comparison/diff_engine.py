"""Pixel-wise comparison between aligned before/after images.

This module REQUIRES a vehicle mask to produce useful results.
Without it you get garbage — every background pixel, warp border,
and lighting change gets flagged as "damage."

The flow:
  1. Validate inputs (shape match, single-channel)
  2. Optional pre-blur to suppress pixel-level noise
  3. Compute raw pixel diff (absdiff, SSIM, or combined)
  4. Mask to vehicle region only
  5. Threshold to binary (fixed, Otsu, or adaptive)
  6. Morphological cleanup (remove noise, fill holes)
  7. Extract contours = candidate damage regions, sorted by area
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════

class DiffMethod(str, Enum):
    """Supported pixel-differencing algorithms."""
    ABSOLUTE = "absolute"
    SSIM = "structural_similarity"
    COMBINED = "combined"  # average of normalized absdiff and SSIM


class ThresholdMethod(str, Enum):
    """How to binarize the diff image."""
    FIXED = "fixed"        # user-specified threshold value
    OTSU = "otsu"          # automatic via Otsu's method
    ADAPTIVE = "adaptive"  # local adaptive (Gaussian-weighted neighborhood)


@dataclass
class DiffResult:
    """Container for comparison outputs.

    Attributes
    ----------
    raw_diff : np.ndarray
        Raw difference image (grayscale, uint8), masked to vehicle region.
    thresh_diff : np.ndarray
        Binary thresholded difference mask (0 or 255).
    cleaned_mask : np.ndarray
        Morphologically cleaned binary mask.
    contours : list
        cv2 contours representing candidate damage regions,
        sorted by area descending (largest first).
    bounding_boxes : list[tuple[int, int, int, int]]
        (x, y, w, h) bounding rect for each contour, same order.
    damage_area_px : int
        Total damaged pixel count in the cleaned mask.
    diff_mean : float
        Mean intensity of the raw diff within the vehicle mask.
        Useful as a global "how different are these images" signal.
    diff_max : int
        Max intensity in the raw diff within the vehicle mask.
    """

    raw_diff: np.ndarray = field(repr=False)
    thresh_diff: np.ndarray = field(repr=False)
    cleaned_mask: np.ndarray = field(repr=False)
    contours: list = field(default_factory=list, repr=False)
    bounding_boxes: list = field(default_factory=list)
    damage_area_px: int = 0
    diff_mean: float = 0.0
    diff_max: int = 0


# ══════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════

class DiffEngine:
    """Computes and post-processes the pixel difference between two images.

    Parameters
    ----------
    config : dict
        The ``comparison`` section of the pipeline config.  Keys:

        diff_method : str
            ``"absolute"`` | ``"structural_similarity"`` | ``"combined"``
            (default ``"absolute"``).
        threshold : int
            Fixed binarization threshold, 0–255 (default 30).
            Ignored when ``threshold_method`` is ``"otsu"``.
        threshold_method : str
            ``"fixed"`` | ``"otsu"`` | ``"adaptive"``  (default ``"fixed"``).
        adaptive_block_size : int
            Block size for adaptive thresholding — must be odd (default 11).
        adaptive_c : int
            Constant subtracted from the adaptive mean (default 5).
        pre_blur : int
            Gaussian blur kernel before differencing; 0 = disabled (default 0).
            Suppresses pixel-level noise so morphology doesn't have to.
        morph_kernel : int
            Structuring element diameter for open/close (default 5).
        morph_iterations : int
            Iterations for each morphological op (default 2).
        min_contour_area : int
            Discard contours smaller than this (px², default 500).
        max_contour_area : int
            Discard contours larger than this (px², default 100000).
    """

    VALID_DIFF_METHODS = {m.value for m in DiffMethod}
    VALID_THRESH_METHODS = {m.value for m in ThresholdMethod}

    def __init__(self, config: dict) -> None:
        self.diff_method = DiffMethod(config.get("diff_method", "absolute"))
        self.threshold: int = config.get("threshold", 30)
        self.threshold_method = ThresholdMethod(
            config.get("threshold_method", "fixed")
        )
        self.adaptive_block: int = config.get("adaptive_block_size", 11)
        self.adaptive_c: int = config.get("adaptive_c", 5)
        self.pre_blur: int = config.get("pre_blur", 0)
        self.morph_kernel: int = config.get("morph_kernel", 5)
        self.morph_iterations: int = config.get("morph_iterations", 2)
        self.min_contour_area: int = config.get("min_contour_area", 500)
        self.max_contour_area: int = config.get("max_contour_area", 100000)

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def compare(
        self,
        before: np.ndarray,
        after: np.ndarray,
        vehicle_mask: np.ndarray | None = None,
    ) -> DiffResult:
        """Compute the masked difference between aligned before/after images.

        Parameters
        ----------
        before : np.ndarray
            Preprocessed + aligned "before" image (grayscale).
        after : np.ndarray
            Preprocessed + aligned "after" image (grayscale, warped).
        vehicle_mask : np.ndarray | None
            Binary mask from VehicleDetector (255 = car, 0 = background).
            **Strongly recommended** — without this, results are mostly noise.

        Returns
        -------
        DiffResult
            Raw diff, thresholded mask, cleaned contours, bounding boxes.

        Raises
        ------
        ValueError
            If images have mismatched shapes or are not single-channel.
        """
        # ── validate inputs ────────────────────────────────────────
        before, after = self._validate_inputs(before, after, vehicle_mask)

        # ── optional pre-blur ──────────────────────────────────────
        if self.pre_blur > 0:
            k = self.pre_blur if self.pre_blur % 2 == 1 else self.pre_blur + 1
            before = cv2.GaussianBlur(before, (k, k), 0)
            after = cv2.GaussianBlur(after, (k, k), 0)

        # ── step 1: compute raw pixel difference ───────────────────
        if self.diff_method == DiffMethod.SSIM:
            raw_diff = self._ssim_diff(before, after)
        elif self.diff_method == DiffMethod.COMBINED:
            raw_diff = self._combined_diff(before, after)
        else:
            raw_diff = self._absolute_diff(before, after)

        # ── step 2: mask to vehicle region ─────────────────────────
        if vehicle_mask is not None:
            binary_mask = self._ensure_binary_mask(vehicle_mask)
            raw_diff = cv2.bitwise_and(raw_diff, raw_diff, mask=binary_mask)
        else:
            binary_mask = None
            logger.warning(
                "No vehicle mask provided — diff will include background "
                "noise, warp borders, and lighting changes."
            )

        # ── compute diff statistics (within mask) ──────────────────
        if binary_mask is not None:
            masked_pixels = raw_diff[binary_mask > 0]
        else:
            masked_pixels = raw_diff.ravel()

        if masked_pixels.size > 0:
            diff_mean = float(masked_pixels.mean())
            diff_max = int(masked_pixels.max())
        else:
            diff_mean, diff_max = 0.0, 0

        # ── step 3: threshold ──────────────────────────────────────
        thresh_diff = self._apply_threshold(raw_diff)

        # re-mask after adaptive threshold (it can activate outside vehicle)
        if binary_mask is not None:
            thresh_diff = cv2.bitwise_and(thresh_diff, thresh_diff, mask=binary_mask)

        # ── step 4: morphological cleanup ──────────────────────────
        cleaned = self._morphological_cleanup(thresh_diff)

        # ── step 5: extract contours (sorted by area desc) ─────────
        contours, bboxes = self._extract_contours(cleaned)

        # pixel-accurate damage area from the mask itself
        damage_area = int(cv2.countNonZero(cleaned))

        logger.debug(
            "Diff complete: method=%s, threshold=%s(%d), "
            "diff_mean=%.1f, diff_max=%d, contours=%d, damage_area=%dpx",
            self.diff_method.value, self.threshold_method.value,
            self.threshold, diff_mean, diff_max, len(contours), damage_area,
        )

        return DiffResult(
            raw_diff=raw_diff,
            thresh_diff=thresh_diff,
            cleaned_mask=cleaned,
            contours=contours,
            bounding_boxes=bboxes,
            damage_area_px=damage_area,
            diff_mean=diff_mean,
            diff_max=diff_max,
        )

    # ══════════════════════════════════════════════════════════════════
    # Input validation
    # ══════════════════════════════════════════════════════════════════

    def _validate_inputs(
        self,
        before: np.ndarray,
        after: np.ndarray,
        vehicle_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure images are grayscale, same shape, and mask matches.

        Auto-converts BGR to grayscale if needed rather than crashing.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (before_gray, after_gray) — guaranteed single-channel, same shape.

        Raises
        ------
        ValueError
            If shapes don't match after conversion.
        """
        # auto-convert to grayscale
        if before.ndim == 3:
            logger.debug("Converting before image from BGR to grayscale.")
            before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        if after.ndim == 3:
            logger.debug("Converting after image from BGR to grayscale.")
            after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        if before.shape != after.shape:
            raise ValueError(
                f"Shape mismatch: before={before.shape}, after={after.shape}. "
                f"Images must be the same size after preprocessing + alignment."
            )

        if vehicle_mask is not None and vehicle_mask.shape[:2] != before.shape[:2]:
            raise ValueError(
                f"Mask shape {vehicle_mask.shape} doesn't match "
                f"image shape {before.shape}."
            )

        return before, after

    # ══════════════════════════════════════════════════════════════════
    # Differencing methods
    # ══════════════════════════════════════════════════════════════════

    def _absolute_diff(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Per-pixel absolute difference.

        |before[i,j] - after[i,j]| for every pixel.
        Fast, simple, works well when alignment is good and lighting
        is consistent.
        """
        return cv2.absdiff(before, after)

    def _ssim_diff(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Structural Similarity Index based difference.

        SSIM considers luminance, contrast, and structure — more robust
        to global brightness shifts than raw pixel diff.  We invert the
        map so that low similarity = high diff value (bright = damage).

        Falls back to absolute diff if scikit-image is not installed.
        """
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            logger.warning(
                "scikit-image not installed — falling back to absolute diff. "
                "Install with: pip install scikit-image"
            )
            return self._absolute_diff(before, after)

        # win_size must be odd and <= min image dimension
        min_dim = min(before.shape[:2])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(win_size, 3)

        _, diff_map = ssim(before, after, win_size=win_size, full=True)

        # invert: 1.0 (identical) → 0, 0.0 (different) → 255
        inverted = (1.0 - diff_map) * 255.0
        return np.clip(inverted, 0, 255).astype(np.uint8)

    def _combined_diff(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Average of normalized absolute diff and SSIM diff.

        Combines the spatial precision of absdiff with the perceptual
        robustness of SSIM.  Both are normalized to [0, 255] before
        averaging.
        """
        abs_diff = self._absolute_diff(before, after)
        ssim_diff = self._ssim_diff(before, after)

        # blend 50/50
        combined = cv2.addWeighted(abs_diff, 0.5, ssim_diff, 0.5, 0)
        return combined

    # ══════════════════════════════════════════════════════════════════
    # Thresholding
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
        """Force a mask to strict binary {0, 255}.

        Dilations and resizing can leave intermediate values that
        confuse bitwise_and.
        """
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _apply_threshold(self, diff: np.ndarray) -> np.ndarray:
        """Binarize the diff image using the configured method.

        Returns
        -------
        np.ndarray
            Binary mask (0 or 255, dtype uint8).
        """
        if self.threshold_method == ThresholdMethod.OTSU:
            # Otsu ignores our self.threshold — it picks the optimal split
            _, binary = cv2.threshold(
                diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary

        elif self.threshold_method == ThresholdMethod.ADAPTIVE:
            block = self.adaptive_block
            if block % 2 == 0:
                block += 1  # must be odd
            block = max(block, 3)  # minimum 3

            return cv2.adaptiveThreshold(
                diff, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=block,
                C=-self.adaptive_c,  # negative C → detect bright regions
            )

        else:  # FIXED
            _, binary = cv2.threshold(
                diff, self.threshold, 255, cv2.THRESH_BINARY
            )
            return binary

    # ══════════════════════════════════════════════════════════════════
    # Morphological cleanup
    # ══════════════════════════════════════════════════════════════════

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Remove noise specks and fill small holes.

        Opening (erode → dilate): removes small bright spots (noise).
        Closing (dilate → erode): fills small dark gaps within damage regions.

        Uses an elliptical kernel which handles diagonal features better
        than a rectangular one.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel, self.morph_kernel),
        )

        # open first to kill noise specks
        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations
        )
        # close to fill small holes within damage regions
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations
        )

        return cleaned

    # ══════════════════════════════════════════════════════════════════
    # Contour extraction
    # ══════════════════════════════════════════════════════════════════

    def _extract_contours(
        self, mask: np.ndarray
    ) -> tuple[list, list[tuple[int, int, int, int]]]:
        """Find contours and bounding rects, filtered by area.

        Returns contours sorted by area descending so the largest
        (most significant) damage regions come first.

        Returns
        -------
        tuple[list, list[tuple]]
            (filtered_contours, bounding_boxes) where each bbox is (x, y, w, h).
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # filter by area bounds
        valid: list[tuple[np.ndarray, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_contour_area <= area <= self.max_contour_area:
                valid.append((c, area))

        # sort by area descending — largest damage first
        valid.sort(key=lambda pair: pair[1], reverse=True)

        filtered = [c for c, _ in valid]
        bboxes = [cv2.boundingRect(c) for c in filtered]

        return filtered, bboxes