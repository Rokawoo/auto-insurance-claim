"""Image preprocessing for the damage assessment pipeline.

Handles resizing, grayscale conversion, Gaussian blur, and optional
CLAHE histogram equalization to normalize lighting across before/after pairs.
"""

from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """Normalizes input images before alignment and comparison.

    Applies a consistent preprocessing chain so that downstream stages
    (feature matching, pixel differencing) are not thrown off by trivial
    differences in resolution, lighting, or sensor noise.

    Parameters
    ----------
    config : dict
        The ``preprocessing`` section of the pipeline config.
    """

    def __init__(self, config: dict) -> None:
        # OpenCV uses (width, height) for resize, but arrays are (height, width).
        self.target_size: tuple[int, int] = tuple(config.get("target_size", [640, 640]))
        self.grayscale: bool = config.get("grayscale", True)
        self.blur_kernel: int = config.get("blur_kernel", 5)
        self.blur_sigma: float = float(config.get("blur_sigma", 0))
        self.clahe_cfg: dict = config.get("clahe", {})

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing chain on a single image."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid or empty input image provided.")

        # 1. Resize (Standardize dimensions)
        result = self._resize(image)

        # 2. Grayscale (Optional)
        if self.grayscale:
            result = self._to_grayscale(result)

        # 3. Blur (Do this BEFORE CLAHE to avoid amplifying sensor noise)
        if self.blur_kernel > 0:
            result = self._blur(result)

        # 4. CLAHE (Normalize contrast)
        if self.clahe_cfg.get("enabled", False):
            result = self._apply_clahe(result)

        return result

    def process_pair(
        self, before: np.ndarray, after: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess a before/after image pair identically."""
        return self.process(before), self.process(after)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to ``self.target_size``.

        Uses INTER_AREA for downscaling (avoids aliasing) and
        INTER_LINEAR for upscaling.
        """
        h, w = image.shape[:2]
        tw, th = self.target_size  # (width, height)

        interp = cv2.INTER_AREA if (h * w > th * tw) else cv2.INTER_LINEAR
        return cv2.resize(image, (tw, th), interpolation=interp)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to single-channel grayscale safely."""
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            return image  # Already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        ksize = self.blur_kernel
        # OpenCV requires Gaussian blur kernels to be odd and positive
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), self.blur_sigma)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast-Limited Adaptive Histogram Equalization."""
        clip = self.clahe_cfg.get("clip_limit", 2.0)
        grid = tuple(self.clahe_cfg.get("tile_grid_size", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)

        if image.ndim == 3:
            # For color images, apply CLAHE only to the Lightness channel in LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # For grayscale images
        return clahe.apply(image)