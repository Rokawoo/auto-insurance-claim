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
        self.target_size: tuple[int, int] = tuple(config.get("target_size", [640, 640]))
        self.grayscale: bool = config.get("grayscale", True)
        self.blur_kernel: int = config.get("blur_kernel", 5)
        self.blur_sigma: int = config.get("blur_sigma", 0)
        self.clahe_cfg: dict = config.get("clahe", {})

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing chain on a single image.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image as loaded by ``cv2.imread``.

        Returns
        -------
        np.ndarray
            Preprocessed image ready for downstream stages.
        """
        # TODO: implement the full chain by calling the helpers below
        #   1. resize
        #   2. convert to grayscale (if configured)
        #   3. apply CLAHE (if configured)
        #   4. gaussian blur

        resized_image = self._resize(image)
        grayscale_image = self._to_grayscale(resized_image)
        clahe_image = self._apply_clahe(grayscale_image)
        blur_image = self._blur(clahe_image)
        return blur_image

    def process_pair(
        self, before: np.ndarray, after: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess a before/after image pair.

        Ensures both images go through identical transforms so they are
        directly comparable in later stages.

        Parameters
        ----------
        before : np.ndarray
            "Before" image (undamaged vehicle).
        after : np.ndarray
            "After" image (damaged vehicle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (processed_before, processed_after)
        """
        before_image = self.process(before)
        after_image = self.process(after)
        # TODO: call self.process on each image and return the pair
        raise (before_image, after_image)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to ``self.target_size`` using area interpolation.

        Parameters
        ----------
        image : np.ndarray
            Input image (any size).

        Returns
        -------
        np.ndarray
            Resized image.
        """
        # TODO: cv2.resize with INTER_AREA for downscale, INTER_LINEAR for upscale
        initial_height = len(image)
        initial_width = len(image[0])
        target_height = self.target_size[0]
        target_width = self.target_size[1]

        scale_height = initial_height / target_height
        scale_width = initial_width / target_width

        # pure upscaling
        if scale_height < 1 and scale_width < 1:
            return cv2.resize(image, (target_height, target_width), interpolation=cv2.INTER_LINEAR)
        
        # For pure downscaling and all the other cases
        return cv2.resize(image, (target_height, target_width), interpolation=cv2.INTER_AREA)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to single-channel grayscale.

        Parameters
        ----------
        image : np.ndarray
            BGR image.

        Returns
        -------
        np.ndarray
            Grayscale image (H, W).
        """
        # TODO: cv2.cvtColor COLOR_BGR2GRAY
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast-Limited Adaptive Histogram Equalization.

        Normalizes local contrast so that lighting differences between
        the before/after pair don't create false diff regions.

        Parameters
        ----------
        image : np.ndarray
            Grayscale image.

        Returns
        -------
        np.ndarray
            Equalized image.
        """
        # TODO: create cv2.createCLAHE with clip_limit and tile_grid_size
        #       from self.clahe_cfg, then apply

        # because it's grayscale, default tile
        clahe = cv2.createCLAHE(clipLimit=self.clahe_cfg["clip_limit"], tileGridSize=self.clahe_cfg["tile_grid_size"])
        result = clahe.apply(image)
        return result

    def _blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for noise reduction.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Blurred image.
        """
        kernel_size = (self.blur_kernel, self.blur_kernel)
        blurred = cv2.GaussianBlur(image, kernel_size, self.blur_sigma)
        # TODO: cv2.GaussianBlur with self.blur_kernel and self.blur_sigma
        return blurred
