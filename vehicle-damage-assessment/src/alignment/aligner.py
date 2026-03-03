"""Spatial alignment of before/after image pairs.

Uses keypoint detection (ORB / SIFT / AKAZE), descriptor matching, and
RANSAC-based homography estimation to warp the "after" image into the
coordinate frame of the "before" image.  This is critical so that pixel
differencing later on reflects *actual damage* rather than camera-angle
differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np


@dataclass
class AlignmentResult:
    """Container for alignment outputs.

    Attributes
    ----------
    warped_after : np.ndarray
        The "after" image warped to match the "before" perspective.
    homography : np.ndarray
        3×3 homography matrix used for the warp.
    num_inliers : int
        Number of RANSAC inlier matches (quality indicator).
    keypoints_before : list
        Detected keypoints in the before image (for visualization).
    keypoints_after : list
        Detected keypoints in the after image (for visualization).
    matches : list
        Good matches after ratio test + RANSAC.
    """
    warped_after: np.ndarray = field(repr=False)
    homography: np.ndarray = field(repr=False)
    num_inliers: int = 0
    keypoints_before: list = field(default_factory=list, repr=False)
    keypoints_after: list = field(default_factory=list, repr=False)
    matches: list = field(default_factory=list, repr=False)


class ImageAligner:
    """Aligns the "after" image to the "before" image via homography.

    Parameters
    ----------
    config : dict
        The ``alignment`` section of the pipeline config.
    """

    FEATURE_DETECTORS = {"orb", "sift", "akaze"}

    def __init__(self, config: dict) -> None:
        self.feature_method: str = config.get("feature_method", "orb")
        self.max_features: int = config.get("max_features", 5000)
        self.match_method: str = config.get("match_method", "bf")
        self.ratio_threshold: float = config.get("ratio_threshold", 0.75)
        self.ransac_thresh: float = config.get("ransac_reproj_threshold", 5.0)
        self.min_match_count: int = config.get("min_match_count", 10)

        if self.feature_method not in self.FEATURE_DETECTORS:
            raise ValueError(
                f"Unknown feature method '{self.feature_method}'. "
                f"Choose from {self.FEATURE_DETECTORS}"
            )

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def align(
        self, before: np.ndarray, after: np.ndarray
    ) -> AlignmentResult:
        """Compute homography and warp *after* to match *before*.

        Parameters
        ----------
        before : np.ndarray
            Preprocessed "before" image (grayscale or BGR).
        after : np.ndarray
            Preprocessed "after" image (grayscale or BGR).

        Returns
        -------
        AlignmentResult
            Warped image + metadata.

        Raises
        ------
        RuntimeError
            If not enough matches are found to compute a reliable homography.
        """
        # TODO:
        #   1. detect keypoints + descriptors in both images
        #   2. match descriptors
        #   3. apply ratio test to filter matches
        #   4. compute homography via RANSAC
        #   5. warp the after image
        #   6. return AlignmentResult
        raise NotImplementedError

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _create_detector(self) -> cv2.Feature2D:
        """Instantiate the configured keypoint detector.

        Returns
        -------
        cv2.Feature2D
            ORB, SIFT, or AKAZE detector instance.
        """
        # TODO: create and return the detector based on self.feature_method
        #   - orb: cv2.ORB_create(nfeatures=self.max_features)
        #   - sift: cv2.SIFT_create(nfeatures=self.max_features)
        #   - akaze: cv2.AKAZE_create()
        raise NotImplementedError

    def _create_matcher(
        self, descriptor_type: int
    ) -> cv2.DescriptorMatcher:
        """Instantiate the descriptor matcher.

        Parameters
        ----------
        descriptor_type : int
            ``cv2.CV_8U`` for binary descriptors (ORB/AKAZE),
            ``cv2.CV_32F`` for float descriptors (SIFT).

        Returns
        -------
        cv2.DescriptorMatcher
            BFMatcher or FLANN matcher.
        """
        # TODO:
        #   - if bf: BFMatcher with appropriate norm (HAMMING for binary, L2 for float)
        #   - if flann: FlannBasedMatcher with appropriate index params
        raise NotImplementedError

    def _match_and_filter(
        self,
        matcher: cv2.DescriptorMatcher,
        desc_before: np.ndarray,
        desc_after: np.ndarray,
    ) -> list:
        """Match descriptors and apply Lowe's ratio test.

        Parameters
        ----------
        matcher : cv2.DescriptorMatcher
            Matcher instance.
        desc_before : np.ndarray
            Descriptors from the before image.
        desc_after : np.ndarray
            Descriptors from the after image.

        Returns
        -------
        list[cv2.DMatch]
            Filtered good matches.
        """
        # TODO:
        #   1. knnMatch with k=2
        #   2. ratio test: keep m if m.distance < ratio * n.distance
        raise NotImplementedError

    def _compute_homography(
        self,
        kp_before: list,
        kp_after: list,
        matches: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate homography via RANSAC.

        Parameters
        ----------
        kp_before : list[cv2.KeyPoint]
        kp_after : list[cv2.KeyPoint]
        matches : list[cv2.DMatch]

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (homography_matrix, inlier_mask)

        Raises
        ------
        RuntimeError
            If fewer than ``self.min_match_count`` matches exist.
        """
        # TODO:
        #   1. extract matched point coords
        #   2. cv2.findHomography with RANSAC
        #   3. validate that enough inliers exist
        raise NotImplementedError

    def _warp_image(
        self, image: np.ndarray, homography: np.ndarray, target_shape: tuple
    ) -> np.ndarray:
        """Warp an image using the computed homography.

        Parameters
        ----------
        image : np.ndarray
            Image to warp (the "after" image).
        homography : np.ndarray
            3×3 transformation matrix.
        target_shape : tuple
            (height, width) of the target (before) image.

        Returns
        -------
        np.ndarray
            Warped image.
        """
        # TODO: cv2.warpPerspective
        raise NotImplementedError
