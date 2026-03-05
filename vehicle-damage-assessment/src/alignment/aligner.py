"""Spatial alignment of before/after image pairs.

PURPOSE
-------
Take two photos of the same vehicle from (potentially) different camera
positions and warp one into the other's coordinate frame so they can be
compared pixel-by-pixel downstream.

LIMITATIONS (being honest)
--------------------------
A single global homography assumes the scene is roughly planar. Cars are
3D objects — a fender curves away from a door panel. If the two photos are
taken from very different angles, no single 2D warp can perfectly align
the whole car. The alignment will be best in the region where most
keypoint matches cluster (usually the largest flat-ish panel visible).

For mildly different viewpoints (same side of the car, small angle shift)
this works well. For dramatically different viewpoints, downstream stages
should rely more on learned detection (YOLO) than pixel differencing.

STRATEGIES
----------
This module provides two alignment approaches:

1. **Homography** (default) — single 3x3 perspective transform.
   Best for: small viewpoint changes, mostly-planar subjects.

2. **Affine** — 6-DOF affine transform (no perspective distortion).
   Best for: when the camera is far from the car (telephoto-ish),
   or when homography produces weird warping artifacts.

The caller can also get quality metrics to decide whether the alignment
is trustworthy enough for pixel differencing, or whether to fall back
to detection-only assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class WarpMethod(Enum):
    """Which geometric transform to estimate."""
    HOMOGRAPHY = "homography"
    AFFINE = "affine"


@dataclass
class AlignmentResult:
    """Everything the aligner produces.

    Attributes
    ----------
    warped_after : np.ndarray
        The "after" image warped into the "before" image's frame.
    transform : np.ndarray
        The estimated transform matrix (3x3 for homography, 2x3 for affine).
    warp_method : WarpMethod
        Which method was used.
    num_inliers : int
        RANSAC inlier count — higher = more confident alignment.
    inlier_ratio : float
        inliers / total_good_matches — closer to 1.0 = cleaner match set.
    reprojection_error : float
        Mean reprojection error of inliers in pixels.
        Lower = more accurate alignment. >5px is suspicious.
    keypoints_before : list
        Detected keypoints in the before image.
    keypoints_after : list
        Detected keypoints in the after image.
    matches : list
        Good matches that passed the ratio test.
    is_reliable : bool
        Whether the alignment meets quality thresholds.
        If False, downstream should not trust pixel differencing.
    """

    warped_after: np.ndarray = field(repr=False)
    transform: np.ndarray = field(repr=False)
    warp_method: WarpMethod = WarpMethod.HOMOGRAPHY
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = float("inf")
    keypoints_before: list = field(default_factory=list, repr=False)
    keypoints_after: list = field(default_factory=list, repr=False)
    matches: list = field(default_factory=list, repr=False)
    is_reliable: bool = False


class ImageAligner:
    """Aligns the "after" image to the "before" image.

    Parameters
    ----------
    config : dict
        The ``alignment`` section of the pipeline config. Keys:

        - feature_method : str — "orb", "sift", or "akaze"
        - max_features : int — max keypoints to detect
        - match_method : str — "bf" or "flann"
        - ratio_threshold : float — Lowe's ratio test threshold (0-1)
        - ransac_reproj_threshold : float — RANSAC inlier pixel tolerance
        - min_match_count : int — minimum matches required
        - warp_method : str — "homography" or "affine"
        - max_reprojection_error : float — above this, alignment is flagged unreliable
        - min_inlier_ratio : float — below this, alignment is flagged unreliable
    """

    FEATURE_DETECTORS = {"orb", "sift", "akaze"}

    def __init__(self, config: dict) -> None:
        self.feature_method: str = config.get("feature_method", "orb")
        self.max_features: int = config.get("max_features", 5000)
        self.match_method: str = config.get("match_method", "bf")
        self.ratio_threshold: float = config.get("ratio_threshold", 0.75)
        self.ransac_thresh: float = config.get("ransac_reproj_threshold", 5.0)
        self.min_match_count: int = config.get("min_match_count", 10)
        self.warp_method: WarpMethod = WarpMethod(config.get("warp_method", "homography"))
        self.max_reproj_error: float = config.get("max_reprojection_error", 5.0)
        self.min_inlier_ratio: float = config.get("min_inlier_ratio", 0.3)

        if self.feature_method not in self.FEATURE_DETECTORS:
            raise ValueError(
                f"Unknown feature method '{self.feature_method}'. "
                f"Choose from {self.FEATURE_DETECTORS}"
            )

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def align(self, before: np.ndarray, after: np.ndarray) -> AlignmentResult:
        """Compute transform and warp *after* to match *before*.

        Parameters
        ----------
        before : np.ndarray
            Reference image (grayscale or BGR).
        after : np.ndarray
            Image to warp (grayscale or BGR).

        Returns
        -------
        AlignmentResult
            Warped image, transform, quality metrics, and reliability flag.

        Raises
        ------
        RuntimeError
            If feature detection or matching fails completely.
        """
        # step 1: detect keypoints + descriptors
        detector = self._create_detector()
        kp_before, desc_before = detector.detectAndCompute(before, None)
        kp_after, desc_after = detector.detectAndCompute(after, None)

        if desc_before is None or desc_after is None:
            raise RuntimeError(
                "Could not detect any features in one or both images. "
                "Images may be blank or too uniform."
            )

        # step 2: match + ratio test
        matcher = self._create_matcher(desc_before.dtype)
        good_matches = self._match_and_filter(matcher, desc_before, desc_after)

        # step 3: estimate transform
        transform, inlier_mask = self._estimate_transform(
            kp_before, kp_after, good_matches
        )

        num_inliers = int(inlier_mask.ravel().sum()) if inlier_mask is not None else 0
        inlier_ratio = num_inliers / max(len(good_matches), 1)

        # step 4: compute reprojection error on inliers
        reproj_error = self._reprojection_error(
            kp_before, kp_after, good_matches, transform, inlier_mask
        )

        # step 5: warp
        h, w = before.shape[:2]
        warped = self._warp_image(after, transform, (h, w))

        # step 6: judge reliability
        is_reliable = (
            num_inliers >= self.min_match_count
            and inlier_ratio >= self.min_inlier_ratio
            and reproj_error <= self.max_reproj_error
        )

        return AlignmentResult(
            warped_after=warped,
            transform=transform,
            warp_method=self.warp_method,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=reproj_error,
            keypoints_before=list(kp_before),
            keypoints_after=list(kp_after),
            matches=good_matches,
            is_reliable=is_reliable,
        )

    # ------------------------------------------------------------------
    # internal: feature detection
    # ------------------------------------------------------------------

    def _create_detector(self) -> cv2.Feature2D:
        """Instantiate the configured keypoint detector."""
        if self.feature_method == "orb":
            return cv2.ORB_create(nfeatures=self.max_features)
        elif self.feature_method == "sift":
            return cv2.SIFT_create(nfeatures=self.max_features)
        elif self.feature_method == "akaze":
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown feature method: {self.feature_method}")

    # ------------------------------------------------------------------
    # internal: matching
    # ------------------------------------------------------------------

    def _create_matcher(self, descriptor_dtype: np.dtype) -> cv2.DescriptorMatcher:
        """Instantiate the descriptor matcher with correct distance norm."""
        is_binary = (descriptor_dtype == np.uint8)
        norm = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2

        if self.match_method == "bf":
            return cv2.BFMatcher(norm, crossCheck=False)

        elif self.match_method == "flann":
            if is_binary:
                index_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6, key_size=12, multi_probe_level=1,
                )
            else:
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            return cv2.FlannBasedMatcher(index_params, dict(checks=50))

        else:
            raise ValueError(f"Unknown match method: {self.match_method}")

    def _match_and_filter(
        self,
        matcher: cv2.DescriptorMatcher,
        desc_before: np.ndarray,
        desc_after: np.ndarray,
    ) -> list[cv2.DMatch]:
        """Match descriptors and apply Lowe's ratio test."""
        raw_matches = matcher.knnMatch(desc_before, desc_after, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)

        return good

    # ------------------------------------------------------------------
    # internal: transform estimation
    # ------------------------------------------------------------------

    def _estimate_transform(
        self,
        kp_before: list,
        kp_after: list,
        matches: list[cv2.DMatch],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate either homography or affine transform via RANSAC.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (transform_matrix, inlier_mask)
        """
        if len(matches) < self.min_match_count:
            raise RuntimeError(
                f"Not enough matches: {len(matches)} found, "
                f"need at least {self.min_match_count}. "
                f"Images may be too different or lack texture."
            )

        pts_before = np.float32(
            [kp_before[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        pts_after = np.float32(
            [kp_after[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        if self.warp_method == WarpMethod.HOMOGRAPHY:
            # 3x3 perspective transform — 8 DOF
            # needs minimum 4 point pairs
            M, mask = cv2.findHomography(
                pts_after, pts_before,
                cv2.RANSAC, self.ransac_thresh,
            )
        else:
            # 2x3 affine transform — 6 DOF (no perspective distortion)
            # needs minimum 3 point pairs, more stable for distant/telephoto shots
            M, mask = cv2.estimateAffinePartial2D(
                pts_after, pts_before,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
            )

        if M is None:
            raise RuntimeError(
                "Transform estimation failed — returned None. "
                "Matches may be degenerate (e.g. all collinear)."
            )

        num_inliers = int(mask.ravel().sum()) if mask is not None else 0
        if num_inliers < 4:
            raise RuntimeError(
                f"Only {num_inliers} inliers — too few for a reliable transform."
            )

        return M, mask

    # ------------------------------------------------------------------
    # internal: quality metrics
    # ------------------------------------------------------------------

    def _reprojection_error(
        self,
        kp_before: list,
        kp_after: list,
        matches: list[cv2.DMatch],
        transform: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> float:
        """Compute mean reprojection error for inlier matches.

        Takes each inlier point in the after image, applies the transform,
        and measures how far it lands from its matched point in the before
        image. Lower = better alignment.

        Returns
        -------
        float
            Mean euclidean distance in pixels for inlier matches.
        """
        if inlier_mask is None or inlier_mask.ravel().sum() == 0:
            return float("inf")

        mask_flat = inlier_mask.ravel().astype(bool)

        pts_after = np.float32(
            [kp_after[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        pts_before = np.float32(
            [kp_before[m.queryIdx].pt for m in matches]
        ).reshape(-1, 2)

        # apply transform to after points
        if transform.shape == (3, 3):
            projected = cv2.perspectiveTransform(pts_after, transform).reshape(-1, 2)
        else:
            # affine: need to add a row to make it work with transform,
            # or just do the math directly
            ones = np.ones((pts_after.shape[0], 1, 1), dtype=np.float32)
            pts_h = np.concatenate([pts_after, ones], axis=2)  # (N, 1, 3)
            # M is 2x3, so projected = pts @ M^T
            projected = pts_h.reshape(-1, 3) @ transform.T  # (N, 2)

        # error only on inliers
        errors = np.linalg.norm(projected[mask_flat] - pts_before[mask_flat], axis=1)
        return float(errors.mean())

    # ------------------------------------------------------------------
    # internal: warping
    # ------------------------------------------------------------------

    def _warp_image(
        self, image: np.ndarray, transform: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:
        """Apply the estimated transform to warp an image.

        Parameters
        ----------
        image : np.ndarray
            The image to warp (the "after" image).
        transform : np.ndarray
            3x3 homography or 2x3 affine matrix.
        target_shape : tuple[int, int]
            (height, width) of the output image.

        Returns
        -------
        np.ndarray
            Warped image.
        """
        h, w = target_shape

        if transform.shape == (3, 3):
            return cv2.warpPerspective(
                image, transform, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            return cv2.warpAffine(
                image, transform, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )