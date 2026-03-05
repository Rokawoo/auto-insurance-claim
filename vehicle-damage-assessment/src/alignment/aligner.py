"""Spatial alignment of before/after image pairs.

PURPOSE
-------
Take two photos of the same vehicle from (potentially) different camera
positions and warp one into the other's coordinate frame so they can be
compared pixel-by-pixel downstream.

ROBUSTNESS FEATURES
-------------------
- Exposure normalization (CLAHE) before feature detection so lighting
  differences between shots don't kill matching.
- Multiple detector support (ORB / SIFT / AKAZE) with automatic fallback:
  if the primary method doesn't get enough matches, tries the next one.
- Both homography (8-DOF) and affine (4-DOF) transforms — affine is more
  stable on 3D subjects like cars where perspective correction can distort.
- Valid-region mask: after warping, black border pixels are masked out so
  downstream diff stages don't flag them as "damage."
- Quality metrics (reprojection error, inlier ratio) with a reliability
  flag the pipeline can use to decide whether to trust pixel differencing
  or fall back to detection-only assessment.

LIMITATIONS
-----------
A single global transform assumes the scene is roughly planar. Cars are
3D — a fender curves away from a door. For mild viewpoint shifts (same
side, small angle change) this works well. For large angle differences,
downstream should lean on YOLO detection rather than pixel differencing.
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
    valid_mask : np.ndarray
        Binary mask (uint8, 0/255) of the region that has actual image
        data after warping. Pixels outside this are black border artifacts
        and should be ignored by any downstream diff/comparison.
    transform : np.ndarray
        The estimated transform matrix (3x3 for homography, 2x3 for affine).
    warp_method : WarpMethod
        Which method produced this result.
    feature_method : str
        Which detector was actually used (may differ from config if fallback kicked in).
    num_inliers : int
        RANSAC inlier count — higher = more confident alignment.
    inlier_ratio : float
        inliers / total_good_matches — closer to 1.0 = cleaner match set.
    reprojection_error : float
        Mean reprojection error of inliers in pixels.
        Lower = more accurate. >5px is suspicious.
    num_matches : int
        Total good matches after ratio test (before RANSAC).
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
    valid_mask: np.ndarray = field(repr=False)
    transform: np.ndarray = field(repr=False)
    warp_method: WarpMethod = WarpMethod.HOMOGRAPHY
    feature_method: str = "orb"
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = float("inf")
    num_matches: int = 0
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

        - feature_method : str — "orb", "sift", or "akaze" (default "orb")
        - max_features : int — max keypoints to detect (default 10000)
        - match_method : str — "bf" or "flann" (default "bf")
        - ratio_threshold : float — Lowe's ratio test threshold (default 0.75)
        - ransac_reproj_threshold : float — RANSAC inlier pixel tolerance (default 5.0)
        - min_match_count : int — minimum matches required (default 10)
        - warp_method : str — "homography" or "affine" (default "homography")
        - max_reprojection_error : float — reliability threshold (default 5.0)
        - min_inlier_ratio : float — reliability threshold (default 0.25)
        - fallback : bool — try other detectors if primary fails (default True)
        - normalize_exposure : bool — apply CLAHE before detection (default True)
    """

    FEATURE_DETECTORS = {"orb", "sift", "akaze"}
    # fallback order: try SIFT (most robust) then AKAZE then ORB
    FALLBACK_ORDER = ["sift", "akaze", "orb"]

    def __init__(self, config: dict) -> None:
        self.feature_method: str = config.get("feature_method", "orb")
        self.max_features: int = config.get("max_features", 10000)
        self.match_method: str = config.get("match_method", "bf")
        self.ratio_threshold: float = config.get("ratio_threshold", 0.75)
        self.ransac_thresh: float = config.get("ransac_reproj_threshold", 5.0)
        self.min_match_count: int = config.get("min_match_count", 10)
        self.warp_method: WarpMethod = WarpMethod(config.get("warp_method", "homography"))
        self.max_reproj_error: float = config.get("max_reprojection_error", 5.0)
        self.min_inlier_ratio: float = config.get("min_inlier_ratio", 0.25)
        self.fallback: bool = config.get("fallback", True)
        self.normalize_exposure: bool = config.get("normalize_exposure", True)

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

        If ``fallback`` is enabled and the primary detector doesn't produce
        enough matches, automatically tries other detectors before giving up.

        Parameters
        ----------
        before : np.ndarray
            Reference image (grayscale or BGR).
        after : np.ndarray
            Image to warp (grayscale or BGR).

        Returns
        -------
        AlignmentResult
            Warped image, valid mask, transform, quality metrics.

        Raises
        ------
        RuntimeError
            If all methods fail to produce a viable alignment.
        """
        # normalize exposure before feature detection
        before_norm = self._normalize(before)
        after_norm = self._normalize(after)

        # build detector order: primary first, then fallbacks
        methods_to_try = [self.feature_method]
        if self.fallback:
            for m in self.FALLBACK_ORDER:
                if m not in methods_to_try:
                    methods_to_try.append(m)

        last_error = None
        for method in methods_to_try:
            try:
                result = self._try_align(before_norm, after_norm, before, after, method)
                return result
            except RuntimeError as e:
                last_error = e
                if method == self.feature_method:
                    # primary failed — worth reporting
                    pass
                continue

        # all methods exhausted
        raise RuntimeError(
            f"Alignment failed with all methods {methods_to_try}. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # internal: core alignment attempt
    # ------------------------------------------------------------------

    def _try_align(
        self,
        before_norm: np.ndarray,
        after_norm: np.ndarray,
        before_orig: np.ndarray,
        after_orig: np.ndarray,
        feature_method: str,
    ) -> AlignmentResult:
        """Single alignment attempt with a specific detector.

        Uses the normalized images for feature detection/matching, but
        warps the original images so output quality isn't degraded.
        """
        # detect + describe
        detector = self._create_detector(feature_method)
        kp_before, desc_before = detector.detectAndCompute(before_norm, None)
        kp_after, desc_after = detector.detectAndCompute(after_norm, None)

        if desc_before is None or desc_after is None:
            raise RuntimeError(
                f"[{feature_method}] no features detected in one or both images."
            )

        if len(kp_before) < self.min_match_count or len(kp_after) < self.min_match_count:
            raise RuntimeError(
                f"[{feature_method}] too few keypoints: "
                f"before={len(kp_before)}, after={len(kp_after)}"
            )

        # match + ratio test
        matcher = self._create_matcher(desc_before.dtype)
        good_matches = self._match_and_filter(matcher, desc_before, desc_after)

        # estimate transform
        transform, inlier_mask = self._estimate_transform(
            kp_before, kp_after, good_matches
        )

        num_inliers = int(inlier_mask.ravel().sum()) if inlier_mask is not None else 0
        inlier_ratio = num_inliers / max(len(good_matches), 1)

        # reprojection error
        reproj_error = self._reprojection_error(
            kp_before, kp_after, good_matches, transform, inlier_mask
        )

        # warp the ORIGINAL after image (not the normalized one)
        h, w = before_orig.shape[:2]
        warped = self._warp_image(after_orig, transform, (h, w))

        # build valid-region mask: warp a white image to see where data lands
        white = np.full(after_orig.shape[:2], 255, dtype=np.uint8)
        valid_mask = self._warp_image(white, transform, (h, w))
        # erode slightly to remove interpolation fuzz at edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

        # reliability
        is_reliable = (
            num_inliers >= self.min_match_count
            and inlier_ratio >= self.min_inlier_ratio
            and reproj_error <= self.max_reproj_error
        )

        return AlignmentResult(
            warped_after=warped,
            valid_mask=valid_mask,
            transform=transform,
            warp_method=self.warp_method,
            feature_method=feature_method,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=reproj_error,
            num_matches=len(good_matches),
            keypoints_before=list(kp_before),
            keypoints_after=list(kp_after),
            matches=good_matches,
            is_reliable=is_reliable,
        )

    # ------------------------------------------------------------------
    # internal: preprocessing
    # ------------------------------------------------------------------

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image for more robust feature detection.

        Applies CLAHE (contrast-limited adaptive histogram equalization)
        to handle exposure/lighting differences between the before and
        after shots. Works on grayscale; if BGR, converts to grayscale
        first since feature detection only uses intensity anyway.

        Returns grayscale uint8 image.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if not self.normalize_exposure:
            return gray

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # ------------------------------------------------------------------
    # internal: feature detection
    # ------------------------------------------------------------------

    def _create_detector(self, method: str) -> cv2.Feature2D:
        """Instantiate a keypoint detector by name."""
        if method == "orb":
            # WTA_K=2 is default — produces binary descriptors compatible with HAMMING
            return cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=1.2,
                nlevels=12,          # more pyramid levels for scale robustness
                edgeThreshold=31,
                patchSize=31,
            )
        elif method == "sift":
            return cv2.SIFT_create(
                nfeatures=self.max_features,
                contrastThreshold=0.03,  # lower = more keypoints (default 0.04)
            )
        elif method == "akaze":
            return cv2.AKAZE_create(
                threshold=0.001,     # lower = more keypoints (default 0.001)
            )
        else:
            raise ValueError(f"Unknown feature method: {method}")

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
                    table_number=12, key_size=20, multi_probe_level=2,
                )
            else:
                index_params = dict(algorithm=1, trees=5)
            return cv2.FlannBasedMatcher(index_params, dict(checks=80))

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
        """Estimate homography or affine transform via RANSAC."""
        if len(matches) < self.min_match_count:
            raise RuntimeError(
                f"Not enough matches: {len(matches)} found, "
                f"need at least {self.min_match_count}."
            )

        pts_before = np.float32(
            [kp_before[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        pts_after = np.float32(
            [kp_after[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        if self.warp_method == WarpMethod.HOMOGRAPHY:
            M, mask = cv2.findHomography(
                pts_after, pts_before,
                cv2.RANSAC, self.ransac_thresh,
            )
        else:
            M, mask = cv2.estimateAffinePartial2D(
                pts_after, pts_before,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
            )

        if M is None:
            raise RuntimeError(
                "Transform estimation returned None — "
                "matches may be degenerate (collinear, etc.)."
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
        """Mean reprojection error on inlier matches (pixels)."""
        if inlier_mask is None or inlier_mask.ravel().sum() == 0:
            return float("inf")

        mask_flat = inlier_mask.ravel().astype(bool)

        pts_after = np.float32(
            [kp_after[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        pts_before = np.float32(
            [kp_before[m.queryIdx].pt for m in matches]
        ).reshape(-1, 2)

        if transform.shape == (3, 3):
            projected = cv2.perspectiveTransform(pts_after, transform).reshape(-1, 2)
        else:
            ones = np.ones((pts_after.shape[0], 1, 1), dtype=np.float32)
            pts_h = np.concatenate([pts_after, ones], axis=2)
            projected = pts_h.reshape(-1, 3) @ transform.T

        errors = np.linalg.norm(projected[mask_flat] - pts_before[mask_flat], axis=1)
        return float(errors.mean())

    # ------------------------------------------------------------------
    # internal: warping
    # ------------------------------------------------------------------

    def _warp_image(
        self, image: np.ndarray, transform: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:
        """Apply the estimated transform to warp an image."""
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