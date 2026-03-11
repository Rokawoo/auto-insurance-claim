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
- Both homography (8-DOF) and affine (6-DOF / 4-DOF) transforms — affine
  is more stable on 3D subjects like cars where perspective correction can
  distort.
- Homography sanity checking: determinant, condition number, and corner-
  mapping validation to reject degenerate transforms that flip/collapse
  the image.
- Spatial distribution check: ensures matched keypoints aren't all
  clustered in one region, which produces unreliable transforms.
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

import logging
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════

class WarpMethod(Enum):
    """Which geometric transform to estimate."""
    HOMOGRAPHY = "homography"       # 8-DOF perspective
    AFFINE = "affine"               # 6-DOF full affine
    AFFINE_PARTIAL = "affine_partial"  # 4-DOF similarity (rotation + uniform scale + translation)


@dataclass
class AlignmentResult:
    """Everything the aligner produces.

    Attributes
    ----------
    warped_after : np.ndarray
        The "after" image warped into the "before" image's frame.
    valid_mask : np.ndarray
        Binary mask (uint8, 0/255) of the region that has actual image
        data after warping.  Pixels outside this are black border
        artifacts and must be excluded from downstream diff/comparison.
    transform : np.ndarray
        The estimated transform matrix (3×3 for homography, 2×3 for affine).
    warp_method : WarpMethod
        Which method produced this result.
    feature_method : str
        Which detector was actually used (may differ from config if
        fallback kicked in).
    num_inliers : int
        RANSAC inlier count — higher = more confident alignment.
    inlier_ratio : float
        inliers / total_good_matches — closer to 1.0 = cleaner match set.
    reprojection_error : float
        Mean reprojection error of inliers in pixels.
        Lower = more accurate.  >5 px is suspicious.
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


# ══════════════════════════════════════════════════════════════════════
# Aligner
# ══════════════════════════════════════════════════════════════════════

class ImageAligner:
    """Aligns the "after" image to the "before" image.

    Parameters
    ----------
    config : dict
        The ``alignment`` section of the pipeline config.  Keys:

        feature_method : str
            ``"orb"`` | ``"sift"`` | ``"akaze"``  (default ``"orb"``)
        max_features : int
            Max keypoints to detect (default 10 000).
        match_method : str
            ``"bf"`` | ``"flann"``  (default ``"bf"``)
        ratio_threshold : float
            Lowe's ratio test threshold (default 0.75).
        ransac_reproj_threshold : float
            RANSAC inlier pixel tolerance (default 5.0).
        min_match_count : int
            Minimum matches required (default 10).
        warp_method : str
            ``"homography"`` | ``"affine"`` | ``"affine_partial"``
            (default ``"homography"``).
        max_reprojection_error : float
            Reliability threshold in pixels (default 5.0).
        min_inlier_ratio : float
            Reliability threshold (default 0.25).
        fallback : bool
            Try other detectors if primary fails (default True).
        normalize_exposure : bool
            Apply CLAHE before detection (default True).
        clahe_clip_limit : float
            CLAHE clip limit for normalization (default 3.0).
        clahe_grid_size : list[int]
            CLAHE tile grid [rows, cols] (default [8, 8]).
    """

    FEATURE_DETECTORS = {"orb", "sift", "akaze"}
    # fallback order: SIFT is most robust, then AKAZE, then ORB
    FALLBACK_ORDER = ["sift", "akaze", "orb"]

    def __init__(self, config: dict) -> None:
        self.feature_method: str = config.get("feature_method", "orb")
        self.max_features: int = config.get("max_features", 10_000)
        self.match_method: str = config.get("match_method", "bf")
        self.ratio_threshold: float = config.get("ratio_threshold", 0.75)
        self.ransac_thresh: float = config.get("ransac_reproj_threshold", 5.0)
        self.min_match_count: int = config.get("min_match_count", 10)
        self.warp_method: WarpMethod = WarpMethod(
            config.get("warp_method", "homography")
        )
        self.max_reproj_error: float = config.get("max_reprojection_error", 5.0)
        self.min_inlier_ratio: float = config.get("min_inlier_ratio", 0.25)
        self.fallback: bool = config.get("fallback", True)
        self.normalize_exposure: bool = config.get("normalize_exposure", True)
        self.clahe_clip: float = config.get("clahe_clip_limit", 3.0)
        self.clahe_grid: tuple[int, int] = tuple(
            config.get("clahe_grid_size", [8, 8])
        )

        if self.feature_method not in self.FEATURE_DETECTORS:
            raise ValueError(
                f"Unknown feature method '{self.feature_method}'. "
                f"Choose from {self.FEATURE_DETECTORS}"
            )

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def align(self, before: np.ndarray, after: np.ndarray) -> AlignmentResult:
        """Compute transform and warp *after* to match *before*.

        If ``fallback`` is enabled and the primary detector doesn't
        produce enough matches, automatically tries other detectors
        before giving up.

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
        before_norm = self._normalize(before)
        after_norm = self._normalize(after)

        # build ordered list of detectors to attempt
        methods_to_try = [self.feature_method]
        if self.fallback:
            for m in self.FALLBACK_ORDER:
                if m not in methods_to_try:
                    methods_to_try.append(m)

        last_error: RuntimeError | None = None
        for method in methods_to_try:
            try:
                result = self._try_align(
                    before_norm, after_norm, before, after, method
                )
                if method != self.feature_method:
                    logger.info(
                        "Primary detector '%s' failed; succeeded with '%s'",
                        self.feature_method, method,
                    )
                return result

            except RuntimeError as exc:
                last_error = exc
                logger.debug(
                    "Alignment with '%s' failed: %s", method, exc
                )
                continue

        raise RuntimeError(
            f"Alignment failed with all methods {methods_to_try}. "
            f"Last error: {last_error}"
        )

    # ══════════════════════════════════════════════════════════════════
    # Core alignment attempt
    # ══════════════════════════════════════════════════════════════════

    def _try_align(
        self,
        before_norm: np.ndarray,
        after_norm: np.ndarray,
        before_orig: np.ndarray,
        after_orig: np.ndarray,
        feature_method: str,
    ) -> AlignmentResult:
        """Single alignment attempt with a specific detector.

        Feature detection/matching runs on the normalized (CLAHE) images,
        but warping is applied to the originals so output quality isn't
        degraded.
        """
        # ── detect + describe ──────────────────────────────────────
        detector = self._create_detector(feature_method)
        kp_before, desc_before = detector.detectAndCompute(before_norm, None)
        kp_after, desc_after = detector.detectAndCompute(after_norm, None)

        if desc_before is None or desc_after is None:
            raise RuntimeError(
                f"[{feature_method}] No descriptors in one or both images."
            )

        if len(kp_before) < self.min_match_count:
            raise RuntimeError(
                f"[{feature_method}] Too few keypoints in before image: "
                f"{len(kp_before)} (need {self.min_match_count})"
            )
        if len(kp_after) < self.min_match_count:
            raise RuntimeError(
                f"[{feature_method}] Too few keypoints in after image: "
                f"{len(kp_after)} (need {self.min_match_count})"
            )

        logger.debug(
            "[%s] Keypoints: before=%d, after=%d",
            feature_method, len(kp_before), len(kp_after),
        )

        # ── match + ratio test ─────────────────────────────────────
        matcher = self._create_matcher(desc_before.dtype)
        good_matches = self._match_and_filter(matcher, desc_before, desc_after)

        logger.debug(
            "[%s] Good matches after ratio test: %d",
            feature_method, len(good_matches),
        )

        # ── check spatial distribution ─────────────────────────────
        self._check_spatial_distribution(
            kp_before, good_matches, before_norm.shape[:2], feature_method
        )

        # ── estimate transform ─────────────────────────────────────
        transform, inlier_mask = self._estimate_transform(
            kp_before, kp_after, good_matches
        )

        num_inliers = int(inlier_mask.ravel().sum()) if inlier_mask is not None else 0
        inlier_ratio = num_inliers / max(len(good_matches), 1)

        # ── validate transform geometry (homography-specific) ──────
        h, w = before_orig.shape[:2]
        if self.warp_method == WarpMethod.HOMOGRAPHY:
            self._validate_homography(transform, (h, w))

        # ── reprojection error ─────────────────────────────────────
        reproj_error = self._reprojection_error(
            kp_before, kp_after, good_matches, transform, inlier_mask
        )

        logger.debug(
            "[%s] Inliers=%d (%.1f%%), reproj_error=%.2fpx",
            feature_method, num_inliers, inlier_ratio * 100, reproj_error,
        )

        # ── warp the ORIGINAL after image ──────────────────────────
        warped = self._warp_image(after_orig, transform, (h, w))

        # ── build valid-region mask ────────────────────────────────
        white = np.full(after_orig.shape[:2], 255, dtype=np.uint8)
        valid_mask = self._warp_image(white, transform, (h, w))
        # erode slightly to remove interpolation fuzz at warp edges
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        valid_mask = cv2.erode(valid_mask, erode_kernel, iterations=1)

        # ── reliability assessment ─────────────────────────────────
        is_reliable = (
            num_inliers >= self.min_match_count
            and inlier_ratio >= self.min_inlier_ratio
            and reproj_error <= self.max_reproj_error
        )

        if not is_reliable:
            logger.warning(
                "[%s] Alignment below reliability thresholds "
                "(inliers=%d, ratio=%.2f, reproj=%.2fpx). "
                "Downstream pixel-diff results may be noisy.",
                feature_method, num_inliers, inlier_ratio, reproj_error,
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

    # ══════════════════════════════════════════════════════════════════
    # Preprocessing
    # ══════════════════════════════════════════════════════════════════

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image for robust feature detection.

        Converts to grayscale (feature detection only uses intensity) and
        optionally applies CLAHE to handle exposure differences between
        the before and after shots.

        Returns
        -------
        np.ndarray
            Grayscale uint8 image.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if not self.normalize_exposure:
            return gray

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.clahe_grid,
        )
        return clahe.apply(gray)

    # ══════════════════════════════════════════════════════════════════
    # Feature detection
    # ══════════════════════════════════════════════════════════════════

    def _create_detector(self, method: str) -> cv2.Feature2D:
        """Instantiate a keypoint detector by name.

        Parameters
        ----------
        method : str
            ``"orb"``, ``"sift"``, or ``"akaze"``.

        Returns
        -------
        cv2.Feature2D

        Raises
        ------
        ValueError
            If *method* is not recognized.
        """
        if method == "orb":
            return cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=1.2,
                nlevels=12,         # extra pyramid levels for scale robustness
                edgeThreshold=31,
                patchSize=31,
            )
        elif method == "sift":
            return cv2.SIFT_create(
                nfeatures=self.max_features,
                contrastThreshold=0.03,  # lower → more keypoints (default 0.04)
            )
        elif method == "akaze":
            return cv2.AKAZE_create(
                threshold=0.001,    # lower → more keypoints (default 0.001)
            )
        else:
            raise ValueError(f"Unknown feature method: {method}")

    # ══════════════════════════════════════════════════════════════════
    # Matching
    # ══════════════════════════════════════════════════════════════════

    def _create_matcher(self, descriptor_dtype: np.dtype) -> cv2.DescriptorMatcher:
        """Instantiate the descriptor matcher with the correct distance norm.

        Binary descriptors (ORB, AKAZE) use Hamming distance;
        float descriptors (SIFT) use L2.

        Parameters
        ----------
        descriptor_dtype : np.dtype
            ``np.uint8`` for binary, ``np.float32`` for float.

        Returns
        -------
        cv2.DescriptorMatcher
        """
        is_binary = descriptor_dtype == np.uint8
        norm = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2

        if self.match_method == "bf":
            # crossCheck=False because we use knnMatch with ratio test
            return cv2.BFMatcher(norm, crossCheck=False)

        elif self.match_method == "flann":
            if is_binary:
                index_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2,
                )
            else:
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=80)
            return cv2.FlannBasedMatcher(index_params, search_params)

        else:
            raise ValueError(f"Unknown match method: {self.match_method}")

    def _match_and_filter(
        self,
        matcher: cv2.DescriptorMatcher,
        desc_before: np.ndarray,
        desc_after: np.ndarray,
    ) -> list[cv2.DMatch]:
        """Match descriptors and apply Lowe's ratio test.

        For each descriptor in *before*, find the two nearest neighbors
        in *after*.  Keep the match only if the best is significantly
        closer than the second-best (ratio < threshold).

        Parameters
        ----------
        matcher : cv2.DescriptorMatcher
        desc_before, desc_after : np.ndarray
            Descriptor arrays.

        Returns
        -------
        list[cv2.DMatch]
            Filtered good matches.
        """
        raw_matches = matcher.knnMatch(desc_before, desc_after, k=2)

        good: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) < 2:
                # can happen when the after image has very few descriptors
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)

        return good

    # ══════════════════════════════════════════════════════════════════
    # Spatial distribution check
    # ══════════════════════════════════════════════════════════════════

    def _check_spatial_distribution(
        self,
        kp_before: list,
        matches: list[cv2.DMatch],
        image_shape: tuple[int, int],
        method_name: str,
    ) -> None:
        """Verify that matched keypoints aren't all clustered together.

        If all matches are concentrated in one region, the estimated
        transform is only valid for that area and will badly distort the
        rest of the image.  We check that matches span at least 25% of
        the image in both x and y.

        Parameters
        ----------
        kp_before : list[cv2.KeyPoint]
        matches : list[cv2.DMatch]
        image_shape : tuple[int, int]
            (height, width)
        method_name : str
            For error messages.

        Raises
        ------
        RuntimeError
            If spatial spread is insufficient.
        """
        if len(matches) < self.min_match_count:
            return  # will be caught by _estimate_transform

        pts = np.float32([kp_before[m.queryIdx].pt for m in matches])
        h, w = image_shape

        x_spread = (pts[:, 0].max() - pts[:, 0].min()) / max(w, 1)
        y_spread = (pts[:, 1].max() - pts[:, 1].min()) / max(h, 1)

        min_spread = 0.25
        if x_spread < min_spread and y_spread < min_spread:
            raise RuntimeError(
                f"[{method_name}] Matches are spatially clustered "
                f"(x_spread={x_spread:.2f}, y_spread={y_spread:.2f}). "
                f"Need at least {min_spread:.0%} spread in one axis "
                f"for a reliable global transform."
            )

    # ══════════════════════════════════════════════════════════════════
    # Transform estimation
    # ══════════════════════════════════════════════════════════════════

    def _estimate_transform(
        self,
        kp_before: list,
        kp_after: list,
        matches: list[cv2.DMatch],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate homography or affine transform via RANSAC.

        Parameters
        ----------
        kp_before, kp_after : list[cv2.KeyPoint]
        matches : list[cv2.DMatch]

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (transform_matrix, inlier_mask)

        Raises
        ------
        RuntimeError
            If too few matches, transform is None, or too few inliers.
        """
        if len(matches) < self.min_match_count:
            raise RuntimeError(
                f"Not enough matches: {len(matches)} found, "
                f"need at least {self.min_match_count}."
            )

        # extract matched point coordinates
        pts_before = np.float32(
            [kp_before[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        pts_after = np.float32(
            [kp_after[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # estimate based on configured warp method
        if self.warp_method == WarpMethod.HOMOGRAPHY:
            M, mask = cv2.findHomography(
                pts_after, pts_before,
                cv2.RANSAC, self.ransac_thresh,
            )
        elif self.warp_method == WarpMethod.AFFINE:
            M, mask = cv2.estimateAffine2D(
                pts_after, pts_before,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
            )
        else:  # AFFINE_PARTIAL — 4-DOF similarity
            M, mask = cv2.estimateAffinePartial2D(
                pts_after, pts_before,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
            )

        if M is None:
            raise RuntimeError(
                "Transform estimation returned None — matches may be "
                "degenerate (collinear points, insufficient spread, etc.)."
            )

        num_inliers = int(mask.ravel().sum()) if mask is not None else 0
        if num_inliers < 4:
            raise RuntimeError(
                f"Only {num_inliers} RANSAC inliers — too few for a "
                f"reliable transform."
            )

        return M, mask

    # ══════════════════════════════════════════════════════════════════
    # Homography validation
    # ══════════════════════════════════════════════════════════════════

    def _validate_homography(
        self,
        H: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        """Reject degenerate homographies that would mangle the image.

        Checks performed:
        1. **Determinant** — should be positive and not too far from 1.0.
           A negative determinant means the image is flipped (orientation
           reversed).  A very small or very large determinant means
           extreme scaling.
        2. **Condition number** — a near-singular matrix will amplify
           noise catastrophically.
        3. **Corner mapping** — warp the four image corners and verify the
           result is still a reasonable convex quadrilateral that hasn't
           collapsed or folded over.

        Parameters
        ----------
        H : np.ndarray
            3×3 homography matrix.
        image_shape : tuple[int, int]
            (height, width) of the reference image.

        Raises
        ------
        RuntimeError
            If any sanity check fails.
        """
        if H.shape != (3, 3):
            return  # not a homography — skip validation

        # 1. determinant check
        det = np.linalg.det(H)
        if det < 0:
            raise RuntimeError(
                f"Homography has negative determinant ({det:.4f}) — "
                f"image would be flipped/mirrored."
            )
        if det < 0.01 or det > 100:
            raise RuntimeError(
                f"Homography determinant ({det:.4f}) indicates extreme "
                f"scaling — transform is likely degenerate."
            )

        # 2. condition number check
        cond = np.linalg.cond(H)
        if cond > 1e6:
            raise RuntimeError(
                f"Homography is near-singular (condition number {cond:.0f}). "
                f"Transform would amplify noise."
            )

        # 3. corner mapping — verify output is a reasonable quad
        h, w = image_shape
        corners_src = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ]).reshape(-1, 1, 2)

        corners_dst = cv2.perspectiveTransform(corners_src, H).reshape(-1, 2)

        # check that corners map inside a reasonable region
        # (allow 50% margin beyond image bounds for mild perspective)
        margin = max(h, w) * 0.5
        if (corners_dst < -margin).any() or (corners_dst[:, 0] > w + margin).any() \
                or (corners_dst[:, 1] > h + margin).any():
            raise RuntimeError(
                "Homography maps corners far outside image bounds — "
                "transform is likely degenerate."
            )

        # check that the quadrilateral hasn't collapsed (area > 10% of original)
        quad_area = cv2.contourArea(corners_dst.astype(np.float32))
        original_area = h * w
        area_ratio = quad_area / max(original_area, 1)

        if area_ratio < 0.1:
            raise RuntimeError(
                f"Homography collapses image to {area_ratio:.1%} of "
                f"original area — transform is degenerate."
            )

        # check convexity — a folded quad means the perspective is impossible
        if not cv2.isContourConvex(corners_dst.astype(np.int32)):
            raise RuntimeError(
                "Homography produces a non-convex quadrilateral — "
                "the transform folds the image over itself."
            )

    # ══════════════════════════════════════════════════════════════════
    # Quality metrics
    # ══════════════════════════════════════════════════════════════════

    def _reprojection_error(
        self,
        kp_before: list,
        kp_after: list,
        matches: list[cv2.DMatch],
        transform: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> float:
        """Mean reprojection error on RANSAC inlier matches.

        Projects the "after" keypoints through the estimated transform
        and measures how far they land from their corresponding "before"
        keypoints.  Only inlier matches are considered.

        Parameters
        ----------
        kp_before, kp_after : list[cv2.KeyPoint]
        matches : list[cv2.DMatch]
        transform : np.ndarray
            3×3 or 2×3 matrix.
        inlier_mask : np.ndarray
            Boolean mask from RANSAC.

        Returns
        -------
        float
            Mean error in pixels.  ``inf`` if no inliers.
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

        # project through transform
        if transform.shape == (3, 3):
            projected = cv2.perspectiveTransform(pts_after, transform)
        else:
            # cv2.transform handles 2×3 affine matrices correctly
            projected = cv2.transform(pts_after, transform)

        projected = projected.reshape(-1, 2)

        errors = np.linalg.norm(
            projected[mask_flat] - pts_before[mask_flat], axis=1
        )
        return float(errors.mean())

    # ══════════════════════════════════════════════════════════════════
    # Warping
    # ══════════════════════════════════════════════════════════════════

    def _warp_image(
        self,
        image: np.ndarray,
        transform: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Apply the estimated transform to warp an image.

        Parameters
        ----------
        image : np.ndarray
            Image to warp (the "after" image or a mask).
        transform : np.ndarray
            3×3 (homography) or 2×3 (affine) matrix.
        target_shape : tuple[int, int]
            (height, width) of the output.

        Returns
        -------
        np.ndarray
            Warped image.  Border pixels are set to 0 (black).
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