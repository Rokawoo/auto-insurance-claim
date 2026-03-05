#!/usr/bin/env python3
"""Visualize alignment results on real car images.

All paths resolve relative to <project_root>/tests/ since that's where
the images and outputs directories live.

Expects:
    <project_root>/tests/images/car A - 1.png
    <project_root>/tests/images/car A - 2.png

Produces:
    <project_root>/tests/outputs/alignment_demo/01_inputs.png
    <project_root>/tests/outputs/alignment_demo/02_keypoints.png
    <project_root>/tests/outputs/alignment_demo/03_matches.png
    <project_root>/tests/outputs/alignment_demo/04_warped_comparison.png
    <project_root>/tests/outputs/alignment_demo/05_method_comparison.png
    <project_root>/tests/outputs/alignment_demo/06_quality_report.txt

Usage (from anywhere):
    python path/to/scripts/demo_alignment.py
    python path/to/scripts/demo_alignment.py --method sift --warp affine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# project root = scripts/ -> vda/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"

sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.alignment.aligner import ImageAligner, AlignmentResult


# ─── defaults ───────────────────────────────────────────────────

DEFAULT_BEFORE = TESTS_DIR / "images" / "car A - 1.png"
DEFAULT_AFTER  = TESTS_DIR / "images" / "car A - 2.png"
DEFAULT_outputs = TESTS_DIR / "outputs" / "alignment_demo"


# ─── helpers ────────────────────────────────────────────────────

def load_or_die(path: Path) -> np.ndarray:
    if not path.exists():
        print(f"\n  ERROR: {path} not found")
        print(f"  place your images at:")
        print(f"    {DEFAULT_BEFORE}")
        print(f"    {DEFAULT_AFTER}")
        sys.exit(1)
    img = cv2.imread(str(path))
    if img is None:
        print(f"\n  ERROR: cv2.imread failed on {path}")
        sys.exit(1)
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def label(img, text, pos=(10, 30), scale=0.65):
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_tile(img, tw, th):
    return cv2.resize(to_bgr(img), (tw, th), interpolation=cv2.INTER_AREA)


def save(path, img):
    cv2.imwrite(str(path), img)
    print(f"    -> {path.name}")


# ─── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alignment demo on real car photos")
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_outputs)
    parser.add_argument("--method", default="orb", choices=["orb", "sift", "akaze"])
    parser.add_argument("--warp", default="homography", choices=["homography", "affine"])
    args = parser.parse_args()

    out = args.outputs
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  ALIGNMENT DEMO")
    print("=" * 65)
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  tests dir:    {TESTS_DIR}")

    # ── load ────────────────────────────────────────────────────
    print(f"\n[1] loading images...")
    before_bgr = load_or_die(args.before)
    after_bgr = load_or_die(args.after)
    print(f"    before: {args.before.name} ({before_bgr.shape[1]}x{before_bgr.shape[0]})")
    print(f"    after:  {args.after.name} ({after_bgr.shape[1]}x{after_bgr.shape[0]})")

    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        print(f"    resizing after to match before...")
        after_bgr = cv2.resize(after_bgr, (before_bgr.shape[1], before_bgr.shape[0]),
                                interpolation=cv2.INTER_AREA)

    before_gray = to_gray(before_bgr)
    after_gray = to_gray(after_bgr)

    save(out / "01_inputs.png", np.hstack([
        label(before_bgr, "BEFORE (car A - 1)"),
        label(after_bgr, "AFTER (car A - 2)"),
    ]))

    # ── align with primary method ───────────────────────────────
    print(f"\n[2] aligning with {args.method.upper()} + {args.warp}...")
    config = {
        "feature_method": args.method,
        "max_features": 8000,
        "match_method": "bf",
        "ratio_threshold": 0.75,
        "ransac_reproj_threshold": 5.0,
        "min_match_count": 10,
        "warp_method": args.warp,
    }

    aligner = ImageAligner(config)
    try:
        result = aligner.align(before_gray, after_gray)
    except RuntimeError as e:
        print(f"\n    FAILED: {e}")
        print(f"    try --method sift or --warp affine")
        sys.exit(1)

    print(f"    keypoints: before={len(result.keypoints_before)}, after={len(result.keypoints_after)}")
    print(f"    matches: {len(result.matches)}")
    print(f"    inliers: {result.num_inliers} ({result.inlier_ratio:.1%})")
    print(f"    reprojection error: {result.reprojection_error:.3f} px")
    print(f"    reliable: {result.is_reliable}")

    # ── keypoints visualization ─────────────────────────────────
    print(f"\n[3] visualizing keypoints...")
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    kp_b = cv2.drawKeypoints(before_bgr, result.keypoints_before[:500], None,
                              color=(0, 255, 0), flags=flags)
    kp_a = cv2.drawKeypoints(after_bgr, result.keypoints_after[:500], None,
                              color=(0, 255, 0), flags=flags)
    save(out / "02_keypoints.png", np.hstack([
        label(kp_b, f"BEFORE — {len(result.keypoints_before)} keypoints"),
        label(kp_a, f"AFTER — {len(result.keypoints_after)} keypoints"),
    ]))

    # ── matches visualization ───────────────────────────────────
    print(f"\n[4] visualizing matches...")
    n_draw = min(80, len(result.matches))
    match_img = cv2.drawMatches(
        before_gray, result.keypoints_before,
        after_gray, result.keypoints_after,
        result.matches[:n_draw], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save(out / "03_matches.png",
         label(match_img, f"{n_draw} of {len(result.matches)} matches shown"))

    # ── warped comparison ───────────────────────────────────────
    print(f"\n[5] warping after image...")
    h, w = before_bgr.shape[:2]
    if result.transform.shape == (3, 3):
        warped_bgr = cv2.warpPerspective(after_bgr, result.transform, (w, h), borderValue=(0, 0, 0))
    else:
        warped_bgr = cv2.warpAffine(after_bgr, result.transform, (w, h), borderValue=(0, 0, 0))

    save(out / "04_warped_comparison.png", np.hstack([
        label(before_bgr, "BEFORE (reference)"),
        label(after_bgr, "AFTER (original)"),
        label(warped_bgr, "AFTER (aligned)"),
    ]))

    # ── compare all 4 method combos ─────────────────────────────
    print(f"\n[6] comparing all method combinations...")

    combos = [
        ("ORB+homo",    "orb",  "homography"),
        ("SIFT+homo",   "sift", "homography"),
        ("ORB+affine",  "orb",  "affine"),
        ("SIFT+affine", "sift", "affine"),
    ]

    naive_diff = cv2.absdiff(before_gray, after_gray)
    _, naive_thresh = cv2.threshold(naive_diff, 30, 255, cv2.THRESH_BINARY)
    naive_count = cv2.countNonZero(naive_thresh)

    tiles = []
    report_lines = [
        "ALIGNMENT QUALITY REPORT",
        "=" * 70,
        f"before: {args.before}",
        f"after:  {args.after}",
        f"image size: {w}x{h}",
        f"naive diff pixels: {naive_count:,} ({naive_count / before_gray.size * 100:.1f}%)",
        "",
        f"{'method':<18} {'matches':>8} {'inliers':>8} {'ratio':>8} {'reproj':>9} "
        f"{'diff_px':>10} {'reduction':>10} {'reliable':>9}",
        "-" * 90,
    ]

    tw, th = 400, 300

    for name, feat, warp in combos:
        cfg = {
            "feature_method": feat,
            "max_features": 8000,
            "match_method": "bf",
            "ratio_threshold": 0.75,
            "ransac_reproj_threshold": 5.0,
            "min_match_count": 10,
            "warp_method": warp,
        }
        try:
            r = ImageAligner(cfg).align(before_gray, after_gray)
            adiff = cv2.absdiff(before_gray, r.warped_after)
            _, athresh = cv2.threshold(adiff, 30, 255, cv2.THRESH_BINARY)
            acount = cv2.countNonZero(athresh)
            reduction = (1 - acount / max(naive_count, 1)) * 100

            if r.transform.shape == (3, 3):
                w_bgr = cv2.warpPerspective(after_bgr, r.transform, (w, h), borderValue=(0, 0, 0))
            else:
                w_bgr = cv2.warpAffine(after_bgr, r.transform, (w, h), borderValue=(0, 0, 0))

            tile = resize_tile(w_bgr, tw, th)
            tile = label(tile, f"{name}", scale=0.5)
            tile = label(tile, f"inliers:{r.num_inliers} reproj:{r.reprojection_error:.1f}px",
                         pos=(10, th - 40), scale=0.4)
            tile = label(tile, f"reduction:{reduction:.0f}% reliable:{r.is_reliable}",
                         pos=(10, th - 15), scale=0.4)
            tiles.append(tile)

            line = (f"{name:<18} {len(r.matches):>8,} {r.num_inliers:>8,} "
                    f"{r.inlier_ratio:>7.1%} {r.reprojection_error:>8.2f}px "
                    f"{acount:>10,} {reduction:>9.1f}% {str(r.is_reliable):>9}")
            report_lines.append(line)
            print(f"    {line}")

        except RuntimeError as e:
            tile = np.full((th, tw, 3), 40, dtype=np.uint8)
            tile = label(tile, f"{name}: FAILED", scale=0.5)
            tiles.append(tile)
            report_lines.append(f"{name:<18} FAILED: {e}")
            print(f"    {name}: FAILED — {e}")

    row1 = np.hstack(tiles[:2])
    row2 = np.hstack(tiles[2:])
    comparison = np.vstack([row1, row2])
    save(out / "05_method_comparison.png", comparison)

    report_text = "\n".join(report_lines)
    report_path = out / "06_quality_report.txt"
    report_path.write_text(report_text)
    print(f"    -> {report_path.name}")

    # ── done ────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    n_files = len(list(out.glob("*")))
    print(f"  DONE — {n_files} files in {out}/")
    print(f"  check 05_method_comparison.png to pick the best method")
    print(f"  check 06_quality_report.txt for the numbers")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()