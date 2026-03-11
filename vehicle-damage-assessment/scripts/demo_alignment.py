#!/usr/bin/env python3
"""Visualize alignment + difference maps on real car images.

All paths resolve from the project root (this script's grandparent dir).
Images read from tests/images/, outputs written to tests/outputs/.

Produces:
    01_inputs.png              side-by-side raw inputs
    02_keypoints.png           detected keypoints on both images
    03_matches.png             matched feature lines
    04_warped_comparison.png   before | after original | after aligned
    05_naive_diff.png          pixel diff WITHOUT alignment
    06_aligned_diff.png        pixel diff WITH alignment (masked to valid region)
    07_diff_heatmaps.png       side-by-side heatmaps
    08_method_comparison.png   all 4 method combos warped + diff
    09_quality_report.txt      metrics table

Usage:
    python scripts/demo_alignment.py
    python scripts/demo_alignment.py --method sift --warp affine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.alignment.aligner import ImageAligner, AlignmentResult, WarpMethod


# ─── defaults ───────────────────────────────────────────────────

DEFAULT_BEFORE = TESTS_DIR / "images" / "car A - 1.png"
DEFAULT_AFTER  = TESTS_DIR / "images" / "car A - 2.png"
DEFAULT_outputs = TESTS_DIR / "outputs" / "alignment_demo"


# ─── helpers ────────────────────────────────────────────────────

def load_or_die(path: Path) -> np.ndarray:
    if not path.exists():
        print(f"\n  ERROR: {path} not found")
        print(f"  expected locations:")
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


def label(img, text, pos=(10, 30), scale=0.6):
    out = to_bgr(img).copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def tile(img, tw, th, lbl=None, sub=None):
    """resize to (tw, th), add optional label + subtitle."""
    t = cv2.resize(to_bgr(img), (tw, th), interpolation=cv2.INTER_AREA)
    if lbl:
        t = label(t, lbl, scale=0.5)
    if sub:
        t = label(t, sub, pos=(10, th - 12), scale=0.38)
    return t


def save(path, img):
    cv2.imwrite(str(path), img)
    print(f"    -> {path.name}")


def masked_diff(before_gray, warped_gray, valid_mask, threshold=25):
    """compute pixel diff only within the valid warped region.

    returns (raw_diff, binary_mask, count_of_flagged_pixels).
    """
    diff = cv2.absdiff(before_gray, warped_gray)
    # zero out anything outside the valid region
    diff = cv2.bitwise_and(diff, valid_mask)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, valid_mask)
    count = cv2.countNonZero(binary)
    return diff, binary, count


# ─── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alignment + diff demo")
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_outputs)
    parser.add_argument("--method", default="orb", choices=["orb", "sift", "akaze"])
    parser.add_argument("--warp", default="homography", choices=["homography", "affine"])
    args = parser.parse_args()

    out = args.outputs
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  ALIGNMENT + DIFFERENCE MAP DEMO")
    print("=" * 65)
    print(f"  project root: {PROJECT_ROOT}")

    # ── 1. load ─────────────────────────────────────────────────
    print(f"\n[1/8] loading images...")
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
    h, w = before_gray.shape[:2]
    total_px = before_gray.size

    save(out / "01_inputs.png", np.hstack([
        label(before_bgr, "BEFORE (car A - 1)"),
        label(after_bgr, "AFTER (car A - 2)"),
    ]))

    # ── 2. align with primary method ────────────────────────────
    print(f"\n[2/8] aligning ({args.method.upper()} + {args.warp}, fallback enabled)...")
    config = {
        "feature_method": args.method,
        "max_features": 10000,
        "match_method": "bf",
        "ratio_threshold": 0.75,
        "ransac_reproj_threshold": 5.0,
        "min_match_count": 10,
        "warp_method": args.warp,
        "fallback": True,
        "normalize_exposure": True,
    }

    aligner = ImageAligner(config)
    try:
        result = aligner.align(before_bgr, after_bgr)
    except RuntimeError as e:
        print(f"\n    FAILED: {e}")
        sys.exit(1)

    used = result.feature_method.upper()
    if result.feature_method != args.method:
        print(f"    NOTE: primary method {args.method.upper()} failed, fell back to {used}")
    print(f"    detector used: {used}")
    print(f"    keypoints: before={len(result.keypoints_before)}, after={len(result.keypoints_after)}")
    print(f"    matches: {result.num_matches}")
    print(f"    inliers: {result.num_inliers} ({result.inlier_ratio:.1%})")
    print(f"    reprojection error: {result.reprojection_error:.3f} px")
    print(f"    reliable: {result.is_reliable}")

    valid_px = cv2.countNonZero(result.valid_mask)
    print(f"    valid region: {valid_px/total_px*100:.1f}% of image")

    # convenience: get the warped gray for diff computations
    warped_gray = to_gray(result.warped_after)

    # ── 3. keypoints ────────────────────────────────────────────
    print(f"\n[3/8] keypoint visualization...")
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    kp_b = cv2.drawKeypoints(before_bgr, result.keypoints_before[:500], None,
                              color=(0, 255, 0), flags=flags)
    kp_a = cv2.drawKeypoints(after_bgr, result.keypoints_after[:500], None,
                              color=(0, 255, 0), flags=flags)
    save(out / "02_keypoints.png", np.hstack([
        label(kp_b, f"BEFORE — {len(result.keypoints_before)} kp ({used})"),
        label(kp_a, f"AFTER — {len(result.keypoints_after)} kp ({used})"),
    ]))

    # ── 4. matches ──────────────────────────────────────────────
    print(f"\n[4/8] match visualization...")
    # draw on the normalized images for clarity
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    norm_b = clahe.apply(before_gray)
    norm_a = clahe.apply(after_gray)

    n_draw = min(100, len(result.matches))
    match_img = cv2.drawMatches(
        norm_b, result.keypoints_before,
        norm_a, result.keypoints_after,
        result.matches[:n_draw], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save(out / "03_matches.png",
         label(match_img, f"{n_draw} of {result.num_matches} matches ({used})"))

    # ── 5. warp comparison ──────────────────────────────────────
    print(f"\n[5/8] warp comparison...")
    save(out / "04_warped_comparison.png", np.hstack([
        label(before_bgr, "BEFORE (reference)"),
        label(after_bgr, "AFTER (original)"),
        label(result.warped_after, "AFTER (aligned)"),
    ]))

    # ── 6. naive diff (no alignment) ────────────────────────────
    print(f"\n[6/8] difference maps...")

    naive_diff = cv2.absdiff(before_gray, after_gray)
    _, naive_binary = cv2.threshold(naive_diff, 25, 255, cv2.THRESH_BINARY)
    naive_count = cv2.countNonZero(naive_binary)

    save(out / "05_naive_diff.png", np.hstack([
        label(naive_diff, f"NAIVE DIFF (raw, no alignment)"),
        label(naive_binary, f"NAIVE BINARY — {naive_count:,}px ({naive_count/total_px*100:.1f}%)"),
    ]))

    # ── 7. aligned + masked diff ────────────────────────────────
    aligned_raw, aligned_binary, aligned_count = masked_diff(
        before_gray, warped_gray, result.valid_mask, threshold=25
    )

    reduction = (1 - aligned_count / max(naive_count, 1)) * 100
    print(f"    naive diff:   {naive_count:>8,} px ({naive_count/total_px*100:.1f}%)")
    print(f"    aligned diff: {aligned_count:>8,} px ({aligned_count/total_px*100:.1f}%) "
          f"[masked to valid region]")
    print(f"    reduction:    {reduction:.1f}%")

    save(out / "06_aligned_diff.png", np.hstack([
        label(aligned_raw, "ALIGNED DIFF (raw, masked)"),
        label(aligned_binary, f"ALIGNED BINARY — {aligned_count:,}px ({reduction:.0f}% reduction)"),
    ]))

    # ── 8. heatmap comparison ───────────────────────────────────
    naive_heat = cv2.applyColorMap(naive_diff, cv2.COLORMAP_JET)
    aligned_heat = cv2.applyColorMap(aligned_raw, cv2.COLORMAP_JET)
    # dim the invalid region in the aligned heatmap so it's obvious
    mask_3ch = cv2.merge([result.valid_mask, result.valid_mask, result.valid_mask])
    aligned_heat = cv2.bitwise_and(aligned_heat, mask_3ch)

    save(out / "07_diff_heatmaps.png", np.hstack([
        label(naive_heat, f"NAIVE HEATMAP ({naive_count:,}px)"),
        label(aligned_heat, f"ALIGNED HEATMAP ({aligned_count:,}px, masked)"),
    ]))

    # ── 9. all 4 method combos ──────────────────────────────────
    print(f"\n[7/8] comparing all method combinations...")

    combos = [
        ("ORB+homo",    "orb",  "homography"),
        ("SIFT+homo",   "sift", "homography"),
        ("ORB+affine",  "orb",  "affine"),
        ("SIFT+affine", "sift", "affine"),
    ]

    tw, th = 400, 300
    warp_tiles = []
    diff_tiles = []
    report_lines = [
        "ALIGNMENT QUALITY REPORT",
        "=" * 95,
        f"before: {args.before}",
        f"after:  {args.after}",
        f"image size: {w}x{h} ({total_px:,} px)",
        f"naive diff: {naive_count:,} px ({naive_count/total_px*100:.1f}%)",
        "",
        f"{'method':<18} {'detector':>8} {'matches':>8} {'inliers':>8} "
        f"{'ratio':>7} {'reproj':>9} {'diff_px':>10} {'reduction':>10} {'reliable':>9}",
        "-" * 95,
    ]

    for name, feat, warp in combos:
        cfg = {
            "feature_method": feat,
            "max_features": 10000,
            "match_method": "bf",
            "ratio_threshold": 0.75,
            "ransac_reproj_threshold": 5.0,
            "min_match_count": 10,
            "warp_method": warp,
            "fallback": False,  # no fallback here — we want to compare each method directly
            "normalize_exposure": True,
        }
        try:
            r = ImageAligner(cfg).align(before_bgr, after_bgr)

            w_gray = to_gray(r.warped_after)
            _, _, acount = masked_diff(before_gray, w_gray, r.valid_mask, threshold=25)
            red = (1 - acount / max(naive_count, 1)) * 100

            # diff heatmap for this method
            adiff_raw, _, _ = masked_diff(before_gray, w_gray, r.valid_mask, threshold=0)
            aheat = cv2.applyColorMap(adiff_raw, cv2.COLORMAP_JET)
            m3 = cv2.merge([r.valid_mask] * 3)
            aheat = cv2.bitwise_and(aheat, m3)

            warp_tiles.append(tile(
                r.warped_after, tw, th,
                lbl=f"{name}",
                sub=f"inliers:{r.num_inliers} reproj:{r.reprojection_error:.1f}px reliable:{r.is_reliable}",
            ))
            diff_tiles.append(tile(
                aheat, tw, th,
                lbl=f"{name} diff",
                sub=f"{acount:,}px flagged ({red:.0f}% reduction)",
            ))

            line = (f"{name:<18} {r.feature_method:>8} {r.num_matches:>8,} "
                    f"{r.num_inliers:>8,} {r.inlier_ratio:>6.1%} "
                    f"{r.reprojection_error:>8.2f}px {acount:>10,} "
                    f"{red:>9.1f}% {str(r.is_reliable):>9}")
            report_lines.append(line)
            print(f"    {line}")

        except RuntimeError as e:
            gray_tile = np.full((th, tw, 3), 40, dtype=np.uint8)
            warp_tiles.append(label(gray_tile, f"{name}: FAILED", scale=0.45))
            diff_tiles.append(label(gray_tile, f"{name}: FAILED", scale=0.45))
            report_lines.append(f"{name:<18} FAILED: {e}")
            print(f"    {name}: FAILED")

    # assemble 2x4 grid: top row = warps, bottom row = diffs
    row_warps = np.hstack(warp_tiles)
    row_diffs = np.hstack(diff_tiles)
    grid = np.vstack([row_warps, row_diffs])
    save(out / "08_method_comparison.png", grid)

    # text report
    report_text = "\n".join(report_lines)
    report_path = out / "09_quality_report.txt"
    report_path.write_text(report_text)
    print(f"    -> {report_path.name}")

    # ── done ────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    n_files = len(list(out.glob("*")))
    print(f"  DONE — {n_files} files saved to:")
    print(f"    {out}")
    print(f"")
    print(f"  key files:")
    print(f"    04_warped_comparison.png  — see if alignment looks right")
    print(f"    07_diff_heatmaps.png     — naive vs aligned diff")
    print(f"    08_method_comparison.png  — pick the best method")
    print(f"    09_quality_report.txt    — numbers for all methods")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
