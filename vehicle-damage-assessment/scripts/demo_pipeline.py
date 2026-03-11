#!/usr/bin/env python3
"""Run the full correct pipeline on real car images.

    Align → Detect Vehicle → Mask → Diff → Visualize

This demo shows why each stage matters by comparing results
with and without the vehicle mask.

Expects:
    ./images/car A - 1.png   (before)
    ./images/car A - 2.png   (after)

Usage:
    cd vda/
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --method sift
    python scripts/demo_pipeline.py --before "images/car A - 1.png" --after "images/car A - 2.png"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.alignment.aligner import ImageAligner
from src.detection.vehicle_detector import VehicleDetector
from src.comparison.diff_engine import DiffEngine


# ── helpers ─────────────────────────────────────────────────────────

def load(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        print(f"\n  ERROR: {p.resolve()} not found")
        sys.exit(1)
    img = cv2.imread(str(p))
    if img is None:
        print(f"\n  ERROR: cv2.imread failed on {p}")
        sys.exit(1)
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img


def label(img, text, pos=(10, 30), scale=0.6):
    out = to_bgr(img).copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save(out_dir, name, img):
    path = out_dir / name
    cv2.imwrite(str(path), img)
    print(f"    -> {name}")


# ── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="images/car A - 1.png")
    parser.add_argument("--after", default="images/car A - 2.png")
    parser.add_argument("--output", default="outputs/pipeline_demo")
    parser.add_argument("--method", default="orb", choices=["orb", "sift", "akaze"])
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  FULL PIPELINE DEMO: Align → Detect → Mask → Diff")
    print("=" * 65)

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Load images
    # ════════════════════════════════════════════════════════════
    print(f"\n[1/6] LOAD")
    before_bgr = load(args.before)
    after_bgr = load(args.after)
    print(f"  before: {before_bgr.shape[1]}x{before_bgr.shape[0]}")
    print(f"  after:  {after_bgr.shape[1]}x{after_bgr.shape[0]}")

    # resize after to match before if needed
    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        print(f"  resizing after to match before...")
        after_bgr = cv2.resize(after_bgr, (before_bgr.shape[1], before_bgr.shape[0]))

    save(out, "01_before.png", label(before_bgr, "BEFORE"))
    save(out, "01_after.png", label(after_bgr, "AFTER"))

    before_gray = to_gray(before_bgr)
    after_gray = to_gray(after_bgr)

    # ════════════════════════════════════════════════════════════
    # STAGE 2: Align
    # ════════════════════════════════════════════════════════════
    print(f"\n[2/6] ALIGN ({args.method.upper()})")
    aligner = ImageAligner({
        "feature_method": args.method,
        "max_features": 8000,
        "match_method": "bf",
        "ratio_threshold": 0.75,
        "ransac_reproj_threshold": 5.0,
        "min_match_count": 10,
    })

    try:
        align_result = aligner.align(before_gray, after_gray)
    except RuntimeError as e:
        print(f"  ALIGNMENT FAILED: {e}")
        print(f"  try --method sift for better robustness")
        sys.exit(1)

    print(f"  matches: {len(align_result.matches)}, inliers: {align_result.num_inliers}")

    # also warp the BGR version for visualization
    h, w = before_bgr.shape[:2]
    warped_bgr = cv2.warpPerspective(after_bgr, align_result.homography, (w, h))

    save(out, "02_warped_after.png", label(warped_bgr, "AFTER (aligned)"))

    # match visualization
    n_draw = min(60, len(align_result.matches))
    match_img = cv2.drawMatches(
        before_gray, align_result.keypoints_before,
        after_gray, align_result.keypoints_after,
        align_result.matches[:n_draw], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save(out, "02_matches.png", match_img)

    # ════════════════════════════════════════════════════════════
    # STAGE 3: Detect vehicle (YOLOv8)
    # ════════════════════════════════════════════════════════════
    print(f"\n[3/6] DETECT VEHICLE (YOLOv8)")
    detector = VehicleDetector({"confidence_threshold": 0.3, "mask_dilation_kernel": 20})
    det_result = detector.detect(before_bgr)

    n_vehicles = len(det_result.boxes)
    print(f"  vehicles found: {n_vehicles}")
    for i in range(n_vehicles):
        box = det_result.boxes[i].astype(int)
        print(f"    #{i+1}: {tuple(box)}, conf={det_result.confidences[i]:.3f}")

    mask_pct = cv2.countNonZero(det_result.vehicle_mask) / det_result.vehicle_mask.size * 100
    print(f"  mask coverage: {mask_pct:.1f}% of image")

    if n_vehicles == 0:
        print("\n  WARNING: no vehicles detected. results will be unmasked (noisy).")

    # draw detection boxes on the image
    det_viz = before_bgr.copy()
    for i in range(n_vehicles):
        x1, y1, x2, y2 = det_result.boxes[i].astype(int)
        cv2.rectangle(det_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(det_viz, f"vehicle {det_result.confidences[i]:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    save(out, "03_detection_boxes.png", det_viz)

    # save the mask itself
    mask_colored = cv2.applyColorMap(det_result.vehicle_mask, cv2.COLORMAP_BONE)
    save(out, "03_vehicle_mask.png", label(mask_colored, f"VEHICLE MASK ({mask_pct:.1f}% coverage)"))

    # overlay mask on image
    mask_overlay = before_bgr.copy()
    red_layer = np.zeros_like(mask_overlay)
    red_layer[:, :, 2] = det_result.vehicle_mask  # red channel
    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, red_layer, 0.3, 0)
    save(out, "03_mask_overlay.png", label(mask_overlay, "VEHICLE MASK OVERLAY"))

    # ════════════════════════════════════════════════════════════
    # STAGE 4: Diff WITHOUT mask (to show why it's bad)
    # ════════════════════════════════════════════════════════════
    print(f"\n[4/6] DIFF — WITHOUT VEHICLE MASK (the bad way)")
    diff_engine = DiffEngine({
        "threshold": 30,
        "morph_kernel": 5,
        "morph_iterations": 2,
        "min_contour_area": 300,
        "max_contour_area": 100000,
    })

    result_unmasked = diff_engine.compare(before_gray, align_result.warped_after, vehicle_mask=None)
    total_px = before_gray.size
    um_pct = result_unmasked.damage_area_px / total_px * 100
    print(f"  regions: {len(result_unmasked.contours)}")
    print(f"  'damage' area: {result_unmasked.damage_area_px:,}px ({um_pct:.1f}%)")
    print(f"  ^ this is mostly noise from background, borders, lighting")

    # visualize
    unmasked_viz = warped_bgr.copy()
    cv2.drawContours(unmasked_viz, result_unmasked.contours, -1, (0, 0, 255), 2)
    for bbox in result_unmasked.bounding_boxes:
        x, y, bw, bh = bbox
        cv2.rectangle(unmasked_viz, (x, y), (x + bw, y + bh), (0, 255, 255), 1)
    save(out, "04_diff_NO_mask.png",
         label(unmasked_viz, f"UNMASKED DIFF: {len(result_unmasked.contours)} regions (NOISY)"))

    unmasked_heat = cv2.applyColorMap(result_unmasked.raw_diff, cv2.COLORMAP_JET)
    save(out, "04_heatmap_NO_mask.png", label(unmasked_heat, "UNMASKED DIFF HEATMAP"))

    # ════════════════════════════════════════════════════════════
    # STAGE 5: Diff WITH mask (the correct way)
    # ════════════════════════════════════════════════════════════
    print(f"\n[5/6] DIFF — WITH VEHICLE MASK (the correct way)")
    result_masked = diff_engine.compare(
        before_gray, align_result.warped_after, vehicle_mask=det_result.vehicle_mask
    )
    m_pct = result_masked.damage_area_px / total_px * 100
    print(f"  regions: {len(result_masked.contours)}")
    print(f"  damage area: {result_masked.damage_area_px:,}px ({m_pct:.1f}%)")

    if result_unmasked.damage_area_px > 0:
        reduction = (1 - result_masked.damage_area_px / result_unmasked.damage_area_px) * 100
        print(f"  noise reduction from masking: {reduction:.1f}%")

    # damage overlay on the warped BGR image
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0), (0, 255, 0)]
    masked_viz = warped_bgr.copy()
    fill_layer = masked_viz.copy()

    for i, c in enumerate(result_masked.contours):
        color = colors[i % len(colors)]
        area = cv2.contourArea(c)
        x, y, bw, bh = result_masked.bounding_boxes[i]

        cv2.drawContours(fill_layer, [c], -1, color, -1)
        cv2.drawContours(masked_viz, [c], -1, color, 2)
        cv2.rectangle(masked_viz, (x, y), (x + bw, y + bh), color, 1)
        cv2.putText(masked_viz, f"#{i+1} {area:.0f}px",
                    (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        print(f"    region #{i+1}: bbox=({x},{y},{bw}x{bh}) area={area:.0f}px")

    damage_viz = cv2.addWeighted(masked_viz, 0.6, fill_layer, 0.4, 0)
    # re-draw outlines crisp on top
    for i, c in enumerate(result_masked.contours):
        cv2.drawContours(damage_viz, [c], -1, colors[i % len(colors)], 2)
    save(out, "05_damage_MASKED.png",
         label(damage_viz, f"MASKED DIFF: {len(result_masked.contours)} damage regions"))

    masked_heat = cv2.applyColorMap(result_masked.raw_diff, cv2.COLORMAP_JET)
    save(out, "05_heatmap_MASKED.png", label(masked_heat, "MASKED DIFF HEATMAP"))

    # ════════════════════════════════════════════════════════════
    # STAGE 6: Summary comparison
    # ════════════════════════════════════════════════════════════
    print(f"\n[6/6] SUMMARY")

    tile_h, tile_w = 300, 400
    def tile(img, txt):
        t = cv2.resize(to_bgr(img), (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        return label(t, txt, scale=0.45)

    row1 = np.hstack([
        tile(before_bgr, "1. BEFORE"),
        tile(after_bgr, "2. AFTER"),
        tile(warped_bgr, "3. ALIGNED"),
    ])
    row2 = np.hstack([
        tile(mask_overlay, f"4. VEHICLE MASK ({mask_pct:.0f}%)"),
        tile(unmasked_heat, f"5. UNMASKED DIFF ({len(result_unmasked.contours)} regions)"),
        tile(damage_viz, f"6. MASKED DIFF ({len(result_masked.contours)} regions)"),
    ])
    summary = np.vstack([row1, row2])
    save(out, "06_summary.png", summary)

    # ── final report ────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  RESULTS")
    print(f"{'=' * 65}")
    print(f"  alignment:     {args.method.upper()}, {align_result.num_inliers} inliers")
    print(f"  vehicles:      {n_vehicles} detected")
    print(f"  mask coverage: {mask_pct:.1f}%")
    print(f"  unmasked diff: {len(result_unmasked.contours)} regions, {result_unmasked.damage_area_px:,}px")
    print(f"  masked diff:   {len(result_masked.contours)} regions, {result_masked.damage_area_px:,}px")
    if result_unmasked.damage_area_px > 0:
        print(f"  masking removed {reduction:.0f}% of false positives")
    print(f"\n  outputs saved to: {out.resolve()}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
