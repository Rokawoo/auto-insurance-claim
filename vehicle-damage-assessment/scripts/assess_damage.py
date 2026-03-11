#!/usr/bin/env python3
"""Assess vehicle damage from before/after images.

This is the main user-facing script.  It runs the full pipeline and
prints a human-readable damage report to the terminal, saves annotated
images and a JSON report to disk.

Usage
-----
    # single pair
    python scripts/assess_damage.py \\
        --before data/raw/before/car_001.jpg \\
        --after  data/raw/after/car_001.jpg

    # with custom config and output directory
    python scripts/assess_damage.py \\
        --before before.jpg --after after.jpg \\
        --config configs/default.yaml \\
        --output results/

    # batch mode (CSV with columns: before_path, after_path)
    python scripts/assess_damage.py \\
        --batch data/pairs.csv \\
        --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.pipeline import DamagePipeline
from src.pipeline.damage_pipeline import PipelineResult


# ── ANSI colors for terminal output ─────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"

SEVERITY_COLORS = {
    "none": C.GREEN,
    "minor": C.CYAN,
    "moderate": C.YELLOW,
    "severe": C.RED,
    "critical": C.MAGENTA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle damage assessment from before/after images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--before", type=str, help="Path to before (undamaged) image")
    parser.add_argument("--after", type=str, help="Path to after (damaged) image")
    parser.add_argument("--batch", type=str, help="CSV file with before_path,after_path columns")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Pipeline config YAML (default: configs/default.yaml)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output files, just print report")
    parser.add_argument("--json", action="store_true",
                        help="Print raw JSON report instead of formatted text")

    args = parser.parse_args()

    if args.batch is None and (args.before is None or args.after is None):
        parser.error("Provide --before and --after, or --batch")

    return args


def print_report(result: PipelineResult) -> None:
    """Print a formatted damage report to the terminal."""
    report = result.report
    summary = report["summary"]
    meta = report["metadata"]

    sev = summary["overall_severity"]
    sev_color = SEVERITY_COLORS.get(sev, C.RESET)

    print()
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}  VEHICLE DAMAGE ASSESSMENT REPORT{C.RESET}")
    print(f"{'═' * 60}")
    print()

    # metadata
    print(f"  {C.GRAY}Before:{C.RESET}  {meta['before_image']}")
    print(f"  {C.GRAY}After:{C.RESET}   {meta['after_image']}")
    print(f"  {C.GRAY}Image:{C.RESET}   {meta['image_dimensions']['width']}×{meta['image_dimensions']['height']}px")

    if not meta.get("alignment_reliable", True):
        print(f"\n  {C.YELLOW}⚠  Alignment quality below threshold — results may be noisy{C.RESET}")

    # overall severity
    print()
    print(f"  {C.BOLD}Overall Severity:{C.RESET}  {sev_color}{sev.upper()}{C.RESET}"
          f"  ({summary['overall_severity_score']:.2f})")
    print(f"  {C.BOLD}Damage Regions:{C.RESET}    {summary['num_damage_regions']}")
    print(f"  {C.BOLD}Total Damage Area:{C.RESET} {summary['total_damage_area_pct']:.1f}% of vehicle")

    # type breakdown
    if summary["damage_type_counts"]:
        print()
        print(f"  {C.BOLD}Damage Type Breakdown:{C.RESET}")
        for dtype, count in sorted(summary["damage_type_counts"].items()):
            print(f"    • {dtype}: {count}")

    # per-region details
    regions = report.get("regions", [])
    if regions:
        print()
        print(f"  {C.BOLD}{'─' * 56}{C.RESET}")
        print(f"  {C.BOLD}Region Details (sorted by severity):{C.RESET}")
        print(f"  {C.BOLD}{'─' * 56}{C.RESET}")

        for r in regions:
            rsev = r["severity_score"]
            if rsev >= 0.55:
                rc = C.RED
            elif rsev >= 0.30:
                rc = C.YELLOW
            else:
                rc = C.CYAN

            bbox = r["bounding_box"]
            print(f"\n  {C.BOLD}Region #{r['region_id']}{C.RESET}")
            print(f"    Type:       {r['damage_type']}")
            print(f"    Severity:   {rc}{rsev:.3f}{C.RESET}")
            print(f"    Area:       {r['area_px']}px ({r['area_pct_of_vehicle']:.2f}% of vehicle)")
            print(f"    Location:   ({bbox['x']}, {bbox['y']}) {bbox['width']}×{bbox['height']}")
            print(f"    Intensity:  {r['mean_diff_intensity']:.1f}/255")
            geom = r["geometry"]
            print(f"    Geometry:   circ={geom['circularity']:.2f}  "
                  f"aspect={geom['aspect_ratio']:.1f}  "
                  f"solid={geom['solidity']:.2f}")

    # warnings
    warnings = report.get("warnings", [])
    if warnings:
        print()
        for w in warnings:
            print(f"  {C.YELLOW}⚠  {w}{C.RESET}")

    print()
    print(f"{'═' * 60}")
    print()


def run_single(args: argparse.Namespace, config: dict) -> None:
    """Run pipeline on a single pair and display results."""
    pipeline = DamagePipeline(config)

    output_dir = None if args.no_save else args.output
    result = pipeline.run(args.before, args.after, output_dir=output_dir)

    if args.json:
        print(json.dumps(result.report, indent=2))
    else:
        print_report(result)

    if not args.no_save:
        print(f"  {C.GREEN}✓ Outputs saved to {args.output}/{C.RESET}\n")


def run_batch(args: argparse.Namespace, config: dict) -> None:
    """Run pipeline on all pairs from a CSV."""
    import pandas as pd

    df = pd.read_csv(args.batch)
    if "before_path" not in df.columns or "after_path" not in df.columns:
        print(f"{C.RED}ERROR: CSV must have 'before_path' and 'after_path' columns{C.RESET}")
        sys.exit(1)

    pairs = list(zip(df["before_path"], df["after_path"]))
    print(f"\n  Processing {len(pairs)} image pairs...\n")

    pipeline = DamagePipeline(config)
    output_dir = None if args.no_save else args.output
    results = pipeline.run_batch(pairs, output_dir=output_dir)

    # print summary table
    print(f"\n{'═' * 60}")
    print(f"  BATCH SUMMARY: {len(results)}/{len(pairs)} processed successfully")
    print(f"{'═' * 60}\n")

    for result in results:
        sev = result.report["summary"]["overall_severity"]
        score = result.report["summary"]["overall_severity_score"]
        n_regions = result.report["summary"]["num_damage_regions"]
        color = SEVERITY_COLORS.get(sev, C.RESET)
        before = Path(result.report["metadata"]["before_image"]).name
        print(f"  {before:30s}  {color}{sev:10s}{C.RESET}  "
              f"score={score:.2f}  regions={n_regions}")

    if not args.no_save:
        print(f"\n  {C.GREEN}✓ All outputs saved to {args.output}/{C.RESET}\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.batch:
        run_batch(args, config)
    else:
        run_single(args, config)


if __name__ == "__main__":
    main()
