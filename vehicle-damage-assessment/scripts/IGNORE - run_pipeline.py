#!/usr/bin/env python3
"""Run the damage assessment pipeline on a before/after image pair.

Usage
-----
    python scripts/run_pipeline.py \\
        --before data/raw/before/car_001.jpg \\
        --after  data/raw/after/car_001.jpg \\
        --config configs/default.yaml \\
        --output outputs/

    # batch mode (CSV file with columns: before_path, after_path)
    python scripts/run_pipeline.py \\
        --batch data/pairs.csv \\
        --config configs/default.yaml \\
        --output outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# add project root to path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.pipeline import DamagePipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed args with before, after, config, output, and batch fields.
    """
    # TODO:
    #   --before (str, required unless --batch)
    #   --after  (str, required unless --batch)
    #   --config (str, default="configs/default.yaml")
    #   --output (str, default="outputs/")
    #   --batch  (str, optional CSV path for batch mode)
    #   --verbose (flag)
    raise NotImplementedError


def run_single(args: argparse.Namespace, config: dict) -> None:
    """Run the pipeline on a single before/after pair.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments (must have .before and .after).
    config : dict
        Loaded pipeline config.
    """
    # TODO:
    #   1. create DamagePipeline(config)
    #   2. pipeline.run(args.before, args.after)
    #   3. print summary from result.report
    #   4. save outputs
    raise NotImplementedError


def run_batch(args: argparse.Namespace, config: dict) -> None:
    """Run the pipeline on all pairs listed in a CSV file.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments (must have .batch pointing to a CSV).
    config : dict
        Loaded pipeline config.
    """
    # TODO:
    #   1. read CSV with pandas (columns: before_path, after_path)
    #   2. create DamagePipeline(config)
    #   3. pipeline.run_batch(pairs, output_dir=args.output)
    #   4. print summary statistics
    raise NotImplementedError


def main() -> None:
    """Entry point."""
    # TODO:
    #   1. parse args
    #   2. load config
    #   3. dispatch to run_single or run_batch
    raise NotImplementedError


if __name__ == "__main__":
    main()
