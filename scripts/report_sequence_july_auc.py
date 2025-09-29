#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Tuple

import numpy as np

from avinor.data_loader import load_training_data
from avinor.metrics import roc_auc_score
from avinor.paths import FEATURE_CACHE_DIR

OOF_PATH = FEATURE_CACHE_DIR / "sequence_oof.npy"
METRICS_PATH = FEATURE_CACHE_DIR / "sequence_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the sequence model AUC for a given month without rerunning simulations unless needed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level passed to the pipeline when refresh is required (default: INFO).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars if the pipeline must run.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=120,
        help=(
            "Number of Monte Carlo simulations when refreshing artifacts (default: 120)."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed for the sequence pipeline when refreshing (default: 123).",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=7,
        help="Calendar month to evaluate (default: 7 for July).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Calendar year to evaluate (default: 2025).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a full sequence pipeline run even if cached artifacts exist.",
    )
    return parser.parse_args()


def run_sequence_pipeline_with_logging(args: argparse.Namespace, logger: logging.Logger) -> Tuple[np.ndarray, Dict[str, float]]:
    from avinor.sequence_pipeline import sequence_pipeline

    logger.info("Running sequence pipeline with %d simulations", args.n_simulations)
    result = sequence_pipeline(
        n_simulations=args.n_simulations,
        random_state=args.random_state,
        show_progress=not args.no_progress,
        logger=logger,
    )
    return result.oof_predictions, result.metrics


def load_cached_artifacts() -> Tuple[np.ndarray, Dict[str, float]]:
    if not OOF_PATH.exists():
        raise FileNotFoundError(OOF_PATH)
    oof = np.load(OOF_PATH)
    metrics: Dict[str, float] = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open() as fh:
            metrics = json.load(fh)
    return oof, metrics


def compute_month_auc(
    *,
    oof: np.ndarray,
    month: int,
    year: int,
) -> Tuple[float, int]:
    training = load_training_data()
    mask = (training["date"].dt.year == year) & (training["date"].dt.month == month)
    if mask.sum() == 0:
        raise ValueError(f"No samples found for {year}-{month:02d} in the training data.")

    preds = oof[mask.to_numpy()]
    targets = training.loc[mask, "target"].to_numpy(dtype=np.float32)
    valid_mask = ~np.isnan(preds)
    preds = preds[valid_mask]
    targets = targets[valid_mask]

    if preds.size == 0:
        raise ValueError(
            "Out-of-fold predictions for the requested period are missing. "
            "Run the sequence pipeline first (e.g., via --refresh)."
        )

    auc = roc_auc_score(targets, preds)
    support = int(targets.size)
    return auc, support


def print_metrics(metrics: Dict[str, float]) -> None:
    if not metrics:
        print("No cached sequence metrics were found.")
        return
    print("Key sequence metrics:")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()

    log_level_value = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level_value,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    if args.refresh:
        oof, metrics = run_sequence_pipeline_with_logging(args, logger)
    else:
        try:
            oof, metrics = load_cached_artifacts()
            logger.info("Loaded cached sequence artifacts from %s", FEATURE_CACHE_DIR)
        except FileNotFoundError:
            logger.error(
                "Cached sequence artifacts not found. Run scripts/run_sequence_pipeline.py "
                "first or rerun this command with --refresh."
            )
            return

    print_metrics(metrics)

    try:
        month_auc, support = compute_month_auc(
            oof=oof,
            month=args.month,
            year=args.year,
        )
    except ValueError as exc:
        print(str(exc))
        return

    label = f"Sequence model AUC {args.year}-{args.month:02d}"
    print(f"{label}: {month_auc:.4f} (n={support})")


if __name__ == "__main__":
    main()
