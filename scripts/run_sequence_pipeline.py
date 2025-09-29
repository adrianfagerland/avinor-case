#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from avinor.sequence_pipeline import sequence_pipeline
from avinor.paths import ARTIFACTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Avinor sequence pipeline")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG). Defaults to INFO.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Launching sequence pipeline")
    result = sequence_pipeline(
        n_simulations=120,
        random_state=123,
        show_progress=not args.no_progress,
    )
    metrics = result.metrics
    summary_keys = [
        key for key in metrics if key.endswith("_overall_auc") or key.endswith("_july")
    ]
    if summary_keys:
        print("Key sequence metrics:")
        for key in sorted(summary_keys):
            value = metrics[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    if "sequence_overall_auc" in metrics:
        overall = metrics["sequence_overall_auc"]
        print(f"Sequence model overall AUC: {overall:.4f}")
    preds = result.predictions.copy()
    preds["date"] = preds["date"].dt.strftime("%Y-%m-%d")
    output_path = ARTIFACTS_DIR / "final_predictions_sequence.csv"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False, float_format="%.3f")
    print(f"Saved sequence model predictions to {output_path}")


if __name__ == "__main__":
    main()
