#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from avinor.pipeline import pipeline
from avinor.paths import ARTIFACTS_DIR


def main() -> None:
    artifacts = pipeline(n_simulations=120, random_state=123)
    metrics = artifacts.get("metrics", {})
    summary_keys = [
        key for key in metrics if key.endswith("_overall_auc") or key.endswith("_july")
    ]
    if summary_keys:
        print("Key training metrics:")
        for key in sorted(summary_keys):
            value = metrics[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    if "ensemble_overall_auc" in metrics:
        overall = metrics["ensemble_overall_auc"]
        july_val = metrics.get("ensemble_july_auc")
        if isinstance(overall, (int, float)) and not isinstance(overall, bool):
            if isinstance(july_val, (int, float)) and not isinstance(july_val, bool):
                print(
                    "Final Ensemble AUCs: "
                    f"overall={overall:.4f}, july={july_val:.4f}"
                )
            else:
                print(f"Final Ensemble overall AUC: {overall:.4f}")
    if "catboost_overall_auc" in metrics:
        overall = metrics["catboost_overall_auc"]
        july_val = metrics.get("catboost_auc_july")
        if isinstance(overall, (int, float)) and not isinstance(overall, bool):
            if isinstance(july_val, (int, float)) and not isinstance(july_val, bool):
                print(
                    "Final CatBoost AUCs: "
                    f"overall={overall:.4f}, july={july_val:.4f}"
                )
            else:
                print(f"Final CatBoost overall AUC: {overall:.4f}")
    if "xgb_overall_auc" in metrics:
        overall = metrics["xgb_overall_auc"]
        july_val = metrics.get("xgb_auc_july")
        if isinstance(overall, (int, float)) and not isinstance(overall, bool):
            if isinstance(july_val, (int, float)) and not isinstance(july_val, bool):
                print(
                    "Final XGBoost AUCs: "
                    f"overall={overall:.4f}, july={july_val:.4f}"
                )
            else:
                print(f"Final XGBoost overall AUC: {overall:.4f}")
    predictions = artifacts["predictions"].copy()
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    output_path = ARTIFACTS_DIR / "final_predictions.csv"
    predictions.to_csv(output_path, index=False, float_format="%.3f")
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
