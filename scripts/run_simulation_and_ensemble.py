#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from avinor.data_loader import (
    load_historical_flights,
    load_inference_template,
    load_schedule_october,
    load_training_data,
)
from avinor.feature_engineering import build_base_index
from avinor.metrics import roc_auc_score
from avinor.models.delay_cancellation import CancellationEstimator, DelaySampler
from avinor.paths import ARTIFACTS_DIR, FEATURE_CACHE_DIR
from avinor.pipeline import (
    determine_ensemble_weight,
    run_simulation_for_period,
)
from avinor.simulation import prepare_schedule

START_DATE = "2022-01-01"


def load_filtered_training() -> pd.DataFrame:
    cached = FEATURE_CACHE_DIR / "training_filtered.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["date"])
    training = load_training_data()
    return training[training["date"] >= START_DATE].reset_index(drop=True)


def main() -> None:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logistic_preds_path = FEATURE_CACHE_DIR / "logistic_preds_inference.npy"
    logistic_oof_path = FEATURE_CACHE_DIR / "logistic_oof.npy"
    if not logistic_preds_path.exists() or not logistic_oof_path.exists():
        raise FileNotFoundError("Logistic predictions not found. Run scripts/train_logistic.py first.")

    logistic_preds = np.load(logistic_preds_path)
    logistic_oof = np.load(logistic_oof_path)

    training = load_filtered_training()
    inference = load_inference_template()
    flights = load_historical_flights()
    schedule_october = prepare_schedule(load_schedule_october())

    delay_sampler = DelaySampler()
    delay_sampler.fit(flights)
    cancellation_model = CancellationEstimator()
    cancellation_model.fit(flights)

    base_inference = build_base_index(inference)
    sim_features_oct = run_simulation_for_period(
        schedule_october,
        base_inference,
        delay_sampler,
        cancellation_model,
        n_simulations=80,
        random_state=123,
    )

    july_mask = (training["date"] >= "2025-07-01") & (training["date"] <= "2025-07-31")
    training_july = training[july_mask].copy()
    base_july = build_base_index(training_july)

    mask_std = (flights["std"] >= "2025-07-01") & (flights["std"] <= "2025-07-31 23:59:59")
    mask_sta = (flights["sta"] >= "2025-07-01") & (flights["sta"] <= "2025-07-31 23:59:59")
    flights_july = flights[mask_std | mask_sta]
    schedule_july = prepare_schedule(
        flights_july[[
            "flight_id",
            "dep_airport",
            "arr_airport",
            "service_type",
            "std",
            "sta",
            "dep_airport_group",
            "arr_airport_group",
        ]]
    )
    schedule_july["cancelled"] = 0

    sim_features_july = run_simulation_for_period(
        schedule_july,
        base_july,
        delay_sampler,
        cancellation_model,
        n_simulations=30,
        random_state=456,
    )

    sim_features_july = training_july[["airport_group", "date", "hour"]].merge(
        sim_features_july,
        on=["airport_group", "date", "hour"],
        how="left",
    )
    sim_mean = sim_features_july["sim_prob_any_overlap"].mean()
    july_sim_preds = sim_features_july["sim_prob_any_overlap"].fillna(
        sim_mean if not np.isnan(sim_mean) else 0.2
    ).to_numpy(dtype=np.float32)
    july_targets = training_july["target"].astype(np.float32)
    july_logistic_preds = logistic_oof[july_mask.to_numpy()]

    ensemble_weight = determine_ensemble_weight(july_targets, july_logistic_preds, july_sim_preds)
    sim_prob_oct = sim_features_oct["sim_prob_any_overlap"].to_numpy(dtype=np.float32)
    final_preds = ensemble_weight * logistic_preds + (1 - ensemble_weight) * sim_prob_oct
    final_preds = np.clip(final_preds, 0.0001, 0.9999)

    predictions = inference[["airport_group", "date", "hour"]].copy()
    predictions["pred"] = final_preds
    predictions.sort_values(["date", "hour", "airport_group"], inplace=True)
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")

    output_path = ARTIFACTS_DIR / "final_predictions.csv"
    predictions.to_csv(output_path, index=False)

    metrics = {
        "ensemble_weight": ensemble_weight,
        "logistic_auc_july": roc_auc_score(july_targets.to_numpy(dtype=np.float32), july_logistic_preds),
        "simulation_auc_july": roc_auc_score(july_targets.to_numpy(dtype=np.float32), july_sim_preds),
    }
    (FEATURE_CACHE_DIR / "simulation_metrics.json").write_text(json.dumps(metrics, indent=2))
    sim_features_oct.to_csv(FEATURE_CACHE_DIR / "sim_features_october.csv", index=False)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
