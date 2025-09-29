#!/usr/bin/env python3
from __future__ import annotations

from avinor.data_loader import load_inference_template, load_training_data
from avinor.paths import FEATURE_CACHE_DIR, MODELS_DIR
from avinor.pipeline import prepare_training_features, train_logistic_with_cv

import joblib
import numpy as np
import pandas as pd

START_DATE = "2022-01-01"


def main() -> None:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    training = load_training_data()
    training = training[training["date"] >= START_DATE].reset_index(drop=True)

    features = training.drop(columns=["target"])
    target = training["target"]

    x_df, y, feature_builder = prepare_training_features(features, target)
    model, oof_preds, metrics = train_logistic_with_cv(x_df, y, training["date"], random_state=42)

    feature_columns = list(x_df.columns)
    joblib.dump(model, MODELS_DIR / "logistic_model.joblib")

    inference = load_inference_template()
    inference_features = feature_builder.transform(inference)
    inference_matrix = inference_features.to_numpy(dtype=np.float32)
    logistic_preds = model.predict_proba(inference_matrix)[:, -1]

    np.save(FEATURE_CACHE_DIR / "logistic_oof.npy", oof_preds.astype(np.float32))
    np.save(FEATURE_CACHE_DIR / "logistic_preds_inference.npy", logistic_preds.astype(np.float32))
    pd.Series({
        "columns": feature_columns,
        "start_date": START_DATE,
        "estimator": "sklearn_logistic_pipeline",
    }).to_json(FEATURE_CACHE_DIR / "logistic_features.json", indent=2)
    pd.Series(metrics).to_json(FEATURE_CACHE_DIR / "logistic_metrics.json", indent=2)
    training.to_csv(FEATURE_CACHE_DIR / "training_filtered.csv", index=False)
    print("Logistic training completed")


if __name__ == "__main__":
    main()
