#!/usr/bin/env python3
"""Generate dataset summary and sequence-model interpretability artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from avinor.data_loader import (
    load_historical_flights,
    load_training_data,
)
from avinor.external_features import merge_external_features
from avinor.feature_engineering import (
    build_base_index,
    build_event_table,
    build_feature_matrix,
    compute_recent_target_features,
)
from avinor.paths import (
    EXTERNAL_PROCESSED_DIR,
    EXTERNAL_RAW_DIR,
    FEATURE_CACHE_DIR,
    MODELS_DIR,
    SCHOOL_HOLIDAYS_PATH,
)
from avinor.pipeline import (
    assemble_design_frame,
    apply_target_encodings,
    compute_temporal_target_encodings,
    prepare_training_features,
)
from avinor.metrics import roc_auc_score
from avinor.sequence_pipeline import (
    MONTH_FEATURE_COLUMNS,
    SequenceModel,
    _compute_monthly_target_stats,
    _merge_monthly_stats,
    build_minute_sequences,
)


ANALYSIS_DIR = FEATURE_CACHE_DIR.parent / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = ANALYSIS_DIR / "model_analysis.json"
SAMPLE_SIZE = 80000
PERMUTATION_REPEATS = 3
BATCH_SIZE = 2048
RANDOM_STATE = 42


@dataclass
class SequenceAssets:
    training_df: pd.DataFrame
    training_matrix: np.ndarray
    training_columns: List[str]
    seq_tensor: np.ndarray  # shape (n_samples, channels, seq_len)
    targets: np.ndarray


def summarize_dataset(training: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    summary["num_rows"] = int(len(training))
    summary["num_airport_groups"] = int(training["airport_group"].nunique())
    summary["date_range"] = {
        "min": training["date"].min().strftime("%Y-%m-%d"),
        "max": training["date"].max().strftime("%Y-%m-%d"),
    }
    summary["target_rate"] = float(training["target"].mean())
    by_group = (
        training.groupby("airport_group")["target"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "target_rate", "count": "rows"})
        .sort_values("target_rate", ascending=False)
    )
    summary["airport_group_breakdown"] = by_group.reset_index().to_dict(orient="records")
    summary["monthly_target_rate"] = _training_monthly_rate(training)
    return summary


def _training_monthly_rate(training: pd.DataFrame) -> Dict[str, float]:
    monthly = (
        training.assign(month=training["date"].dt.to_period("M"))
        .groupby("month")["target"]
        .mean()
        .sort_index()
    )
    return {str(idx): float(val) for idx, val in monthly.items()}


def load_simulation_features(name: str) -> pd.DataFrame:
    parquet_path = FEATURE_CACHE_DIR / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Expected cached simulation features at {parquet_path}")


def prepare_sequence_assets() -> SequenceAssets:
    training = load_training_data()
    flights = load_historical_flights()

    base_training = build_base_index(training)
    schedule_events_all = build_event_table(flights, use_actual=True)
    schedule_features_training = build_feature_matrix(
        base_training,
        schedule_events_all,
        training_df=training,
    )

    holiday_path = EXTERNAL_PROCESSED_DIR / "calendar_public_holidays.csv"
    weather_history_path = EXTERNAL_RAW_DIR / "weather_history.parquet"
    external_training = merge_external_features(
        base_training,
        weather_path=weather_history_path if weather_history_path.exists() else None,
        holiday_path=holiday_path if holiday_path.exists() else None,
        school_holiday_path=SCHOOL_HOLIDAYS_PATH if SCHOOL_HOLIDAYS_PATH.exists() else None,
    )

    training_design = assemble_design_frame(
        original=training.drop(columns=["target"]),
        base=base_training,
        schedule_features=schedule_features_training,
        external_features=external_training,
    )
    training_design = training_design.merge(
        training[["airport_group", "date", "hour", "target"]],
        on=["airport_group", "date", "hour"],
        how="left",
    )

    recent_training_features, recent_defaults = compute_recent_target_features(training)
    recent_feature_cols = [
        col
        for col in recent_training_features.columns
        if col not in {"airport_group", "date", "hour"}
    ]
    if recent_feature_cols:
        training_design = training_design.merge(
            recent_training_features,
            on=["airport_group", "date", "hour"],
            how="left",
        )
        for col in recent_feature_cols:
            if col in training_design:
                values = training_design[col].astype(np.float32)
                if np.isnan(values).any():
                    mean_val = float(np.nanmean(values))
                    if np.isnan(mean_val):
                        mean_val = float(training["target"].mean())
                    values = np.where(np.isnan(values), mean_val, values)
                training_design[col] = values
    else:
        recent_defaults = pd.DataFrame()

    sim_features_training = load_simulation_features("sim_features_training")
    training_design = training_design.merge(
        sim_features_training,
        on=["airport_group", "date", "hour"],
        how="left",
    )

    month_stats, month_global = _compute_monthly_target_stats(training)
    if not month_stats.empty:
        training_design = _merge_monthly_stats(training_design, month_stats, month_global)

    target_enc_columns = [col for col in ("airport_group", "feat_season") if col in training.columns]
    enc_input = training[target_enc_columns].reset_index(drop=True)
    target_enc_df, _, _ = compute_temporal_target_encodings(
        enc_input,
        training["target"].reset_index(drop=True),
        training["date"].reset_index(drop=True),
        target_enc_columns,
    )
    training_design = pd.concat([training_design.reset_index(drop=True), target_enc_df], axis=1)

    x_df, y, feature_builder = prepare_training_features(
        training_design.drop(columns=["target"]),
        training_design["target"].astype(np.float32),
    )
    training_matrix = x_df.to_numpy(dtype=np.float32)
    columns = list(x_df.columns)

    # Minute sequences
    training_events = build_event_table(flights, use_actual=True)
    sequences, dep_sequences, arr_sequences = build_minute_sequences(base_training, training_events)
    seq_train = np.stack([sequences, dep_sequences, arr_sequences], axis=1)

    return SequenceAssets(
        training_df=training,
        training_matrix=training_matrix,
        training_columns=columns,
        seq_tensor=seq_train,
        targets=training["target"].to_numpy(dtype=np.float32),
    )


def load_sequence_model(static_dim: int, channels: int) -> SequenceModel:
    model_path = MODELS_DIR / "sequence_model.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    model = SequenceModel(seq_channels=channels, static_dim=static_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_probabilities(
    model: SequenceModel,
    seq_array: np.ndarray,
    static_array: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.to(device)
    n_rows = seq_array.shape[0]
    preds: List[np.ndarray] = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        seq_batch = torch.from_numpy(seq_array[start:end]).float().to(device)
        static_batch = torch.from_numpy(static_array[start:end]).float().to(device)
        with torch.no_grad():
            logits = model(seq_batch, static_batch)
            batch_preds = torch.sigmoid(logits).cpu().numpy()
        preds.append(batch_preds)
    return np.concatenate(preds)


def logistic_top_features() -> List[Dict[str, float]]:
    pipeline = joblib.load(MODELS_DIR / "logistic_model.joblib")
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    feature_meta_path = FEATURE_CACHE_DIR / "logistic_features.json"
    columns = json.loads(feature_meta_path.read_text())["columns"]
    coefs = model.coef_[0]
    if scaler.with_std:
        scaled = coefs / scaler.scale_
    else:
        scaled = coefs
    items = [
        {
            "feature": name,
            "coefficient": float(coef),
            "abs_coefficient": float(abs(coef)),
        }
        for name, coef in zip(columns, scaled)
    ]
    items.sort(key=lambda x: x["abs_coefficient"], reverse=True)
    return items[:25]


def select_feature_indices(columns: List[str], top_names: Iterable[str]) -> List[int]:
    name_to_index = {name: idx for idx, name in enumerate(columns)}
    indices = [name_to_index[name] for name in top_names if name in name_to_index]
    return indices


def permutation_importance(
    model: SequenceModel,
    seq_sample: np.ndarray,
    static_sample: np.ndarray,
    targets: np.ndarray,
    feature_columns: List[str],
    feature_indices: List[int],
    baseline_auc: float,
    device: torch.device,
    repeats: int = PERMUTATION_REPEATS,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(RANDOM_STATE)
    importances: List[Dict[str, float]] = []
    n_sample = len(targets)
    for idx in feature_indices:
        drops: List[float] = []
        for _ in range(repeats):
            permuted = static_sample.copy()
            permuted[:, idx] = rng.permutation(permuted[:, idx])
            preds = predict_probabilities(model, seq_sample, permuted, device)
            auc = roc_auc_score(targets, preds)
            drops.append(baseline_auc - auc)
        importances.append(
            {
                "feature": feature_columns[idx],
                "auc_drop": float(np.mean(drops)),
                "std_drop": float(np.std(drops)),
            }
        )
    importances.sort(key=lambda x: x["auc_drop"], reverse=True)
    return importances


def channel_ablation(
    model: SequenceModel,
    seq_sample: np.ndarray,
    static_sample: np.ndarray,
    targets: np.ndarray,
    baseline_auc: float,
    device: torch.device,
) -> List[Dict[str, float]]:
    channel_names = ["concurrency", "departures", "arrivals"]
    results: List[Dict[str, float]] = []
    for channel, name in enumerate(channel_names):
        modified = seq_sample.copy()
        modified[:, channel, :] = 0.0
        preds = predict_probabilities(model, modified, static_sample, device)
        auc = roc_auc_score(targets, preds)
        results.append(
            {
                "channel": name,
                "auc_drop": float(baseline_auc - auc),
                "auc": float(auc),
            }
        )
    results.sort(key=lambda x: x["auc_drop"], reverse=True)
    return results


def concurrency_profile(
    seq_sample: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
) -> List[Dict[str, float]]:
    max_concurrency = seq_sample[:, 0, :].max(axis=1)
    bins = np.clip(max_concurrency.astype(int), 0, 12)
    df = pd.DataFrame(
        {
            "max_concurrency": bins,
            "prediction": preds,
            "target": targets,
        }
    )
    grouped = df.groupby("max_concurrency").agg(
        mean_pred=("prediction", "mean"),
        mean_target=("target", "mean"),
        count=("prediction", "size"),
    )
    grouped = grouped[grouped["count"] >= 50]
    records = []
    for _, row in grouped.reset_index().sort_values("max_concurrency").iterrows():
        records.append(
            {
                "max_concurrency": int(row["max_concurrency"]),
                "mean_prediction": float(row["mean_pred"]),
                "mean_target_rate": float(row["mean_target"]),
                "count": int(row["count"]),
            }
        )
    return records


def sample_subset(
    seq_array: np.ndarray,
    static_array: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = len(targets)
    if n_rows <= SAMPLE_SIZE:
        indices = np.arange(n_rows)
    else:
        indices = rng.choice(n_rows, size=SAMPLE_SIZE, replace=False)
    indices.sort()
    return seq_array[indices], static_array[indices], targets[indices]


def main() -> None:
    training = load_training_data()
    dataset_summary = summarize_dataset(training)

    assets = prepare_sequence_assets()
    model = load_sequence_model(
        static_dim=assets.training_matrix.shape[1],
        channels=assets.seq_tensor.shape[1],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_preds = predict_probabilities(
        model,
        assets.seq_tensor,
        assets.training_matrix,
        device,
    )
    baseline_auc = roc_auc_score(assets.targets, baseline_preds)

    seq_sample, static_sample, target_sample = sample_subset(
        assets.seq_tensor,
        assets.training_matrix,
        assets.targets,
    )
    sample_preds = predict_probabilities(model, seq_sample, static_sample, device)
    sample_auc = roc_auc_score(target_sample, sample_preds)

    top_logistic = logistic_top_features()
    top_feature_names = [item["feature"] for item in top_logistic[:15]]
    feature_indices = select_feature_indices(assets.training_columns, top_feature_names)
    feature_importance = permutation_importance(
        model,
        seq_sample,
        static_sample,
        target_sample,
        assets.training_columns,
        feature_indices,
        sample_auc,
        device,
    )

    channel_importance = channel_ablation(
        model,
        seq_sample,
        static_sample,
        target_sample,
        sample_auc,
        device,
    )

    concurrency_stats = concurrency_profile(seq_sample, sample_preds, target_sample)

    payload = {
        "dataset": dataset_summary,
        "sequence_model": {
            "baseline_auc_full": float(baseline_auc),
            "baseline_auc_sample": float(sample_auc),
            "static_feature_importance": feature_importance,
            "channel_importance": channel_importance,
            "concurrency_profile": concurrency_stats,
        },
        "logistic_reference": top_logistic,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
