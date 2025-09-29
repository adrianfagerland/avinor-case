from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.neural_network import MLPClassifier as SkMLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

from .data_loader import (
    load_historical_flights,
    load_inference_template,
    load_schedule_october,
    load_training_data,
)
from .external_features import merge_external_features
from .feature_engineering import (
    build_base_index,
    build_event_table,
    build_feature_matrix,
    compute_recent_target_features,
)
from .metrics import roc_auc_score
from .models.delay_cancellation import CancellationEstimator, DelaySampler
from .paths import (
    EXTERNAL_PROCESSED_DIR,
    EXTERNAL_RAW_DIR,
    FEATURE_CACHE_DIR,
    MODELS_DIR,
    SCHOOL_HOLIDAYS_PATH,
)
from .simulation import run_monte_carlo
from .tabular import TabularFeatureBuilder

LGBM_PARAM_GRID: Tuple[Dict[str, Any], ...] = (
    {
        "learning_rate": 0.05,
        "num_leaves": 48,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
    },
    {
        "learning_rate": 0.03,
        "num_leaves": 96,
        "min_child_samples": 35,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.6,
    },
    {
        "learning_rate": 0.08,
        "num_leaves": 64,
        "min_child_samples": 25,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.15,
        "reg_lambda": 0.15,
    },
    {
        "learning_rate": 0.04,
        "num_leaves": 80,
        "min_child_samples": 45,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.4,
    },
)

BEST_LGBM_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.04,
    "num_leaves": 80,
    "min_child_samples": 45,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 0.4,
}


CATBOOST_PARAM_GRID: Tuple[Dict[str, Any], ...] = (
    {
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "bagging_temperature": 0.2,
    },
    {
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "bagging_temperature": 0.3,
    },
    {
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "bagging_temperature": 0.1,
    },
    {
        "learning_rate": 0.06,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.6,
    },
)

BEST_CATBOOST_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.04,
    "depth": 7,
    "l2_leaf_reg": 6.0,
    "bagging_temperature": 0.3,
}

TARGET_ENCODING_CANDIDATES: Tuple[str, ...] = (
    "airport_group",
    "feat_season",
    "airport_hour_bucket",
    "airport_dow_bucket",
)


def _log_progress(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


@contextmanager
def timed_step(name: str) -> Any:
    start = time.perf_counter()
    timestamp = time.strftime("%H:%M:%S")
    _log_progress(f"[{timestamp}] ▶ {name}...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        end_ts = time.strftime("%H:%M:%S")
        _log_progress(f"[{end_ts}] ✓ {name} completed in {elapsed:.1f}s")


def _simulation_cache_paths(name: str) -> tuple[Path, Path]:
    data_path = FEATURE_CACHE_DIR / f"{name}.parquet"
    meta_path = FEATURE_CACHE_DIR / f"{name}_meta.json"
    return data_path, meta_path


def load_cached_simulation(name: str, config: Dict[str, Any]) -> pd.DataFrame | None:
    data_path, meta_path = _simulation_cache_paths(name)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    if meta != config:
        return None
    _log_progress(f"[simulation] Reusing cached {name} results (n={config.get('n_simulations')})")
    return pd.read_parquet(data_path)


def cache_simulation(name: str, df: pd.DataFrame, config: Dict[str, Any]) -> None:
    data_path, meta_path = _simulation_cache_paths(name)
    df.to_parquet(data_path, index=False)
    meta_path.write_text(json.dumps(config, indent=2))


def assemble_design_frame(
    original: pd.DataFrame,
    base: pd.DataFrame,
    schedule_features: pd.DataFrame,
    external_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    components = [
        original.reset_index(drop=True),
        base.drop(columns=[col for col in ["interval_start", "interval_end"] if col in base.columns]).reset_index(drop=True),
        schedule_features.reset_index(drop=True),
    ]
    if external_features is not None and not external_features.empty:
        components.append(external_features.reset_index(drop=True))

    combined = pd.concat(components, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def add_airport_time_buckets(df: pd.DataFrame) -> None:
    required_columns = {"airport_group", "hour", "date"}
    if not required_columns.issubset(df.columns):
        return
    airport_values = df["airport_group"].astype(str)
    hour_values = df["hour"].astype(int).astype(str).str.zfill(2)
    dow_values = df["date"].dt.dayofweek.astype(int).astype(str)
    df["airport_hour_bucket"] = (airport_values + "_h" + hour_values).astype(str)
    df["airport_dow_bucket"] = (airport_values + "_d" + dow_values).astype(str)


def prepare_training_features(
    features: pd.DataFrame,
    target: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series, TabularFeatureBuilder]:
    if target is None:
        if "target" not in features.columns:
            msg = "target series must be provided or column named 'target' must exist"
            raise ValueError(msg)
        target = features["target"]
        features = features.drop(columns=["target"])

    feature_builder = TabularFeatureBuilder()
    x_df = feature_builder.fit_transform(features)
    y = target.astype(np.float32)
    return x_df, y, feature_builder


def compute_temporal_target_encodings(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.Series,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, float]], float]:
    folds = temporal_cv_indices(dates)
    global_mean = float(target.mean())
    encoded = pd.DataFrame(index=features.index)
    mapping: Dict[str, Dict[Any, float]] = {}

    for column in columns:
        if column not in features.columns:
            continue
        col_values = features[column]
        out = pd.Series(global_mean, index=features.index, dtype=np.float32)
        for train_mask, val_mask in folds:
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            train_df = pd.DataFrame(
                {
                    column: col_values.iloc[train_idx].to_numpy(),
                    "target": target.iloc[train_idx].to_numpy(),
                }
            )
            stats = train_df.groupby(column)["target"].mean()
            mapped = col_values.iloc[val_idx].map(stats).fillna(global_mean)
            out.iloc[val_idx] = mapped.astype(np.float32)
        out = out.fillna(global_mean)
        encoded[f"{column}_target_enc"] = out.astype(np.float32)
        overall_stats = (
            pd.DataFrame({column: col_values.to_numpy(), "target": target.to_numpy()})
            .groupby(column)["target"]
            .mean()
            .to_dict()
        )
        mapping[column] = overall_stats
    return encoded, mapping, global_mean


def apply_target_encodings(
    features: pd.DataFrame,
    mapping: Dict[str, Dict[Any, float]],
    global_mean: float,
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=features.index)
    for column, value_map in mapping.items():
        if column not in features.columns:
            continue
        encoded[f"{column}_target_enc"] = (
            features[column].map(value_map).fillna(global_mean).astype(np.float32)
        )
    return encoded


def temporal_cv_indices(dates: pd.Series, n_val_periods: int = 4) -> List[tuple[np.ndarray, np.ndarray]]:
    periods = dates.dt.to_period("M")
    period_strings = periods.astype(str)
    period_values = period_strings.to_numpy()
    unique_periods = period_strings.sort_values().unique()
    folds: List[tuple[np.ndarray, np.ndarray]] = []
    for period in unique_periods[-n_val_periods:]:
        val_mask = period_values == period
        train_mask = period_values < period
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append((train_mask, val_mask))
    return folds


def create_logistic_pipeline(random_state: int = 42) -> SkPipeline:
    return SkPipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                SkLogisticRegression(
                    solver="saga",
                    penalty="l2",
                    C=2.0,
                    max_iter=400,
                    tol=1e-4,
                    random_state=random_state,
                ),
            ),
        ]
    )


def create_mlp_pipeline(random_state: int = 42) -> SkPipeline:
    return SkPipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                SkMLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=0.001,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=10,
                    verbose=False,
                    random_state=random_state,
                ),
            ),
        ]
    )


def create_lgbm_estimator(random_state: int = 42, **overrides: Any) -> LGBMClassifier:
    params: Dict[str, Any] = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 600,
        "num_leaves": 64,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": random_state,
    }
    params.update(overrides)
    return LGBMClassifier(**params)


def create_catboost_estimator(random_state: int = 42, **overrides: Any) -> CatBoostClassifier:
    params: Dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "bagging_temperature": 0.2,
        "border_count": 254,
        "iterations": 1000,
        "od_type": "Iter",
        "od_wait": 40,
        "verbose": False,
        "random_state": random_state,
    }
    params.update(overrides)
    return CatBoostClassifier(**params)


def create_xgb_estimator(random_state: int = 42, **overrides: Any) -> XGBClassifier:
    params: Dict[str, Any] = {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 5.0,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "n_estimators": 900,
        "random_state": random_state,
        "verbosity": 0,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def tune_catboost_with_cv(
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    random_state: int,
    param_grid: Sequence[Dict[str, Any]] = CATBOOST_PARAM_GRID,
) -> Tuple[Any, np.ndarray, Dict[str, float], Dict[str, Any]]:
    best_auc = -np.inf
    best_model: Any | None = None
    best_oof: np.ndarray | None = None
    best_metrics: Dict[str, float] | None = None
    best_params: Dict[str, Any] | None = None
    for idx, params in enumerate(param_grid, start=1):
        _log_progress(
            f"[catboost] Candidate {idx}/{len(param_grid)}: "
            + ", ".join(f"{k}={v}" for k, v in params.items())
        )
        model, oof, metrics = train_estimator_with_cv(
            lambda: create_catboost_estimator(random_state=random_state, **params),
            x_df,
            y,
            dates,
            prefix="catboost",
        )
        overall_auc = metrics.get("catboost_overall_auc", float("nan"))
        if np.isnan(overall_auc):
            continue
        if overall_auc > best_auc:
            best_auc = overall_auc
            best_model = model
            best_oof = oof
            best_metrics = metrics
            best_params = dict(params)
            _log_progress(f"[catboost] New best overall AUC: {overall_auc:.4f}")
    if best_model is None or best_oof is None or best_metrics is None or best_params is None:
        raise RuntimeError("CatBoost hyperparameter search failed to produce a valid model")
    best_metrics["catboost_best_params"] = json.dumps(best_params)
    return best_model, best_oof, best_metrics, best_params


def train_xgb_with_cv(
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    random_state: int = 42,
    overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Any, np.ndarray, Dict[str, float]]:
    estimator_kwargs = overrides or {}
    return train_estimator_with_cv(
        lambda: create_xgb_estimator(random_state=random_state, **estimator_kwargs),
        x_df,
        y,
        dates,
        prefix="xgb",
    )


def tune_lgbm_with_cv(
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    random_state: int,
    param_grid: Sequence[Dict[str, Any]] = LGBM_PARAM_GRID,
) -> Tuple[Any, np.ndarray, Dict[str, float], Dict[str, Any]]:
    best_auc = -np.inf
    best_model: Any | None = None
    best_oof: np.ndarray | None = None
    best_metrics: Dict[str, float] | None = None
    best_params: Dict[str, Any] | None = None
    for idx, params in enumerate(param_grid, start=1):
        _log_progress(
            f"[lgbm] Hyperparameter candidate {idx}/{len(param_grid)}: "
            + ", ".join(f"{k}={v}" for k, v in params.items())
        )
        model, oof, metrics = train_estimator_with_cv(
            lambda: create_lgbm_estimator(random_state=random_state, **params),
            x_df,
            y,
            dates,
            prefix="lgbm",
        )
        overall_auc = metrics.get("lgbm_overall_auc", float("nan"))
        if np.isnan(overall_auc):
            continue
        if overall_auc > best_auc:
            best_auc = overall_auc
            best_model = model
            best_oof = oof
            best_metrics = metrics
            best_params = dict(params)
            _log_progress(f"[lgbm] New best overall AUC: {overall_auc:.4f}")
    if best_model is None or best_oof is None or best_metrics is None or best_params is None:
        raise RuntimeError("LightGBM hyperparameter search failed to produce a valid model")
    best_metrics["lgbm_best_params"] = json.dumps(best_params)
    return best_model, best_oof, best_metrics, best_params


def _extract_positive_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        if probs.ndim == 2:
            return probs[:, -1].astype(np.float32)
        return probs.astype(np.float32)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features).astype(np.float32, copy=False)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Estimator must implement predict_proba or decision_function")


def train_estimator_with_cv(
    estimator_factory: Callable[[], Any],
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    prefix: str,
) -> tuple[Any, np.ndarray, Dict[str, float]]:
    x_array = x_df.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)

    folds = temporal_cv_indices(dates)
    num_folds = len(folds)
    oof = np.zeros_like(y_array)
    coverage = np.zeros_like(y_array, dtype=bool)
    metrics: Dict[str, float] = {}

    if num_folds == 0:
        _log_progress(f"[{prefix}] No temporal CV folds generated; training only on full dataset.")

    iterable = list(enumerate(folds, start=1))
    if num_folds > 0 and tqdm is not None:
        fold_iter = tqdm(
            iterable,
            total=num_folds,
            desc=f"{prefix} CV",
            unit="fold",
            leave=True,
        )
    else:
        fold_iter = iterable

    for idx, (train_mask, val_mask) in fold_iter:
        train_count = int(train_mask.sum())
        val_count = int(val_mask.sum())
        if tqdm is None or num_folds == 0:
            _log_progress(
                f"[{prefix}] Starting CV fold {idx}/{num_folds} "
                f"(train={train_count}, val={val_count})"
            )
        model = estimator_factory()
        with timed_step(f"{prefix} fold {idx} fit"):
            model.fit(x_array[train_mask], y_array[train_mask])
        preds = _extract_positive_scores(model, x_array[val_mask])
        oof[val_mask] = preds
        coverage[val_mask] = True
        fold_auc = roc_auc_score(y_array[val_mask], preds)
        metrics[f"{prefix}_fold_{idx}_auc"] = fold_auc
        if tqdm is not None and num_folds > 0:
            fold_iter.set_postfix({"train": train_count, "val": val_count, "auc": f"{fold_auc:.4f}"})
        else:
            _log_progress(f"[{prefix}] Fold {idx} AUC: {fold_auc:.4f}")

    if coverage.any():
        overall_auc = roc_auc_score(y_array[coverage], oof[coverage])
        metrics[f"{prefix}_overall_auc"] = overall_auc
        _log_progress(f"[{prefix}] Overall OOF AUC: {overall_auc:.4f}")
    else:
        metrics[f"{prefix}_overall_auc"] = float("nan")
        _log_progress(f"[{prefix}] No validation predictions collected; overall AUC set to NaN")

    final_model = estimator_factory()
    _log_progress(
        f"[{prefix}] Training final model on full dataset "
        f"(n={x_array.shape[0]}, features={x_array.shape[1]})"
    )
    with timed_step(f"{prefix} final fit"):
        final_model.fit(x_array, y_array)

    return final_model, oof, metrics


def train_logistic_with_cv(
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    random_state: int = 42,
) -> tuple[Any, np.ndarray, Dict[str, float]]:
    return train_estimator_with_cv(
        lambda: create_logistic_pipeline(random_state=random_state),
        x_df,
        y,
        dates,
        prefix="logistic",
    )


def train_mlp_with_cv(
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    random_state: int = 42,
) -> tuple[Any, np.ndarray, Dict[str, float]]:
    return train_estimator_with_cv(
        lambda: create_mlp_pipeline(random_state=random_state),
        x_df,
        y,
        dates,
        prefix="mlp",
    )


def train_lgbm_with_cv(
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    random_state: int = 42,
    overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Any, np.ndarray, Dict[str, float]]:
    estimator_kwargs = overrides or {}
    return train_estimator_with_cv(
        lambda: create_lgbm_estimator(random_state=random_state, **estimator_kwargs),
        x_df,
        y,
        dates,
        prefix="lgbm",
    )


def train_catboost_with_cv(
    x_df: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    random_state: int = 42,
    overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Any, np.ndarray, Dict[str, float]]:
    estimator_kwargs = overrides or {}
    return train_estimator_with_cv(
        lambda: create_catboost_estimator(random_state=random_state, **estimator_kwargs),
        x_df,
        y,
        dates,
        prefix="catboost",
    )


def run_simulation_for_period(
    schedule: pd.DataFrame,
    base_df: pd.DataFrame,
    delay_sampler: DelaySampler,
    cancellation_model: CancellationEstimator,
    n_simulations: int = 200,
    random_state: int = 42,
    progress_name: Optional[str] = None,
    *,
    min_simulations: Optional[int] = None,
    max_error: float = 0.003,
    adaptive: bool = True,
    stratified: bool = True,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    sim_features = run_monte_carlo(
        schedule=schedule,
        base_index=base_df,
        delay_sampler=delay_sampler,
        cancellation_model=cancellation_model,
        n_simulations=n_simulations,
        random_state=random_state,
        progress_name=progress_name,
        min_simulations=min_simulations,
        max_error=max_error,
        adaptive=adaptive,
        stratified=stratified,
        n_jobs=n_jobs,
    )
    return sim_features


def pipeline(
    n_simulations: int = 150,
    random_state: int = 42,
    training_simulations: Optional[int] = None,
    tune_lgbm: bool = False,
    tune_catboost: bool = False,
    include_xgb: bool = True,
    xgb_params: Optional[Dict[str, Any]] = None,
    catboost_params: Optional[Dict[str, Any]] = None,
    lgbm_params: Optional[Dict[str, Any]] = None,
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, float] = {}

    with timed_step("Load datasets"):
        training = load_training_data()
        inference = load_inference_template()
        flights = load_historical_flights()
        schedule_october = load_schedule_october()

    add_airport_time_buckets(training)
    add_airport_time_buckets(inference)

    with timed_step("Build training feature matrices"):
        schedule_events_all = build_event_table(flights, use_actual=False)
        base_training = build_base_index(training)
        schedule_features_training = build_feature_matrix(
            base_training, schedule_events_all, training_df=training
        )

    holiday_path = EXTERNAL_PROCESSED_DIR / "calendar_public_holidays.csv"
    weather_history_path = EXTERNAL_RAW_DIR / "weather_history.parquet"
    weather_forecast_path = EXTERNAL_RAW_DIR / "weather_forecast.parquet"

    with timed_step("Merge training external features"):
        external_training = merge_external_features(
            base_training,
            weather_path=weather_history_path if weather_history_path.exists() else None,
            holiday_path=holiday_path if holiday_path.exists() else None,
            school_holiday_path=SCHOOL_HOLIDAYS_PATH if SCHOOL_HOLIDAYS_PATH.exists() else None,
        )

    global_target_mean = float(training["target"].mean()) if not training.empty else 0.0

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

    recent_training_features, recent_defaults = compute_recent_target_features(
        training,
        windows=(24, 72, 168, 336),
    )
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
                col_values = training_design[col].astype(np.float32)
                mean_val = float(np.nanmean(col_values)) if np.isnan(col_values).any() else None
                if mean_val is None or np.isnan(mean_val):
                    mean_val = global_target_mean
                if np.isnan(col_values).any():
                    col_values = np.where(np.isnan(col_values), mean_val, col_values)
                training_design[col] = col_values.astype(np.float32)
        recent_group_defaults = (
            recent_training_features
            .sort_values(["airport_group", "date", "hour"])
            .groupby("airport_group", as_index=False)
            .tail(1)
        )
    else:
        recent_group_defaults = pd.DataFrame()

    with timed_step("Fit delay and cancellation models"):
        delay_sampler = DelaySampler()
        delay_sampler.fit(flights)
        cancellation_model = CancellationEstimator()
        cancellation_model.fit(flights)

    schedule_training = flights[[
        "flight_id",
        "dep_airport",
        "arr_airport",
        "service_type",
        "std",
        "sta",
        "dep_airport_group",
        "arr_airport_group",
        "cancelled",
    ]].copy()
    if "cancelled" not in schedule_training.columns:
        schedule_training["cancelled"] = 0
    schedule_training["cancelled"] = schedule_training["cancelled"].fillna(0).astype(int)

    train_simulations = int(training_simulations or max(n_simulations, 120))
    sim_jobs = max(1, os.cpu_count() - 1)
    train_min_sim = max(60, min(train_simulations, int(train_simulations * 0.6)))
    infer_min_sim = max(40, min(int(n_simulations), int(n_simulations * 0.6)))
    train_sim_kwargs = {
        "min_simulations": train_min_sim,
        "max_error": 0.0025,
        "adaptive": True,
        "stratified": True,
        "n_jobs": sim_jobs,
    }
    infer_sim_kwargs = {
        "min_simulations": infer_min_sim,
        "max_error": 0.003,
        "adaptive": True,
        "stratified": True,
        "n_jobs": sim_jobs,
    }
    train_sim_config = {
        "kind": "training",
        "n_simulations": int(train_simulations),
        "random_state": int(random_state),
        "schedule_rows": int(len(schedule_training)),
        "minute_features": True,
        "delay_sampler": delay_sampler.fingerprint(),
        "cancellation_model": cancellation_model.fingerprint(),
    }
    sim_features_training = load_cached_simulation("sim_features_training", train_sim_config)
    if sim_features_training is None:
        with timed_step(f"Run training simulation ({train_simulations} draws)"):
            sim_features_training = run_simulation_for_period(
                schedule_training,
                base_training,
                delay_sampler,
                cancellation_model,
                n_simulations=train_simulations,
                random_state=random_state,
                progress_name="Training simulation",
                **train_sim_kwargs,
            )
        cache_simulation("sim_features_training", sim_features_training, train_sim_config)

    training_design = training_design.merge(
        sim_features_training,
        on=["airport_group", "date", "hour"],
        how="left",
    )
    sim_feature_cols = [col for col in sim_features_training.columns if col.startswith("sim_")]
    sim_feature_means: Dict[str, float] = {}
    for col in sim_feature_cols:
        if col in training_design.columns:
            col_mean = float(training_design[col].mean())
            sim_feature_means[col] = col_mean
            training_design[col] = training_design[col].fillna(col_mean)
    sim_train_array = (
        training_design["sim_prob_any_overlap"].to_numpy(dtype=np.float32)
        if "sim_prob_any_overlap" in training_design.columns
        else np.zeros(len(training_design), dtype=np.float32)
    )

    target_enc_columns = [
        col for col in TARGET_ENCODING_CANDIDATES if col in training.columns
    ]
    enc_input = training[target_enc_columns].reset_index(drop=True)
    target_enc_df, target_enc_map, target_global_mean = compute_temporal_target_encodings(
        enc_input,
        training["target"].reset_index(drop=True),
        training["date"].reset_index(drop=True),
        target_enc_columns,
    )
    training_design = pd.concat([training_design.reset_index(drop=True), target_enc_df], axis=1)

    with timed_step("Prepare training design matrix"):
        target_series = training_design["target"].astype(np.float32)
        x_df, y, feature_builder = prepare_training_features(
            training_design.drop(columns=["target"]),
            target_series,
        )
        feature_columns = list(x_df.columns)

    with timed_step("Train logistic model (sklearn)"):
        logistic_model, logistic_oof, logistic_metrics = train_logistic_with_cv(
            x_df,
            y,
            training["date"],
            random_state=random_state,
        )
        metrics.update(logistic_metrics)
        joblib.dump(logistic_model, MODELS_DIR / "logistic_model.joblib")
        np.save(FEATURE_CACHE_DIR / "logistic_oof.npy", logistic_oof.astype(np.float32))
        (FEATURE_CACHE_DIR / "logistic_metrics.json").write_text(
            pd.Series(logistic_metrics).to_json(indent=2)
        )

    with timed_step("Train MLP model (sklearn)"):
        mlp_model, mlp_oof, mlp_metrics = train_mlp_with_cv(
            x_df,
            y,
            training["date"],
            random_state=random_state,
        )
        metrics.update(mlp_metrics)
        joblib.dump(mlp_model, MODELS_DIR / "mlp_model.joblib")
        np.save(FEATURE_CACHE_DIR / "mlp_oof.npy", mlp_oof.astype(np.float32))
        (FEATURE_CACHE_DIR / "mlp_metrics.json").write_text(
            pd.Series(mlp_metrics).to_json(indent=2)
        )

    if tune_lgbm:
        with timed_step("Tune LightGBM model"):
            lgbm_model, lgbm_oof, lgbm_metrics, best_lgbm_params = tune_lgbm_with_cv(
                x_df,
                y,
                training["date"],
                random_state=random_state,
            )
            lgbm_metrics["lgbm_best_params"] = json.dumps(best_lgbm_params)
    else:
        effective_lgbm_params = {
            **BEST_LGBM_PARAMS,
            **(lgbm_params or {}),
        }
        with timed_step("Train LightGBM model"):
            lgbm_model, lgbm_oof, lgbm_metrics = train_lgbm_with_cv(
                x_df,
                y,
                training["date"],
                random_state=random_state,
                overrides=effective_lgbm_params,
            )
        lgbm_metrics["lgbm_best_params"] = json.dumps(effective_lgbm_params)
    metrics.update(lgbm_metrics)
    joblib.dump(lgbm_model, MODELS_DIR / "lgbm_model.joblib")
    np.save(FEATURE_CACHE_DIR / "lgbm_oof.npy", lgbm_oof.astype(np.float32))
    (FEATURE_CACHE_DIR / "lgbm_metrics.json").write_text(
        pd.Series(lgbm_metrics).to_json(indent=2)
    )

    if tune_catboost:
        with timed_step("Tune CatBoost model"):
            catboost_model, catboost_oof, catboost_metrics, best_cat_params = tune_catboost_with_cv(
                x_df,
                y,
                training["date"],
                random_state=random_state,
            )
            catboost_metrics["catboost_best_params"] = json.dumps(best_cat_params)
    else:
        effective_cat_params = {
            **BEST_CATBOOST_PARAMS,
            **(catboost_params or {}),
        }
        with timed_step("Train CatBoost model"):
            catboost_model, catboost_oof, catboost_metrics = train_catboost_with_cv(
                x_df,
                y,
                training["date"],
                random_state=random_state,
                overrides=effective_cat_params,
            )
        catboost_metrics["catboost_best_params"] = json.dumps(effective_cat_params)
    metrics.update(catboost_metrics)
    joblib.dump(catboost_model, MODELS_DIR / "catboost_model.cbm")
    np.save(FEATURE_CACHE_DIR / "catboost_oof.npy", catboost_oof.astype(np.float32))
    (FEATURE_CACHE_DIR / "catboost_metrics.json").write_text(
        pd.Series(catboost_metrics).to_json(indent=2)
    )

    if include_xgb:
        effective_xgb_params = xgb_params or {}
        with timed_step("Train XGBoost model"):
            xgb_model, xgb_oof, xgb_metrics = train_xgb_with_cv(
                x_df,
                y,
                training["date"],
                random_state=random_state,
                overrides=effective_xgb_params,
            )
        metrics.update(xgb_metrics)
        joblib.dump(xgb_model, MODELS_DIR / "xgb_model.ubj")
        np.save(FEATURE_CACHE_DIR / "xgb_oof.npy", xgb_oof.astype(np.float32))
        (FEATURE_CACHE_DIR / "xgb_metrics.json").write_text(
            pd.Series(xgb_metrics).to_json(indent=2)
        )
    else:
        xgb_model = None
        xgb_oof = np.zeros_like(y.to_numpy(dtype=np.float32))

    (FEATURE_CACHE_DIR / "logistic_features.json").write_text(
        pd.Series({"columns": feature_columns, "model": "sklearn_pipeline"}).to_json(indent=2)
    )

    with timed_step("Build inference design matrix"):
        schedule_events_october = build_event_table(schedule_october, use_actual=False)
        base_inference = build_base_index(inference)
        schedule_features_inference = build_feature_matrix(
            base_inference, schedule_events_october, training_df=training
        )
        external_inference = merge_external_features(
            base_inference,
            weather_path=weather_forecast_path if weather_forecast_path.exists() else None,
            holiday_path=holiday_path if holiday_path.exists() else None,
            school_holiday_path=SCHOOL_HOLIDAYS_PATH if SCHOOL_HOLIDAYS_PATH.exists() else None,
        )
        inference_design = assemble_design_frame(
            original=inference,
            base=base_inference,
            schedule_features=schedule_features_inference,
            external_features=external_inference,
        )
        enc_input_inference = inference[target_enc_columns].reset_index(drop=True)
        enc_inference = apply_target_encodings(
            enc_input_inference,
            target_enc_map,
            target_global_mean,
        )
        inference_design = pd.concat(
            [inference_design.reset_index(drop=True), enc_inference], axis=1
        )
        if recent_feature_cols:
            recent_defaults_hour = recent_defaults.drop(columns=["date"], errors="ignore")
            inference_design = inference_design.merge(
                recent_defaults_hour,
                on=["airport_group", "hour"],
                how="left",
            )
            if not recent_group_defaults.empty:
                fallback_cols = [
                    col
                    for col in recent_group_defaults.columns
                    if col not in {"airport_group", "date", "hour"}
                ]
                if fallback_cols:
                    inference_design = inference_design.merge(
                        recent_group_defaults[["airport_group", *fallback_cols]],
                        on="airport_group",
                        how="left",
                        suffixes=("", "_group_fallback"),
                    )
                    for col in fallback_cols:
                        fallback_col = f"{col}_group_fallback"
                        if fallback_col in inference_design:
                            inference_design[col] = inference_design[col].astype(np.float32)
                            inference_design[col] = inference_design[col].fillna(
                                inference_design[fallback_col]
                            )
                            inference_design.drop(columns=[fallback_col], inplace=True)
            for col in recent_feature_cols:
                if col in inference_design:
                    col_values = inference_design[col].astype(np.float32)
                    mean_val = float(np.nanmean(col_values)) if np.isnan(col_values).any() else None
                    if mean_val is None or np.isnan(mean_val):
                        mean_val = global_target_mean
                    if np.isnan(col_values).any():
                        col_values = np.where(np.isnan(col_values), mean_val, col_values)
                    inference_design[col] = col_values.astype(np.float32)

    infer_sim_config = {
        "kind": "inference",
        "n_simulations": int(n_simulations),
        "random_state": int(random_state),
        "schedule_rows": int(len(schedule_october)),
        "minute_features": True,
        "delay_sampler": delay_sampler.fingerprint(),
        "cancellation_model": cancellation_model.fingerprint(),
    }
    sim_features_oct = load_cached_simulation("sim_features_october", infer_sim_config)
    if sim_features_oct is None:
        with timed_step(f"Run October simulation ({n_simulations} draws)"):
            sim_features_oct = run_simulation_for_period(
                schedule_october,
                base_inference,
                delay_sampler,
                cancellation_model,
                n_simulations=n_simulations,
                random_state=random_state,
                progress_name="Inference simulation",
                **infer_sim_kwargs,
            )
        cache_simulation("sim_features_october", sim_features_oct, infer_sim_config)

    inference_design = inference_design.merge(
        sim_features_oct,
        on=["airport_group", "date", "hour"],
        how="left",
    )
    for col in sim_feature_cols:
        if col in inference_design.columns:
            fill_value = sim_feature_means.get(col, float(inference_design[col].mean()))
            inference_design[col] = inference_design[col].fillna(fill_value)
    sim_inference_array = (
        inference_design["sim_prob_any_overlap"].to_numpy(dtype=np.float32)
        if "sim_prob_any_overlap" in inference_design.columns
        else np.zeros(len(inference_design), dtype=np.float32)
    )

    inference_features = feature_builder.transform(inference_design)
    inference_matrix = inference_features.to_numpy(dtype=np.float32)

    with timed_step("Score inference rows"):
        logistic_preds = _extract_positive_scores(logistic_model, inference_matrix)
        mlp_preds = _extract_positive_scores(mlp_model, inference_matrix)
        lgbm_preds = _extract_positive_scores(lgbm_model, inference_matrix)
        catboost_preds = _extract_positive_scores(catboost_model, inference_matrix)
        if include_xgb and xgb_model is not None:
            xgb_preds = _extract_positive_scores(xgb_model, inference_matrix)
        else:
            xgb_preds = np.zeros_like(catboost_preds, dtype=np.float32)
        np.save(
            FEATURE_CACHE_DIR / "logistic_preds_inference.npy",
            logistic_preds.astype(np.float32),
        )
        np.save(
            FEATURE_CACHE_DIR / "mlp_preds_inference.npy",
            mlp_preds.astype(np.float32),
        )
        np.save(
            FEATURE_CACHE_DIR / "lgbm_preds_inference.npy",
            lgbm_preds.astype(np.float32),
        )
        np.save(
            FEATURE_CACHE_DIR / "catboost_preds_inference.npy",
            catboost_preds.astype(np.float32),
        )
        if include_xgb and xgb_model is not None:
            np.save(
                FEATURE_CACHE_DIR / "xgb_preds_inference.npy",
                xgb_preds.astype(np.float32),
            )

    july_mask = (training["date"] >= pd.Timestamp("2025-07-01")) & (
        training["date"] <= pd.Timestamp("2025-07-31")
    )
    july_mask_array = july_mask.to_numpy()

    training_july = training[july_mask].copy()

    sim_features_july = sim_features_training[
        (sim_features_training["date"] >= pd.Timestamp("2025-07-01"))
        & (sim_features_training["date"] <= pd.Timestamp("2025-07-31"))
    ].copy()
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
    july_logistic_preds = logistic_oof[july_mask_array]
    july_mlp_preds = mlp_oof[july_mask_array]
    july_lgbm_preds = lgbm_oof[july_mask_array]
    july_catboost_preds = catboost_oof[july_mask_array]

    base_train_components: list[np.ndarray] = []
    base_infer_components: list[np.ndarray] = []
    base_names: list[str] = []

    def add_component(name: str, train_vec: np.ndarray, infer_vec: np.ndarray) -> None:
        base_names.append(name)
        base_train_components.append(train_vec.astype(np.float32, copy=False))
        base_infer_components.append(infer_vec.astype(np.float32, copy=False))

    target_array = y.to_numpy(dtype=np.float32)
    add_component("logistic", logistic_oof, logistic_preds)
    add_component("mlp", mlp_oof, mlp_preds)
    add_component("catboost", catboost_oof, catboost_preds)
    add_component("lgbm", lgbm_oof, lgbm_preds)
    if include_xgb and xgb_model is not None:
        add_component("xgb", xgb_oof, xgb_preds)
    add_component("simulation", sim_train_array, sim_inference_array)

    ensemble_train = None
    ensemble_infer = None
    ensemble_metrics: Dict[str, float] = {}
    ensemble_model = None
    if len(base_names) >= 2:
        stack_train = np.column_stack(base_train_components)
        stack_infer = np.column_stack(base_infer_components)
        ensemble_model = SkLogisticRegression(
            penalty="l2",
            C=0.5,
            solver="lbfgs",
            max_iter=500,
            random_state=random_state,
        )
        ensemble_model.fit(stack_train, target_array)
        ensemble_train = ensemble_model.predict_proba(stack_train)[:, 1].astype(np.float32)
        ensemble_infer = ensemble_model.predict_proba(stack_infer)[:, 1].astype(np.float32)
        ensemble_overall_auc = roc_auc_score(target_array, ensemble_train)
        if july_mask_array.any():
            ensemble_july_auc = roc_auc_score(
                july_targets,
                ensemble_train[july_mask_array],
            )
        else:
            ensemble_july_auc = float("nan")
        ensemble_metrics = {
            "ensemble_overall_auc": ensemble_overall_auc,
            "ensemble_july_auc": ensemble_july_auc,
            "ensemble_intercept": float(ensemble_model.intercept_[0]),
        }
        for name, coef in zip(base_names, ensemble_model.coef_.ravel()):
            ensemble_metrics[f"ensemble_coef_{name}"] = float(coef)

    if ensemble_metrics:
        metrics.update(ensemble_metrics)

    candidate_models: list[dict[str, Any]] = []

    def register_candidate(
        name: str,
        auc_key: str,
        preds: np.ndarray | None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if preds is None:
            return
        auc_val = metrics.get(auc_key)
        if auc_val is None:
            return
        try:
            auc_float = float(auc_val)
        except (TypeError, ValueError):
            return
        if np.isnan(auc_float):
            return
        entry: Dict[str, Any] = {
            "name": name,
            "auc": auc_float,
            "preds": preds,
        }
        if extra:
            entry.update(extra)
        candidate_models.append(entry)

    register_candidate("logistic", "logistic_overall_auc", logistic_preds)
    register_candidate("mlp", "mlp_overall_auc", mlp_preds)
    register_candidate("lgbm", "lgbm_overall_auc", lgbm_preds)
    register_candidate("catboost", "catboost_overall_auc", catboost_preds)
    if include_xgb and xgb_model is not None:
        register_candidate("xgb", "xgb_overall_auc", xgb_preds)
    if ensemble_infer is not None and ensemble_metrics:
        register_candidate(
            "ensemble",
            "ensemble_overall_auc",
            ensemble_infer,
            extra={
                "model": ensemble_model,
                "train_oof": ensemble_train,
            },
        )

    if candidate_models:
        best_candidate = max(candidate_models, key=lambda item: item["auc"])
    else:
        best_candidate = {
            "name": "catboost",
            "auc": float(metrics.get("catboost_overall_auc", float("nan"))),
            "preds": catboost_preds,
        }

    final_model_label = best_candidate["name"]
    final_overall_auc = float(best_candidate["auc"])
    final_preds = np.clip(best_candidate["preds"], 0.0001, 0.9999)

    if final_model_label == "ensemble":
        model_obj = best_candidate.get("model")
        train_oof = best_candidate.get("train_oof")
        if model_obj is not None:
            joblib.dump(model_obj, MODELS_DIR / "ensemble_model.joblib")
        if train_oof is not None:
            np.save(
                FEATURE_CACHE_DIR / "ensemble_oof.npy",
                np.asarray(train_oof, dtype=np.float32),
            )
        np.save(
            FEATURE_CACHE_DIR / "ensemble_preds_inference.npy",
            np.asarray(best_candidate["preds"], dtype=np.float32),
        )

    result = inference[["airport_group", "date", "hour"]].copy()
    result["pred"] = final_preds
    result.sort_values(["date", "hour", "airport_group"], inplace=True)

    metrics["logistic_auc_july"] = roc_auc_score(july_targets, july_logistic_preds)
    metrics["mlp_auc_july"] = roc_auc_score(july_targets, july_mlp_preds)
    metrics["lgbm_auc_july"] = roc_auc_score(july_targets, july_lgbm_preds)
    metrics["catboost_auc_july"] = roc_auc_score(july_targets, july_catboost_preds)
    metrics["simulation_auc_july"] = roc_auc_score(july_targets, july_sim_preds)
    metrics["final_model"] = final_model_label
    metrics["final_overall_auc"] = final_overall_auc

    summary_keys = [
        key
        for key in metrics
        if key.endswith("_overall_auc")
        or key.endswith("_july")
    ]
    if summary_keys:
        summary_parts = []
        for key in sorted(summary_keys):
            value = metrics[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                summary_parts.append(f"{key}={value:.4f}")
            else:
                summary_parts.append(f"{key}={value}")
        _log_progress(f"[metrics] {' | '.join(summary_parts)}")
    if "ensemble_overall_auc" in metrics:
        overall = metrics["ensemble_overall_auc"]
        july_val = metrics.get("ensemble_july_auc", float("nan"))
        _log_progress(
            f"[metrics] Final Ensemble overall AUC: {overall:.4f} | July AUC: {july_val:.4f}"
        )
    elif "catboost_overall_auc" in metrics:
        overall = metrics["catboost_overall_auc"]
        july_val = metrics.get("catboost_auc_july", float("nan"))
        _log_progress(
            f"[metrics] Final CatBoost overall AUC: {overall:.4f} | July AUC: {july_val:.4f}"
        )
    _log_progress(
        f"[metrics] Final model used: {final_model_label} | Overall AUC: "
        f"{final_overall_auc:.4f}"
    )

    use_ensemble = final_model_label == "ensemble"

    metrics_path = FEATURE_CACHE_DIR / "training_metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)

    return {
        "predictions": result,
        "metrics": metrics,
        "logistic_oof": logistic_oof,
        "mlp_oof": mlp_oof,
        "lgbm_oof": lgbm_oof,
        "catboost_oof": catboost_oof,
        "xgb_oof": xgb_oof,
        "ensemble_used": use_ensemble,
    }



__all__ = [
    "pipeline",
    "prepare_training_features",
    "compute_temporal_target_encodings",
    "apply_target_encodings",
    "create_logistic_pipeline",
    "create_mlp_pipeline",
    "create_lgbm_estimator",
    "create_catboost_estimator",
    "create_xgb_estimator",
    "tune_lgbm_with_cv",
    "tune_catboost_with_cv",
    "train_estimator_with_cv",
    "train_logistic_with_cv",
    "train_mlp_with_cv",
    "train_lgbm_with_cv",
    "train_catboost_with_cv",
    "train_xgb_with_cv",
]
