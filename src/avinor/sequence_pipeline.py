from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .data_loader import (
    load_historical_flights,
    load_inference_template,
    load_schedule_october,
    load_training_data,
)
from .feature_engineering import (
    build_base_index,
    build_event_table,
    build_feature_matrix,
    build_minute_sequences,
    compute_historical_baselines,
    compute_recent_target_features,
)
from .models.delay_cancellation import CancellationEstimator, DelaySampler
from .pipeline import (
    assemble_design_frame,
    apply_target_encodings,
    cache_simulation,
    compute_temporal_target_encodings,
    load_cached_simulation,
    prepare_training_features,
    run_simulation_for_period,
    temporal_cv_indices,
)
from .external_features import merge_external_features
from .metrics import roc_auc_score
from .paths import (
    EXTERNAL_PROCESSED_DIR,
    EXTERNAL_RAW_DIR,
    FEATURE_CACHE_DIR,
    MODELS_DIR,
    SCHOOL_HOLIDAYS_PATH,
)


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
SIM_MINUTE_PREFIX = "sim_concurrency_minute_"
CONTEXT_HOURS = 3
SEQ_LEN = 60
PAIRWISE_WEIGHT = 0.3
PAIRWISE_MARGIN = 0.0
EARLY_STOP_PATIENCE = 5
EARLY_STOP_DELTA = 1e-4
SEQ_ATTENTION_HEADS = 4
SEQ_DROPOUT = 0.2
MONTH_FEATURE_COLUMNS = [
    "month_target_mean",
    "month_target_median",
    "month_target_std",
    "month_target_count",
]


@dataclass
class TargetFeatureArtifacts:
    month_stats: pd.DataFrame
    month_global: pd.DataFrame
    recent_defaults_hour: pd.DataFrame
    recent_group_defaults: pd.DataFrame
    recent_feature_cols: List[str]
    recent_fill_values: Dict[str, float]
    target_enc_map: Dict[str, Dict[Any, float]]
    target_global_mean: float


def _fill_recent_features(
    design: pd.DataFrame,
    feature_cols: Sequence[str],
    defaults_hour: pd.DataFrame,
    defaults_group: pd.DataFrame,
    fill_values: Dict[str, float],
) -> pd.DataFrame:
    if feature_cols:
        if not defaults_hour.empty:
            merged = design.merge(
                defaults_hour,
                on=["airport_group", "hour"],
                how="left",
                suffixes=("", "_hour_default"),
            )
            for col in feature_cols:
                fallback = f"{col}_hour_default"
                if fallback in merged.columns:
                    merged[col] = merged[col].fillna(merged[fallback])
                    merged.drop(columns=[fallback], inplace=True)
            design = merged

        if not defaults_group.empty:
            merged = design.merge(
                defaults_group,
                on="airport_group",
                how="left",
                suffixes=("", "_group_default"),
            )
            for col in feature_cols:
                fallback = f"{col}_group_default"
                if fallback in merged.columns:
                    merged[col] = merged[col].fillna(merged[fallback])
                    merged.drop(columns=[fallback], inplace=True)
            design = merged

        for col in feature_cols:
            fill_value = fill_values.get(col, 0.0)
            design[col] = design[col].fillna(fill_value).astype(np.float32)

    return design


def _prepare_recent_artifacts(
    recent_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if recent_features.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    feature_cols = [
        col
        for col in recent_features.columns
        if col not in {"airport_group", "date", "hour"}
    ]

    defaults_hour = recent_features.sort_values(["airport_group", "hour", "date"])
    defaults_hour = (
        defaults_hour.groupby(["airport_group", "hour"], as_index=False).tail(1)
    )
    defaults_hour = defaults_hour.drop(columns=["date"], errors="ignore")

    group_defaults = (
        recent_features.sort_values(["airport_group", "date", "hour"])
        .groupby("airport_group", as_index=False)
        .tail(1)
    )
    group_defaults = group_defaults.drop(columns=["date", "hour"], errors="ignore")

    fill_values: Dict[str, float] = {}
    for col in feature_cols:
        col_mean = float(recent_features[col].dropna().mean())
        if not np.isfinite(col_mean):
            col_mean = 0.0
        fill_values[col] = col_mean

    return defaults_hour, group_defaults, fill_values


def _ensure_month_columns(design: pd.DataFrame) -> pd.DataFrame:
    for col in MONTH_FEATURE_COLUMNS:
        if col not in design.columns:
            design[col] = np.float32(0.0)
    return design


def _merge_month_stats_safe(
    design: pd.DataFrame,
    month_stats: pd.DataFrame,
    month_global: pd.DataFrame,
) -> pd.DataFrame:
    design = design.drop(columns=[col for col in MONTH_FEATURE_COLUMNS if col in design.columns])
    if month_stats.empty and month_global.empty:
        return _ensure_month_columns(design)
    merged = _merge_monthly_stats(design, month_stats, month_global)
    return merged


def build_target_dependent_features(
    base_design: pd.DataFrame,
    base_index: pd.DataFrame,
    training_df: pd.DataFrame,
    train_mask: Optional[np.ndarray],
    *,
    target_enc_columns: Sequence[str],
) -> tuple[pd.DataFrame, TargetFeatureArtifacts]:
    design = base_design.copy()
    if train_mask is None:
        train_subset = training_df
    else:
        train_subset = training_df.loc[train_mask]

    month_stats, month_global = _compute_monthly_target_stats(train_subset)
    design = _merge_month_stats_safe(design, month_stats, month_global)

    if not train_subset.empty:
        historical = compute_historical_baselines(base_index, train_subset)
        historical = historical.reset_index(drop=True).astype(np.float32)
        design = design.reset_index(drop=True)
        design = pd.concat([design, historical], axis=1)
    else:
        for col in [
            "hist_rate_group_hour",
            "hist_support_group_hour",
            "hist_rate_month_hour",
            "hist_rate_dow_hour",
        ]:
            if col not in design.columns:
                design[col] = np.float32(0.0)

    recent_features, _ = compute_recent_target_features(train_subset)
    feature_cols = [
        col
        for col in recent_features.columns
        if col not in {"airport_group", "date", "hour"}
    ]
    if feature_cols:
        design = design.drop(columns=feature_cols, errors="ignore")
        design = design.merge(
            recent_features,
            on=["airport_group", "date", "hour"],
            how="left",
        )
        defaults_hour, group_defaults, fill_values = _prepare_recent_artifacts(recent_features)
        design = _fill_recent_features(design, feature_cols, defaults_hour, group_defaults, fill_values)
    else:
        defaults_hour = pd.DataFrame()
        group_defaults = pd.DataFrame()
        fill_values = {}

    if feature_cols:
        for col in feature_cols:
            design[col] = design[col].astype(np.float32)

    for column in target_enc_columns:
        enc_col = f"{column}_target_enc"
        if enc_col not in design.columns:
            design[enc_col] = np.nan

    if target_enc_columns and not train_subset.empty:
        enc_input = training_df.loc[train_subset.index, target_enc_columns]
        target_series = training_df.loc[train_subset.index, "target"]
        date_series = training_df.loc[train_subset.index, "date"]
        (
            target_enc_df,
            target_enc_map,
            target_global_mean,
        ) = compute_temporal_target_encodings(
            enc_input,
            target_series,
            date_series,
            target_enc_columns,
        )
        for col in target_enc_df.columns:
            design.loc[target_enc_df.index, col] = target_enc_df[col].astype(np.float32)
    else:
        target_enc_map = {}
        target_global_mean = float(train_subset["target"].mean()) if not train_subset.empty else 0.0

    if target_enc_columns:
        enc_full = apply_target_encodings(
            design[target_enc_columns],
            target_enc_map,
            target_global_mean,
        )
        for col in enc_full.columns:
            design[col] = design[col].fillna(enc_full[col].astype(np.float32))

    artifacts = TargetFeatureArtifacts(
        month_stats=month_stats,
        month_global=month_global,
        recent_defaults_hour=defaults_hour,
        recent_group_defaults=group_defaults,
        recent_feature_cols=feature_cols,
        recent_fill_values=fill_values,
        target_enc_map=target_enc_map,
        target_global_mean=target_global_mean,
    )

    return design.reset_index(drop=True), artifacts


def apply_feature_artifacts_to_inference(
    base_design: pd.DataFrame,
    base_index: pd.DataFrame,
    training_df: pd.DataFrame,
    artifacts: TargetFeatureArtifacts,
    target_enc_columns: Sequence[str],
) -> pd.DataFrame:
    design = base_design.copy()
    design = _merge_month_stats_safe(design, artifacts.month_stats, artifacts.month_global)
    historical = compute_historical_baselines(base_index, training_df)
    historical = historical.reset_index(drop=True).astype(np.float32)
    design = design.reset_index(drop=True)
    design = pd.concat([design, historical], axis=1)

    feature_cols = artifacts.recent_feature_cols
    if feature_cols:
        for col in feature_cols:
            if col not in design.columns:
                design[col] = np.nan
        design = _fill_recent_features(
            design,
            feature_cols,
            artifacts.recent_defaults_hour,
            artifacts.recent_group_defaults,
            artifacts.recent_fill_values,
        )
        for col in feature_cols:
            design[col] = design[col].astype(np.float32)

    for column in target_enc_columns:
        enc_col = f"{column}_target_enc"
        if enc_col not in design.columns:
            design[enc_col] = np.nan
    if target_enc_columns:
        enc_full = apply_target_encodings(
            design[target_enc_columns],
            artifacts.target_enc_map,
            artifacts.target_global_mean,
        )
        for col in enc_full.columns:
            design[col] = enc_full[col].astype(np.float32)

    return design.reset_index(drop=True)


def _progress(iterable: Iterable[T], desc: str, show: bool) -> Iterable[T]:
    """Wrap an iterable with tqdm when progress display is enabled."""
    if show:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def _drop_sim_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if col.startswith(SIM_MINUTE_PREFIX)]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def _pairwise_auc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return logits.new_tensor(0.0)
    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]
    diff = neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1) + PAIRWISE_MARGIN
    loss = torch.relu(diff)
    return loss.mean()


def _extract_simulation_sequence(df: pd.DataFrame, stat: str) -> Optional[np.ndarray]:
    cols = [f"sim_concurrency_minute_{minute:02d}_{stat}" for minute in range(SEQ_LEN)]
    if not all(col in df.columns for col in cols):
        return None
    return df[cols].to_numpy(dtype=np.float32)


def _build_context_sequences(
    base: pd.DataFrame,
    sequences: np.ndarray,
    context_hours: int = CONTEXT_HOURS,
) -> np.ndarray:
    if context_hours <= 1:
        return sequences

    n_samples, n_channels, seq_len = sequences.shape
    context = np.zeros((n_samples, n_channels, seq_len * context_hours), dtype=sequences.dtype)

    base_indices = base.reset_index(drop=False)
    base_indices.sort_values(["airport_group", "date", "hour"], inplace=True)

    for _, group_df in base_indices.groupby("airport_group", sort=False):
        ordered_idx = group_df["index"].to_numpy(dtype=int)
        for position, idx in enumerate(ordered_idx):
            for ctx_offset in range(context_hours):
                shift = context_hours - 1 - ctx_offset
                source_pos = position - shift
                if source_pos < 0:
                    continue
                src_idx = ordered_idx[source_pos]
                start = ctx_offset * seq_len
                end = start + seq_len
                context[idx, :, start:end] = sequences[src_idx]

    return context


def _compute_monthly_target_stats(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if training.empty:
        empty = pd.DataFrame(columns=["airport_group", "month", *MONTH_FEATURE_COLUMNS])
        global_empty = pd.DataFrame(columns=["month", *MONTH_FEATURE_COLUMNS])
        return empty, global_empty

    month_series = training["date"].dt.month.astype("int16")
    group_stats = (
        training.assign(month=month_series)
        .groupby(["airport_group", "month"], as_index=False)["target"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            count="count",
        )
    )
    group_stats.rename(
        columns={
            "mean": "month_target_mean",
            "median": "month_target_median",
            "std": "month_target_std",
            "count": "month_target_count",
        },
        inplace=True,
    )
    group_stats["month_target_std"] = group_stats["month_target_std"].fillna(0.0)

    global_stats = (
        training.assign(month=month_series)
        .groupby("month", as_index=False)["target"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            count="count",
        )
    )
    global_stats.rename(
        columns={
            "mean": "month_target_mean",
            "median": "month_target_median",
            "std": "month_target_std",
            "count": "month_target_count",
        },
        inplace=True,
    )
    global_stats["month_target_std"] = global_stats["month_target_std"].fillna(0.0)
    return group_stats, global_stats


def _merge_monthly_stats(
    design: pd.DataFrame,
    month_stats: pd.DataFrame,
    month_global: pd.DataFrame,
) -> pd.DataFrame:
    if month_stats.empty:
        for col in MONTH_FEATURE_COLUMNS:
            if col not in design.columns:
                default_value = 0.0 if col != "month_target_count" else 0.0
                design[col] = np.float32(default_value)
        return design

    merged = design.merge(month_stats, on=["airport_group", "month"], how="left")
    if not month_global.empty:
        global_map = month_global.set_index("month")
        for col in MONTH_FEATURE_COLUMNS:
            if col in merged:
                fallback = merged["month"].map(global_map[col])
                merged[col] = merged[col].fillna(fallback)
    for col in MONTH_FEATURE_COLUMNS:
        if col in merged:
            if col == "month_target_count":
                merged[col] = merged[col].fillna(0).astype(np.float32)
            else:
                merged[col] = merged[col].fillna(0.0).astype(np.float32)
    return merged


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device != device:
        return tensor.to(device)
    return tensor


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = SEQ_DROPOUT,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        self.residual_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_bn(self.residual(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + residual
        out = self.activation(out)
        return out


class SequenceAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = SEQ_ATTENTION_HEADS,
        dropout: float = SEQ_DROPOUT,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class SequenceModel(nn.Module):
    def __init__(self, seq_channels: int, static_dim: int) -> None:
        super().__init__()

        conv_channels = [seq_channels, 128, 128, 192, 256]
        dilations = [1, 2, 4, 8]
        self.conv_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    conv_channels[i],
                    conv_channels[i + 1],
                    dilation=dilations[i],
                )
                for i in range(len(conv_channels) - 1)
            ]
        )

        self.attention = SequenceAttentionBlock(conv_channels[-1])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        if static_dim > 0:
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(SEQ_DROPOUT),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(SEQ_DROPOUT),
            )
            self.static_proj = nn.Linear(128, conv_channels[-1])
        else:
            self.static_net = None
            self.static_proj = None

        seq_dim = conv_channels[-1]
        if self.static_net is not None:
            combined_dim = seq_dim * 4 + 128
        else:
            combined_dim = seq_dim * 3

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, sequences: torch.Tensor, static: Optional[torch.Tensor]) -> torch.Tensor:
        x = sequences
        for block in self.conv_blocks:
            x = block(x)

        # attention expects shape (batch, seq_len, channels)
        x_attn = x.permute(0, 2, 1)
        x_attn = self.attention(x_attn)
        x = x_attn.permute(0, 2, 1)

        avg_pool = self.avg_pool(x).squeeze(-1)
        max_pool = self.max_pool(x).squeeze(-1)
        tail_features = x[:, :, -1]

        if self.static_net is not None and static is not None:
            static_repr = self.static_net(static)
            gate = torch.sigmoid(self.static_proj(static_repr))
            gated_seq = avg_pool * gate
            combined = torch.cat([avg_pool, max_pool, tail_features, gated_seq, static_repr], dim=1)
        else:
            combined = torch.cat([avg_pool, max_pool, tail_features], dim=1)

        logits = self.head(combined)
        return logits.squeeze(-1)


@dataclass
class SequencePipelineResult:
    predictions: pd.DataFrame
    metrics: Dict[str, float]
    oof_predictions: np.ndarray
    model_path: Path


def sequence_pipeline(
    n_simulations: int = 150,
    random_state: int = 42,
    max_epochs: int = 40,
    tune_lr: float = 1e-3,
    batch_size: int = 512,
    *,
    show_progress: bool = True,
    logger: Optional[logging.Logger] = None,
) -> SequencePipelineResult:
    logger = logger or LOGGER
    logger.info("Starting sequence pipeline with %d simulations", n_simulations)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    logger.info("Loading raw datasets")
    training = load_training_data()
    inference = load_inference_template()
    flights = load_historical_flights()
    schedule_october = load_schedule_october()
    folds = temporal_cv_indices(training["date"])

    train_last_date = training["date"].max()
    inference_first_date = inference["date"].min()
    if pd.notna(train_last_date) and pd.notna(inference_first_date):
        gap_days = (inference_first_date - train_last_date).days
        if gap_days > 30:
            logger.warning(
                "Training data ends on %s while inference starts on %s (%d-day gap)",
                train_last_date.date(),
                inference_first_date.date(),
                gap_days,
            )

    logger.info("Computing training feature bases")
    base_training = build_base_index(training)
    schedule_events_all = build_event_table(flights, use_actual=False)
    schedule_features_training = build_feature_matrix(
        base_training,
        schedule_events_all,
        training_df=None,
    )

    holiday_path = EXTERNAL_PROCESSED_DIR / "calendar_public_holidays.csv"
    school_holiday_path = SCHOOL_HOLIDAYS_PATH
    weather_history_path = EXTERNAL_RAW_DIR / "weather_history.parquet"
    weather_forecast_path = EXTERNAL_RAW_DIR / "weather_forecast.parquet"

    external_training = merge_external_features(
        base_training,
        weather_path=weather_history_path if weather_history_path.exists() else None,
        holiday_path=holiday_path if holiday_path.exists() else None,
        school_holiday_path=school_holiday_path if school_holiday_path.exists() else None,
    )

    logger.info("Assembling training design frame")
    training_design_base = assemble_design_frame(
        original=training.drop(columns=["target"]),
        base=base_training,
        schedule_features=schedule_features_training,
        external_features=external_training,
    )
    training_design_base = training_design_base.merge(
        training[["airport_group", "date", "hour", "target"]],
        on=["airport_group", "date", "hour"],
        how="left",
    )

    logger.info("Fitting delay and cancellation models")
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

    train_simulations = max(n_simulations, 120)
    train_sim_config = {
        "kind": "training",
        "n_simulations": int(train_simulations),
        "random_state": int(random_state),
        "schedule_rows": int(len(schedule_training)),
        "minute_features": True,
    }
    sim_features_training = load_cached_simulation("sim_features_training", train_sim_config)
    if sim_features_training is not None:
        logger.info(
            "Loaded cached training simulation features (n_simulations=%d)",
            train_simulations,
        )
    else:
        logger.info("Running %d Monte Carlo simulations for training", train_simulations)
        sim_features_training = run_simulation_for_period(
            schedule_training,
            base_training,
            delay_sampler,
            cancellation_model,
            n_simulations=train_simulations,
            random_state=random_state,
            progress_name="Sequence training simulation" if show_progress else None,
        )
        cache_simulation("sim_features_training", sim_features_training, train_sim_config)
    training_design_base = training_design_base.merge(
        sim_features_training,
        on=["airport_group", "date", "hour"],
        how="left",
    )

    target_enc_columns = [col for col in ("airport_group", "feat_season") if col in training.columns]
    logger.info("Preparing full-training features")
    full_design, feature_artifacts = build_target_dependent_features(
        training_design_base,
        base_training,
        training,
        train_mask=None,
        target_enc_columns=target_enc_columns,
    )

    target_series = training["target"].astype(np.float32)
    logger.info("Preparing model training features")
    full_design_static = _drop_sim_sequence_columns(full_design)
    x_df, y, feature_builder = prepare_training_features(
        full_design_static.drop(columns=["target"]),
        target_series,
    )
    training_matrix_full = x_df.to_numpy(dtype=np.float32)

    logger.info("Building inference design frame")
    base_inference = build_base_index(inference)
    schedule_events_october = build_event_table(schedule_october, use_actual=False)
    schedule_features_inference = build_feature_matrix(
        base_inference,
        schedule_events_october,
        training_df=None,
    )
    external_inference = merge_external_features(
        base_inference,
        weather_path=weather_forecast_path if weather_forecast_path.exists() else None,
        holiday_path=holiday_path if holiday_path.exists() else None,
        school_holiday_path=school_holiday_path if school_holiday_path.exists() else None,
    )
    inference_design_base = assemble_design_frame(
        original=inference,
        base=base_inference,
        schedule_features=schedule_features_inference,
        external_features=external_inference,
    )
    infer_sim_config = {
        "kind": "inference",
        "n_simulations": int(n_simulations),
        "random_state": int(random_state),
        "schedule_rows": int(len(schedule_october)),
        "minute_features": True,
    }
    infer_sim_features = load_cached_simulation("sim_features_october", infer_sim_config)
    if infer_sim_features is not None:
        logger.info(
            "Loaded cached inference simulation features (n_simulations=%d)",
            n_simulations,
        )
    else:
        logger.info("Running %d Monte Carlo simulations for inference", n_simulations)
        infer_sim_features = run_simulation_for_period(
            schedule_october,
            base_inference,
            delay_sampler,
            cancellation_model,
            n_simulations=n_simulations,
            random_state=random_state,
            progress_name="Sequence inference simulation" if show_progress else None,
        )
        cache_simulation("sim_features_october", infer_sim_features, infer_sim_config)
    inference_design_base = inference_design_base.merge(
        infer_sim_features,
        on=["airport_group", "date", "hour"],
        how="left",
    )

    inference_design = apply_feature_artifacts_to_inference(
        inference_design_base,
        base_inference,
        training,
        feature_artifacts,
        target_enc_columns,
    )

    inference_design_static = _drop_sim_sequence_columns(inference_design)
    inference_features = feature_builder.transform(inference_design_static)
    inference_matrix = inference_features.to_numpy(dtype=np.float32)

    y_array = y.to_numpy(dtype=np.float32)

    logger.info("Generating minute-level sequences")
    # Align feature availability with the competition setting: at inference
    # time we only have the scheduled plan (plus simulations). To avoid a
    # train/test mismatch we also build the training sequences from scheduled
    # movements and rely on the Monte Carlo features + external data to bridge
    # to actual concurrency.
    training_events = build_event_table(flights, use_actual=False)
    train_concurrency, train_dep, train_arr = build_minute_sequences(
        base_training,
        training_events,
        seq_len=SEQ_LEN,
    )

    seq_components_train: List[np.ndarray] = [
        train_concurrency,
        train_dep,
        train_arr,
    ]
    sim_mean_train = _extract_simulation_sequence(training_design_base, "mean")
    if sim_mean_train is not None:
        seq_components_train.append(sim_mean_train)
    sim_std_train = _extract_simulation_sequence(training_design_base, "std")
    if sim_std_train is not None:
        seq_components_train.append(sim_std_train)
    seq_train = np.stack(seq_components_train, axis=1).astype(np.float32)
    seq_train = _build_context_sequences(base_training, seq_train, CONTEXT_HOURS).astype(
        np.float32,
        copy=False,
    )

    inference_events = build_event_table(schedule_october, use_actual=False)
    inf_concurrency, inf_dep, inf_arr = build_minute_sequences(
        base_inference,
        inference_events,
        seq_len=SEQ_LEN,
    )
    seq_components_infer: List[np.ndarray] = [
        inf_concurrency,
        inf_dep,
        inf_arr,
    ]
    sim_mean_infer = _extract_simulation_sequence(inference_design_base, "mean")
    if sim_mean_infer is not None:
        seq_components_infer.append(sim_mean_infer)
    sim_std_infer = _extract_simulation_sequence(inference_design_base, "std")
    if sim_std_infer is not None:
        seq_components_infer.append(sim_std_infer)
    seq_infer = np.stack(seq_components_infer, axis=1).astype(np.float32)
    seq_infer = _build_context_sequences(base_inference, seq_infer, CONTEXT_HOURS).astype(
        np.float32,
        copy=False,
    )

    logger.info("Starting temporal cross-validation")
    num_folds = len(folds)
    oof = np.full(len(training), np.nan, dtype=np.float32)
    metrics: Dict[str, float] = {}
    best_epochs: List[int] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (train_mask, val_mask) in enumerate(
        _progress(folds, desc="CV folds", show=show_progress),
        start=1,
    ):
        fold_design, _ = build_target_dependent_features(
            training_design_base,
            base_training,
            training,
            train_mask=train_mask,
            target_enc_columns=target_enc_columns,
        )
        fold_design_static = _drop_sim_sequence_columns(fold_design)
        fold_features = feature_builder.transform(fold_design_static.drop(columns=["target"]))
        fold_matrix = fold_features.to_numpy(dtype=np.float32)

        def make_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
            seq_tensor = torch.from_numpy(seq_train[mask])
            static_tensor = torch.from_numpy(fold_matrix[mask])
            target_tensor = torch.from_numpy(y_array[mask])
            dataset = TensorDataset(seq_tensor, static_tensor, target_tensor)
            if shuffle:
                targets_np = target_tensor.numpy()
                num_pos = (targets_np > 0.5).sum()
                num_neg = targets_np.size - num_pos
                if num_pos > 0 and num_neg > 0:
                    pos_w = 0.5 / num_pos
                    neg_w = 0.5 / num_neg
                    weights = torch.where(
                        target_tensor > 0.5,
                        torch.full_like(target_tensor, pos_w, dtype=torch.double),
                        torch.full_like(target_tensor, neg_w, dtype=torch.double),
                    )
                    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
                    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(train_mask, shuffle=True)
        val_loader = make_loader(val_mask, shuffle=False)

        model = SequenceModel(seq_channels=seq_train.shape[1], static_dim=fold_matrix.shape[1])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=tune_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        targets_subset = y_array[train_mask]
        pos_fraction = float(targets_subset.mean())
        pos_weight_value = (1.0 - pos_fraction) / max(pos_fraction, 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

        best_auc = -np.inf
        best_state: Dict[str, Any] | None = None
        best_epoch = 0
        no_improve = 0

        epoch_iter = _progress(
            range(1, max_epochs + 1),
            desc=f"Fold {fold_idx} epochs",
            show=show_progress,
        )

        for epoch in epoch_iter:
            model.train()
            for seq_batch, static_batch, target_batch in train_loader:
                seq_batch = _to_device(seq_batch.float(), device)
                static_batch = _to_device(static_batch.float(), device)
                target_batch = _to_device(target_batch.float(), device)

                optimizer.zero_grad()
                logits = model(seq_batch, static_batch)
                bce_loss = criterion(logits, target_batch)
                pairwise_loss = _pairwise_auc_loss(logits, target_batch)
                loss = bce_loss + PAIRWISE_WEIGHT * pairwise_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for seq_batch, static_batch, target_batch in val_loader:
                    seq_batch = _to_device(seq_batch.float(), device)
                    static_batch = _to_device(static_batch.float(), device)
                    logits = model(seq_batch, static_batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.append(probs)
                    val_targets.append(target_batch.numpy())
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_auc = roc_auc_score(val_targets, val_preds)
            stop_early = False
            if val_auc > best_auc + EARLY_STOP_DELTA:
                best_auc = val_auc
                best_epoch = epoch
                best_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    logger.info(
                        "Early stopping triggered on fold %d at epoch %d",
                        fold_idx,
                        epoch,
                    )
                    stop_early = True

            scheduler.step()
            if stop_early:
                break

        logger.info(
            "Fold %d finished with AUC %.4f at epoch %d",
            fold_idx,
            best_auc,
            best_epoch,
        )

        if best_state is not None:
            model.load_state_dict(best_state)
        best_epochs.append(best_epoch)

        metrics[f"sequence_fold_{fold_idx}_auc"] = best_auc

        model.eval()
        val_loader = make_loader(val_mask, shuffle=False)
        preds = []
        with torch.no_grad():
            for seq_batch, static_batch, _ in val_loader:
                seq_batch = _to_device(seq_batch.float(), device)
                static_batch = _to_device(static_batch.float(), device)
                logits = model(seq_batch, static_batch)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        oof[val_mask] = np.concatenate(preds)

    valid_oof_mask = ~np.isnan(oof)
    if valid_oof_mask.any():
        overall_auc = roc_auc_score(
            y.to_numpy(dtype=np.float32)[valid_oof_mask],
            oof[valid_oof_mask],
        )
        metrics["sequence_overall_auc"] = overall_auc
        logger.info(
            "Cross-validation overall AUC: %.4f (computed on %d validation samples)",
            overall_auc,
            int(valid_oof_mask.sum()),
        )
    else:
        metrics["sequence_overall_auc"] = float("nan")
        logger.warning("Cross-validation overall AUC skipped (no validation samples)")

    best_epoch_final = int(np.median(best_epochs)) if best_epochs else max_epochs
    best_epoch_final = max(best_epoch_final, 3)
    logger.info("Training final model for %d epochs", best_epoch_final)

    final_model = SequenceModel(seq_channels=seq_train.shape[1], static_dim=training_matrix_full.shape[1])
    final_model.to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=tune_lr, weight_decay=1e-4)
    scheduler_final = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, best_epoch_final),
    )
    global_pos_rate = float(training["target"].mean())
    pos_weight_value = (1.0 - global_pos_rate) / max(global_pos_rate, 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    seq_tensor_full = torch.from_numpy(seq_train).float()
    static_tensor_full = torch.from_numpy(training_matrix_full).float()
    target_tensor_full = torch.from_numpy(y_array).float()
    full_dataset = TensorDataset(seq_tensor_full, static_tensor_full, target_tensor_full)

    num_pos_full = (target_tensor_full > 0.5).sum().item()
    num_neg_full = target_tensor_full.numel() - num_pos_full
    if num_pos_full > 0 and num_neg_full > 0:
        pos_w = 0.5 / num_pos_full
        neg_w = 0.5 / num_neg_full
        weights_full = torch.where(
            target_tensor_full > 0.5,
            torch.full_like(target_tensor_full, pos_w, dtype=torch.double),
            torch.full_like(target_tensor_full, neg_w, dtype=torch.double),
        )
        sampler_full = WeightedRandomSampler(weights_full, num_samples=len(full_dataset), replacement=True)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler_full)
    else:
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    final_epoch_iter = _progress(
        range(best_epoch_final),
        desc="Final model epochs",
        show=show_progress,
    )

    for _ in final_epoch_iter:
        final_model.train()
        for seq_batch, static_batch, target_batch in full_loader:
            seq_batch = _to_device(seq_batch, device)
            static_batch = _to_device(static_batch, device)
            target_batch = _to_device(target_batch, device)

            optimizer.zero_grad()
            logits = final_model(seq_batch, static_batch)
            bce_loss = criterion(logits, target_batch)
            pairwise_loss = _pairwise_auc_loss(logits, target_batch)
            loss = bce_loss + PAIRWISE_WEIGHT * pairwise_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler_final.step()

    final_model.eval()
    logger.info("Generating inference predictions")
    with torch.no_grad():
        seq_tensor = torch.from_numpy(seq_infer).float().to(device)
        static_tensor = torch.from_numpy(inference_matrix).float().to(device)
        logits = final_model(seq_tensor, static_tensor)
        predictions = torch.sigmoid(logits).cpu().numpy()

    oof_path = FEATURE_CACHE_DIR / "sequence_oof.npy"
    np.save(oof_path, oof.astype(np.float32))
    model_path = MODELS_DIR / "sequence_model.pt"
    torch.save(final_model.state_dict(), model_path)

    preds_df = inference[["airport_group", "date", "hour"]].copy()
    preds_df["pred"] = predictions

    metrics_path = FEATURE_CACHE_DIR / "sequence_metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)

    logger.info("Sequence pipeline completed. Model saved to %s", model_path)

    return SequencePipelineResult(
        predictions=preds_df,
        metrics=metrics,
        oof_predictions=oof,
        model_path=model_path,
    )
