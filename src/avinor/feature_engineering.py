from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from .calendar_features import add_calendar_features
from .feature_utils import add_cyclical_features, safe_divide

MINUTE_NS = 60 * 10**9
OPERATIONS_WINDOW = {
    "dep": (-15, 8),
    "arr": (-16, 5),
}


@dataclass
class OverlapMetrics:
    max_concurrency: float
    mean_concurrency: float
    median_concurrency: float
    p90_concurrency: float
    minutes_ge2: float
    minutes_ge3: float
    overlap_pairs_per_hour: float
    near_conflict_minutes: float
    min_gap_minutes: float
    near_conflict_flag: int


SEASON_MAP = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


# ---------------------------------------------------------------------------
# Flight events
# ---------------------------------------------------------------------------


def build_event_table(flights: pd.DataFrame, *, use_actual: bool) -> pd.DataFrame:
    flights = flights.copy()
    for col in ["std", "sta", "atd", "ata"]:
        if col in flights.columns:
            flights[col] = pd.to_datetime(flights[col], errors="coerce")
        else:
            flights[col] = pd.NaT

    if "cancelled" not in flights.columns:
        flights["cancelled"] = 0

    dep_columns = [
        "flight_id",
        "service_type",
        "dep_airport_group",
        "std",
        "atd",
        "cancelled",
    ]
    dep = flights[dep_columns].rename(
        columns={
            "dep_airport_group": "airport_group",
            "std": "scheduled_time",
            "atd": "actual_time",
        }
    )
    dep["movement"] = "dep"

    arr_columns = [
        "flight_id",
        "service_type",
        "arr_airport_group",
        "sta",
        "ata",
        "cancelled",
    ]
    arr = flights[arr_columns].rename(
        columns={
            "arr_airport_group": "airport_group",
            "sta": "scheduled_time",
            "ata": "actual_time",
        }
    )
    arr["movement"] = "arr"

    events = pd.concat([dep, arr], ignore_index=True)
    events = events[events["airport_group"].notna()]
    events = events[events["airport_group"] != "NA"]
    events = events[events["cancelled"] == 0]

    base_time = events["actual_time"] if use_actual else events["scheduled_time"]
    events["base_time"] = base_time.where(base_time.notna(), events["scheduled_time"])
    events.drop(columns=["cancelled"], inplace=True)
    events.dropna(subset=["base_time"], inplace=True)

    return events


# ---------------------------------------------------------------------------
# Base index construction
# ---------------------------------------------------------------------------


def build_base_index(df: pd.DataFrame) -> pd.DataFrame:
    base = df[["airport_group", "date", "hour"]].drop_duplicates().copy()
    base.sort_values(["date", "hour", "airport_group"], inplace=True)
    base.reset_index(drop=True, inplace=True)

    interval_start = pd.to_datetime(base["date"]) + pd.to_timedelta(base["hour"], unit="h")
    base["interval_start"] = interval_start
    base["interval_end"] = base["interval_start"] + pd.Timedelta(hours=1)

    add_cyclical_features(base, "hour", 24, prefix="hour")
    base["dayofweek"] = base["interval_start"].dt.dayofweek.astype("int8")
    add_cyclical_features(base, "dayofweek", 7, prefix="dow")
    base["month"] = base["interval_start"].dt.month.astype("int8")
    add_cyclical_features(base, "month", 12, prefix="month")

    calendar = add_calendar_features(base, "interval_start")
    base = pd.concat([base, calendar], axis=1)
    base["season"] = base["month"].map(SEASON_MAP).astype("category")
    return base


# ---------------------------------------------------------------------------
# Schedule features
# ---------------------------------------------------------------------------


def _counts_for_group(
    subset: pd.DataFrame,
    start_minutes: np.ndarray,
    horizons_next: Iterable[int],
    horizons_prev: Iterable[int],
) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    for movement in ("dep", "arr"):
        minutes = (
            subset.loc[subset["movement"] == movement, "base_time"].dt.floor("min").astype("int64")
        ) // MINUTE_NS
        minutes = np.sort(minutes.astype(np.int64))
        for horizon in horizons_next:
            right = np.searchsorted(minutes, start_minutes + horizon, side="left")
            left = np.searchsorted(minutes, start_minutes, side="left")
            results[f"{movement}_next_{horizon}m"] = (right - left).astype(np.int16)
        for horizon in horizons_prev:
            right = np.searchsorted(minutes, start_minutes, side="left")
            left = np.searchsorted(minutes, start_minutes - horizon, side="left")
            results[f"{movement}_prev_{horizon}m"] = (right - left).astype(np.int16)
    return results


def compute_schedule_features(base: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=base.index)

    hourly_columns = [
        "scheduled_departures",
        "scheduled_arrivals",
        "scheduled_movements",
    ]
    for col in hourly_columns:
        features[col] = 0

    window_keys = [
        "dep_next_30m",
        "dep_next_60m",
        "dep_next_90m",
        "dep_prev_30m",
        "dep_prev_60m",
        "arr_next_30m",
        "arr_next_60m",
        "arr_next_90m",
        "arr_prev_30m",
        "arr_prev_60m",
    ]
    for key in window_keys:
        features[f"scheduled_{key}"] = 0

    for group, subset in events.groupby("airport_group"):
        mask = base["airport_group"] == group
        if not mask.any():
            continue

        interval_hours = base.loc[mask, "interval_start"].dt.floor("h")
        start_minutes = (
            base.loc[mask, "interval_start"].astype("int64") // MINUTE_NS
        ).astype(np.int64)

        hourly = (
            subset.assign(hour_bucket=subset["base_time"].dt.floor("h"))
            .groupby(["hour_bucket", "movement"])
            .size()
            .unstack(fill_value=0)
        )
        hourly["total"] = hourly.sum(axis=1)
        aligned = hourly.reindex(interval_hours).fillna(0)

        features.loc[mask, "scheduled_departures"] = aligned.get("dep", 0).to_numpy()
        features.loc[mask, "scheduled_arrivals"] = aligned.get("arr", 0).to_numpy()
        features.loc[mask, "scheduled_movements"] = aligned["total"].to_numpy()

        counts = _counts_for_group(
            subset,
            start_minutes=start_minutes,
            horizons_next=(30, 60, 90),
            horizons_prev=(30, 60),
        )
        for key, values in counts.items():
            features.loc[mask, f"scheduled_{key}"] = values

    features["scheduled_dep_ratio"] = safe_divide(
        features["scheduled_departures"], features["scheduled_movements"]
    )
    features["scheduled_arr_ratio"] = safe_divide(
        features["scheduled_arrivals"], features["scheduled_movements"]
    )
    features["scheduled_movements_prev30_ratio"] = safe_divide(
        features["scheduled_dep_prev_30m"] + features["scheduled_arr_prev_30m"], 30
    )
    return features


# ---------------------------------------------------------------------------
# Overlap features via minute-level timeline
# ---------------------------------------------------------------------------


def _with_operational_windows(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    offsets = events["movement"].map(OPERATIONS_WINDOW)
    start_offsets = offsets.apply(lambda x: x[0])
    end_offsets = offsets.apply(lambda x: x[1])
    events["window_start"] = events["base_time"] + pd.to_timedelta(start_offsets, unit="m")
    events["window_end"] = events["base_time"] + pd.to_timedelta(end_offsets, unit="m")
    events["window_start"] = events["window_start"].dt.floor("min")
    events["window_end"] = events["window_end"].dt.floor("min")
    return events


def compute_overlap_features(
    base: pd.DataFrame,
    events: pd.DataFrame,
    context: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    feature_names = [
        "overlap_max",
        "overlap_mean",
        "overlap_median",
        "overlap_p90",
        "overlap_minutes_ge2",
        "overlap_minutes_ge3",
        "overlap_pairs_per_hour",
        "overlap_near_conflict_minutes",
        "overlap_min_gap_minutes",
        "overlap_near_conflict_flag",
    ]
    features = pd.DataFrame(0.0, index=base.index, columns=feature_names)
    features["overlap_min_gap_minutes"] = 999.0

    if events.empty:
        return features

    events = _with_operational_windows(events)
    events["start_minute"] = (
        events["window_start"].astype("int64") // MINUTE_NS
    ).astype(np.int64)
    events["end_minute"] = (
        events["window_end"].astype("int64") // MINUTE_NS
    ).astype(np.int64)
    if context is not None:
        group_indices = context.get("group_indices")
        interval_start_ns = context.get("interval_start_ns")
    if context is None or group_indices is None or interval_start_ns is None:
        group_indices = base.groupby("airport_group", sort=False).indices
        interval_start_ns = base["interval_start"].astype("int64").to_numpy()
    feature_positions = {name: idx for idx, name in enumerate(features.columns)}
    feature_values = features.to_numpy()

    for group, subset in events.groupby("airport_group", sort=False):
        indices = group_indices.get(group)
        if indices is None or subset.empty:
            continue

        idx_array = np.asarray(indices, dtype=np.int64)

        starts = subset["start_minute"].to_numpy(dtype=np.int64)
        ends = subset["end_minute"].to_numpy(dtype=np.int64)
        valid = ends > starts
        if not valid.any():
            continue
        starts = starts[valid]
        ends = ends[valid]

        hour_starts = interval_start_ns[idx_array] // MINUTE_NS
        hour_ends = hour_starts + 60

        timeline_min = int(min(starts.min(), hour_starts.min()))
        timeline_max = int(max(ends.max(), hour_ends.max()))
        diff = np.zeros(timeline_max - timeline_min + 2, dtype=np.int32)

        start_idx = starts - timeline_min
        end_idx = ends - timeline_min
        np.add.at(diff, start_idx, 1)
        np.add.at(diff, end_idx, -1)
        concurrency = np.cumsum(diff[:-1])
        pair_cache = concurrency * (concurrency - 1) / 2.0

        n_hours = len(hour_starts)
        max_arr = np.zeros(n_hours, dtype=np.float32)
        mean_arr = np.zeros(n_hours, dtype=np.float32)
        median_arr = np.zeros(n_hours, dtype=np.float32)
        p90_arr = np.zeros(n_hours, dtype=np.float32)
        ge2_arr = np.zeros(n_hours, dtype=np.float32)
        ge3_arr = np.zeros(n_hours, dtype=np.float32)
        pairs_arr = np.zeros(n_hours, dtype=np.float32)
        near_minutes_arr = np.zeros(n_hours, dtype=np.float32)
        min_gap_arr = np.full(n_hours, 999.0, dtype=np.float32)
        flag_arr = np.zeros(n_hours, dtype=np.float32)

        for i, (hs, he) in enumerate(zip(hour_starts, hour_ends)):
            start_rel = int(hs - timeline_min)
            end_rel = int(he - timeline_min)
            if end_rel <= start_rel:
                continue

            slice_conc = concurrency[start_rel:end_rel]
            if slice_conc.size == 0:
                continue

            max_arr[i] = slice_conc.max()
            mean_arr[i] = slice_conc.mean()
            median_arr[i] = float(np.median(slice_conc))
            p90_arr[i] = float(np.quantile(slice_conc, 0.9))
            ge2_arr[i] = np.sum(slice_conc >= 2)
            ge3_arr[i] = np.sum(slice_conc >= 3)
            pairs_arr[i] = pair_cache[start_rel:end_rel].sum() / 60.0
            near_minutes_arr[i] = np.sum((slice_conc >= 2) & (slice_conc <= 3))

            active_mask = (starts < he) & (ends > hs)
            active_starts = np.sort(starts[active_mask])
            if active_starts.size >= 2:
                min_gap = float(np.diff(active_starts).min())
                min_gap_arr[i] = min_gap
                if min_gap <= 3:
                    flag_arr[i] = 1

        feature_values[idx_array, feature_positions["overlap_max"]] = max_arr
        feature_values[idx_array, feature_positions["overlap_mean"]] = mean_arr
        feature_values[idx_array, feature_positions["overlap_median"]] = median_arr
        feature_values[idx_array, feature_positions["overlap_p90"]] = p90_arr
        feature_values[idx_array, feature_positions["overlap_minutes_ge2"]] = ge2_arr
        feature_values[idx_array, feature_positions["overlap_minutes_ge3"]] = ge3_arr
        feature_values[idx_array, feature_positions["overlap_pairs_per_hour"]] = pairs_arr
        feature_values[idx_array, feature_positions["overlap_near_conflict_minutes"]] = near_minutes_arr
        feature_values[idx_array, feature_positions["overlap_min_gap_minutes"]] = min_gap_arr
        feature_values[idx_array, feature_positions["overlap_near_conflict_flag"]] = flag_arr

    features.loc[:, :] = feature_values
    features.fillna(0.0, inplace=True)
    return features


# ---------------------------------------------------------------------------
# Historical baselines
# ---------------------------------------------------------------------------


def build_minute_sequences(
    base: pd.DataFrame,
    events: pd.DataFrame,
    seq_len: int = 60,
    context: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return minute-level concurrency, departures, and arrivals for each base row."""
    sequences = np.zeros((len(base), seq_len), dtype=np.float32)
    dep_sequences = np.zeros_like(sequences)
    arr_sequences = np.zeros_like(sequences)

    if events.empty:
        return sequences, dep_sequences, arr_sequences

    events = _with_operational_windows(events.copy())
    events.sort_values("base_time", inplace=True)

    events["start_minute"] = (
        events["window_start"].astype("int64") // MINUTE_NS
    ).astype(np.int64)
    events["end_minute"] = (
        events["window_end"].astype("int64") // MINUTE_NS
    ).astype(np.int64)

    if context is not None:
        group_indices = context.get("group_indices")
        interval_start_ns = context.get("interval_start_ns")
    else:
        group_indices = None
        interval_start_ns = None

    if group_indices is None:
        group_indices = base.groupby("airport_group", sort=False).indices
    if interval_start_ns is None:
        interval_start_ns = base["interval_start"].astype("int64").to_numpy()

    for group, subset in events.groupby("airport_group"):
        indices = group_indices.get(group)
        if indices is None or subset.empty:
            continue

        idx_array = np.asarray(indices, dtype=np.int64)
        hour_starts = interval_start_ns[idx_array] // MINUTE_NS
        hour_ends = hour_starts + seq_len

        starts = subset["start_minute"].to_numpy(dtype=np.int64)
        ends = subset["end_minute"].to_numpy(dtype=np.int64)
        movements = subset["movement"].to_numpy()

        valid = ends > starts
        starts = starts[valid]
        ends = ends[valid]
        movements = movements[valid]
        if starts.size == 0:
            continue

        timeline_min = int(min(starts.min(), hour_starts.min()))
        timeline_max = int(max(ends.max(), hour_ends.max()))
        length = timeline_max - timeline_min + 2

        diff_total = np.zeros(length, dtype=np.int32)
        diff_dep = np.zeros(length, dtype=np.int32)
        diff_arr = np.zeros(length, dtype=np.int32)

        start_idx = starts - timeline_min
        end_idx = ends - timeline_min

        np.add.at(diff_total, start_idx, 1)
        np.add.at(diff_total, end_idx, -1)

        dep_mask = movements == "dep"
        arr_mask = movements == "arr"
        if dep_mask.any():
            np.add.at(diff_dep, start_idx[dep_mask], 1)
            np.add.at(diff_dep, end_idx[dep_mask], -1)
        if arr_mask.any():
            np.add.at(diff_arr, start_idx[arr_mask], 1)
            np.add.at(diff_arr, end_idx[arr_mask], -1)

        concurrency = np.cumsum(diff_total[:-1])
        dep_counts = np.cumsum(diff_dep[:-1])
        arr_counts = np.cumsum(diff_arr[:-1])

        for local_idx, (hs, he, global_idx) in enumerate(zip(hour_starts, hour_ends, idx_array)):
            start_rel = int(hs - timeline_min)
            end_rel = int(he - timeline_min)
            segment = concurrency[start_rel:end_rel]
            dep_segment = dep_counts[start_rel:end_rel]
            arr_segment = arr_counts[start_rel:end_rel]
            if segment.size != seq_len:
                padded = np.zeros(seq_len, dtype=np.float32)
                dep_padded = np.zeros(seq_len, dtype=np.float32)
                arr_padded = np.zeros(seq_len, dtype=np.float32)
                size = min(seq_len, segment.size)
                padded[:size] = segment[:size]
                dep_padded[:size] = dep_segment[:size]
                arr_padded[:size] = arr_segment[:size]
                sequences[global_idx] = padded
                dep_sequences[global_idx] = dep_padded
                arr_sequences[global_idx] = arr_padded
            else:
                sequences[global_idx] = segment.astype(np.float32)
                dep_sequences[global_idx] = dep_segment.astype(np.float32)
                arr_sequences[global_idx] = arr_segment.astype(np.float32)

    return sequences, dep_sequences, arr_sequences




def compute_historical_baselines(base: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.DataFrame(index=base.index)
    hist = training_df.copy()
    hist["month"] = hist["date"].dt.month
    hist["dayofweek"] = hist["date"].dt.dayofweek

    group_hour = hist.groupby(["airport_group", "hour"])["target"].agg(["mean", "count"])
    idx = pd.MultiIndex.from_arrays([base["airport_group"], base["hour"]])
    aligned = group_hour.reindex(idx)
    baseline["hist_rate_group_hour"] = aligned["mean"].fillna(group_hour["mean"].mean()).to_numpy()
    baseline["hist_support_group_hour"] = aligned["count"].fillna(0).to_numpy()

    month_hour = hist.groupby(["month", "hour"])["target"].mean()
    idx_month = pd.MultiIndex.from_arrays([base["month"], base["hour"]])
    baseline["hist_rate_month_hour"] = month_hour.reindex(idx_month).fillna(month_hour.mean()).to_numpy()

    dow_hour = hist.groupby(["dayofweek", "hour"])["target"].mean()
    idx_dow = pd.MultiIndex.from_arrays([base["dayofweek"], base["hour"]])
    baseline["hist_rate_dow_hour"] = dow_hour.reindex(idx_dow).fillna(dow_hour.mean()).to_numpy()

    return baseline


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------


def build_feature_matrix(
    base_index: pd.DataFrame,
    schedule_events: pd.DataFrame,
    training_df: pd.DataFrame | None = None,
    include_overlap: bool = True,
) -> pd.DataFrame:
    events = schedule_events.copy()
    schedule_features = compute_schedule_features(base_index, events)
    blocks = [schedule_features]

    if include_overlap:
        overlap = compute_overlap_features(base_index, events)
        blocks.append(overlap)

    if training_df is not None:
        historical = compute_historical_baselines(base_index, training_df)
        blocks.append(historical)

    features = pd.concat(blocks, axis=1)
    features.fillna(0.0, inplace=True)
    return features


def _hours_since_positive(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float32)
    last_positive = -1
    for idx, val in enumerate(values):
        if not np.isnan(val) and val >= 0.5:
            last_positive = idx
        if last_positive >= 0:
            result[idx] = float(idx - last_positive)
    return result


def compute_recent_target_features(
    training_df: pd.DataFrame,
    windows: Sequence[int] = (24, 72, 168),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if training_df.empty:
        empty = training_df[["airport_group", "date", "hour"]].copy()
        return empty, empty

    df = training_df.sort_values(["airport_group", "date", "hour"]).copy()
    group = df.groupby("airport_group", group_keys=False)

    feature_columns: list[str] = []
    shifted_target = group["target"].shift(1)
    df["target_prev_hour"] = shifted_target
    feature_columns.append("target_prev_hour")

    for window in windows:
        window = int(window)
        col = f"target_rolling_mean_{window}h"
        df[col] = (
            shifted_target
            .rolling(window=window, min_periods=max(1, window // 4))
            .mean()
        )
        feature_columns.append(col)
        std_col = f"target_rolling_std_{window}h"
        df[std_col] = (
            shifted_target
            .rolling(window=window, min_periods=max(2, window // 4))
            .std()
        )
        feature_columns.append(std_col)

    df["hours_since_positive"] = group["target"].transform(
        lambda s: _hours_since_positive(s.shift(1).to_numpy())
    )
    feature_columns.append("hours_since_positive")

    recent_features = df[["airport_group", "date", "hour", *feature_columns]].copy()

    defaults = (
        recent_features.sort_values(["airport_group", "hour", "date"])
        .groupby(["airport_group", "hour"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return recent_features, defaults


__all__ = [
    "build_event_table",
    "build_base_index",
    "build_feature_matrix",
    "build_minute_sequences",
    "compute_recent_target_features",
]
