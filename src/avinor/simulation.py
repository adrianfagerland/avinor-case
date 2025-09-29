from __future__ import annotations

import hashlib
import math
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

from .feature_engineering import (
    build_base_index,
    build_minute_sequences,
    compute_overlap_features,
)
from .models.delay_cancellation import CancellationEstimator, DelaySampler


def prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.copy()
    df["std"] = pd.to_datetime(df["std"], errors="coerce")
    df["sta"] = pd.to_datetime(df["sta"], errors="coerce")
    df["service_type"] = df["service_type"].fillna("unknown")
    return df


def build_simulated_events(
    schedule: pd.DataFrame,
    dep_delays: np.ndarray,
    arr_delays: np.ndarray,
    cancellations: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "flight_id",
        "service_type",
        "airport_group",
        "scheduled_time",
        "actual_time",
        "movement",
        "base_time",
    ]

    if not len(schedule) or not np.any(~cancellations):
        return pd.DataFrame(columns=columns)

    active_mask = ~cancellations
    active_schedule = schedule.loc[active_mask].copy()
    if active_schedule.empty:
        return pd.DataFrame(columns=columns)

    # Ensure service type is always available for downstream merges.
    active_schedule["service_type"] = active_schedule["service_type"].fillna("unknown")

    dep_series = dep_delays[active_mask]
    arr_series = arr_delays[active_mask]

    def _build_subset(
        group_col: str,
        time_col: str,
        delays: np.ndarray,
        movement: str,
    ) -> pd.DataFrame:
        if group_col not in active_schedule or time_col not in active_schedule:
            return pd.DataFrame(columns=columns)

        group_values = active_schedule[group_col]
        time_values = active_schedule[time_col]

        valid_mask = (
            group_values.notna()
            & (group_values != "NA")
            & time_values.notna()
        )
        if not valid_mask.any():
            return pd.DataFrame(columns=columns)

        subset = active_schedule.loc[valid_mask, [
            "flight_id",
            "service_type",
            group_col,
            time_col,
        ]].copy()

        scheduled = pd.to_datetime(subset[time_col])
        actual = scheduled + pd.to_timedelta(delays[valid_mask], unit="m")

        subset.rename(columns={group_col: "airport_group", time_col: "scheduled_time"}, inplace=True)
        subset["scheduled_time"] = scheduled
        subset["actual_time"] = actual
        subset["movement"] = movement
        subset["base_time"] = actual

        return subset[[
            "flight_id",
            "service_type",
            "airport_group",
            "scheduled_time",
            "actual_time",
            "movement",
            "base_time",
        ]]

    dep_events = _build_subset("dep_airport_group", "std", dep_series, "dep")
    arr_events = _build_subset("arr_airport_group", "sta", arr_series, "arr")

    if dep_events.empty and arr_events.empty:
        return pd.DataFrame(columns=columns)

    return pd.concat([dep_events, arr_events], ignore_index=True, copy=False)


SIM_METRICS = [
    "overlap_max",
    "overlap_mean",
    "overlap_median",
    "overlap_p90",
    "overlap_minutes_ge2",
    "overlap_minutes_ge3",
    "overlap_pairs_per_hour",
    "overlap_near_conflict_minutes",
]


def run_monte_carlo(
    schedule: pd.DataFrame,
    base_index: pd.DataFrame,
    delay_sampler: DelaySampler,
    cancellation_model: CancellationEstimator,
    n_simulations: int = 200,
    random_state: int = 42,
    progress_name: str | None = None,
    *,
    min_simulations: int | None = None,
    max_error: float = 0.003,
    adaptive: bool = True,
    stratified: bool = True,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    schedule = prepare_schedule(schedule)
    if schedule.empty:
        empty = base_index[["airport_group", "date", "hour"]].reset_index(drop=True).copy()
        empty["simulations_used"] = 0
        return empty

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")

    seq_len = 60
    min_required = min_simulations or min(80, n_simulations)
    min_required = max(1, min(min_required, n_simulations))

    max_workers = n_jobs
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = max(1, min(4, cpu // 2))
    if progress_name is not None:
        message = (
            f"[simulation] {progress_name}: using {max_workers} worker"
            f"{'s' if max_workers != 1 else ''} (target={n_simulations})"
        )
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)
    base_context = _precompute_base_context(base_index)

    # Shared low-discrepancy uniforms for stratified sampling and common random numbers.
    uniforms_cancel = _generate_shared_uniforms(schedule, n_simulations, random_state, salt=11, stratified=stratified)
    uniforms_dep = _generate_shared_uniforms(schedule, n_simulations, random_state + 1, salt=23, stratified=stratified)
    uniforms_arr = _generate_shared_uniforms(schedule, n_simulations, random_state + 2, salt=37, stratified=stratified)

    seed_sequence = np.random.SeedSequence(random_state)
    child_seeds = seed_sequence.spawn(n_simulations)

    metric_stats: Dict[str, Dict[str, np.ndarray]] = {}
    sim_count = 0
    converged = False

    progress = None
    if tqdm is not None and progress_name is not None:
        progress = tqdm(total=n_simulations, desc=progress_name, unit="sim", leave=False)

    def update_metric(name: str, values: np.ndarray, count: int) -> None:
        if name not in metric_stats:
            metric_stats[name] = {
                "mean": np.zeros_like(values, dtype=np.float64),
                "M2": np.zeros_like(values, dtype=np.float64),
            }
        stats = metric_stats[name]
        delta = values.astype(np.float64, copy=False) - stats["mean"]
        stats["mean"] += delta / count
        stats["M2"] += delta * (values.astype(np.float64, copy=False) - stats["mean"])

    def simulate_once(sim_idx: int) -> Dict[str, np.ndarray]:
        rng_local = np.random.default_rng(child_seeds[sim_idx])
        cancellations = cancellation_model.sample(
            schedule,
            rng_local,
            uniform_draws=uniforms_cancel[sim_idx],
        )
        dep_delays = delay_sampler.sample(
            schedule,
            "dep",
            rng_local,
            uniform_draws=uniforms_dep[sim_idx],
        )
        arr_delays = delay_sampler.sample(
            schedule,
            "arr",
            rng_local,
            uniform_draws=uniforms_arr[sim_idx],
        )

        events = build_simulated_events(schedule, dep_delays, arr_delays, cancellations)
        overlap = compute_overlap_features(base_index, events, context=base_context)

        metrics: Dict[str, np.ndarray] = {}
        for metric in SIM_METRICS:
            metrics[f"sim_{metric}"] = overlap[metric].to_numpy(dtype=np.float32)

        prob_any = (overlap["overlap_minutes_ge2"].to_numpy(dtype=np.float32) > 0).astype(np.float32)
        metrics["sim_prob_any_overlap"] = prob_any

        sequences, dep_seq, arr_seq = build_minute_sequences(
            base_index,
            events,
            seq_len=seq_len,
            context=base_context,
        )
        sequences = sequences.astype(np.float32, copy=False)
        dep_seq = dep_seq.astype(np.float32, copy=False)
        arr_seq = arr_seq.astype(np.float32, copy=False)

        metrics["sim_concurrency_mean"] = sequences.mean(axis=1)
        metrics["sim_concurrency_peak"] = sequences.max(axis=1)
        metrics["sim_concurrency_std"] = sequences.std(axis=1)
        metrics["sim_concurrency_area"] = sequences.sum(axis=1)
        metrics["sim_dep_activity"] = dep_seq.sum(axis=1)
        metrics["sim_arr_activity"] = arr_seq.sum(axis=1)

        return metrics

    def process_metrics(result: Dict[str, np.ndarray], count: int) -> None:
        for name, values in result.items():
            update_metric(name, values, count)

    if max_workers == 1:
        for sim_idx in range(n_simulations):
            metrics = simulate_once(sim_idx)
            sim_count += 1
            process_metrics(metrics, sim_count)
            if progress is not None:
                progress.update(1)
            if adaptive and sim_count >= min_required and _has_converged(metric_stats, sim_count, max_error):
                converged = True
                break
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: set = set()
            next_idx = 0

            while (next_idx < n_simulations or in_flight) and not converged:
                while next_idx < n_simulations and len(in_flight) < max_workers and not converged:
                    future = executor.submit(simulate_once, next_idx)
                    in_flight.add(future)
                    next_idx += 1

                if not in_flight:
                    break

                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                in_flight = set(in_flight)
                for future in done:
                    metrics = future.result()
                    sim_count += 1
                    process_metrics(metrics, sim_count)
                    if progress is not None:
                        progress.update(1)
                    if adaptive and sim_count >= min_required and _has_converged(metric_stats, sim_count, max_error):
                        converged = True
                        break

            # Attempt to cancel outstanding futures to shorten executor shutdown.
            for future in in_flight:
                future.cancel()

    if progress is not None:
        # If we converged early, advance to total for visual consistency.
        progress.update(max(0, n_simulations - progress.n))
        progress.close()

    if sim_count == 0:
        empty = base_index[["airport_group", "date", "hour"]].reset_index(drop=True).copy()
        empty["simulations_used"] = 0
        return empty

    summary = base_index[["airport_group", "date", "hour"]].reset_index(drop=True).copy()
    for name, stats in metric_stats.items():
        mean = stats["mean"].astype(np.float32)
        if sim_count > 1:
            variance = stats["M2"] / (sim_count - 1)
        else:
            variance = np.zeros_like(stats["mean"])
        std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        summary[f"{name}_mean"] = mean
        summary[f"{name}_std"] = std

    if "sim_prob_any_overlap_mean" in summary.columns:
        summary["sim_prob_any_overlap"] = summary["sim_prob_any_overlap_mean"]

    summary["simulations_used"] = sim_count
    return summary


def _precompute_base_context(base_index: pd.DataFrame) -> Dict[str, np.ndarray]:
    context: Dict[str, np.ndarray] = {}
    context["group_indices"] = base_index.groupby("airport_group", sort=False).indices
    if "interval_start" in base_index:
        context["interval_start_ns"] = base_index["interval_start"].astype("int64").to_numpy()
    else:
        starts = pd.to_datetime(base_index["date"]) + pd.to_timedelta(base_index["hour"], unit="h")
        context["interval_start_ns"] = starts.astype("int64").to_numpy()
    return context


def _series_to_unit(values: pd.Series, salt: int) -> np.ndarray:
    arr = values.fillna("").astype(str).to_numpy()
    out = np.empty(len(arr), dtype=np.float64)
    salt_bytes = str(salt).encode("utf-8")
    for idx, val in enumerate(arr):
        digest = hashlib.blake2b(val.encode("utf-8"), digest_size=8, person=salt_bytes)
        out[idx] = int.from_bytes(digest.digest(), "little") / 2**64
    return out.astype(np.float32)


def _generate_shared_uniforms(
    schedule: pd.DataFrame,
    n_simulations: int,
    random_state: int,
    *,
    salt: int,
    stratified: bool,
) -> np.ndarray:
    rng = np.random.default_rng(random_state + salt)
    base = (np.arange(n_simulations, dtype=np.float64) + 0.5) / n_simulations

    if "flight_id" in schedule:
        offsets = _series_to_unit(schedule["flight_id"], salt)
    else:
        offsets = _series_to_unit(pd.Series(schedule.index.astype(str)), salt)

    draws = (base[:, None] + offsets[None, :]) % 1.0

    if stratified:
        jitter = rng.uniform(-0.5 / n_simulations, 0.5 / n_simulations, size=draws.shape)
        draws = np.clip(draws + jitter, 1e-6, 1 - 1e-6)
    else:
        draws = rng.random(size=draws.shape, dtype=np.float64)

    return draws.astype(np.float32)


def _has_converged(
    metric_stats: Dict[str, Dict[str, np.ndarray]],
    sim_count: int,
    max_error: float,
) -> bool:
    if sim_count < 2:
        return False
    prob_stats = metric_stats.get("sim_prob_any_overlap")
    if prob_stats is None:
        return False
    variance = prob_stats["M2"] / max(sim_count - 1, 1)
    stderr = np.sqrt(np.maximum(variance, 0.0)) / math.sqrt(sim_count)
    return float(np.nanmax(stderr)) <= max_error


def _resolve_workers(n_jobs: int | None) -> int:
    if n_jobs is not None:
        return max(1, n_jobs)
    cpu = os.cpu_count() or 1
    if cpu <= 1:
        return 1
    return max(1, min(8, cpu - 1))


__all__ = [
    "run_monte_carlo",
    "build_simulated_events",
    "prepare_schedule",
    "SIM_METRICS",
]
