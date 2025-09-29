from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class DelayDistributions:
    by_service_context: Dict[Tuple[str, str, str, int, str], np.ndarray]
    by_group_context: Dict[Tuple[str, str, int, str], np.ndarray]
    by_movement_context: Dict[Tuple[str, int, str], np.ndarray]
    by_service: Dict[Tuple[str, str, str, int], np.ndarray]
    by_group: Dict[Tuple[str, str, int], np.ndarray]
    by_movement_hour: Dict[Tuple[str, int], np.ndarray]
    by_movement: Dict[str, np.ndarray]


class DelaySampler:
    def __init__(
        self,
        clip_low: float = -60.0,
        clip_high: float = 180.0,
        block_bucket_edges: Sequence[float] = (60, 120, 180, 240, 360),
        congestion_bucket_edges: Sequence[int] = (5, 10, 20, 40),
    ) -> None:
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.distributions: DelayDistributions | None = None

        block_edges = sorted(float(edge) for edge in block_bucket_edges if edge > 0)
        self.block_bucket_edges: tuple[float, ...] = tuple(block_edges)
        block_labels = [f"<= {int(edge)}" for edge in block_edges]
        block_labels.append(f"> {int(block_edges[-1])}" if block_edges else "> 0")
        self.block_bucket_labels: tuple[str, ...] = tuple(block_labels)

        congestion_edges = sorted(int(edge) for edge in congestion_bucket_edges if edge > 0)
        self.congestion_bucket_edges: tuple[int, ...] = tuple(congestion_edges)
        congestion_labels = [f"<= {edge}" for edge in congestion_edges]
        congestion_labels.append(f"> {congestion_edges[-1]}" if congestion_edges else "> 0")
        self.congestion_bucket_labels: tuple[str, ...] = tuple(congestion_labels)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, flights: pd.DataFrame) -> None:
        flights = flights.copy()
        flights["std"] = pd.to_datetime(flights.get("std"), errors="coerce")
        flights["sta"] = pd.to_datetime(flights.get("sta"), errors="coerce")
        flights["atd"] = pd.to_datetime(flights.get("atd"), errors="coerce")
        flights["ata"] = pd.to_datetime(flights.get("ata"), errors="coerce")
        flights["block_minutes"] = (flights["sta"] - flights["std"]).dt.total_seconds() / 60.0

        records: list[pd.DataFrame] = []
        if "dep_airport_group" in flights:
            dep = flights[flights["dep_airport_group"].notna()].copy()
            dep = dep[dep["dep_airport_group"] != "NA"]
            dep["movement"] = "dep"
            dep["group"] = dep["dep_airport_group"]
            dep["scheduled_time"] = dep["std"]
            dep["actual_time"] = dep["atd"]
            records.append(dep)
        if "arr_airport_group" in flights:
            arr = flights[flights["arr_airport_group"].notna()].copy()
            arr = arr[arr["arr_airport_group"] != "NA"]
            arr["movement"] = "arr"
            arr["group"] = arr["arr_airport_group"]
            arr["scheduled_time"] = arr["sta"]
            arr["actual_time"] = arr["ata"]
            records.append(arr)

        if not records:
            self.distributions = DelayDistributions({}, {}, {}, {}, {}, {}, {})
            return

        data = pd.concat(records, ignore_index=True)
        data = data[data["scheduled_time"].notna()]
        data = data[data["actual_time"].notna()]
        data["service_type"] = data["service_type"].fillna("unknown")
        data["hour"] = data["scheduled_time"].dt.hour
        data["delay_minutes"] = (
            (data["actual_time"] - data["scheduled_time"]).dt.total_seconds() / 60.0
        ).clip(self.clip_low, self.clip_high)

        data["block_bucket"] = self._bucket_block_series(data["block_minutes"])
        data["congestion"] = (
            data.groupby(["group", "movement", "hour"], observed=True)["scheduled_time"].transform("count")
        )
        data["congestion_bucket"] = self._bucket_congestion_series(data["congestion"].astype(float))
        data["context_bucket"] = self._compose_context(
            data["block_bucket"], data["congestion_bucket"]
        )

        svc_context: Dict[Tuple[str, str, str, int, str], list[float]] = defaultdict(list)
        grp_context: Dict[Tuple[str, str, int, str], list[float]] = defaultdict(list)
        mv_context: Dict[Tuple[str, int, str], list[float]] = defaultdict(list)
        svc: Dict[Tuple[str, str, str, int], list[float]] = defaultdict(list)
        grp: Dict[Tuple[str, str, int], list[float]] = defaultdict(list)
        mv_hour: Dict[Tuple[str, int], list[float]] = defaultdict(list)
        mv_only: Dict[str, list[float]] = defaultdict(list)

        for row in data.itertuples(index=False):
            ctx = getattr(row, "context_bucket", "unknown") or "unknown"
            key_svc_ctx = (row.group, row.movement, row.service_type, int(row.hour), ctx)
            key_grp_ctx = (row.group, row.movement, int(row.hour), ctx)
            key_mv_ctx = (row.movement, int(row.hour), ctx)
            key_svc = (row.group, row.movement, row.service_type, int(row.hour))
            key_grp = (row.group, row.movement, int(row.hour))
            key_mv_hour = (row.movement, int(row.hour))

            value = float(row.delay_minutes)
            svc_context[key_svc_ctx].append(value)
            grp_context[key_grp_ctx].append(value)
            mv_context[key_mv_ctx].append(value)
            svc[key_svc].append(value)
            grp[key_grp].append(value)
            mv_hour[key_mv_hour].append(value)
            mv_only[row.movement].append(value)

        def to_array(storage: Dict[Tuple, list[float]]) -> Dict[Tuple, np.ndarray]:
            return {
                key: np.sort(np.array(values, dtype=np.float32))
                for key, values in storage.items()
                if len(values) >= 5
            }

        self.distributions = DelayDistributions(
            by_service_context=to_array(svc_context),
            by_group_context=to_array(grp_context),
            by_movement_context=to_array(mv_context),
            by_service=to_array(svc),
            by_group=to_array(grp),
            by_movement_hour=to_array(mv_hour),
            by_movement={k: np.sort(np.array(v, dtype=np.float32)) for k, v in mv_only.items()},
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        schedule: pd.DataFrame,
        movement: str,
        rng: np.random.Generator,
        uniform_draws: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.distributions is None:
            raise RuntimeError("DelaySampler must be fitted before sampling")

        delays = np.zeros(len(schedule), dtype=np.float32)
        if len(schedule) == 0:
            return delays

        if movement == "dep":
            group_col = "dep_airport_group"
            time_col = "std"
        else:
            group_col = "arr_airport_group"
            time_col = "sta"

        group_values = schedule[group_col].to_numpy()
        service_values = schedule["service_type"].fillna("unknown").to_numpy()
        time_series = pd.to_datetime(schedule[time_col], errors="coerce")
        time_hours = time_series.dt.hour.to_numpy()
        valid_mask = (
            pd.notna(group_values)
            & (group_values != "NA")
            & pd.notna(time_series).to_numpy()
        )
        valid_indices = np.where(valid_mask)[0]

        block_minutes = (pd.to_datetime(schedule.get("sta"), errors="coerce") - pd.to_datetime(schedule.get("std"), errors="coerce")).dt.total_seconds() / 60.0
        block_buckets = np.array([
            self._bucket_block_time(val) for val in np.asarray(block_minutes)
        ], dtype=object)

        congestion_counts: Dict[Tuple[str, int], int] = {}
        for idx in valid_indices:
            key = (group_values[idx], int(time_hours[idx]))
            congestion_counts[key] = congestion_counts.get(key, 0) + 1

        for idx in valid_indices:
            group = group_values[idx]
            service = service_values[idx] or "unknown"
            hour = int(time_hours[idx])
            block_bucket = str(block_buckets[idx]) if idx < block_buckets.size else "unknown"
            congestion = congestion_counts.get((group, hour), 1)
            congestion_bucket = self._bucket_congestion_value(congestion)
            context_bucket = self._compose_context(block_bucket, congestion_bucket)

            dist = self._lookup(group, movement, service, hour, context_bucket)
            if dist.size == 0:
                continue

            if uniform_draws is not None and idx < uniform_draws.size:
                u = float(np.clip(uniform_draws[idx], 1e-6, 1 - 1e-6))
                pos = int(min(len(dist) - 1, np.floor(u * len(dist))))
                delays[idx] = dist[pos]
            else:
                delays[idx] = rng.choice(dist)

        return delays

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup(
        self,
        group: str,
        movement: str,
        service_type: str,
        hour: int,
        context_bucket: str,
    ) -> np.ndarray:
        if self.distributions is None:
            raise RuntimeError("DelaySampler must be fitted before sampling")

        ctx_key = (group, movement, service_type, hour, context_bucket)
        if ctx_key in self.distributions.by_service_context:
            return self.distributions.by_service_context[ctx_key]

        grp_ctx_key = (group, movement, hour, context_bucket)
        if grp_ctx_key in self.distributions.by_group_context:
            return self.distributions.by_group_context[grp_ctx_key]

        mv_ctx_key = (movement, hour, context_bucket)
        if mv_ctx_key in self.distributions.by_movement_context:
            return self.distributions.by_movement_context[mv_ctx_key]

        svc_key = (group, movement, service_type, hour)
        if svc_key in self.distributions.by_service:
            return self.distributions.by_service[svc_key]

        grp_key = (group, movement, hour)
        if grp_key in self.distributions.by_group:
            return self.distributions.by_group[grp_key]

        mv_hour_key = (movement, hour)
        if mv_hour_key in self.distributions.by_movement_hour:
            return self.distributions.by_movement_hour[mv_hour_key]

        return self.distributions.by_movement.get(movement, np.array([0.0], dtype=np.float32))

    def _bucket_block_series(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(index=series.index, dtype="object")
        if not self.block_bucket_edges:
            result = pd.Series("unknown", index=series.index, dtype="object")
            result.loc[series.notna()] = self.block_bucket_labels[-1]
            return result
        buckets = (-np.inf, *self.block_bucket_edges, np.inf)
        categorized = pd.cut(series, bins=buckets, labels=self.block_bucket_labels, include_lowest=True)
        return categorized.astype(str).fillna("unknown")

    def _bucket_congestion_series(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(index=series.index, dtype="object")
        if not self.congestion_bucket_edges:
            result = pd.Series("unknown", index=series.index, dtype="object")
            result.loc[series.notna()] = self.congestion_bucket_labels[-1]
            return result
        buckets = (-np.inf, *self.congestion_bucket_edges, np.inf)
        categorized = pd.cut(series, bins=buckets, labels=self.congestion_bucket_labels, include_lowest=True)
        return categorized.astype(str).fillna("unknown")

    def _bucket_block_time(self, minutes: float | np.floating | None) -> str:
        if minutes is None or np.isnan(minutes) or minutes <= 0:
            return "unknown"
        if not self.block_bucket_edges:
            return self.block_bucket_labels[-1]
        idx = int(np.digitize([minutes], self.block_bucket_edges, right=True)[0])
        idx = min(idx, len(self.block_bucket_labels) - 1)
        return self.block_bucket_labels[idx]

    def _bucket_congestion_value(self, value: float | int | None) -> str:
        if value is None or np.isnan(value) or value <= 0:
            return "unknown"
        if not self.congestion_bucket_edges:
            return self.congestion_bucket_labels[-1]
        idx = int(np.digitize([value], self.congestion_bucket_edges, right=True)[0])
        idx = min(idx, len(self.congestion_bucket_labels) - 1)
        return self.congestion_bucket_labels[idx]

    @staticmethod
    def _compose_context(block_bucket: str, congestion_bucket: str) -> str:
        return f"{block_bucket}|{congestion_bucket}"

    def fingerprint(self) -> Dict[str, object]:
        return {
            "clip_low": float(self.clip_low),
            "clip_high": float(self.clip_high),
            "block_bucket_edges": list(self.block_bucket_edges),
            "congestion_bucket_edges": list(self.congestion_bucket_edges),
            "version": 3,
        }


class CancellationEstimator:
    def __init__(
        self,
        alpha: float = 1.0,
        congestion_bucket_edges: Sequence[int] = (5, 10, 20, 40),
    ) -> None:
        self.alpha = alpha
        self.congestion_bucket_edges: tuple[int, ...] = tuple(sorted(int(edge) for edge in congestion_bucket_edges if edge > 0))
        labels = [f"<= {edge}" for edge in self.congestion_bucket_edges]
        labels.append(f"> {self.congestion_bucket_edges[-1]}" if self.congestion_bucket_edges else "> 0")
        self.congestion_bucket_labels: tuple[str, ...] = tuple(labels)

        self.by_service_bucket: Dict[Tuple[str, str, str, int, str], float] = {}
        self.by_group_bucket: Dict[Tuple[str, str, int, str], float] = {}
        self.by_movement_bucket: Dict[Tuple[str, int, str], float] = {}
        self.by_service: Dict[Tuple[str, str, str, int], float] = {}
        self.by_group: Dict[Tuple[str, str, int], float] = {}
        self.by_movement_hour: Dict[Tuple[str, int], float] = {}
        self.by_movement: Dict[str, float] = {}
        self.global_rate: float = 0.05

    @staticmethod
    def _rate(count_cancel: int, total: int, alpha: float) -> float:
        return (count_cancel + alpha) / (total + 2 * alpha)

    def fit(self, flights: pd.DataFrame) -> None:
        flights = flights.copy()
        flights["std"] = pd.to_datetime(flights.get("std"))
        flights["sta"] = pd.to_datetime(flights.get("sta"))
        flights["cancelled"] = flights.get("cancelled", 0).fillna(0).astype(int)
        flights["service_type"] = flights.get("service_type", "unknown").fillna("unknown")

        records: list[pd.DataFrame] = []
        if "dep_airport_group" in flights:
            dep = flights[flights["dep_airport_group"].notna()].copy()
            dep = dep[dep["dep_airport_group"] != "NA"]
            dep["movement"] = "dep"
            dep["airport_group"] = dep["dep_airport_group"]
            dep["scheduled_time"] = dep["std"]
            records.append(dep)
        if "arr_airport_group" in flights:
            arr = flights[flights["arr_airport_group"].notna()].copy()
            arr = arr[arr["arr_airport_group"] != "NA"]
            arr["movement"] = "arr"
            arr["airport_group"] = arr["arr_airport_group"]
            arr["scheduled_time"] = arr["sta"]
            records.append(arr)

        if not records:
            self.global_rate = 0.05
            return

        data = pd.concat(records, ignore_index=True)
        data = data[data["scheduled_time"].notna()]
        if data.empty:
            self.global_rate = 0.05
            return

        data["hour"] = data["scheduled_time"].dt.hour
        data["congestion"] = (
            data.groupby(["airport_group", "movement", "hour"], observed=True)["scheduled_time"].transform("count")
        )
        data["congestion_bucket"] = self._bucket_congestion_series(data["congestion"].astype(float))

        self.global_rate = data["cancelled"].mean() if len(data) else 0.05

        def aggregate(level_cols: Iterable[str]) -> Dict[Tuple, float]:
            grouped_counts = (
                data.groupby(list(level_cols))["cancelled"].agg(["sum", "count"]).reset_index()
            )
            result: Dict[Tuple, float] = {}
            for row in grouped_counts.itertuples(index=False):
                key = tuple(getattr(row, col) for col in grouped_counts.columns[:-2])
                rate = self._rate(int(row.sum), int(row.count), self.alpha)
                result[key] = rate
            return result

        self.by_service_bucket = aggregate(["airport_group", "movement", "service_type", "hour", "congestion_bucket"])
        self.by_group_bucket = aggregate(["airport_group", "movement", "hour", "congestion_bucket"])
        self.by_movement_bucket = aggregate(["movement", "hour", "congestion_bucket"])
        self.by_service = aggregate(["airport_group", "movement", "service_type", "hour"])
        self.by_group = aggregate(["airport_group", "movement", "hour"])
        self.by_movement_hour = aggregate(["movement", "hour"])
        self.by_movement = aggregate(["movement"])

    def _lookup(self, group: str, movement: str, service_type: str, hour: int, congestion_bucket: str) -> float:
        key_service_bucket = (group, movement, service_type, hour, congestion_bucket)
        if key_service_bucket in self.by_service_bucket:
            return self.by_service_bucket[key_service_bucket]

        key_group_bucket = (group, movement, hour, congestion_bucket)
        if key_group_bucket in self.by_group_bucket:
            return self.by_group_bucket[key_group_bucket]

        key_mv_bucket = (movement, hour, congestion_bucket)
        if key_mv_bucket in self.by_movement_bucket:
            return self.by_movement_bucket[key_mv_bucket]

        key_service = (group, movement, service_type, hour)
        if key_service in self.by_service:
            return self.by_service[key_service]

        key_group = (group, movement, hour)
        if key_group in self.by_group:
            return self.by_group[key_group]

        key_mv_hour = (movement, hour)
        if key_mv_hour in self.by_movement_hour:
            return self.by_movement_hour[key_mv_hour]

        return self.by_movement.get(movement, self.global_rate)

    def sample(
        self,
        schedule: pd.DataFrame,
        rng: np.random.Generator,
        uniform_draws: np.ndarray | None = None,
    ) -> np.ndarray:
        n_rows = len(schedule)
        if n_rows == 0:
            return np.zeros(0, dtype=bool)

        services = schedule.get("service_type", "unknown").fillna("unknown").to_numpy()
        dep_groups = schedule.get("dep_airport_group", pd.Series([None] * n_rows)).to_numpy()
        dep_times = pd.to_datetime(schedule.get("std"), errors="coerce")
        dep_hours = dep_times.dt.hour.to_numpy()
        dep_valid = (
            pd.notna(dep_groups)
            & (dep_groups != "NA")
            & pd.notna(dep_times).to_numpy()
        )

        arr_groups = schedule.get("arr_airport_group", pd.Series([None] * n_rows)).to_numpy()
        arr_times = pd.to_datetime(schedule.get("sta"), errors="coerce")
        arr_hours = arr_times.dt.hour.to_numpy()
        arr_valid = (
            pd.notna(arr_groups)
            & (arr_groups != "NA")
            & pd.notna(arr_times).to_numpy()
        )

        probs = np.full(n_rows, self.global_rate, dtype=np.float32)

        congestion_counts_dep: Dict[Tuple[str, int], int] = {}
        for idx in np.where(dep_valid)[0]:
            key = (dep_groups[idx], int(dep_hours[idx]))
            congestion_counts_dep[key] = congestion_counts_dep.get(key, 0) + 1

        congestion_counts_arr: Dict[Tuple[str, int], int] = {}
        for idx in np.where(arr_valid)[0]:
            key = (arr_groups[idx], int(arr_hours[idx]))
            congestion_counts_arr[key] = congestion_counts_arr.get(key, 0) + 1

        dep_indices = np.where(dep_valid)[0]
        for idx in dep_indices:
            group = dep_groups[idx]
            hour = int(dep_hours[idx])
            service = services[idx] or "unknown"
            congestion_bucket = self._bucket_congestion_value(congestion_counts_dep.get((group, hour), 1))
            probs[idx] = self._lookup(group, "dep", service, hour, congestion_bucket)

        fallback_indices = np.where(~dep_valid & arr_valid)[0]
        for idx in fallback_indices:
            group = arr_groups[idx]
            hour = int(arr_hours[idx])
            service = services[idx] or "unknown"
            congestion_bucket = self._bucket_congestion_value(congestion_counts_arr.get((group, hour), 1))
            probs[idx] = self._lookup(group, "arr", service, hour, congestion_bucket)

        if uniform_draws is not None and uniform_draws.size >= n_rows:
            uniforms = np.clip(uniform_draws[:n_rows], 1e-6, 1 - 1e-6)
            cancellations = uniforms < probs
        else:
            cancellations = rng.random(n_rows) < probs
        return cancellations

    def _bucket_congestion_series(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(index=series.index, dtype="object")
        if not self.congestion_bucket_edges:
            result = pd.Series("unknown", index=series.index, dtype="object")
            result.loc[series.notna()] = self.congestion_bucket_labels[-1]
            return result
        buckets = (-np.inf, *self.congestion_bucket_edges, np.inf)
        categorized = pd.cut(series, bins=buckets, labels=self.congestion_bucket_labels, include_lowest=True)
        return categorized.astype(str).fillna("unknown")

    def _bucket_congestion_value(self, value: float | int | None) -> str:
        if value is None or np.isnan(value) or value <= 0:
            return "unknown"
        if not self.congestion_bucket_edges:
            return self.congestion_bucket_labels[-1]
        idx = int(np.digitize([value], self.congestion_bucket_edges, right=True)[0])
        idx = min(idx, len(self.congestion_bucket_labels) - 1)
        return self.congestion_bucket_labels[idx]

    def fingerprint(self) -> Dict[str, object]:
        return {
            "alpha": float(self.alpha),
            "congestion_bucket_edges": list(self.congestion_bucket_edges),
            "version": 2,
        }


__all__ = ["DelaySampler", "CancellationEstimator"]
