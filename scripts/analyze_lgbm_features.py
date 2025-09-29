#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib

from avinor.paths import FEATURE_CACHE_DIR, MODELS_DIR


def load_feature_columns() -> List[str]:
    meta_path = FEATURE_CACHE_DIR / "logistic_features.json"
    meta = json.loads(meta_path.read_text())
    columns = meta.get("columns") or meta.get("features")
    if not columns:
        raise RuntimeError("Could not find training columns in logistic_features.json")
    return list(columns)


def main() -> None:
    model_path = MODELS_DIR / "lgbm_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)
    columns = load_feature_columns()

    # LightGBM sklearn wrapper exposes split-based importance via feature_importances_
    split_importance = getattr(model, "feature_importances_", None)
    if split_importance is None:
        raise RuntimeError("LGBM model does not have feature_importances_")

    split_importance = split_importance.tolist()

    # Gain-based importance if available via booster_
    gain_importance = None
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            gain_importance = booster.feature_importance(importance_type="gain").tolist()
        except Exception:
            gain_importance = None

    if len(columns) != len(split_importance):
        raise RuntimeError(
            f"Columns length ({len(columns)}) != importances length ({len(split_importance)})"
        )

    items: List[Dict[str, Any]] = []
    for name, split_val in zip(columns, split_importance):
        item: Dict[str, Any] = {"feature": name, "split_importance": float(split_val)}
        if gain_importance is not None:
            item["gain_importance"] = float(gain_importance[len(items)])
        items.append(item)

    def sort_key(row: Dict[str, Any]) -> float:
        if gain_importance is not None:
            return row.get("gain_importance", 0.0)
        return row.get("split_importance", 0.0)

    items.sort(key=sort_key, reverse=True)

    # Normalize importances for readability
    total_split = sum(x["split_importance"] for x in items) or 1.0
    for x in items:
        x["split_importance_pct"] = x["split_importance"] / total_split
    if gain_importance is not None:
        total_gain = sum(x.get("gain_importance", 0.0) for x in items) or 1.0
        for x in items:
            x["gain_importance_pct"] = x.get("gain_importance", 0.0) / total_gain

    # Aggregate by coarse feature families for readability
    def family(name: str) -> str:
        if name.startswith("sim_"):
            return "simulation"
        if name.startswith("overlap_"):
            return "overlap"
        if name.startswith("scheduled_") or name in {"scheduled_departures", "scheduled_arrivals", "scheduled_movements", "feat_sched_concurrence", "feat_sched_flights_cnt"}:
            return "schedule_counts"
        if name.startswith("hist_") or name.endswith("_target_enc"):
            return "history_encoding"
        if name.startswith("target_") or name in {"target_prev_hour", "hours_since_positive"}:
            return "recent_target"
        if (
            name.startswith("holiday_name_")
            or name.startswith("school_holiday_name_")
            or name in {"is_weekend", "is_public_holiday", "is_school_holiday", "weekofyear", "dayofyear", "is_month_start", "is_month_end"}
        ):
            return "calendar"
        if (
            "air_temperature" in name
            or name.startswith("wind_")
            or name.startswith("precipitation_")
            or "cloud_area_fraction" in name
            or "dew_point" in name
            or "relative_humidity" in name
            or name.startswith("weather_")
            or name.startswith("probability_of_precipitation")
        ):
            return "weather"
        if name.startswith("airport_group_"):
            return "airport_group_onehot"
        if name.startswith("airport_hour_bucket_"):
            return "airport_hour_bucket_onehot"
        if name.startswith("airport_dow_bucket_"):
            return "airport_dow_bucket_onehot"
        if name in {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "hour", "dayofweek", "month"}:
            return "time_cyclical"
        return "other"

    families: Dict[str, Dict[str, float]] = {}
    for it in items:
        fam = family(it["feature"])
        f = families.setdefault(fam, {"split": 0.0, "gain": 0.0})
        f["split"] += it["split_importance_pct"]
        if gain_importance is not None:
            f["gain"] += it.get("gain_importance_pct", 0.0)

    group_importance = [
        {
            "family": k,
            "split_importance_pct": v["split"],
            **({"gain_importance_pct": v["gain"]} if gain_importance is not None else {}),
        }
        for k, v in families.items()
    ]
    group_importance.sort(key=lambda r: r.get("gain_importance_pct", r["split_importance_pct"]), reverse=True)

    out = {
        "model": "lgbm",
        "n_features": len(items),
        "top_features": items[:30],
        "group_importance": group_importance,
    }

    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lgbm_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
