from __future__ import annotations

import numpy as np
import pandas as pd


def parse_naive_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def to_local_minutes(series: pd.Series) -> pd.Series:
    return (series.view(np.int64) // (60 * 10**9)).astype("Int64")


def floor_to_minute(series: pd.Series) -> pd.Series:
    return series.dt.floor("T")


def ceil_to_minute(series: pd.Series) -> pd.Series:
    return series.dt.ceil("T")


def extract_time_parts(timestamp: pd.Series) -> pd.DataFrame:
    ts = timestamp
    return pd.DataFrame(
        {
            "date": ts.dt.floor("D"),
            "hour": ts.dt.hour.astype("int16"),
            "minute": ts.dt.minute.astype("int16"),
            "dayofweek": ts.dt.dayofweek.astype("int8"),
            "month": ts.dt.month.astype("int8"),
            "year": ts.dt.year.astype("int16"),
        }
    )


def season_from_month(month_series: pd.Series) -> pd.Series:
    mapping = {
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
    return month_series.map(mapping).astype("category")


__all__ = [
    "parse_naive_timestamp",
    "to_local_minutes",
    "floor_to_minute",
    "ceil_to_minute",
    "extract_time_parts",
    "season_from_month",
]
