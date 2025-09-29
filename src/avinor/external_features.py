from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def load_weather_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather forecast file not found: {path}")
    df = pd.read_parquet(path)
    if "forecast_time" not in df.columns:
        combined: pd.Series | None = None

        if {"date", "hour"}.issubset(df.columns):
            date_values = pd.to_datetime(df["date"], errors="coerce")
            hour_values = pd.to_numeric(df["hour"], errors="coerce")
            combined = date_values + pd.to_timedelta(hour_values, unit="h")
        else:
            for alt in ("valid_time", "time", "timestamp"):
                if alt in df.columns:
                    combined = pd.to_datetime(df[alt], errors="coerce")
                    break

        if combined is None:
            raise ValueError("Weather forecast parquet missing 'forecast_time' column")

        df = df.copy()
        df["forecast_time"] = combined

    df["forecast_time"] = pd.to_datetime(df["forecast_time"], errors="coerce")
    df = df.dropna(subset=["forecast_time"])
    df["date"] = df["forecast_time"].dt.floor("h").dt.date.astype("datetime64[ns]")
    df["hour"] = df["forecast_time"].dt.hour.astype("int16")
    return df


def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    numeric_features = {
        "air_temperature": ["mean", "min", "max"],
        "wind_speed": ["mean", "max"],
        "wind_speed_of_gust": ["max"],
        "relative_humidity": ["mean"],
        "dew_point_temperature": ["mean"],
        "cloud_area_fraction": ["mean"],
        "cloud_area_fraction_low": ["mean"],
        "precipitation_amount_1h": ["mean", "max"],
        "probability_of_precipitation_1h": ["mean"],
    }

    group_cols = ["airport_group", "date", "hour"]
    for column in numeric_features:
        if column not in weather.columns:
            weather[column] = np.nan

    aggregated = weather.groupby(group_cols).agg(numeric_features)
    aggregated.columns = ["{}_{}".format(col, stat) for col, stat in aggregated.columns]

    aggregated = aggregated.reset_index()

    def compute_complex_flags(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["weather_high_wind"] = (df.get("wind_speed_of_gust_max", 0).fillna(0) >= 15.0).astype(int)
        df["weather_heavy_precip"] = (df.get("precipitation_amount_1h_max", 0).fillna(0) >= 3.0).astype(int)
        df["weather_low_ceiling"] = (df.get("cloud_area_fraction_low_mean", 0).fillna(0) >= 80).astype(int)
        df["weather_complex_score"] = (
            0.4 * df["weather_high_wind"]
            + 0.4 * df["weather_heavy_precip"]
            + 0.2 * df["weather_low_ceiling"]
        )
        df["weather_ifr_proxy"] = ((
            df["weather_low_ceiling"] == 1
        ) | (
            df.get("relative_humidity_mean", 0).fillna(0) >= 90
        )).astype(int)
        return df

    aggregated = compute_complex_flags(aggregated)

    symbol_cols = [col for col in weather.columns if col.startswith("symbol_code")]
    if symbol_cols:
        # Determine dominant symbol per group-hour and encode into broad classes.
        symbol_df = (
            weather.groupby(group_cols)[symbol_cols]
            .agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else np.nan)
            .reset_index()
        )
        for col in symbol_cols:
            new_col = f"dominant_{col}"
            symbol_df[new_col] = symbol_df[col].astype(str)
            del symbol_df[col]
        aggregated = aggregated.merge(symbol_df, on=group_cols, how="left")

        def classify_symbol(series: pd.Series | None, keywords: Iterable[str]) -> pd.Series:
            if series is None:
                return pd.Series(0, index=aggregated.index)
            return series.fillna("").str.contains("|".join(keywords)).astype(int)

        aggregated["weather_symbol_thunder"] = classify_symbol(
            aggregated.get("dominant_symbol_code_1h"), ["thunder"]
        )
        aggregated["weather_symbol_snow"] = classify_symbol(
            aggregated.get("dominant_symbol_code_1h"), ["snow"]
        )
        aggregated["weather_symbol_rain"] = classify_symbol(
            aggregated.get("dominant_symbol_code_1h"), ["rain", "sleet"]
        )
    return aggregated


def load_public_holidays(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Holiday calendar not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date.astype("datetime64[ns]")
    return df


def merge_external_features(
    base: pd.DataFrame,
    weather_path: Path | None = None,
    holiday_path: Path | None = None,
    school_holiday_path: Path | None = None,
) -> pd.DataFrame:
    external = pd.DataFrame(index=base.index)
    key_cols = ["airport_group", "date", "hour"]

    if weather_path is not None and weather_path.exists():
        weather_ts = load_weather_timeseries(weather_path)
        weather_agg = aggregate_weather(weather_ts)
        joined = base[key_cols].merge(weather_agg, on=key_cols, how="left")
        weather_cols: List[str] = [col for col in joined.columns if col not in key_cols]
        for col in weather_cols:
            external[col] = joined[col].to_numpy()

    if holiday_path is not None and holiday_path.exists():
        holidays = load_public_holidays(holiday_path)
        holidays = holidays.set_index("date")
        matched = base[["date"]].join(holidays, on="date")
        if "name" in matched:
            external["holiday_name"] = matched["name"].fillna("")
        if "is_public_holiday" in matched:
            external["is_public_holiday_official"] = matched["is_public_holiday"].fillna(0).astype(int)
        if "name" in matched:
            external["is_school_holiday_official"] = matched["name"].fillna("").str.contains(
                "ferie", case=False
            ).astype(int)

    if school_holiday_path is not None and school_holiday_path.exists():
        school_df = pd.read_csv(school_holiday_path, parse_dates=["date"])
        school_df["date"] = school_df["date"].dt.date.astype("datetime64[ns]")

        join_cols = ["date"]
        if "airport_group" in school_df.columns:
            join_cols = ["airport_group", "date"]
        school_df = school_df.drop_duplicates(join_cols)

        feature_cols: List[str] = []
        if "is_school_holiday" in school_df.columns:
            school_df["is_school_holiday_calendar"] = school_df["is_school_holiday"].astype(int)
            feature_cols.append("is_school_holiday_calendar")
        if "name" in school_df.columns:
            school_df["school_holiday_name"] = school_df["name"].fillna("")
            feature_cols.append("school_holiday_name")

        if not feature_cols:
            school_df["is_school_holiday_calendar"] = 1
            feature_cols.append("is_school_holiday_calendar")

        merged = base[join_cols].merge(school_df[join_cols + feature_cols], on=join_cols, how="left")
        for col in feature_cols:
            values = merged[col]
            if values.dtype == object:
                external[col] = values.fillna("")
            else:
                external[col] = values.fillna(0).astype(int)

    return external


__all__ = [
    "load_weather_timeseries",
    "aggregate_weather",
    "load_public_holidays",
    "merge_external_features",
]
