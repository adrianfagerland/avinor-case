#!/usr/bin/env python3
"""Fetch hourly weather forecasts from MET Norway locationforecast API."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests

MET_ENDPOINT = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
USER_AGENT = "avinor-competition-bot"
DEFAULT_TIMEOUT = 60
DEFAULT_SLEEP_SECONDS = 1.0


def load_coordinates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing airport metadata at {path}")
    df = pd.read_csv(path)
    required_base = {"airport_group", "iata_code", "latitude_deg", "longitude_deg"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"Airport metadata missing columns: {missing_base}")

    if "airport" not in df.columns:
        # Older versions of fetch_airport_metadata.py only output iata_code; use it as identifier.
        df["airport"] = df["iata_code"].fillna(df.get("icao_code", ""))

    # Drop rows without coordinates or identifiers.
    df = df.dropna(subset=["latitude_deg", "longitude_deg", "airport"])
    df["airport"] = df["airport"].astype(str)
    return df


def fetch_forecast(lat: float, lon: float) -> Dict[str, Any]:
    params = {"lat": float(lat), "lon": float(lon)}
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(MET_ENDPOINT, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def parse_timeseries(timeseries: Iterable[Dict[str, Any]], airport_id: str, group: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in timeseries:
        timestamp = pd.to_datetime(entry.get("time"))
        data = entry.get("data", {})
        instant = data.get("instant", {}).get("details", {})
        next_1h = data.get("next_1_hours", {})
        next_6h = data.get("next_6_hours", {})
        next_12h = data.get("next_12_hours", {})
        row: Dict[str, Any] = {
            "airport_group": group,
            "airport": airport_id,
            "forecast_time": timestamp,
            "air_temperature": instant.get("air_temperature"),
            "wind_speed": instant.get("wind_speed"),
            "wind_from_direction": instant.get("wind_from_direction"),
            "wind_speed_of_gust": instant.get("wind_speed_of_gust"),
            "relative_humidity": instant.get("relative_humidity"),
            "cloud_area_fraction": instant.get("cloud_area_fraction"),
            "cloud_area_fraction_high": instant.get("cloud_area_fraction_high"),
            "cloud_area_fraction_medium": instant.get("cloud_area_fraction_medium"),
            "cloud_area_fraction_low": instant.get("cloud_area_fraction_low"),
            "dew_point_temperature": instant.get("dew_point_temperature"),
            "precipitation_amount_1h": (next_1h.get("details") or {}).get("precipitation_amount"),
            "probability_of_precipitation_1h": (next_1h.get("details") or {}).get("probability_of_precipitation"),
            "precipitation_amount_6h": (next_6h.get("details") or {}).get("precipitation_amount"),
            "probability_of_precipitation_6h": (next_6h.get("details") or {}).get("probability_of_precipitation"),
            "precipitation_amount_12h": (next_12h.get("details") or {}).get("precipitation_amount"),
            "symbol_code_1h": (next_1h.get("summary") or {}).get("symbol_code"),
            "symbol_code_6h": (next_6h.get("summary") or {}).get("symbol_code"),
            "symbol_code_12h": (next_12h.get("summary") or {}).get("symbol_code"),
        }
        rows.append(row)
    return rows


def fetch_all_forecasts(metadata: pd.DataFrame, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in metadata.itertuples(index=False):
        airport_id = getattr(row, "airport")
        group = getattr(row, "airport_group")
        lat = getattr(row, "latitude_deg")
        lon = getattr(row, "longitude_deg")
        if pd.isna(lat) or pd.isna(lon):
            continue
        forecast_json = fetch_forecast(lat, lon)
        timeseries = forecast_json.get("properties", {}).get("timeseries", [])
        records.extend(parse_timeseries(timeseries, airport_id=airport_id, group=group))
        time.sleep(sleep_seconds)
    df = pd.DataFrame.from_records(records)
    return df


def select_time_window(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if start is not None:
        df = df[df["forecast_time"] >= start]
    if end is not None:
        df = df[df["forecast_time"] <= end]
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MET Norway weather forecasts for competition airports")
    parser.add_argument(
        "--airport-metadata",
        type=Path,
        default=Path("data/external/processed/airport_metadata.csv"),
        help="Path to per-airport metadata CSV (output from fetch_airport_metadata.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/raw/weather_forecast.parquet"),
        help="Where to store raw forecast timeseries",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Inclusive start datetime (ISO-8601). If omitted, keep all available forecasts",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Inclusive end datetime (ISO-8601)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay between API calls to respect rate limits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_coordinates(args.airport_metadata)
    df = fetch_all_forecasts(metadata, sleep_seconds=args.sleep_seconds)
    if df.empty:
        raise RuntimeError("No weather data fetched; check API response or coordinates")

    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    start = pd.to_datetime(args.start) if args.start else None
    end = pd.to_datetime(args.end) if args.end else None
    df = select_time_window(df, start=start, end=end)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} forecast rows to {args.output}")


if __name__ == "__main__":
    main()
