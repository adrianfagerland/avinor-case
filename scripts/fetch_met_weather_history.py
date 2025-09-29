#!/usr/bin/env python3
"""Fetch historical hourly weather observations for Avinor airport groups using the MET Norway Frost API."""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

FROST_SOURCES_URL = "https://frost.met.no/sources/v0.jsonld"
FROST_OBSERVATIONS_URL = "https://frost.met.no/observations/v0.jsonld"
DEFAULT_ELEMENTS = [
    "air_temperature",
    "wind_speed",
    "wind_from_direction",
    "wind_speed_of_gust",
    "relative_humidity",
    "cloud_area_fraction",
    "cloud_area_fraction_low",
    "cloud_area_fraction_medium",
    "cloud_area_fraction_high",
    "dew_point_temperature",
    "precipitation_amount",
]


@dataclass
class AirportMeta:
    airport_group: str
    airport: str
    latitude: float
    longitude: float


class FrostClient:
    def __init__(self, client_id: str, client_secret: str | None = None, *, max_retries: int = 3, sleep_seconds: float = 0.5) -> None:
        self.auth = (client_id, client_secret or "")
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def _request(self, method: str, url: str, *, params: Optional[Dict[str, str]] = None) -> Dict:
        for attempt in range(1, self.max_retries + 1):
            response = requests.request(method, url, params=params, auth=self.auth, timeout=60)
            if response.status_code == 429:
                time.sleep(self.sleep_seconds * attempt)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:  # type: ignore[attr-defined]
                if attempt == self.max_retries:
                    detail = response.text
                    raise RuntimeError(f"HTTP {response.status_code} for {url}: {detail}") from exc
                time.sleep(self.sleep_seconds * attempt)
                continue
            return response.json()
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts")

    def find_station(self, airport: AirportMeta, *, elements: Iterable[str], valid_from: str, valid_to: str) -> str:
        params = {
            "geometry": f"nearest(POINT({airport.longitude} {airport.latitude}))",
            "types": "SensorSystem",
            "country": "NO",
            "validtime": f"{valid_from[:10]}/{valid_to[:10]}",
        }
        payload = self._request("GET", FROST_SOURCES_URL, params=params)
        data = payload.get("data", [])
        if not data:
            raise RuntimeError(
                f"No observation station found for airport {airport.airport} ({airport.airport_group})"
            )
        return data[0]["id"]

    def fetch_observations(self, station_id: str, *, start: str, end: str, elements: Iterable[str]) -> List[Dict]:
        records: List[Dict] = []
        elements_param = ",".join(elements)
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        start_dt = start_ts.tz_convert("UTC") if start_ts.tzinfo else start_ts.tz_localize("UTC")
        end_dt = end_ts.tz_convert("UTC") if end_ts.tzinfo else end_ts.tz_localize("UTC")
        chunk = pd.Timedelta(days=120)
        current_start = start_dt

        def fmt(ts: pd.Timestamp) -> str:
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        while current_start <= end_dt:
            current_end = min(current_start + chunk, end_dt)
            params = {
                "sources": station_id,
                "referencetime": f"{fmt(current_start)}/{fmt(current_end)}",
                "timeresolutions": "PT1H",
                "elements": elements_param,
            }
            next_url: Optional[str] = None
            while True:
                payload = self._request(
                    "GET",
                    next_url or FROST_OBSERVATIONS_URL,
                    params=params if next_url is None else None,
                )
                records.extend(payload.get("data", []))
                next_url = payload.get("nextLink")
                if not next_url:
                    break
                time.sleep(self.sleep_seconds)
            current_start = current_end + pd.Timedelta(hours=1)
        return records


def parse_airport_metadata(path: Path) -> List[AirportMeta]:
    if not path.exists():
        raise FileNotFoundError(f"Airport metadata not found: {path}")
    df = pd.read_csv(path)
    required = {"airport_group", "iata_code", "latitude_deg", "longitude_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Airport metadata missing columns: {missing}")
    if "airport" in df.columns:
        airport_id = df["airport"].fillna(df["iata_code"])
    else:
        airport_id = df["iata_code"]
    records: List[AirportMeta] = []
    for row in df.itertuples(index=False):
        airport = getattr(row, "airport", getattr(row, "iata_code"))
        airport_str = str(airport)
        records.append(
            AirportMeta(
                airport_group=str(getattr(row, "airport_group")),
                airport=airport_str,
                latitude=float(getattr(row, "latitude_deg")),
                longitude=float(getattr(row, "longitude_deg")),
            )
        )
    return records


def records_to_dataframe(
    records: List[Dict],
    airport: AirportMeta,
    station_id: str,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for entry in records:
        reference_time = entry.get("referenceTime")
        if not reference_time:
            continue
        row: Dict[str, object] = {
            "airport_group": airport.airport_group,
            "airport": airport.airport,
            "station_id": station_id,
            "observation_time": pd.to_datetime(reference_time),
        }
        for obs in entry.get("observations", []):
            elem = obs.get("elementId")
            value = obs.get("value")
            if elem and value is not None:
                row[elem] = value
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    return df


def aggregate_to_airport_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["observation_time"] = df["observation_time"].dt.floor("h")
    df["date"] = df["observation_time"].dt.date.astype("datetime64[ns]")
    df["hour"] = df["observation_time"].dt.hour.astype("int16")
    group_cols = ["airport_group", "airport", "date", "hour"]
    value_cols = [
        col
        for col in df.columns
        if col not in group_cols + ["station_id", "observation_time"]
    ]
    aggregated = df.groupby(group_cols)[value_cols].mean().reset_index()
    aggregated["forecast_time"] = pd.to_datetime(aggregated["date"]) + pd.to_timedelta(aggregated["hour"], unit="h")
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical hourly weather observations from MET Norway Frost API")
    parser.add_argument(
        "--airport-metadata",
        type=Path,
        default=Path("data/external/processed/airport_metadata.csv"),
        help="CSV produced by fetch_airport_metadata.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/raw/weather_history.parquet"),
        help="Destination parquet file",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01T00:00:00Z",
        help="ISO-8601 start timestamp (UTC)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-07-31T23:59:59Z",
        help="ISO-8601 end timestamp (UTC)",
    )
    parser.add_argument(
        "--elements",
        type=str,
        default=",".join(DEFAULT_ELEMENTS),
        help="Comma-separated Frost element IDs",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Frost API client ID (can also be supplied via FROST_CLIENT_ID env)",
    )
    parser.add_argument(
        "--client-secret",
        type=str,
        default=None,
        help="Frost API client secret (optional, env FROST_CLIENT_SECRET)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between paginated requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client_id = args.client_id or os.getenv("FROST_CLIENT_ID")
    client_secret = args.client_secret or os.getenv("FROST_CLIENT_SECRET")
    if not client_id:
        raise SystemExit("Frost client ID is required (pass --client-id or set FROST_CLIENT_ID)")

    elements = [elem.strip() for elem in args.elements.split(",") if elem.strip()]
    airports = parse_airport_metadata(args.airport_metadata)
    client = FrostClient(client_id, client_secret, sleep_seconds=args.sleep)

    frames: List[pd.DataFrame] = []
    station_map: Dict[str, str] = {}

    for airport in tqdm(airports, desc="Airports"):
        station_id = client.find_station(
            airport,
            elements=elements,
            valid_from=args.start,
            valid_to=args.end,
        )
        station_map[airport.airport] = station_id
        observations = client.fetch_observations(
            station_id,
            start=args.start,
            end=args.end,
            elements=elements,
        )
        df = records_to_dataframe(observations, airport, station_id)
        frames.append(df)

    if not frames:
        raise SystemExit("No weather observations downloaded")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["airport_group", "airport", "observation_time"], inplace=True)
    aggregated = aggregate_to_airport_hour(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(args.output, index=False)

    summary = aggregated.groupby("airport_group").agg(
        rows=("hour", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    print("Saved", len(aggregated), "rows to", args.output)
    print("Station mapping:")
    for airport, station in station_map.items():
        print(f"  {airport}: {station}")
    print("Coverage summary:\n", summary)


if __name__ == "__main__":
    main()
