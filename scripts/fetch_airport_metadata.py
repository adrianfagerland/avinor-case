#!/usr/bin/env python3
"""Fetch airport metadata (coordinates and ICAO codes) for competition airports."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

OURAIRPORTS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/master/airports.csv"
RESPONSIBLE_USER_AGENT = "avinor-competition-bot/1.0 (contact: data-team@example.com)"


def load_airport_groups(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing airport group mapping: {path}")
    df = pd.read_csv(path)
    required = {"airport_group", "airport"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required} in {path}")
    return df


def download_ourairports() -> pd.DataFrame:
    response = requests.get(OURAIRPORTS_URL, headers={"User-Agent": RESPONSIBLE_USER_AGENT}, timeout=60)
    response.raise_for_status()
    # Use StringIO to handle CSV text in-memory without creating temporary files.
    data = io.StringIO(response.text)
    df = pd.read_csv(data)
    return df


def filter_competition_airports(ourairports: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    needed = mapping["airport"].str.upper().unique()
    filtered = ourairports[ourairports["iata_code"].str.upper().isin(needed)].copy()
    # Some small STOL airports may not have IATA codes in the dataset; fallback via ident match.
    missing = set(needed) - set(filtered["iata_code"].str.upper())
    if missing:
        ident_matches = ourairports[ourairports["ident"].str.upper().isin(missing)]
        filtered = pd.concat([filtered, ident_matches], ignore_index=True)
    filtered = filtered.drop_duplicates(subset=["ident"])
    return filtered


def enrich_with_groups(filtered: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mapping = mapping.rename(columns={"airport": "iata_code"})
    mapping["iata_code"] = mapping["iata_code"].str.upper()
    filtered["iata_code"] = filtered["iata_code"].str.upper()
    merged = mapping.merge(filtered, on="iata_code", how="left", suffixes=("", "_our"))
    expected_columns = [
        "airport_group",
        "iata_code",
        "ident",
        "name",
        "municipality",
        "latitude_deg",
        "longitude_deg",
        "elevation_ft",
        "type",
        "iso_country",
    ]
    missing_cols = [col for col in expected_columns if col not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in merged airport metadata")
    merged = merged.rename(columns={"ident": "icao_code", "name": "airport_name", "type": "airport_type"})
    return merged


def aggregate_by_group(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["latitude_deg", "longitude_deg", "elevation_ft"]
    grouped = df.groupby("airport_group")[numeric_cols].mean().reset_index()
    grouped = grouped.rename(columns={
        "latitude_deg": "group_latitude_deg",
        "longitude_deg": "group_longitude_deg",
        "elevation_ft": "group_elevation_ft",
    })
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch airport metadata for competition airports")
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("data/airportgroups.csv"),
        help="Path to airport group mapping CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external"),
        help="Directory to store processed metadata",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Persist the full OurAirports dataset for traceability",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    processed_dir = args.output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_airport_groups(args.mapping_path)
    ourairports = download_ourairports()
    if args.keep_raw:
        raw_path = raw_dir / "ourairports_airports.csv"
        ourairports.to_csv(raw_path, index=False)

    filtered = filter_competition_airports(ourairports, mapping)
    enriched = enrich_with_groups(filtered, mapping)
    grouped = aggregate_by_group(enriched)

    enriched_path = processed_dir / "airport_metadata.csv"
    grouped_path = processed_dir / "airport_group_summary.csv"

    enriched.to_csv(enriched_path, index=False)
    grouped.to_csv(grouped_path, index=False)

    print(f"Saved airport metadata to {enriched_path}")
    print(f"Saved airport group summary to {grouped_path}")


if __name__ == "__main__":
    main()
