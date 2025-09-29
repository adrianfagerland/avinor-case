#!/usr/bin/env python3
"""Generate Norwegian public holiday calendar for specified years."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import holidays
except ImportError as exc:  # pragma: no cover - imported at runtime
    raise SystemExit(
        "The 'holidays' package is required. Install it with `pip install holidays`."
    ) from exc


DEFAULT_YEARS = [2023, 2024, 2025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Norwegian public holiday calendar")
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=DEFAULT_YEARS,
        help="Years to include in the calendar (default: 2023-2025)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/processed/calendar_public_holidays.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    norway_holidays = holidays.Norway(years=args.years)
    rows = [
        {
            "date": pd.Timestamp(date),
            "name": name,
            "is_public_holiday": True,
        }
        for date, name in norway_holidays.items()
    ]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(df)} holidays to {args.output}")


if __name__ == "__main__":
    main()
