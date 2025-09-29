from __future__ import annotations

import pandas as pd

from .paths import DATA_DIR


TRAINING_PATH = DATA_DIR / "training_data.csv"
HISTORICAL_PATH = DATA_DIR / "historical_flights.csv"
SCHEDULE_PATH = DATA_DIR / "schedule_oct2025.csv"
INFERENCE_TEMPLATE_PATH = DATA_DIR / "inference_data_oct2025.csv"
AIRPORT_GROUPS_PATH = DATA_DIR / "airportgroups.csv"
PREDS_TEMPLATE_PATH = DATA_DIR / "preds_mal.csv"


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(TRAINING_PATH, parse_dates=["date"])
    df.sort_values(["date", "hour", "airport_group"], inplace=True)
    return df


def load_historical_flights() -> pd.DataFrame:
    datetime_columns = ["std", "sta", "atd", "ata"]
    df = pd.read_csv(HISTORICAL_PATH, parse_dates=datetime_columns)
    df.sort_values("std", inplace=True)
    return df


def load_schedule_october() -> pd.DataFrame:
    datetime_columns = ["std", "sta"]
    df = pd.read_csv(SCHEDULE_PATH, parse_dates=datetime_columns)
    df.sort_values("std", inplace=True)
    return df


def load_inference_template() -> pd.DataFrame:
    df = pd.read_csv(INFERENCE_TEMPLATE_PATH, parse_dates=["date"])
    df.sort_values(["date", "hour", "airport_group"], inplace=True)
    return df


def load_airport_groups() -> pd.DataFrame:
    return pd.read_csv(AIRPORT_GROUPS_PATH)


def load_submission_template() -> pd.DataFrame:
    return pd.read_csv(PREDS_TEMPLATE_PATH)


__all__ = [
    "load_training_data",
    "load_historical_flights",
    "load_schedule_october",
    "load_inference_template",
    "load_airport_groups",
    "load_submission_template",
]
