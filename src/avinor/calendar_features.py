from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalendarConfig:
    winter_break_weeks: Iterable[int] = (8, 9)
    autumn_break_weeks: Iterable[int] = (40, 41)


CONFIG = CalendarConfig()


@lru_cache(maxsize=None)
def easter_date(year: int) -> dt.date:
    """Compute Easter Sunday for a given year (Western)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def norwegian_public_holidays(year: int) -> set[dt.date]:
    easter = easter_date(year)
    holidays = {
        dt.date(year, 1, 1),  # New Year
        dt.date(year, 5, 1),  # Labour Day
        dt.date(year, 5, 17),  # Constitution Day
        dt.date(year, 12, 24),
        dt.date(year, 12, 25),
        dt.date(year, 12, 26),
    }
    holidays.update(
        {
            easter - dt.timedelta(days=3),  # Maundy Thursday
            easter - dt.timedelta(days=2),  # Good Friday
            easter + dt.timedelta(days=1),  # Easter Monday
            easter + dt.timedelta(days=39),  # Ascension Day
            easter + dt.timedelta(days=49),  # Pentecost
            easter + dt.timedelta(days=50),  # Whit Monday
        }
    )
    return holidays


def is_public_holiday(dates: pd.Series) -> pd.Series:
    holidays_cache: dict[int, set[dt.date]] = {}
    def lookup(date: pd.Timestamp) -> bool:
        year = date.year
        if year not in holidays_cache:
            holidays_cache[year] = norwegian_public_holidays(year)
        return date.date() in holidays_cache[year]
    return dates.apply(lookup)


def is_school_holiday(dates: pd.Series) -> pd.Series:
    weeks = dates.dt.isocalendar().week.astype(int)
    month = dates.dt.month
    day = dates.dt.day

    winter_break = weeks.isin(CONFIG.winter_break_weeks) & (month <= 3)
    autumn_break = weeks.isin(CONFIG.autumn_break_weeks) & (month >= 9)
    summer_holiday = (month >= 6) & (month <= 8)
    christmas = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 5))
    return winter_break | autumn_break | summer_holiday | christmas


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    features = pd.DataFrame(index=df.index)
    features["is_weekend"] = dates.dt.dayofweek.isin({5, 6}).astype(int)
    features["is_public_holiday"] = is_public_holiday(dates).astype(int)
    features["is_school_holiday"] = is_school_holiday(dates).astype(int)
    features["weekofyear"] = dates.dt.isocalendar().week.astype("int16")
    features["dayofyear"] = dates.dt.dayofyear.astype("int16")
    features["is_month_start"] = dates.dt.is_month_start.astype(int)
    features["is_month_end"] = dates.dt.is_month_end.astype(int)
    return features


__all__ = [
    "CalendarConfig",
    "add_calendar_features",
    "is_public_holiday",
    "is_school_holiday",
]
