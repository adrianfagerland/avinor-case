from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclical_features(df: pd.DataFrame, column: str, period: int, prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or column
    values = df[column].astype(float)
    df[f"{prefix}_sin"] = np.sin(2 * np.pi * values / period)
    df[f"{prefix}_cos"] = np.cos(2 * np.pi * values / period)
    return df


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    if not isinstance(denominator, pd.Series):
        denominator = pd.Series(denominator, index=numerator.index)
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).fillna(0.0)


__all__ = ["add_cyclical_features", "safe_divide"]
