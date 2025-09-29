from __future__ import annotations

import pandas as pd

CAT_COLUMNS = [
    "airport_group",
    "season",
    "holiday_name",
    "school_holiday_name",
    "feat_season",
    "airport_hour_bucket",
    "airport_dow_bucket",
]
DROP_COLUMNS = ["date", "interval_start", "interval_end"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


class TabularFeatureBuilder:
    def __init__(self) -> None:
        self.columns_: list[str] | None = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_time_features(df)
        df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])
        df = pd.get_dummies(df, columns=[col for col in CAT_COLUMNS if col in df.columns])
        df = df.fillna(0.0)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self._prepare(df)
        self.columns_ = list(transformed.columns)
        return transformed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("TabularFeatureBuilder must be fitted before transform")
        transformed = self._prepare(df)
        for column in self.columns_:
            if column not in transformed.columns:
                transformed[column] = 0.0
        extra_columns = [col for col in transformed.columns if col not in self.columns_]
        if extra_columns:
            transformed = transformed.drop(columns=extra_columns)
        return transformed[self.columns_]


__all__ = ["TabularFeatureBuilder", "add_time_features"]
