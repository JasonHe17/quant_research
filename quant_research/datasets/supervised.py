"""Supervised learning dataset builders for alpha research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from quant_research.factors import FactorResult


@dataclass(frozen=True, slots=True)
class ForwardReturnLabelConfig:
    """Configuration for signal-time aligned forward return labels.

    A label at timestamp ``t`` uses the price after ``entry_lag_bars`` as entry
    and the price after ``entry_lag_bars + horizon_bars`` as exit. For CN A-share
    T+1 research, use a horizon that reaches at least the next trading day.
    """

    name: str = "forward_return"
    horizon_bars: int = 48
    entry_lag_bars: int = 1
    price_column: str = "close_price"
    timestamp_column: str = "bar_end_time"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.entry_lag_bars < 0:
            raise ValueError("entry_lag_bars must be non-negative")
        if not self.price_column:
            raise ValueError("price_column must be non-empty")
        if not self.timestamp_column:
            raise ValueError("timestamp_column must be non-empty")


def build_alpha_feature_matrix(
    factors: Iterable[FactorResult | pd.DataFrame],
    *,
    value_column: str = "factor_value",
) -> pd.DataFrame:
    """Pivot long alpha factor results into one row per timestamp/instrument."""

    frames = [_normalize_factor_frame(item, value_column=value_column) for item in factors]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "instrument_id"])
    long_frame = pd.concat(frames, ignore_index=True)
    duplicate_keys = long_frame.duplicated(
        ["timestamp", "instrument_id", "factor_name"],
        keep=False,
    )
    if duplicate_keys.any():
        duplicates = long_frame.loc[
            duplicate_keys,
            ["timestamp", "instrument_id", "factor_name"],
        ].head(5)
        raise ValueError(f"duplicate factor observations: {duplicates.to_dict('records')}")
    matrix = (
        long_frame.pivot(
            index=["timestamp", "instrument_id"],
            columns="factor_name",
            values=value_column,
        )
        .reset_index()
        .sort_values(["timestamp", "instrument_id"])
        .reset_index(drop=True)
    )
    matrix.columns.name = None
    return matrix


def build_forward_return_labels(
    bars: pd.DataFrame,
    config: ForwardReturnLabelConfig,
) -> pd.DataFrame:
    """Build forward return labels aligned to the original bar timestamp."""

    required = ("instrument_id", config.timestamp_column, config.price_column)
    _require_columns(bars, required)
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "instrument_id",
                config.name,
                "entry_timestamp",
                "exit_timestamp",
                "entry_price",
                "exit_price",
            ]
        )
    frame = bars.loc[:, list(required)].copy()
    frame = frame.sort_values(["instrument_id", config.timestamp_column])
    frame[config.price_column] = frame[config.price_column].astype(float)
    grouped = frame.groupby("instrument_id", sort=False)
    entry_shift = -config.entry_lag_bars
    exit_shift = -(config.entry_lag_bars + config.horizon_bars)
    frame["entry_timestamp"] = grouped[config.timestamp_column].shift(
        periods=entry_shift
    )
    frame["exit_timestamp"] = grouped[config.timestamp_column].shift(periods=exit_shift)
    frame["entry_price"] = grouped[config.price_column].shift(periods=entry_shift)
    frame["exit_price"] = grouped[config.price_column].shift(periods=exit_shift)
    frame[config.name] = frame["exit_price"] / frame["entry_price"] - 1.0
    frame["timestamp"] = frame[config.timestamp_column]
    output = frame.loc[
        frame[config.name].notna(),
        [
            "timestamp",
            "instrument_id",
            config.name,
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
        ],
    ]
    return output.reset_index(drop=True)


def add_cross_sectional_label_rank(
    labels: pd.DataFrame,
    *,
    label_column: str = "forward_return",
    rank_column: str = "label_rank",
    ascending: bool = False,
    pct: bool = True,
) -> pd.DataFrame:
    """Add timestamp-wise ranks for ranking/classification model targets."""

    _require_columns(labels, ("timestamp", "instrument_id", label_column))
    ranked = labels.copy()
    ranked[rank_column] = ranked.groupby("timestamp", sort=False)[label_column].rank(
        method="average",
        ascending=ascending,
        pct=pct,
    )
    return ranked


def join_alpha_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Inner join feature matrix and labels on timestamp/instrument_id."""

    _require_columns(features, ("timestamp", "instrument_id"))
    _require_columns(labels, ("timestamp", "instrument_id"))
    joined = features.merge(labels, on=["timestamp", "instrument_id"], how="inner")
    return joined.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True)


def _normalize_factor_frame(
    item: FactorResult | pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    if isinstance(item, FactorResult):
        frame = item.frame.copy()
        if "factor_name" not in frame.columns:
            frame["factor_name"] = item.factor_name
    else:
        frame = item.copy()
    if "timestamp" not in frame.columns and "bar_end_time" in frame.columns:
        frame = frame.rename(columns={"bar_end_time": "timestamp"})
    required = ("factor_name", "timestamp", "instrument_id", value_column)
    _require_columns(frame, required)
    return frame.loc[:, list(required)].copy()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
