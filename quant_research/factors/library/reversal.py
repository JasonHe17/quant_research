"""Short-horizon reversal factors."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research.factors.base import Factor, FactorContext


@dataclass(frozen=True, slots=True)
class FiveMinuteReversalFactor(Factor):
    """Baseline A: 5-minute cross-sectional short-term reversal score.

    The factor value is the negative lookback return, so recent underperformers
    receive larger scores. Optional rolling volume/turnover filters can be used
    to keep illiquid bars out of the cross-section.
    """

    lookback_bars: int = 1
    liquidity_window_bars: int | None = None
    min_avg_volume: float | None = None
    min_avg_turnover: float | None = None
    timestamp_column: str = "bar_end_time"

    def __post_init__(self) -> None:
        if self.lookback_bars <= 0:
            raise ValueError("lookback_bars must be positive")
        if self.liquidity_window_bars is not None and self.liquidity_window_bars <= 0:
            raise ValueError("liquidity_window_bars must be positive")
        if self.min_avg_volume is not None and self.min_avg_volume < 0:
            raise ValueError("min_avg_volume must be non-negative")
        if self.min_avg_turnover is not None and self.min_avg_turnover < 0:
            raise ValueError("min_avg_turnover must be non-negative")

    def compute(self, context: FactorContext) -> pd.DataFrame:
        if context.frequency != "5m":
            raise ValueError("FiveMinuteReversalFactor requires context.frequency='5m'")
        fields = ["instrument_id", self.timestamp_column, "close_price"]
        if self.min_avg_volume is not None:
            fields.append("volume")
        if self.min_avg_turnover is not None:
            fields.append("turnover")
        bars = context.data.get_bars(
            list(context.symbols),
            start=context.start,
            end=context.end,
            frequency=context.frequency,
            adjustment="raw",
            market=context.market,
            asset_type=context.asset_type,
            fields=fields,
            cache=False,
        )
        _require_columns(bars, tuple(fields))
        if bars.empty:
            return pd.DataFrame(
                columns=[
                    "instrument_id",
                    "timestamp",
                    "factor_value",
                    "lookback_return",
                ]
            )
        frame = bars.sort_values(["instrument_id", self.timestamp_column]).copy()
        frame["close_price"] = frame["close_price"].astype(float)
        grouped = frame.groupby("instrument_id", sort=False)
        frame["lookback_return"] = grouped["close_price"].pct_change(
            periods=self.lookback_bars
        )
        frame["factor_value"] = -frame["lookback_return"]
        if self.min_avg_volume is not None:
            frame["avg_volume"] = _rolling_mean(
                grouped["volume"],
                window=self._liquidity_window(),
            )
            frame = frame.loc[frame["avg_volume"] >= self.min_avg_volume]
        if self.min_avg_turnover is not None:
            frame["avg_turnover"] = _rolling_mean(
                grouped["turnover"],
                window=self._liquidity_window(),
            )
            frame = frame.loc[frame["avg_turnover"] >= self.min_avg_turnover]
        frame = frame.loc[frame["factor_value"].notna()].copy()
        frame["timestamp"] = frame[self.timestamp_column]
        columns = ["instrument_id", "timestamp", "factor_value", "lookback_return"]
        optional_columns = [
            column for column in ("avg_volume", "avg_turnover") if column in frame
        ]
        return frame.loc[:, columns + optional_columns].reset_index(drop=True)

    def _liquidity_window(self) -> int:
        return self.liquidity_window_bars or self.lookback_bars


def _rolling_mean(series_groupby: object, *, window: int) -> pd.Series:
    result = series_groupby.transform(
        lambda values: values.astype(float).rolling(window, min_periods=window).mean()
    )
    return result.astype(float)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
