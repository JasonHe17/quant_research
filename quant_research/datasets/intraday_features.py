"""Intraday alpha feature builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_VALID_GROUPS = {
    "reversal",
    "momentum",
    "volatility",
    "price_position",
    "range_volatility",
    "efficiency",
    "volume",
    "turnover",
    "bar_return",
    "liquidity_impact",
    "vwap_deviation",
    "all",
}


@dataclass(frozen=True, slots=True)
class IntradayFeatureConfig:
    """Configuration for 5-minute intraday alpha feature generation."""

    factor_groups: tuple[str, ...] = ("reversal",)
    reversal_lookback_bars: tuple[int, ...] = (1, 3, 6)
    momentum_lookback_bars: tuple[int, ...] = (3, 6, 12)
    volatility_windows: tuple[int, ...] = (6, 12, 24)
    price_position_windows: tuple[int, ...] = (48,)
    range_volatility_windows: tuple[int, ...] = (12, 48)
    efficiency_windows: tuple[int, ...] = (12, 48)
    volume_windows: tuple[int, ...] = (12, 48)
    turnover_windows: tuple[int, ...] = (12, 48)
    vwap_deviation_windows: tuple[int, ...] = (48,)

    def __post_init__(self) -> None:
        unknown = set(self.factor_groups) - _VALID_GROUPS
        if unknown:
            raise ValueError(f"unknown factor groups: {sorted(unknown)}")
        for name, values in (
            ("reversal_lookback_bars", self.reversal_lookback_bars),
            ("momentum_lookback_bars", self.momentum_lookback_bars),
            ("volatility_windows", self.volatility_windows),
            ("price_position_windows", self.price_position_windows),
            ("range_volatility_windows", self.range_volatility_windows),
            ("efficiency_windows", self.efficiency_windows),
            ("volume_windows", self.volume_windows),
            ("turnover_windows", self.turnover_windows),
            ("vwap_deviation_windows", self.vwap_deviation_windows),
        ):
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must be positive")


def build_intraday_feature_matrix(
    bars: pd.DataFrame,
    config: IntradayFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build wide 5-minute intraday alpha features from OHLCV bars."""

    config = config or IntradayFeatureConfig()
    groups = _expanded_groups(config.factor_groups)
    required = ["instrument_id", "bar_end_time", "close_price"]
    if groups & {"bar_return", "liquidity_impact"}:
        required.append("open_price")
    if groups & {"price_position", "range_volatility"}:
        required.extend(["high_price", "low_price"])
    if "volume" in groups:
        required.append("volume")
    if groups & {"turnover", "liquidity_impact", "vwap_deviation"}:
        required.append("turnover")
    if groups & {"vwap_deviation"}:
        required.append("volume")
    _require_columns(bars, tuple(required))
    frame = bars.sort_values(["instrument_id", "bar_end_time"]).copy()
    frame["close_price"] = frame["close_price"].astype(float)
    grouped = frame.groupby("instrument_id", sort=False)
    output = frame.loc[:, ["bar_end_time", "instrument_id"]].rename(
        columns={"bar_end_time": "timestamp"}
    )
    feature_columns: list[str] = []
    one_bar_return = grouped["close_price"].pct_change(periods=1)
    if "bar_return" in groups:
        frame["open_price"] = frame["open_price"].astype(float)
        column = "intraday_bar_return_5m"
        output[column] = frame["close_price"] / frame["open_price"] - 1.0
        feature_columns.append(column)
    if "reversal" in groups:
        for lookback_bars in config.reversal_lookback_bars:
            column = f"intraday_reversal_5m_lb{lookback_bars}"
            output[column] = -grouped["close_price"].pct_change(periods=lookback_bars)
            feature_columns.append(column)
    if "momentum" in groups:
        for lookback_bars in config.momentum_lookback_bars:
            column = f"intraday_momentum_5m_lb{lookback_bars}"
            output[column] = grouped["close_price"].pct_change(periods=lookback_bars)
            feature_columns.append(column)
    if "volatility" in groups:
        for window in config.volatility_windows:
            column = f"intraday_volatility_5m_w{window}"
            output[column] = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).std()
            )
            feature_columns.append(column)
    if "price_position" in groups:
        frame["high_price"] = frame["high_price"].astype(float)
        frame["low_price"] = frame["low_price"].astype(float)
        high_grouped = frame.groupby("instrument_id", sort=False)["high_price"]
        low_grouped = frame.groupby("instrument_id", sort=False)["low_price"]
        for window in config.price_position_windows:
            rolling_high = high_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).max()
            )
            rolling_low = low_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).min()
            )
            range_width = rolling_high - rolling_low
            column = f"intraday_range_position_5m_w{window}"
            output[column] = (
                (frame["close_price"] - rolling_low) / range_width.where(range_width != 0.0)
            ) - 0.5
            feature_columns.append(column)
    if "range_volatility" in groups:
        frame["high_price"] = frame["high_price"].astype(float)
        frame["low_price"] = frame["low_price"].astype(float)
        valid_low = frame["low_price"].where(frame["low_price"] > 0.0)
        log_range = (frame["high_price"] / valid_low).where(valid_low.notna())
        log_range = np.log(log_range)
        for window in config.range_volatility_windows:
            column = f"intraday_range_volatility_5m_w{window}"
            output[column] = log_range.groupby(frame["instrument_id"]).transform(
                lambda values: (
                    values.pow(2).rolling(window, min_periods=window).mean()
                ).pow(0.5)
            )
            feature_columns.append(column)
    if "efficiency" in groups:
        close_shifted = grouped["close_price"].shift
        absolute_price_change = grouped["close_price"].diff().abs()
        for window in config.efficiency_windows:
            directional_move = (frame["close_price"] - close_shifted(window)).abs()
            path_length = absolute_price_change.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            column = f"intraday_efficiency_ratio_5m_w{window}"
            output[column] = directional_move / path_length.where(path_length != 0.0)
            feature_columns.append(column)
    if "volume" in groups:
        frame["volume"] = frame["volume"].astype(float)
        volume_grouped = frame.groupby("instrument_id", sort=False)["volume"]
        for window in config.volume_windows:
            mean = volume_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).mean()
            )
            column = f"intraday_volume_ratio_5m_w{window}"
            output[column] = frame["volume"] / mean - 1.0
            feature_columns.append(column)
    if "turnover" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        turnover_grouped = frame.groupby("instrument_id", sort=False)["turnover"]
        for window in config.turnover_windows:
            mean = turnover_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).mean()
            )
            std = turnover_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).std()
            )
            ratio_column = f"intraday_turnover_ratio_5m_w{window}"
            zscore_column = f"intraday_turnover_zscore_5m_w{window}"
            output[ratio_column] = frame["turnover"] / mean - 1.0
            output[zscore_column] = (frame["turnover"] - mean) / std
            feature_columns.extend([ratio_column, zscore_column])
    if "liquidity_impact" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        column = "intraday_amihud_5m"
        output[column] = one_bar_return.abs() / frame["turnover"].replace(0.0, pd.NA)
        feature_columns.append(column)
    if "vwap_deviation" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        frame["volume"] = frame["volume"].astype(float)
        turnover_grouped = frame.groupby("instrument_id", sort=False)["turnover"]
        volume_grouped = frame.groupby("instrument_id", sort=False)["volume"]
        for window in config.vwap_deviation_windows:
            rolling_turnover = turnover_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            rolling_volume = volume_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            rolling_vwap = rolling_turnover / rolling_volume.where(rolling_volume != 0.0)
            column = f"intraday_vwap_deviation_5m_w{window}"
            output[column] = frame["close_price"] / rolling_vwap - 1.0
            feature_columns.append(column)
    return output.loc[output[feature_columns].notna().any(axis=1)].reset_index(
        drop=True
    )


def _expanded_groups(groups: tuple[str, ...]) -> set[str]:
    selected = set(groups)
    if "all" not in selected:
        return selected
    return _VALID_GROUPS - {"all"}


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
