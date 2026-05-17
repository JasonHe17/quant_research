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
    "downside_volatility",
    "return_skewness",
    "money_flow",
    "signed_turnover_imbalance",
    "risk_adjusted_momentum",
    "volume_confirmed_momentum",
    "intraday_gap",
    "market_downside_beta",
    "limit_pressure_resilience",
    "return_turnover_correlation",
    "negative_return_persistence",
    "sell_pressure_absorption",
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
    downside_volatility_windows: tuple[int, ...] = (12, 48)
    return_skewness_windows: tuple[int, ...] = (12, 48)
    money_flow_windows: tuple[int, ...] = (12, 48)
    signed_turnover_imbalance_windows: tuple[int, ...] = (12, 48)
    risk_adjusted_momentum_windows: tuple[int, ...] = (12, 48)
    volume_confirmed_momentum_windows: tuple[int, ...] = (12, 48)
    return_turnover_correlation_windows: tuple[int, ...] = (12, 48)
    negative_return_persistence_windows: tuple[int, ...] = (48,)
    sell_pressure_absorption_windows: tuple[int, ...] = (48,)
    market_downside_beta_windows: tuple[int, ...] = (48,)
    limit_pressure_resilience_windows: tuple[int, ...] = (48,)

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
            ("downside_volatility_windows", self.downside_volatility_windows),
            ("return_skewness_windows", self.return_skewness_windows),
            ("money_flow_windows", self.money_flow_windows),
            (
                "signed_turnover_imbalance_windows",
                self.signed_turnover_imbalance_windows,
            ),
            ("risk_adjusted_momentum_windows", self.risk_adjusted_momentum_windows),
            ("volume_confirmed_momentum_windows", self.volume_confirmed_momentum_windows),
            (
                "return_turnover_correlation_windows",
                self.return_turnover_correlation_windows,
            ),
            (
                "negative_return_persistence_windows",
                self.negative_return_persistence_windows,
            ),
            ("sell_pressure_absorption_windows", self.sell_pressure_absorption_windows),
            ("market_downside_beta_windows", self.market_downside_beta_windows),
            (
                "limit_pressure_resilience_windows",
                self.limit_pressure_resilience_windows,
            ),
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
    if groups & {"bar_return", "liquidity_impact", "intraday_gap"}:
        required.append("open_price")
    if groups & {"price_position", "range_volatility"}:
        required.extend(["high_price", "low_price"])
    if groups & {"volume", "volume_confirmed_momentum"}:
        required.append("volume")
    if "money_flow" in groups:
        required.extend(["high_price", "low_price", "volume"])
    if groups & {"turnover", "liquidity_impact", "vwap_deviation"}:
        required.append("turnover")
    if groups & {"signed_turnover_imbalance", "return_turnover_correlation", "sell_pressure_absorption"}:
        required.append("turnover")
    if groups & {"vwap_deviation"}:
        required.append("volume")
    if "limit_pressure_resilience" in groups:
        required.extend(["limit_up_open", "limit_down_open"])
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
    if "risk_adjusted_momentum" in groups:
        for window in config.risk_adjusted_momentum_windows:
            momentum = grouped["close_price"].pct_change(periods=window)
            volatility = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).std()
            )
            column = f"intraday_risk_adjusted_momentum_5m_w{window}"
            output[column] = momentum / volatility.where(volatility != 0.0)
            feature_columns.append(column)
    if "volume_confirmed_momentum" in groups:
        frame["volume"] = frame["volume"].astype(float)
        volume_grouped = frame.groupby("instrument_id", sort=False)["volume"]
        for window in config.volume_confirmed_momentum_windows:
            momentum = grouped["close_price"].pct_change(periods=window)
            volume_mean = volume_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).mean()
            )
            volume_ratio = frame["volume"] / volume_mean.where(volume_mean != 0.0) - 1.0
            column = f"intraday_volume_confirmed_momentum_5m_w{window}"
            output[column] = momentum * volume_ratio
            feature_columns.append(column)
    if "volatility" in groups:
        for window in config.volatility_windows:
            column = f"intraday_volatility_5m_w{window}"
            output[column] = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).std()
            )
            feature_columns.append(column)
    if "downside_volatility" in groups:
        downside_return = one_bar_return.clip(upper=0.0)
        for window in config.downside_volatility_windows:
            column = f"intraday_downside_volatility_5m_w{window}"
            output[column] = downside_return.groupby(frame["instrument_id"]).transform(
                lambda values: (
                    values.pow(2).rolling(window, min_periods=window).mean()
                ).pow(0.5)
            )
            feature_columns.append(column)
    if "negative_return_persistence" in groups:
        negative_return = one_bar_return.lt(0.0).astype(float).where(
            one_bar_return.notna()
        )
        for window in config.negative_return_persistence_windows:
            column = f"intraday_negative_return_persistence_5m_w{window}"
            output[column] = negative_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).mean()
            )
            feature_columns.append(column)
    if "sell_pressure_absorption" in groups:
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        downside_turnover = frame["turnover"].astype(float).where(
            one_bar_return.lt(0.0), 0.0
        ).where(
            one_bar_return.notna()
        )
        for window in config.sell_pressure_absorption_windows:
            rolling_turnover = downside_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            rolling_downside_return = downside_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            column = f"intraday_sell_pressure_absorption_5m_w{window}"
            output[column] = rolling_turnover / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            feature_columns.append(column)
    if "market_downside_beta" in groups:
        market_return = one_bar_return.groupby(frame["bar_end_time"]).transform("median")
        downside_market_return = market_return.where(market_return < 0.0, 0.0)
        downside_covariance = one_bar_return * downside_market_return
        downside_variance = downside_market_return.pow(2)
        for window in config.market_downside_beta_windows:
            rolling_covariance = downside_covariance.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_variance = downside_variance.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            column = f"intraday_market_downside_beta_5m_w{window}"
            output[column] = rolling_covariance / rolling_variance.where(
                rolling_variance != 0.0
            )
            feature_columns.append(column)
    if "limit_pressure_resilience" in groups:
        limit_up_rate = frame["limit_up_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_down_rate = frame["limit_down_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_pressure = (limit_down_rate - limit_up_rate).clip(lower=0.0)
        pressure_weighted_return = one_bar_return * limit_pressure
        for window in config.limit_pressure_resilience_windows:
            rolling_weighted_return = pressure_weighted_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_pressure = limit_pressure.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            column = f"intraday_limit_pressure_resilience_5m_w{window}"
            output[column] = rolling_weighted_return / rolling_pressure.where(
                rolling_pressure != 0.0
            )
            feature_columns.append(column)
    if "return_skewness" in groups:
        for window in config.return_skewness_windows:
            column = f"intraday_return_skewness_5m_w{window}"
            output[column] = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).skew()
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
    if "intraday_gap" in groups:
        frame["open_price"] = frame["open_price"].astype(float)
        previous_close = grouped["close_price"].shift(1)
        column = "intraday_gap_5m"
        output[column] = frame["open_price"] / previous_close.where(
            previous_close != 0.0
        ) - 1.0
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
    if "money_flow" in groups:
        frame["high_price"] = frame["high_price"].astype(float)
        frame["low_price"] = frame["low_price"].astype(float)
        frame["volume"] = frame["volume"].astype(float)
        bar_range = frame["high_price"] - frame["low_price"]
        close_location = (
            (2.0 * frame["close_price"] - frame["high_price"] - frame["low_price"])
            / bar_range.where(bar_range != 0.0)
        )
        signed_volume = close_location * frame["volume"]
        for window in config.money_flow_windows:
            rolling_signed_volume = signed_volume.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            rolling_volume = frame.groupby("instrument_id", sort=False)["volume"].transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            column = f"intraday_money_flow_5m_w{window}"
            output[column] = rolling_signed_volume / rolling_volume.where(
                rolling_volume != 0.0
            )
            feature_columns.append(column)
    if "signed_turnover_imbalance" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        signed_turnover = np.sign(one_bar_return) * frame["turnover"]
        turnover_grouped = frame.groupby("instrument_id", sort=False)["turnover"]
        for window in config.signed_turnover_imbalance_windows:
            rolling_signed_turnover = signed_turnover.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_turnover = turnover_grouped.transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            column = f"intraday_signed_turnover_imbalance_5m_w{window}"
            output[column] = rolling_signed_turnover / rolling_turnover.where(
                rolling_turnover != 0.0
            )
            feature_columns.append(column)
    if "return_turnover_correlation" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        for window in config.return_turnover_correlation_windows:
            column = f"intraday_return_turnover_corr_5m_w{window}"
            output[column] = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).corr(
                    frame.loc[values.index, "turnover"]
                )
            )
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
