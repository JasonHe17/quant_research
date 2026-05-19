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
    "market_state",
    "market_downside_beta",
    "breadth_resilience",
    "limit_pressure_resilience",
    "return_turnover_correlation",
    "negative_return_persistence",
    "sell_pressure_absorption",
    "downside_turnover_decay",
    "sell_pressure_recovery",
    "daily_moving_average",
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
    downside_turnover_decay_windows: tuple[int, ...] = (48,)
    sell_pressure_recovery_windows: tuple[int, ...] = (48,)
    daily_moving_average_windows: tuple[int, ...] = (5, 10, 20)
    daily_moving_average_pairs: tuple[tuple[int, int], ...] = ((5, 20), (10, 20))
    market_downside_beta_windows: tuple[int, ...] = (48,)
    market_state_windows: tuple[int, ...] = (48,)
    breadth_resilience_windows: tuple[int, ...] = (48,)
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
            ("downside_turnover_decay_windows", self.downside_turnover_decay_windows),
            ("sell_pressure_recovery_windows", self.sell_pressure_recovery_windows),
            ("daily_moving_average_windows", self.daily_moving_average_windows),
            ("market_downside_beta_windows", self.market_downside_beta_windows),
            ("market_state_windows", self.market_state_windows),
            ("breadth_resilience_windows", self.breadth_resilience_windows),
            (
                "limit_pressure_resilience_windows",
                self.limit_pressure_resilience_windows,
            ),
        ):
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must be positive")
        if any(short <= 0 or long <= 0 for short, long in self.daily_moving_average_pairs):
            raise ValueError("daily_moving_average_pairs values must be positive")
        if any(short >= long for short, long in self.daily_moving_average_pairs):
            raise ValueError("daily_moving_average_pairs must be ordered short < long")


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
    if groups & {
        "signed_turnover_imbalance",
        "return_turnover_correlation",
        "sell_pressure_absorption",
        "downside_turnover_decay",
        "sell_pressure_recovery",
    }:
        required.append("turnover")
    if groups & {"vwap_deviation"}:
        required.append("volume")
    if groups & {"limit_pressure_resilience", "market_state"}:
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
    if "downside_turnover_decay" in groups:
        downside_turnover = frame["turnover"].astype(float).where(
            one_bar_return.lt(0.0), 0.0
        ).where(one_bar_return.notna())
        for window in config.downside_turnover_decay_windows:
            half_window = window // 2
            if half_window <= 0:
                raise ValueError("downside_turnover_decay_windows values must be at least 2")
            recent_downside_turnover = downside_turnover.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(half_window, min_periods=half_window).sum()
            )
            previous_downside_turnover = recent_downside_turnover.groupby(
                frame["instrument_id"]
            ).shift(half_window)
            total_downside_turnover = (
                previous_downside_turnover + recent_downside_turnover
            )
            column = f"intraday_downside_turnover_decay_5m_w{window}"
            output[column] = (
                previous_downside_turnover - recent_downside_turnover
            ) / total_downside_turnover.where(total_downside_turnover != 0.0)
            feature_columns.append(column)
    if "sell_pressure_recovery" in groups:
        positive_return = one_bar_return.clip(lower=0.0).where(one_bar_return.notna())
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        turnover = frame["turnover"].astype(float).where(one_bar_return.notna())
        upside_turnover = turnover.where(one_bar_return.gt(0.0), 0.0)
        for window in config.sell_pressure_recovery_windows:
            rolling_positive_return = positive_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_downside_return = downside_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_upside_turnover = upside_turnover.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_turnover = turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            recovery_ratio = rolling_positive_return / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            upside_participation = rolling_upside_turnover / rolling_turnover.where(
                rolling_turnover != 0.0
            )
            column = f"intraday_sell_pressure_recovery_5m_w{window}"
            output[column] = recovery_ratio * upside_participation
            feature_columns.append(column)
    if "daily_moving_average" in groups:
        _add_daily_moving_average_features(
            frame,
            output,
            feature_columns,
            windows=config.daily_moving_average_windows,
            pairs=config.daily_moving_average_pairs,
        )
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
    if "market_state" in groups:
        market_return = one_bar_return.groupby(frame["bar_end_time"]).transform("median")
        up_rate = one_bar_return.gt(0.0).where(one_bar_return.notna()).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_up_rate = frame["limit_up_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_down_rate = frame["limit_down_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        market_downside = (-market_return).clip(lower=0.0)
        weak_breadth = (0.5 - up_rate).clip(lower=0.0)
        limit_pressure = (limit_down_rate - limit_up_rate).clip(lower=0.0)
        state_columns = {
            "market_state_return_5m": market_return,
            "market_state_downside_5m": market_downside,
            "market_state_breadth_5m": up_rate,
            "market_state_weak_breadth_5m": weak_breadth,
            "market_state_limit_down_rate_5m": limit_down_rate,
            "market_state_limit_pressure_5m": limit_pressure,
        }
        for column, values in state_columns.items():
            output[column] = values
            feature_columns.append(column)
        for window in config.market_state_windows:
            for base_name, values in (
                ("downside_mean", market_downside),
                ("weak_breadth_mean", weak_breadth),
                ("limit_pressure_mean", limit_pressure),
            ):
                column = f"market_state_{base_name}_5m_w{window}"
                output[column] = _rolling_timestamp_state(
                    frame["bar_end_time"],
                    values,
                    window,
                )
                feature_columns.append(column)
    if "breadth_resilience" in groups:
        up_rate = one_bar_return.gt(0.0).where(one_bar_return.notna()).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        weak_breadth_pressure = (0.5 - up_rate).clip(lower=0.0)
        pressure_weighted_return = one_bar_return * weak_breadth_pressure
        for window in config.breadth_resilience_windows:
            rolling_weighted_return = pressure_weighted_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_pressure = weak_breadth_pressure.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            column = f"intraday_breadth_resilience_5m_w{window}"
            output[column] = rolling_weighted_return / rolling_pressure.where(
                rolling_pressure != 0.0
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


def _rolling_timestamp_state(
    timestamps: pd.Series,
    values: pd.Series,
    window: int,
) -> pd.Series:
    timestamp_values = (
        pd.DataFrame({"timestamp": timestamps, "value": values})
        .drop_duplicates("timestamp")
        .set_index("timestamp")["value"]
        .sort_index()
    )
    rolling_values = timestamp_values.rolling(window, min_periods=window).mean()
    return timestamps.map(rolling_values)


def _add_daily_moving_average_features(
    frame: pd.DataFrame,
    output: pd.DataFrame,
    feature_columns: list[str],
    *,
    windows: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
) -> None:
    session_date = _session_date(frame)
    daily_close = (
        frame.assign(_session_date=session_date)
        .sort_values(["instrument_id", "_session_date", "bar_end_time"])
        .groupby(["instrument_id", "_session_date"], sort=False)
        .tail(1)
        .loc[:, ["instrument_id", "_session_date", "close_price"]]
        .rename(columns={"close_price": "daily_close"})
        .reset_index(drop=True)
    )
    daily_close["daily_close"] = daily_close["daily_close"].astype(float)
    daily_grouped = daily_close.groupby("instrument_id", sort=False)["daily_close"]
    daily_features = daily_close.loc[:, ["instrument_id", "_session_date"]].copy()
    ma_by_window: dict[int, pd.Series] = {}
    unique_windows = tuple(dict.fromkeys(windows))
    for window in unique_windows:
        ma = daily_grouped.transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        ma_by_window[window] = ma
        previous_ma = ma.groupby(daily_close["instrument_id"]).shift(1)
        deviation_column = f"intraday_daily_ma_deviation_5m_d{window}"
        slope_column = f"intraday_daily_ma_slope_5m_d{window}"
        daily_features[deviation_column] = (
            daily_close["daily_close"] / ma.where(ma != 0.0) - 1.0
        )
        daily_features[slope_column] = ma / previous_ma.where(previous_ma != 0.0) - 1.0

    for short, long in pairs:
        if short not in ma_by_window:
            ma_by_window[short] = daily_grouped.transform(
                lambda values, short=short: values.rolling(
                    short, min_periods=short
                ).mean()
            )
        if long not in ma_by_window:
            ma_by_window[long] = daily_grouped.transform(
                lambda values, long=long: values.rolling(long, min_periods=long).mean()
            )
        column = f"intraday_daily_ma_spread_5m_s{short}_l{long}"
        daily_features[column] = (
            ma_by_window[short] / ma_by_window[long].where(ma_by_window[long] != 0.0)
            - 1.0
        )

    ribbon_windows = tuple(sorted(ma_by_window))
    if ribbon_windows:
        ribbon = pd.concat([ma_by_window[window] for window in ribbon_windows], axis=1)
        valid_ribbon = ribbon.notna().all(axis=1) & ribbon.gt(0.0).all(axis=1)
        ribbon_mean = ribbon.mean(axis=1).where(valid_ribbon)
        daily_features["intraday_daily_ma_ribbon_position_5m"] = (
            daily_close["daily_close"] / ribbon_mean.where(ribbon_mean != 0.0) - 1.0
        )
        daily_features["intraday_daily_ma_ribbon_dispersion_5m"] = (
            np.log(ribbon.where(valid_ribbon, np.nan)).std(axis=1, ddof=0)
        )
        trend_components = []
        for index, short in enumerate(ribbon_windows):
            for long in ribbon_windows[index + 1 :]:
                spread = ma_by_window[short] / ma_by_window[long].where(
                    ma_by_window[long] != 0.0
                ) - 1.0
                trend_components.append(np.sign(spread).where(spread.notna()))
        if trend_components:
            daily_features["intraday_daily_ma_ribbon_trend_score_5m"] = pd.concat(
                trend_components,
                axis=1,
            ).mean(axis=1)

    daily_feature_columns = [
        column
        for column in daily_features.columns
        if column not in {"instrument_id", "_session_date"}
    ]
    daily_features.loc[:, daily_feature_columns] = daily_features.groupby(
        "instrument_id", sort=False
    )[daily_feature_columns].shift(1)
    row_keys = pd.DataFrame(
        {
            "_row_id": frame.index,
            "instrument_id": frame["instrument_id"],
            "_session_date": session_date,
        }
    )
    aligned = (
        row_keys.merge(daily_features, on=["instrument_id", "_session_date"], how="left")
        .sort_values("_row_id")
        .reset_index(drop=True)
    )
    for column in daily_feature_columns:
        output[column] = aligned[column]
        feature_columns.append(column)


def _session_date(frame: pd.DataFrame) -> pd.Series:
    if "trade_date" in frame.columns:
        return frame["trade_date"].astype(str)
    parsed = pd.to_datetime(frame["bar_end_time"], errors="coerce", format="ISO8601")
    if parsed.notna().all():
        return parsed.dt.strftime("%Y-%m-%d")
    return frame["bar_end_time"].astype(str)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
