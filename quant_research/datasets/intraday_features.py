"""Intraday alpha feature builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_VALID_GROUPS = {
    "reversal",
    "cross_sectional_reversal",
    "conditional_reversal",
    "eod_reversal",
    "momentum",
    "volatility",
    "volatility_state_change",
    "price_position",
    "range_volatility",
    "efficiency",
    "volume",
    "volume_distribution_shape",
    "turnover",
    "turnover_stability",
    "liquidity_reliability",
    "liquidity_reliability_recovery",
    "liquidity_reliability_recovery_balance",
    "bar_return",
    "liquidity_impact",
    "vwap_deviation",
    "downside_volatility",
    "return_skewness",
    "lottery_max",
    "money_flow",
    "signed_turnover_imbalance",
    "risk_adjusted_momentum",
    "volume_confirmed_momentum",
    "intraday_gap",
    "market_state",
    "market_downside_beta",
    "breadth_resilience",
    "breadth_shock_residual_resilience",
    "limit_pressure_resilience",
    "regime_conditioned",
    "return_turnover_correlation",
    "negative_return_persistence",
    "sell_pressure_absorption",
    "downside_turnover_decay",
    "sell_pressure_recovery",
    "sell_pressure_exhaustion",
    "sell_pressure_exhaustion_persistence",
    "microstructure_exhaustion_alert",
    "microstructure_recovery_speed",
    "same_slot_intraday_memory",
    "overnight_intraday_tug_of_war",
    "weak_tape_overnight_gap",
    "sell_pressure_quality_state",
    "event_shock_proxy",
    "daily_moving_average",
    "all",
}


@dataclass(frozen=True, slots=True)
class IntradayFeatureConfig:
    """Configuration for 5-minute intraday alpha feature generation."""

    factor_groups: tuple[str, ...] = ("reversal",)
    reversal_lookback_bars: tuple[int, ...] = (1, 6, 12, 24)
    cross_sectional_reversal_lookback_bars: tuple[int, ...] = (1, 6, 12, 24)
    conditional_reversal_lookback_bars: tuple[int, ...] = (1, 6, 12, 24)
    conditional_reversal_state_windows: tuple[int, ...] = (12,)
    conditional_reversal_low_vol_quantile: float = 0.5
    conditional_reversal_min_volume_ratio: float = 0.0
    eod_reversal_lookback_bars: tuple[int, ...] = (1, 6, 12, 24)
    eod_reversal_tail_bars: int = 6
    eod_reversal_weight: float = 1.5
    momentum_lookback_bars: tuple[int, ...] = (3, 6, 12)
    volatility_windows: tuple[int, ...] = (6, 12, 24)
    volatility_state_change_specs: tuple[tuple[int, int], ...] = ((12, 48), (24, 96))
    price_position_windows: tuple[int, ...] = (48,)
    range_volatility_windows: tuple[int, ...] = (12, 48)
    efficiency_windows: tuple[int, ...] = (12, 48)
    volume_windows: tuple[int, ...] = (12, 48)
    volume_distribution_windows: tuple[int, ...] = (24, 48, 96)
    turnover_windows: tuple[int, ...] = (12, 48)
    turnover_stability_windows: tuple[int, ...] = (48,)
    liquidity_reliability_windows: tuple[int, ...] = (48,)
    liquidity_reliability_recovery_specs: tuple[tuple[int, int, int], ...] = (
        (48, 12, 24),
    )
    liquidity_reliability_recovery_balance_specs: tuple[tuple[int, int, int], ...] = (
        (48, 12, 24),
    )
    vwap_deviation_windows: tuple[int, ...] = (48,)
    downside_volatility_windows: tuple[int, ...] = (12, 48)
    return_skewness_windows: tuple[int, ...] = (12, 48)
    lottery_max_windows: tuple[int, ...] = (24, 48, 96)
    money_flow_windows: tuple[int, ...] = (12, 48)
    signed_turnover_imbalance_windows: tuple[int, ...] = (12, 48)
    risk_adjusted_momentum_windows: tuple[int, ...] = (12, 48)
    volume_confirmed_momentum_windows: tuple[int, ...] = (12, 48)
    return_turnover_correlation_windows: tuple[int, ...] = (12, 48)
    negative_return_persistence_windows: tuple[int, ...] = (48,)
    sell_pressure_absorption_windows: tuple[int, ...] = (48,)
    downside_turnover_decay_windows: tuple[int, ...] = (48,)
    sell_pressure_recovery_windows: tuple[int, ...] = (48,)
    sell_pressure_exhaustion_windows: tuple[int, ...] = (48,)
    sell_pressure_exhaustion_persistence_specs: tuple[tuple[int, int, int], ...] = (
        (96, 24, 48),
    )
    microstructure_exhaustion_windows: tuple[int, ...] = (48,)
    microstructure_recovery_windows: tuple[int, ...] = (24, 48)
    microstructure_recovery_acceleration_specs: tuple[tuple[int, int], ...] = (
        (12, 48),
        (24, 96),
    )
    same_slot_memory_windows: tuple[int, ...] = (5, 20)
    weak_tape_gap_windows: tuple[int, ...] = (48,)
    sell_pressure_quality_windows: tuple[int, ...] = (48,)
    event_shock_windows: tuple[int, ...] = (48,)
    daily_moving_average_windows: tuple[int, ...] = (5, 10, 20)
    daily_moving_average_pairs: tuple[tuple[int, int], ...] = ((5, 20), (10, 20))
    market_downside_beta_windows: tuple[int, ...] = (48,)
    market_state_windows: tuple[int, ...] = (48,)
    breadth_resilience_windows: tuple[int, ...] = (48,)
    breadth_shock_residual_resilience_windows: tuple[int, ...] = (48,)
    limit_pressure_resilience_windows: tuple[int, ...] = (48,)
    regime_conditioned_lookback_bars: tuple[int, ...] = (12, 24)
    regime_conditioned_state_windows: tuple[int, ...] = (48,)

    def __post_init__(self) -> None:
        unknown = set(self.factor_groups) - _VALID_GROUPS
        if unknown:
            raise ValueError(f"unknown factor groups: {sorted(unknown)}")
        for name, values in (
            ("reversal_lookback_bars", self.reversal_lookback_bars),
            (
                "cross_sectional_reversal_lookback_bars",
                self.cross_sectional_reversal_lookback_bars,
            ),
            (
                "conditional_reversal_lookback_bars",
                self.conditional_reversal_lookback_bars,
            ),
            (
                "conditional_reversal_state_windows",
                self.conditional_reversal_state_windows,
            ),
            ("eod_reversal_lookback_bars", self.eod_reversal_lookback_bars),
            ("momentum_lookback_bars", self.momentum_lookback_bars),
            ("volatility_windows", self.volatility_windows),
            ("price_position_windows", self.price_position_windows),
            ("range_volatility_windows", self.range_volatility_windows),
            ("efficiency_windows", self.efficiency_windows),
            ("volume_windows", self.volume_windows),
            ("volume_distribution_windows", self.volume_distribution_windows),
            ("turnover_windows", self.turnover_windows),
            ("turnover_stability_windows", self.turnover_stability_windows),
            ("liquidity_reliability_windows", self.liquidity_reliability_windows),
            ("vwap_deviation_windows", self.vwap_deviation_windows),
            ("downside_volatility_windows", self.downside_volatility_windows),
            ("return_skewness_windows", self.return_skewness_windows),
            ("lottery_max_windows", self.lottery_max_windows),
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
            ("sell_pressure_exhaustion_windows", self.sell_pressure_exhaustion_windows),
            (
                "microstructure_exhaustion_windows",
                self.microstructure_exhaustion_windows,
            ),
            ("microstructure_recovery_windows", self.microstructure_recovery_windows),
            ("same_slot_memory_windows", self.same_slot_memory_windows),
            ("weak_tape_gap_windows", self.weak_tape_gap_windows),
            ("sell_pressure_quality_windows", self.sell_pressure_quality_windows),
            ("event_shock_windows", self.event_shock_windows),
            ("daily_moving_average_windows", self.daily_moving_average_windows),
            ("market_downside_beta_windows", self.market_downside_beta_windows),
            ("market_state_windows", self.market_state_windows),
            ("breadth_resilience_windows", self.breadth_resilience_windows),
            (
                "breadth_shock_residual_resilience_windows",
                self.breadth_shock_residual_resilience_windows,
            ),
            (
                "limit_pressure_resilience_windows",
                self.limit_pressure_resilience_windows,
            ),
            (
                "regime_conditioned_lookback_bars",
                self.regime_conditioned_lookback_bars,
            ),
            (
                "regime_conditioned_state_windows",
                self.regime_conditioned_state_windows,
            ),
        ):
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must be positive")
        if not 0.0 < self.conditional_reversal_low_vol_quantile <= 1.0:
            raise ValueError(
                "conditional_reversal_low_vol_quantile must be in (0, 1]"
            )
        if self.conditional_reversal_min_volume_ratio <= -1.0:
            raise ValueError(
                "conditional_reversal_min_volume_ratio must be greater than -1"
            )
        if self.eod_reversal_tail_bars <= 0:
            raise ValueError("eod_reversal_tail_bars must be positive")
        if self.eod_reversal_weight <= 0.0:
            raise ValueError("eod_reversal_weight must be positive")
        if any(value <= 1 for value in self.volume_distribution_windows):
            raise ValueError("volume_distribution_windows values must be at least 2")
        if any(short <= 0 or long <= 0 for short, long in self.daily_moving_average_pairs):
            raise ValueError("daily_moving_average_pairs values must be positive")
        if any(short >= long for short, long in self.daily_moving_average_pairs):
            raise ValueError("daily_moving_average_pairs must be ordered short < long")
        for short_window, long_window in self.volatility_state_change_specs:
            if short_window <= 1 or long_window <= 1:
                raise ValueError(
                    "volatility_state_change_specs values must be at least 2"
                )
            if short_window >= long_window:
                raise ValueError(
                    "volatility_state_change_specs must be ordered short < long"
                )
        for short_window, long_window in self.microstructure_recovery_acceleration_specs:
            if short_window <= 1 or long_window <= 1:
                raise ValueError(
                    "microstructure_recovery_acceleration_specs values must be at least 2"
                )
            if short_window >= long_window:
                raise ValueError(
                    "microstructure_recovery_acceleration_specs must be ordered short < long"
                )
        for long_window, short_window, medium_window in (
            self.sell_pressure_exhaustion_persistence_specs
        ):
            if long_window <= 1 or short_window <= 1 or medium_window <= 1:
                raise ValueError(
                    "sell_pressure_exhaustion_persistence_specs values must be at least 2"
                )
            if long_window <= max(short_window, medium_window):
                raise ValueError(
                    "sell_pressure_exhaustion_persistence_specs must be ordered "
                    "long > short and long > medium"
                )
        if any(value <= 1 for value in self.microstructure_exhaustion_windows):
            raise ValueError(
                "microstructure_exhaustion_windows values must be at least 2"
            )
        for long_window, capacity_window, recovery_window in (
            self.liquidity_reliability_recovery_specs
        ):
            if long_window <= 1 or capacity_window <= 1 or recovery_window <= 1:
                raise ValueError(
                    "liquidity_reliability_recovery_specs values must be at least 2"
                )
            if long_window <= max(capacity_window, recovery_window):
                raise ValueError(
                    "liquidity_reliability_recovery_specs must be ordered "
                    "long > capacity and long > recovery"
                )
        for long_window, capacity_window, recovery_window in (
            self.liquidity_reliability_recovery_balance_specs
        ):
            if long_window <= 1 or capacity_window <= 1 or recovery_window <= 1:
                raise ValueError(
                    "liquidity_reliability_recovery_balance_specs values must be at least 2"
                )
            if long_window <= max(capacity_window, recovery_window):
                raise ValueError(
                    "liquidity_reliability_recovery_balance_specs must be ordered "
                    "long > capacity and long > recovery"
                )


def build_intraday_feature_matrix(
    bars: pd.DataFrame,
    config: IntradayFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build wide 5-minute intraday alpha features from OHLCV bars."""

    config = config or IntradayFeatureConfig()
    groups = _expanded_groups(config.factor_groups)
    required = ["instrument_id", "bar_end_time", "close_price"]
    if groups & {
        "bar_return",
        "liquidity_impact",
        "intraday_gap",
        "overnight_intraday_tug_of_war",
        "weak_tape_overnight_gap",
        "event_shock_proxy",
    }:
        required.append("open_price")
    if groups & {"price_position", "range_volatility"}:
        required.extend(["high_price", "low_price"])
    if groups & {"volume", "volume_confirmed_momentum", "conditional_reversal"}:
        required.append("volume")
    if "volume_distribution_shape" in groups:
        required.append("volume")
    if "money_flow" in groups:
        required.extend(["high_price", "low_price", "volume"])
    if groups & {
        "turnover",
        "turnover_stability",
        "liquidity_reliability",
        "liquidity_reliability_recovery",
        "liquidity_reliability_recovery_balance",
        "liquidity_impact",
        "vwap_deviation",
        "event_shock_proxy",
    }:
        required.append("turnover")
    if groups & {
        "signed_turnover_imbalance",
        "return_turnover_correlation",
        "sell_pressure_absorption",
        "downside_turnover_decay",
        "sell_pressure_recovery",
        "sell_pressure_exhaustion",
        "sell_pressure_exhaustion_persistence",
        "microstructure_exhaustion_alert",
        "microstructure_recovery_speed",
        "sell_pressure_quality_state",
    }:
        required.append("turnover")
    if groups & {"vwap_deviation"}:
        required.append("volume")
    if groups & {
        "limit_pressure_resilience",
        "market_state",
        "event_shock_proxy",
        "regime_conditioned",
        "microstructure_exhaustion_alert",
    }:
        required.extend(["limit_up_open", "limit_down_open"])
    _require_columns(bars, tuple(required))
    frame = bars.sort_values(["instrument_id", "bar_end_time"]).copy()
    frame["close_price"] = frame["close_price"].astype(float)
    grouped = frame.groupby("instrument_id", sort=False)
    output = _FeatureOutput(
        frame.loc[:, ["bar_end_time", "instrument_id"]].rename(
            columns={"bar_end_time": "timestamp"}
        )
    )
    feature_columns: list[str] = []
    one_bar_return = grouped["close_price"].pct_change(periods=1)
    lookback_returns: dict[int, pd.Series] = {}

    def lookback_return(lookback_bars: int) -> pd.Series:
        if lookback_bars not in lookback_returns:
            lookback_returns[lookback_bars] = grouped["close_price"].pct_change(
                periods=lookback_bars
            )
        return lookback_returns[lookback_bars]

    if "bar_return" in groups:
        frame["open_price"] = frame["open_price"].astype(float)
        column = "intraday_bar_return_5m"
        output[column] = frame["close_price"] / frame["open_price"] - 1.0
        feature_columns.append(column)
    if "reversal" in groups:
        for lookback_bars in config.reversal_lookback_bars:
            column = f"intraday_reversal_5m_lb{lookback_bars}"
            output[column] = -lookback_return(lookback_bars)
            feature_columns.append(column)
    if "cross_sectional_reversal" in groups:
        for lookback_bars in config.cross_sectional_reversal_lookback_bars:
            returns = lookback_return(lookback_bars)
            market_return = returns.groupby(frame["bar_end_time"]).transform("median")
            column = f"intraday_cross_sectional_reversal_5m_lb{lookback_bars}"
            output[column] = -(returns - market_return)
            feature_columns.append(column)
    if "conditional_reversal" in groups:
        frame["volume"] = frame["volume"].astype(float)
        volume_grouped = frame.groupby("instrument_id", sort=False)["volume"]
        low_vol_by_window: dict[int, pd.Series] = {}
        volume_confirmed_by_window: dict[int, pd.Series] = {}
        for window in config.conditional_reversal_state_windows:
            realized_volatility = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values, window=window: values.rolling(
                    window,
                    min_periods=window,
                ).std()
            )
            low_vol_threshold = realized_volatility.groupby(
                frame["bar_end_time"]
            ).transform(
                lambda values: values.quantile(
                    config.conditional_reversal_low_vol_quantile
                )
            )
            low_vol_by_window[window] = (
                realized_volatility.notna()
                & low_vol_threshold.notna()
                & (realized_volatility <= low_vol_threshold)
            )
            volume_mean = volume_grouped.transform(
                lambda values, window=window: values.rolling(
                    window,
                    min_periods=window,
                ).mean()
            )
            volume_ratio = frame["volume"] / volume_mean.where(volume_mean != 0.0) - 1.0
            volume_confirmed_by_window[window] = (
                volume_ratio >= config.conditional_reversal_min_volume_ratio
            )
        for lookback_bars in config.conditional_reversal_lookback_bars:
            returns = lookback_return(lookback_bars)
            reversal = -returns
            recent_loser = returns < 0.0
            for window in config.conditional_reversal_state_windows:
                column = (
                    "intraday_low_vol_volume_confirmed_reversal_5m_"
                    f"lb{lookback_bars}_w{window}"
                )
                output[column] = reversal.where(
                    recent_loser
                    & low_vol_by_window[window]
                    & volume_confirmed_by_window[window]
                )
                feature_columns.append(column)
    if "eod_reversal" in groups:
        eod_tail = _session_tail_mask(frame, tail_bars=config.eod_reversal_tail_bars)
        for lookback_bars in config.eod_reversal_lookback_bars:
            column = (
                f"intraday_eod_reversal_5m_lb{lookback_bars}"
                f"_tail{config.eod_reversal_tail_bars}"
            )
            output[column] = (
                -lookback_return(lookback_bars) * config.eod_reversal_weight
            ).where(eod_tail)
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
    if "volatility_state_change" in groups:
        realized_volatility_by_window: dict[int, pd.Series] = {}
        unique_windows = tuple(
            sorted(
                {
                    window
                    for spec in config.volatility_state_change_specs
                    for window in spec
                }
            )
        )
        for window in unique_windows:
            realized_volatility_by_window[window] = one_bar_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).std())
        for short_window, long_window in config.volatility_state_change_specs:
            short_volatility = realized_volatility_by_window[short_window]
            long_volatility = realized_volatility_by_window[long_window]
            relative_column = (
                "intraday_volatility_state_change_5m_"
                f"s{short_window}_l{long_window}"
            )
            trend_column = (
                "intraday_volatility_state_trend_5m_"
                f"s{short_window}_l{long_window}"
            )
            output[relative_column] = (
                short_volatility / long_volatility.where(long_volatility != 0.0) - 1.0
            )
            output[trend_column] = (
                short_volatility
                - short_volatility.groupby(frame["instrument_id"]).shift(short_window)
            ) / long_volatility.where(long_volatility != 0.0)
            feature_columns.extend([relative_column, trend_column])
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
    if "sell_pressure_quality_state" in groups:
        positive_return = one_bar_return.clip(lower=0.0).where(one_bar_return.notna())
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        turnover = frame["turnover"].astype(float).where(one_bar_return.notna())
        downside_turnover = turnover.where(one_bar_return.lt(0.0), 0.0)
        up_rate = one_bar_return.gt(0.0).where(one_bar_return.notna()).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        weak_breadth_pressure = (0.5 - up_rate).clip(lower=0.0)
        for window in config.sell_pressure_quality_windows:
            rolling_downside_return = downside_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_downside_turnover = downside_turnover.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_positive_return = positive_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_weak_breadth = weak_breadth_pressure.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).mean())
            absorption = rolling_downside_turnover / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            absorption_score = np.log1p(absorption.clip(lower=0.0))
            recovery_ratio = rolling_positive_return / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            recovery_balance = (
                2.0
                * recovery_ratio
                / (1.0 + recovery_ratio * recovery_ratio)
            ).where(recovery_ratio >= 0.0)
            tape_pressure = (2.0 * rolling_weak_breadth).clip(lower=0.0, upper=1.0)
            tape_quality = 1.0 - tape_pressure
            quality_column = f"intraday_sell_pressure_absorption_quality_5m_w{window}"
            risk_column = f"intraday_false_absorption_risk_5m_w{window}"
            output[quality_column] = (
                absorption_score * recovery_balance * tape_quality
            )
            output[risk_column] = (
                absorption_score * (1.0 - recovery_balance) * (1.0 + tape_pressure)
            )
            feature_columns.extend([quality_column, risk_column])
    if "microstructure_exhaustion_alert" in groups:
        turnover = frame["turnover"].astype(float).where(one_bar_return.notna())
        positive_return = one_bar_return.clip(lower=0.0).where(one_bar_return.notna())
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        downside_turnover = turnover.where(one_bar_return.lt(0.0), 0.0)
        limit_up_rate = frame["limit_up_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_down_rate = frame["limit_down_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_pressure = (limit_down_rate - limit_up_rate).clip(lower=0.0)
        for window in config.microstructure_exhaustion_windows:
            half_window = window // 2
            if half_window <= 0:
                raise ValueError(
                    "microstructure_exhaustion_windows values must be at least 2"
                )
            rolling_downside_turnover = downside_turnover.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_downside_return = downside_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            recent_downside_turnover = downside_turnover.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(
                    half_window,
                    min_periods=half_window,
                ).sum()
            )
            previous_downside_turnover = recent_downside_turnover.groupby(
                frame["instrument_id"]
            ).shift(half_window)
            recent_downside_return = downside_return.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(
                    half_window,
                    min_periods=half_window,
                ).sum()
            )
            previous_downside_return = recent_downside_return.groupby(
                frame["instrument_id"]
            ).shift(half_window)
            recent_positive_return = positive_return.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(
                    half_window,
                    min_periods=half_window,
                ).sum()
            )
            previous_positive_return = recent_positive_return.groupby(
                frame["instrument_id"]
            ).shift(half_window)

            absorption_load = np.log1p(
                (
                    rolling_downside_turnover
                    / rolling_downside_return.where(rolling_downside_return != 0.0)
                ).clip(lower=0.0)
            )
            absorption_speed = recent_downside_turnover / previous_downside_turnover.where(
                previous_downside_turnover != 0.0
            )
            recent_impact = recent_downside_return / recent_downside_turnover.where(
                recent_downside_turnover != 0.0
            )
            previous_impact = previous_downside_return / previous_downside_turnover.where(
                previous_downside_turnover != 0.0
            )
            impact_total = recent_impact + previous_impact
            impact_acceleration = (
                (recent_impact - previous_impact)
                / impact_total.where(impact_total != 0.0)
            ).clip(lower=0.0)
            previous_recovery = (
                previous_positive_return
                / previous_downside_return.where(previous_downside_return != 0.0)
            )
            recent_recovery = (
                recent_positive_return
                / recent_downside_return.where(recent_downside_return != 0.0)
            )
            recovery_total = previous_recovery.abs() + recent_recovery.abs()
            recovery_decay = (
                (previous_recovery - recent_recovery)
                / recovery_total.where(recovery_total != 0.0)
            ).clip(lower=0.0).fillna(0.0)
            lagged_limit_pressure = _lagged_rolling_timestamp_state(
                frame["bar_end_time"],
                limit_pressure,
                window,
            )
            limit_withdrawal_proxy = (
                limit_pressure + (limit_pressure - lagged_limit_pressure).clip(lower=0.0)
            )
            impact_strain = (
                absorption_load
                * np.log1p(absorption_speed.clip(lower=0.0))
                * impact_acceleration
            )
            strain_column = f"intraday_microstructure_impact_strain_5m_w{window}"
            alert_column = f"intraday_microstructure_exhaustion_alert_5m_w{window}"
            output[strain_column] = impact_strain
            output[alert_column] = impact_strain * (1.0 + recovery_decay) * (
                1.0 + limit_withdrawal_proxy
            )
            feature_columns.extend([strain_column, alert_column])
    if groups & {"sell_pressure_exhaustion", "sell_pressure_exhaustion_persistence"}:
        positive_return = one_bar_return.clip(lower=0.0).where(one_bar_return.notna())
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        turnover = frame["turnover"].astype(float).where(one_bar_return.notna())
        upside_turnover = turnover.where(one_bar_return.gt(0.0), 0.0)
        downside_turnover = turnover.where(one_bar_return.lt(0.0), 0.0)
        exhaustion_windows = set()
        if "sell_pressure_exhaustion" in groups:
            exhaustion_windows.update(config.sell_pressure_exhaustion_windows)
        if "sell_pressure_exhaustion_persistence" in groups:
            for spec in config.sell_pressure_exhaustion_persistence_specs:
                exhaustion_windows.update(spec)
        exhaustion_by_window: dict[int, pd.Series] = {}
        for window in sorted(exhaustion_windows):
            half_window = window // 2
            if half_window <= 0:
                raise ValueError("sell_pressure_exhaustion_windows values must be at least 2")
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
            recovery_ratio = rolling_positive_return / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            upside_participation = rolling_upside_turnover / rolling_turnover.where(
                rolling_turnover != 0.0
            )
            downside_decay = (
                previous_downside_turnover - recent_downside_turnover
            ) / total_downside_turnover.where(total_downside_turnover != 0.0)
            exhaustion = (
                np.log1p(recovery_ratio.clip(lower=0.0))
                * upside_participation
                * downside_decay.clip(lower=0.0)
            )
            exhaustion_by_window[window] = exhaustion
            if "sell_pressure_exhaustion" in groups and (
                window in config.sell_pressure_exhaustion_windows
            ):
                column = f"intraday_sell_pressure_exhaustion_5m_w{window}"
                output[column] = exhaustion
                feature_columns.append(column)
        if "sell_pressure_exhaustion_persistence" in groups:
            for long_window, short_window, medium_window in (
                config.sell_pressure_exhaustion_persistence_specs
            ):
                column = (
                    "intraday_sell_pressure_exhaustion_persistence_5m_"
                    f"l{long_window}_s{short_window}_m{medium_window}"
                )
                output[column] = exhaustion_by_window[long_window] - 0.5 * (
                    exhaustion_by_window[short_window]
                    + exhaustion_by_window[medium_window]
                )
                feature_columns.append(column)
    if "same_slot_intraday_memory" in groups:
        market_return = one_bar_return.groupby(frame["bar_end_time"]).transform("median")
        residual_return = one_bar_return - market_return
        slot = _intraday_slot(frame)
        for window in config.same_slot_memory_windows:
            column = f"intraday_same_slot_residual_return_5m_d{window}"
            output[column] = residual_return.groupby(
                [frame["instrument_id"], slot],
                sort=False,
            ).transform(
                lambda values: values.shift(1).rolling(
                    window,
                    min_periods=window,
                ).mean()
            )
            feature_columns.append(column)
    if "overnight_intraday_tug_of_war" in groups:
        frame["open_price"] = frame["open_price"].astype(float)
        session_state = _align_session_state(frame)
        overnight_gap = (
            session_state["session_open"]
            / session_state["previous_session_close"].where(
                session_state["previous_session_close"] != 0.0
            )
            - 1.0
        )
        intraday_from_open = (
            frame["close_price"]
            / session_state["session_open"].where(session_state["session_open"] != 0.0)
            - 1.0
        )
        output["intraday_overnight_gap_5m"] = overnight_gap
        output["intraday_overnight_gap_down_recovery_5m"] = (
            (-overnight_gap).clip(lower=0.0) * intraday_from_open.clip(lower=0.0)
        )
        output["intraday_overnight_gap_up_fade_5m"] = (
            overnight_gap.clip(lower=0.0) * (-intraday_from_open).clip(lower=0.0)
        )
        output["intraday_overnight_intraday_disagreement_5m"] = (
            -overnight_gap * intraday_from_open
        )
        feature_columns.extend(
            [
                "intraday_overnight_gap_5m",
                "intraday_overnight_gap_down_recovery_5m",
                "intraday_overnight_gap_up_fade_5m",
                "intraday_overnight_intraday_disagreement_5m",
            ]
        )
    if "weak_tape_overnight_gap" in groups:
        frame["open_price"] = frame["open_price"].astype(float)
        session_state = _align_session_state(frame)
        overnight_gap = (
            session_state["session_open"]
            / session_state["previous_session_close"].where(
                session_state["previous_session_close"] != 0.0
            )
            - 1.0
        )
        intraday_from_open = (
            frame["close_price"]
            / session_state["session_open"].where(session_state["session_open"] != 0.0)
            - 1.0
        )
        market_return = one_bar_return.groupby(frame["bar_end_time"]).transform("median")
        up_rate = one_bar_return.gt(0.0).where(one_bar_return.notna()).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        market_downside = (-market_return).clip(lower=0.0)
        weak_breadth = (0.5 - up_rate).clip(lower=0.0)
        positive_gap = overnight_gap.clip(lower=0.0)
        gap_up_fade = positive_gap * (-intraday_from_open).clip(lower=0.0)
        gap_down_recovery = (
            (-overnight_gap).clip(lower=0.0) * intraday_from_open.clip(lower=0.0)
        )
        for window in config.weak_tape_gap_windows:
            weak_breadth_state = _rolling_timestamp_state(
                frame["bar_end_time"],
                weak_breadth,
                window,
            )
            downside_state = _rolling_timestamp_state(
                frame["bar_end_time"],
                market_downside,
                window,
            )
            weak_tape_score = weak_breadth_state + np.log1p(
                100.0 * downside_state
            )
            gap_up_risk_column = f"intraday_weak_tape_gap_up_risk_5m_w{window}"
            gap_up_fade_risk_column = (
                f"intraday_weak_tape_gap_up_fade_risk_5m_w{window}"
            )
            gap_down_recovery_risk_column = (
                f"intraday_weak_tape_gap_down_recovery_risk_5m_w{window}"
            )
            output[gap_up_risk_column] = positive_gap * weak_tape_score
            output[gap_up_fade_risk_column] = gap_up_fade * (1.0 + weak_tape_score)
            output[gap_down_recovery_risk_column] = (
                gap_down_recovery * weak_tape_score
            )
            feature_columns.extend(
                [
                    gap_up_risk_column,
                    gap_up_fade_risk_column,
                    gap_down_recovery_risk_column,
                ]
            )
    if "event_shock_proxy" in groups:
        frame["open_price"] = frame["open_price"].astype(float)
        frame["turnover"] = frame["turnover"].astype(float)
        residual_return = _cross_sectional_residual_return(
            one_bar_return,
            frame["bar_end_time"],
        )
        market_return = one_bar_return.groupby(frame["bar_end_time"]).transform(
            "median"
        )
        down_rate = one_bar_return.lt(0.0).where(one_bar_return.notna()).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        sync_down_pressure = market_return.clip(upper=0.0).abs() * down_rate

        limit_up_rate = frame["limit_up_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_down_rate = frame["limit_down_open"].astype(bool).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        limit_up_diffusion = (
            limit_up_rate
            - _lagged_rolling_timestamp_state(
                frame["bar_end_time"],
                limit_up_rate,
                1,
            )
        ).clip(lower=0.0)
        limit_down_diffusion = (
            limit_down_rate
            - _lagged_rolling_timestamp_state(
                frame["bar_end_time"],
                limit_down_rate,
                1,
            )
        ).clip(lower=0.0)
        limit_diffusion_pressure = (
            limit_up_rate
            + limit_down_rate
            + limit_up_diffusion
            + limit_down_diffusion
        )

        log_turnover = np.log1p(frame["turnover"].clip(lower=0.0))
        previous_close = grouped["close_price"].shift(1)
        open_jump = frame["open_price"] / previous_close.where(
            previous_close != 0.0
        ) - 1.0
        intraday_from_open = (
            frame["close_price"] / frame["open_price"].where(frame["open_price"] != 0.0)
            - 1.0
        )
        open_jump_magnitude = open_jump.abs()

        for window in config.event_shock_windows:
            sync_column = f"intraday_event_sync_down_resilience_5m_w{window}"
            output[sync_column] = _rolling_weighted_mean(
                residual_return,
                sync_down_pressure,
                frame["instrument_id"],
                window,
            )
            limit_column = f"intraday_event_limit_diffusion_resilience_5m_w{window}"
            output[limit_column] = _rolling_weighted_mean(
                residual_return,
                limit_diffusion_pressure,
                frame["instrument_id"],
                window,
            )

            turnover_mean = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values, window=window: values.shift(1)
                .rolling(window, min_periods=window)
                .mean()
            )
            turnover_std = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values, window=window: values.shift(1)
                .rolling(window, min_periods=window)
                .std()
            )
            turnover_dislocation = (
                (log_turnover - turnover_mean) / turnover_std.where(turnover_std != 0.0)
            ).abs()
            turnover_column = (
                f"intraday_event_turnover_dislocation_recovery_5m_w{window}"
            )
            output[turnover_column] = _rolling_weighted_mean(
                one_bar_return,
                turnover_dislocation,
                frame["instrument_id"],
                window,
            )

            open_jump_column = (
                f"intraday_event_open_jump_recovery_quality_5m_w{window}"
            )
            output[open_jump_column] = _rolling_weighted_mean(
                intraday_from_open,
                open_jump_magnitude,
                frame["instrument_id"],
                window,
            )
            feature_columns.extend(
                [
                    sync_column,
                    limit_column,
                    turnover_column,
                    open_jump_column,
                ]
            )
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
    if "breadth_shock_residual_resilience" in groups:
        market_return = one_bar_return.groupby(frame["bar_end_time"]).transform("median")
        up_rate = one_bar_return.gt(0.0).where(one_bar_return.notna()).groupby(
            frame["bar_end_time"]
        ).transform("mean")
        residual_return = one_bar_return - market_return
        weak_breadth_pressure = (0.5 - up_rate).clip(lower=0.0)
        for window in config.breadth_shock_residual_resilience_windows:
            lagged_breadth_mean = _lagged_rolling_timestamp_state(
                frame["bar_end_time"],
                up_rate,
                window,
            )
            breadth_shock = (lagged_breadth_mean - up_rate).clip(lower=0.0)
            stress_weight = (weak_breadth_pressure + breadth_shock).where(
                one_bar_return.notna()
            )
            weighted_residual = residual_return * stress_weight
            rolling_weighted_residual = weighted_residual.groupby(
                frame["instrument_id"]
            ).transform(lambda values: values.rolling(window, min_periods=window).sum())
            rolling_stress = stress_weight.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).sum()
            )
            column = f"intraday_breadth_shock_residual_resilience_5m_w{window}"
            output[column] = rolling_weighted_residual / rolling_stress.where(
                rolling_stress != 0.0
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
    if "regime_conditioned" in groups:
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
        weak_breadth = (0.5 - up_rate).clip(lower=0.0)
        downside_momentum = (-market_return).clip(lower=0.0)
        limit_pressure = (limit_down_rate - limit_up_rate).clip(lower=0.0)
        regime_probability_by_window: dict[int, pd.Series] = {}
        for window in config.regime_conditioned_state_windows:
            weak_breadth_state = _rolling_timestamp_state(
                frame["bar_end_time"],
                weak_breadth,
                window,
            )
            downside_state = _rolling_timestamp_state(
                frame["bar_end_time"],
                downside_momentum,
                window,
            )
            limit_pressure_state = _rolling_timestamp_state(
                frame["bar_end_time"],
                limit_pressure,
                window,
            )
            regime_probability = (
                (weak_breadth_state / 0.5).clip(lower=0.0, upper=1.0)
                + np.tanh(200.0 * downside_state.clip(lower=0.0))
                + (limit_pressure_state / 0.2).clip(lower=0.0, upper=1.0)
            ) / 3.0
            regime_probability = regime_probability.where(one_bar_return.notna())
            regime_probability_by_window[window] = regime_probability
            probability_column = f"market_regime_stress_probability_5m_w{window}"
            output[probability_column] = regime_probability
            feature_columns.append(probability_column)
        for lookback_bars in config.regime_conditioned_lookback_bars:
            returns = lookback_return(lookback_bars)
            market_lookback_return = returns.groupby(frame["bar_end_time"]).transform(
                "median"
            )
            residual_momentum = returns - market_lookback_return
            residual_reversal = -residual_momentum
            for window, regime_probability in regime_probability_by_window.items():
                stress_reversal_column = (
                    "intraday_regime_stress_reversal_5m_"
                    f"lb{lookback_bars}_w{window}"
                )
                stress_momentum_column = (
                    "intraday_regime_stress_momentum_5m_"
                    f"lb{lookback_bars}_w{window}"
                )
                calm_reversal_column = (
                    "intraday_regime_calm_reversal_5m_"
                    f"lb{lookback_bars}_w{window}"
                )
                switch_reversal_column = (
                    "intraday_regime_switch_reversal_5m_"
                    f"lb{lookback_bars}_w{window}"
                )
                output[stress_reversal_column] = residual_reversal * regime_probability
                output[stress_momentum_column] = residual_momentum * regime_probability
                output[calm_reversal_column] = residual_reversal * (
                    1.0 - regime_probability
                )
                output[switch_reversal_column] = residual_reversal * (
                    1.0 - 2.0 * regime_probability
                )
                feature_columns.extend(
                    [
                        stress_reversal_column,
                        stress_momentum_column,
                        calm_reversal_column,
                        switch_reversal_column,
                    ]
                )
    if "return_skewness" in groups:
        for window in config.return_skewness_windows:
            column = f"intraday_return_skewness_5m_w{window}"
            output[column] = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).skew()
            )
            feature_columns.append(column)
    if "lottery_max" in groups:
        for window in config.lottery_max_windows:
            rolling_max_return = one_bar_return.groupby(frame["instrument_id"]).transform(
                lambda values, window=window: values.rolling(
                    window,
                    min_periods=window,
                ).max()
            )
            column = f"intraday_lottery_max_5m_w{window}"
            output[column] = _inverted_cross_sectional_percentile(
                rolling_max_return,
                frame["bar_end_time"],
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
    if "volume_distribution_shape" in groups:
        frame["volume"] = frame["volume"].astype(float)
        volume_grouped = frame.groupby("instrument_id", sort=False)["volume"]
        for window in config.volume_distribution_windows:
            rolling_mean = volume_grouped.transform(
                lambda values, window=window: values.rolling(
                    window,
                    min_periods=window,
                ).mean()
            )
            rolling_std = volume_grouped.transform(
                lambda values, window=window: values.rolling(
                    window,
                    min_periods=window,
                ).std()
            )
            zscore = (frame["volume"] - rolling_mean) / rolling_std.where(
                rolling_std != 0.0
            )
            burstiness_column = f"intraday_volume_burstiness_5m_w{window}"
            u_shape_column = f"intraday_volume_u_shape_5m_w{window}"
            back_loaded_column = f"intraday_volume_back_loaded_5m_w{window}"
            concentration_column = f"intraday_volume_concentration_5m_w{window}"
            output[burstiness_column] = zscore.abs()
            half_window = window // 2
            back_volume = volume_grouped.transform(
                lambda values, half_window=half_window: values.rolling(
                    half_window,
                    min_periods=half_window,
                ).sum()
            )
            front_volume = back_volume.groupby(frame["instrument_id"]).shift(half_window)
            total_volume = front_volume + back_volume
            back_share = back_volume / total_volume.where(total_volume != 0.0)
            output[back_loaded_column] = back_share - 0.5
            rolling_volume_sum = volume_grouped.transform(
                lambda values, window=window: values.rolling(
                    window,
                    min_periods=window,
                ).sum()
            )
            volume_share = frame["volume"] / rolling_volume_sum.where(
                rolling_volume_sum != 0.0
            )
            rolling_squared_volume = frame["volume"].pow(2).groupby(
                frame["instrument_id"]
            ).transform(
                lambda values, window=window: values.rolling(window, min_periods=window).sum()
            )
            output[concentration_column] = rolling_squared_volume / rolling_volume_sum.pow(
                2
            ).where(rolling_volume_sum != 0.0)
            output[u_shape_column] = output[burstiness_column] + output[
                concentration_column
            ]
            feature_columns.extend(
                [
                    burstiness_column,
                    u_shape_column,
                    back_loaded_column,
                    concentration_column,
                ]
            )
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
    if "turnover_stability" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        log_turnover = np.log1p(frame["turnover"].clip(lower=0.0))
        for window in config.turnover_stability_windows:
            rolling_mean = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).mean()
            )
            rolling_std = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).std()
            )
            column = f"intraday_turnover_stability_5m_w{window}"
            output[column] = rolling_mean / rolling_std.where(rolling_std != 0.0)
            feature_columns.append(column)
    if "liquidity_reliability" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        log_turnover = np.log1p(frame["turnover"].clip(lower=0.0))
        for window in config.liquidity_reliability_windows:
            rolling_mean = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).mean()
            )
            rolling_std = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(window, min_periods=window).std()
            )
            column = f"intraday_liquidity_reliability_5m_w{window}"
            output[column] = rolling_mean - rolling_std
            feature_columns.append(column)
    if "liquidity_reliability_recovery" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        log_turnover = np.log1p(frame["turnover"].clip(lower=0.0))
        positive_return = one_bar_return.clip(lower=0.0).where(one_bar_return.notna())
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        for long_window, capacity_window, recovery_window in (
            config.liquidity_reliability_recovery_specs
        ):
            long_mean = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(long_window, min_periods=long_window).mean()
            )
            long_std = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(long_window, min_periods=long_window).std()
            )
            recent_capacity = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(
                    capacity_window,
                    min_periods=capacity_window,
                ).mean()
            )
            rolling_positive_return = positive_return.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(
                    recovery_window,
                    min_periods=recovery_window,
                ).sum()
            )
            rolling_downside_return = downside_return.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(
                    recovery_window,
                    min_periods=recovery_window,
                ).sum()
            )
            low_reliability_premium = -(long_mean - long_std)
            relative_capacity = recent_capacity / long_mean.where(long_mean != 0.0)
            capacity_gate = np.log1p(recent_capacity.clip(lower=0.0)) * (
                relative_capacity.clip(lower=0.0, upper=2.0)
            )
            recovery_ratio = rolling_positive_return / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            recovery_confirmation = np.log1p(recovery_ratio.clip(lower=0.0))
            column = (
                "intraday_liquidity_reliability_recovery_5m_"
                f"l{long_window}_c{capacity_window}_r{recovery_window}"
            )
            output[column] = (
                low_reliability_premium
                * capacity_gate
                * recovery_confirmation
            )
            feature_columns.append(column)
    if "liquidity_reliability_recovery_balance" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        log_turnover = np.log1p(frame["turnover"].clip(lower=0.0))
        positive_return = one_bar_return.clip(lower=0.0).where(one_bar_return.notna())
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        for long_window, capacity_window, recovery_window in (
            config.liquidity_reliability_recovery_balance_specs
        ):
            long_mean = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(long_window, min_periods=long_window).mean()
            )
            long_std = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(long_window, min_periods=long_window).std()
            )
            recent_capacity = log_turnover.groupby(frame["instrument_id"]).transform(
                lambda values: values.rolling(
                    capacity_window,
                    min_periods=capacity_window,
                ).mean()
            )
            rolling_positive_return = positive_return.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(
                    recovery_window,
                    min_periods=recovery_window,
                ).sum()
            )
            rolling_downside_return = downside_return.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values: values.rolling(
                    recovery_window,
                    min_periods=recovery_window,
                ).sum()
            )
            low_reliability_score = _softplus(long_std - long_mean)
            relative_capacity = recent_capacity / long_mean.where(long_mean != 0.0)
            capacity_balance = (
                2.0
                * relative_capacity
                / (1.0 + relative_capacity * relative_capacity)
            ).where(relative_capacity >= 0.0)
            capacity_quality = np.log1p(recent_capacity.clip(lower=0.0)) * (
                capacity_balance
            )
            recovery_ratio = rolling_positive_return / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            recovery_balance = (
                2.0
                * recovery_ratio
                / (1.0 + recovery_ratio * recovery_ratio)
            ).where(recovery_ratio >= 0.0)
            column = (
                "intraday_liquidity_reliability_recovery_balance_5m_"
                f"l{long_window}_c{capacity_window}_r{recovery_window}"
            )
            output[column] = (
                low_reliability_score
                * capacity_quality
                * recovery_balance
            )
            feature_columns.append(column)
    if "microstructure_recovery_speed" in groups:
        frame["turnover"] = frame["turnover"].astype(float)
        positive_return = one_bar_return.clip(lower=0.0).where(one_bar_return.notna())
        downside_return = one_bar_return.clip(upper=0.0).abs().where(
            one_bar_return.notna()
        )
        turnover = frame["turnover"].where(one_bar_return.notna())
        downside_turnover = turnover.where(one_bar_return.lt(0.0), 0.0)
        recovery_windows = set(config.microstructure_recovery_windows)
        for spec in config.microstructure_recovery_acceleration_specs:
            recovery_windows.update(spec)
        recovery_by_window: dict[int, pd.Series] = {}
        for window in sorted(recovery_windows):
            rolling_positive_return = positive_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values, window=window: values.rolling(window, min_periods=window).sum())
            rolling_downside_return = downside_return.groupby(
                frame["instrument_id"]
            ).transform(lambda values, window=window: values.rolling(window, min_periods=window).sum())
            recovery_ratio = rolling_positive_return / rolling_downside_return.where(
                rolling_downside_return != 0.0
            )
            half_window = window // 2
            if half_window <= 0:
                raise ValueError("microstructure_recovery_windows values must be at least 2")
            recent_downside_turnover = downside_turnover.groupby(
                frame["instrument_id"]
            ).transform(
                lambda values, half_window=half_window: values.rolling(
                    half_window,
                    min_periods=half_window,
                ).sum()
            )
            previous_downside_turnover = recent_downside_turnover.groupby(
                frame["instrument_id"]
            ).shift(half_window)
            total_downside_turnover = previous_downside_turnover + recent_downside_turnover
            pressure_relief = (
                previous_downside_turnover - recent_downside_turnover
            ) / total_downside_turnover.where(total_downside_turnover != 0.0)
            recovery_speed = np.log1p(recovery_ratio.clip(lower=0.0)) * (
                1.0 + pressure_relief.clip(lower=0.0)
            )
            recovery_by_window[window] = recovery_speed
            if window in config.microstructure_recovery_windows:
                column = f"intraday_microstructure_recovery_speed_5m_w{window}"
                output[column] = recovery_speed
                feature_columns.append(column)
        for short_window, long_window in (
            config.microstructure_recovery_acceleration_specs
        ):
            column = (
                "intraday_microstructure_recovery_acceleration_5m_"
                f"s{short_window}_l{long_window}"
            )
            output[column] = recovery_by_window[short_window] - recovery_by_window[
                long_window
            ]
            feature_columns.append(column)
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
    result = output.to_frame(feature_columns)
    return result.loc[result[feature_columns].notna().any(axis=1)].reset_index(
        drop=True
    )


class _FeatureOutput:
    """Collect feature columns and materialize once to avoid DataFrame fragmentation."""

    def __init__(self, base: pd.DataFrame) -> None:
        self._base = base
        self._features: dict[str, object] = {}

    def __setitem__(self, column: str, values: object) -> None:
        self._features[column] = values

    def __getitem__(self, column: str) -> object:
        if column in self._features:
            return self._features[column]
        return self._base[column]

    def to_frame(self, feature_columns: list[str]) -> pd.DataFrame:
        del feature_columns
        if not self._features:
            return self._base.copy()
        ordered_columns = list(self._features)
        features = pd.DataFrame(
            {column: self._features[column] for column in ordered_columns},
            index=self._base.index,
        )
        return pd.concat([self._base, features], axis=1)


def _expanded_groups(groups: tuple[str, ...]) -> set[str]:
    selected = set(groups)
    if "all" not in selected:
        return selected
    return _VALID_GROUPS - {"all"}


def _inverted_cross_sectional_percentile(
    values: pd.Series,
    timestamps: pd.Series,
) -> pd.Series:
    """Rank by timestamp so high raw values receive low scores."""

    ranks = values.groupby(timestamps).rank(method="average", na_option="keep")
    counts = values.notna().groupby(timestamps).transform("sum")
    denominator = (counts - 1).where(counts > 1)
    percentile = (ranks - 1.0) / denominator
    percentile = percentile.where(counts > 1, 0.5)
    return (1.0 - percentile).where(values.notna())


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


def _lagged_rolling_timestamp_state(
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
    rolling_values = timestamp_values.shift(1).rolling(window, min_periods=window).mean()
    return timestamps.map(rolling_values)


def _cross_sectional_residual_return(
    returns: pd.Series,
    timestamps: pd.Series,
) -> pd.Series:
    market_return = returns.groupby(timestamps).transform("median")
    return returns - market_return


def _rolling_weighted_mean(
    values: pd.Series,
    weights: pd.Series,
    instruments: pd.Series,
    window: int,
) -> pd.Series:
    aligned_weights = weights.where(values.notna())
    weighted_values = values * aligned_weights
    numerator = weighted_values.groupby(instruments).transform(
        lambda series: series.rolling(window, min_periods=window).sum()
    )
    denominator = aligned_weights.groupby(instruments).transform(
        lambda series: series.rolling(window, min_periods=window).sum()
    )
    return numerator / denominator.where(denominator != 0.0)


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


def _align_session_state(frame: pd.DataFrame) -> pd.DataFrame:
    session_date = _session_date(frame)
    sorted_frame = (
        frame.assign(_session_date=session_date, _row_id=frame.index)
        .sort_values(["instrument_id", "_session_date", "bar_end_time"])
        .copy()
    )
    session_grouped = sorted_frame.groupby(
        ["instrument_id", "_session_date"],
        sort=False,
    )
    session_open = session_grouped["open_price"].first()
    session_close = session_grouped["close_price"].last()
    daily_state = (
        pd.DataFrame(
            {
                "session_open": session_open,
                "session_close": session_close,
            }
        )
        .reset_index()
        .sort_values(["instrument_id", "_session_date"])
        .reset_index(drop=True)
    )
    daily_state["previous_session_close"] = daily_state.groupby(
        "instrument_id",
        sort=False,
    )["session_close"].shift(1)
    row_keys = pd.DataFrame(
        {
            "_row_id": frame.index,
            "instrument_id": frame["instrument_id"],
            "_session_date": session_date,
        }
    )
    aligned = (
        row_keys.merge(
            daily_state.loc[
                :,
                [
                    "instrument_id",
                    "_session_date",
                    "session_open",
                    "previous_session_close",
                ],
            ],
            on=["instrument_id", "_session_date"],
            how="left",
        )
        .sort_values("_row_id")
        .set_index("_row_id")
    )
    return aligned.loc[frame.index, ["session_open", "previous_session_close"]]


def _intraday_slot(frame: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(frame["bar_end_time"], errors="coerce", format="ISO8601")
    if parsed.notna().all():
        return parsed.dt.strftime("%H:%M:%S")
    return frame["bar_end_time"].astype(str)


def _session_tail_mask(frame: pd.DataFrame, *, tail_bars: int) -> pd.Series:
    parsed = pd.to_datetime(frame["bar_end_time"], errors="coerce", format="ISO8601")
    if parsed.notna().all():
        minutes = parsed.dt.hour * 60 + parsed.dt.minute
        session_close_minutes = 15 * 60
        first_tail_minutes = session_close_minutes - 5 * (tail_bars - 1)
        return (
            (minutes >= first_tail_minutes) & (minutes <= session_close_minutes)
        ).astype(bool)
    session_date = _session_date(frame)
    sorted_frame = (
        frame.assign(_session_date=session_date, _row_id=frame.index)
        .sort_values(["instrument_id", "_session_date", "bar_end_time"])
        .copy()
    )
    session_grouped = sorted_frame.groupby(
        ["instrument_id", "_session_date"],
        sort=False,
    )
    session_size = session_grouped["_row_id"].transform("size")
    position_in_session = session_grouped.cumcount()
    bars_to_session_end = session_size - position_in_session - 1
    tail_mask = bars_to_session_end < tail_bars
    return (
        pd.Series(tail_mask.to_numpy(), index=sorted_frame["_row_id"])
        .reindex(frame.index)
        .fillna(False)
        .astype(bool)
    )


def _session_date(frame: pd.DataFrame) -> pd.Series:
    if "trade_date" in frame.columns:
        return frame["trade_date"].astype(str)
    parsed = pd.to_datetime(frame["bar_end_time"], errors="coerce", format="ISO8601")
    if parsed.notna().all():
        return parsed.dt.strftime("%Y-%m-%d")
    return frame["bar_end_time"].astype(str)


def _softplus(values: pd.Series) -> pd.Series:
    return np.log1p(np.exp(-values.abs())) + values.clip(lower=0.0)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
