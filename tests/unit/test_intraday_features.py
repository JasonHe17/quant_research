from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_research.datasets import IntradayFeatureConfig, build_intraday_feature_matrix


def test_build_intraday_feature_matrix_generates_heterogeneous_features() -> None:
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "open_price": float(10 + i),
                "high_price": float(11 + i),
                "low_price": float(9 + i),
                "close_price": float(10 + i + 1),
                "volume": float(100 + i * 10),
                "turnover": float((100 + i * 10) * (10 + i + 1)),
                "limit_up_open": False,
                "limit_down_open": False,
            }
            for i in range(6)
        ]
    )
    config = IntradayFeatureConfig(
        factor_groups=(
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
            "return_turnover_correlation",
            "negative_return_persistence",
            "sell_pressure_absorption",
            "downside_turnover_decay",
            "sell_pressure_recovery",
            "sell_pressure_exhaustion",
            "sell_pressure_exhaustion_persistence",
            "microstructure_recovery_speed",
            "same_slot_intraday_memory",
            "overnight_intraday_tug_of_war",
            "weak_tape_overnight_gap",
            "sell_pressure_quality_state",
            "event_shock_proxy",
            "daily_moving_average",
        ),
        reversal_lookback_bars=(1,),
        cross_sectional_reversal_lookback_bars=(1,),
        conditional_reversal_lookback_bars=(1,),
        conditional_reversal_state_windows=(3,),
        eod_reversal_lookback_bars=(1,),
        eod_reversal_tail_bars=2,
        momentum_lookback_bars=(2,),
        volatility_windows=(3,),
        volatility_state_change_specs=((2, 4),),
        price_position_windows=(3,),
        range_volatility_windows=(3,),
        efficiency_windows=(3,),
        volume_windows=(3,),
        volume_distribution_windows=(4,),
        turnover_windows=(3,),
        turnover_stability_windows=(3,),
        liquidity_reliability_windows=(3,),
        liquidity_reliability_recovery_specs=((4, 2, 3),),
        liquidity_reliability_recovery_balance_specs=((4, 2, 3),),
        vwap_deviation_windows=(3,),
        downside_volatility_windows=(3,),
        return_skewness_windows=(3,),
        money_flow_windows=(3,),
        signed_turnover_imbalance_windows=(3,),
        risk_adjusted_momentum_windows=(3,),
        volume_confirmed_momentum_windows=(3,),
        return_turnover_correlation_windows=(3,),
        negative_return_persistence_windows=(3,),
        sell_pressure_absorption_windows=(3,),
        downside_turnover_decay_windows=(4,),
        sell_pressure_recovery_windows=(3,),
        sell_pressure_exhaustion_windows=(4,),
        sell_pressure_exhaustion_persistence_specs=((4, 2, 3),),
        microstructure_recovery_windows=(4,),
        microstructure_recovery_acceleration_specs=((2, 4),),
        same_slot_memory_windows=(1,),
        weak_tape_gap_windows=(1,),
        sell_pressure_quality_windows=(3,),
        event_shock_windows=(3,),
        daily_moving_average_windows=(2, 3),
        daily_moving_average_pairs=((2, 3),),
        market_state_windows=(3,),
        market_downside_beta_windows=(3,),
        breadth_resilience_windows=(3,),
        breadth_shock_residual_resilience_windows=(3,),
        limit_pressure_resilience_windows=(3,),
    )

    features = build_intraday_feature_matrix(bars, config)

    assert "intraday_reversal_5m_lb1" in features
    assert "intraday_cross_sectional_reversal_5m_lb1" in features
    assert "intraday_low_vol_volume_confirmed_reversal_5m_lb1_w3" in features
    assert "intraday_eod_reversal_5m_lb1_tail2" in features
    assert "intraday_momentum_5m_lb2" in features
    assert "intraday_volatility_5m_w3" in features
    assert "intraday_volatility_state_change_5m_s2_l4" in features
    assert "intraday_volatility_state_trend_5m_s2_l4" in features
    assert "intraday_range_position_5m_w3" in features
    assert "intraday_range_volatility_5m_w3" in features
    assert "intraday_efficiency_ratio_5m_w3" in features
    assert "intraday_volume_ratio_5m_w3" in features
    assert "intraday_volume_burstiness_5m_w4" in features
    assert "intraday_volume_u_shape_5m_w4" in features
    assert "intraday_volume_back_loaded_5m_w4" in features
    assert "intraday_volume_concentration_5m_w4" in features
    assert "intraday_turnover_zscore_5m_w3" in features
    assert "intraday_turnover_stability_5m_w3" in features
    assert "intraday_liquidity_reliability_5m_w3" in features
    assert "intraday_liquidity_reliability_recovery_5m_l4_c2_r3" in features
    assert "intraday_liquidity_reliability_recovery_balance_5m_l4_c2_r3" in features
    assert "intraday_bar_return_5m" in features
    assert "intraday_amihud_5m" in features
    assert "intraday_vwap_deviation_5m_w3" in features
    assert "intraday_downside_volatility_5m_w3" in features
    assert "intraday_return_skewness_5m_w3" in features
    assert "intraday_money_flow_5m_w3" in features
    assert "intraday_signed_turnover_imbalance_5m_w3" in features
    assert "intraday_risk_adjusted_momentum_5m_w3" in features
    assert "intraday_volume_confirmed_momentum_5m_w3" in features
    assert "intraday_gap_5m" in features
    assert "market_state_downside_mean_5m_w3" in features
    assert "intraday_market_downside_beta_5m_w3" in features
    assert "intraday_breadth_resilience_5m_w3" in features
    assert "intraday_breadth_shock_residual_resilience_5m_w3" in features
    assert "intraday_limit_pressure_resilience_5m_w3" in features
    assert "intraday_return_turnover_corr_5m_w3" in features
    assert "intraday_negative_return_persistence_5m_w3" in features
    assert "intraday_sell_pressure_absorption_5m_w3" in features
    assert "intraday_downside_turnover_decay_5m_w4" in features
    assert "intraday_sell_pressure_recovery_5m_w3" in features
    assert "intraday_sell_pressure_exhaustion_5m_w4" in features
    assert "intraday_sell_pressure_exhaustion_persistence_5m_l4_s2_m3" in features
    assert "intraday_microstructure_recovery_speed_5m_w4" in features
    assert "intraday_microstructure_recovery_acceleration_5m_s2_l4" in features
    assert "intraday_same_slot_residual_return_5m_d1" in features
    assert "intraday_overnight_gap_down_recovery_5m" in features
    assert "intraday_weak_tape_gap_up_risk_5m_w1" in features
    assert "intraday_sell_pressure_absorption_quality_5m_w3" in features
    assert "intraday_false_absorption_risk_5m_w3" in features
    assert "intraday_event_sync_down_resilience_5m_w3" in features
    assert "intraday_event_limit_diffusion_resilience_5m_w3" in features
    assert "intraday_event_turnover_dislocation_recovery_5m_w3" in features
    assert "intraday_event_open_jump_recovery_quality_5m_w3" in features
    assert "intraday_daily_ma_deviation_5m_d3" in features
    assert "intraday_daily_ma_spread_5m_s2_l3" in features
    assert features.loc[0, "intraday_bar_return_5m"] == pytest.approx(0.1)
    assert features["intraday_reversal_5m_lb1"].notna().sum() == 5
    assert features["intraday_range_position_5m_w3"].iloc[-1] == pytest.approx(0.5)
    assert features["intraday_efficiency_ratio_5m_w3"].iloc[-1] == pytest.approx(1.0)
    assert features["intraday_vwap_deviation_5m_w3"].notna().sum() == 4
    assert features["intraday_downside_volatility_5m_w3"].iloc[-1] == pytest.approx(0.0)
    assert features["intraday_money_flow_5m_w3"].iloc[-1] == pytest.approx(1.0)
    assert features["intraday_signed_turnover_imbalance_5m_w3"].iloc[-1] == pytest.approx(
        1.0
    )
    assert features["intraday_gap_5m"].iloc[-1] == pytest.approx(0.0)
    assert features["intraday_risk_adjusted_momentum_5m_w3"].notna().sum() >= 1
    assert features["intraday_volume_confirmed_momentum_5m_w3"].notna().sum() >= 1
    assert features["intraday_return_turnover_corr_5m_w3"].notna().sum() >= 1
    assert features["intraday_negative_return_persistence_5m_w3"].iloc[-1] == pytest.approx(
        0.0
    )


def test_build_intraday_feature_matrix_supports_all_group_alias() -> None:
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "open_price": float(10 + i),
                "high_price": float(12 + i),
                "low_price": float(9 + i),
                "close_price": float(11 + i),
                "volume": float(100 + i),
                "turnover": float((100 + i) * (11 + i)),
                "limit_up_open": False,
                "limit_down_open": False,
            }
            for i in range(50)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(factor_groups=("all",)),
    )

    assert "intraday_momentum_5m_lb12" in features
    assert "intraday_reversal_5m_lb24" in features
    assert "intraday_volatility_state_change_5m_s24_l96" in features
    assert "intraday_volume_burstiness_5m_w96" in features
    assert "intraday_volume_back_loaded_5m_w96" in features
    assert "intraday_microstructure_recovery_speed_5m_w48" in features
    assert "intraday_microstructure_recovery_acceleration_5m_s24_l96" in features
    assert "intraday_cross_sectional_reversal_5m_lb24" in features
    assert "intraday_low_vol_volume_confirmed_reversal_5m_lb24_w12" in features
    assert "intraday_eod_reversal_5m_lb24_tail6" in features
    assert "intraday_range_position_5m_w48" in features
    assert "intraday_range_volatility_5m_w48" in features
    assert "intraday_efficiency_ratio_5m_w48" in features
    assert "intraday_turnover_ratio_5m_w48" in features
    assert "intraday_turnover_stability_5m_w48" in features
    assert "intraday_liquidity_reliability_5m_w48" in features
    assert "intraday_liquidity_reliability_recovery_5m_l48_c12_r24" in features
    assert "intraday_liquidity_reliability_recovery_balance_5m_l48_c12_r24" in features
    assert "intraday_vwap_deviation_5m_w48" in features
    assert "intraday_downside_volatility_5m_w48" in features
    assert "intraday_return_skewness_5m_w48" in features
    assert "intraday_money_flow_5m_w48" in features
    assert "intraday_signed_turnover_imbalance_5m_w48" in features
    assert "intraday_risk_adjusted_momentum_5m_w48" in features
    assert "intraday_volume_confirmed_momentum_5m_w48" in features
    assert "intraday_gap_5m" in features
    assert "market_state_downside_mean_5m_w48" in features
    assert "intraday_market_downside_beta_5m_w48" in features
    assert "intraday_breadth_resilience_5m_w48" in features
    assert "intraday_breadth_shock_residual_resilience_5m_w48" in features
    assert "intraday_limit_pressure_resilience_5m_w48" in features
    assert "intraday_return_turnover_corr_5m_w48" in features
    assert "intraday_negative_return_persistence_5m_w48" in features
    assert "intraday_sell_pressure_absorption_5m_w48" in features
    assert "intraday_downside_turnover_decay_5m_w48" in features
    assert "intraday_sell_pressure_recovery_5m_w48" in features
    assert "intraday_sell_pressure_exhaustion_5m_w48" in features
    assert "intraday_sell_pressure_exhaustion_persistence_5m_l96_s24_m48" in features
    assert "intraday_same_slot_residual_return_5m_d5" in features
    assert "intraday_overnight_intraday_disagreement_5m" in features
    assert "intraday_sell_pressure_absorption_quality_5m_w48" in features
    assert "intraday_event_sync_down_resilience_5m_w48" in features
    assert "intraday_event_open_jump_recovery_quality_5m_w48" in features
    assert "intraday_daily_ma_deviation_5m_d20" in features
    assert "intraday_daily_ma_ribbon_trend_score_5m" in features
    assert not features.empty


def test_negative_return_persistence_counts_only_past_intraday_losses() -> None:
    closes = [10.0, 9.0, 9.5, 9.0, 8.8]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
            }
            for i, close in enumerate(closes)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("negative_return_persistence",),
            negative_return_persistence_windows=(3,),
        ),
    )

    assert features["intraday_negative_return_persistence_5m_w3"].tolist() == pytest.approx(
        [2.0 / 3.0, 2.0 / 3.0]
    )


def test_cross_sectional_reversal_demeans_market_move() -> None:
    bars = pd.DataFrame(
        [
            {"instrument_id": "a", "bar_end_time": "t0", "close_price": 100.0},
            {"instrument_id": "b", "bar_end_time": "t0", "close_price": 100.0},
            {"instrument_id": "c", "bar_end_time": "t0", "close_price": 100.0},
            {"instrument_id": "a", "bar_end_time": "t1", "close_price": 90.0},
            {"instrument_id": "b", "bar_end_time": "t1", "close_price": 110.0},
            {"instrument_id": "c", "bar_end_time": "t1", "close_price": 105.0},
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("cross_sectional_reversal",),
            cross_sectional_reversal_lookback_bars=(1,),
        ),
    )

    values = features.set_index(["instrument_id", "timestamp"])
    column = "intraday_cross_sectional_reversal_5m_lb1"

    assert values.loc[("a", "t1"), column] == pytest.approx(0.15)
    assert values.loc[("b", "t1"), column] == pytest.approx(-0.05)
    assert values.loc[("c", "t1"), column] == pytest.approx(-0.0)


def test_conditional_reversal_only_scores_low_vol_volume_confirmed_losers() -> None:
    rows = []
    closes = {
        "low-vol-loser": [100.0, 99.0, 98.0, 98.5, 98.0],
        "high-vol-loser": [100.0, 80.0, 100.0, 78.0, 70.0],
        "low-vol-winner": [100.0, 101.0, 102.0, 102.5, 103.0],
    }
    for instrument_id, prices in closes.items():
        for index, close in enumerate(prices):
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "bar_end_time": f"t{index}",
                    "close_price": close,
                    "volume": 100.0,
                }
            )
    bars = pd.DataFrame(rows)

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("conditional_reversal",),
            conditional_reversal_lookback_bars=(1,),
            conditional_reversal_state_windows=(3,),
            conditional_reversal_min_volume_ratio=-0.01,
        ),
    )

    values = features.set_index(["instrument_id", "timestamp"])
    column = "intraday_low_vol_volume_confirmed_reversal_5m_lb1_w3"

    assert values.loc[("low-vol-loser", "t4"), column] == pytest.approx(
        -(98.0 / 98.5 - 1.0)
    )
    assert ("high-vol-loser", "t4") not in values.index
    assert ("low-vol-winner", "t4") not in values.index


def test_eod_reversal_uses_clock_tail_when_timestamps_are_parseable() -> None:
    times = [
        "2024-01-02T14:25:00+08:00",
        "2024-01-02T14:30:00+08:00",
        "2024-01-02T14:35:00+08:00",
        "2024-01-02T15:00:00+08:00",
    ]
    closes = [100.0, 101.0, 99.0, 98.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "a",
                "bar_end_time": timestamp,
                "close_price": close,
            }
            for timestamp, close in zip(times, closes, strict=True)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("eod_reversal",),
            eod_reversal_lookback_bars=(1,),
            eod_reversal_tail_bars=6,
            eod_reversal_weight=2.0,
        ),
    )

    values = features.set_index("timestamp")
    column = "intraday_eod_reversal_5m_lb1_tail6"

    assert "2024-01-02T14:30:00+08:00" not in values.index
    assert values.loc["2024-01-02T14:35:00+08:00", column] == pytest.approx(
        -2.0 * (99.0 / 101.0 - 1.0)
    )
    assert values.loc["2024-01-02T15:00:00+08:00", column] == pytest.approx(
        -2.0 * (98.0 / 99.0 - 1.0)
    )


def test_turnover_stability_uses_log_turnover_signal_to_noise() -> None:
    turnovers = [100.0, 120.0, 90.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": 10.0 + i,
                "turnover": turnover,
            }
            for i, turnover in enumerate(turnovers)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("turnover_stability",),
            turnover_stability_windows=(3,),
        ),
    )

    values = pd.Series([math.log1p(value) for value in turnovers])
    expected = values.mean() / values.std()
    column = "intraday_turnover_stability_5m_w3"

    assert features[column].iloc[-1] == pytest.approx(expected)


def test_liquidity_reliability_uses_conservative_log_turnover_bound() -> None:
    turnovers = [100.0, 120.0, 90.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": 10.0 + i,
                "turnover": turnover,
            }
            for i, turnover in enumerate(turnovers)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("liquidity_reliability",),
            liquidity_reliability_windows=(3,),
        ),
    )

    values = pd.Series([math.log1p(value) for value in turnovers])
    expected = values.mean() - values.std()
    column = "intraday_liquidity_reliability_5m_w3"

    assert features[column].iloc[-1] == pytest.approx(expected)


def test_liquidity_reliability_recovery_requires_capacity_and_recovery() -> None:
    closes = [10.0, 9.0, 9.5, 9.75]
    turnovers = [100.0, 80.0, 120.0, 150.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("liquidity_reliability_recovery",),
            liquidity_reliability_recovery_specs=((4, 2, 3),),
        ),
    )

    log_turnover = pd.Series([math.log1p(value) for value in turnovers])
    long_values = log_turnover.iloc[0:4]
    recent_capacity = log_turnover.iloc[2:4].mean()
    long_mean = long_values.mean()
    low_reliability_premium = -(long_mean - long_values.std())
    relative_capacity = recent_capacity / long_mean
    capacity_gate = math.log1p(recent_capacity) * min(max(relative_capacity, 0.0), 2.0)
    positive_return = (9.5 - 9.0) / 9.0 + (9.75 - 9.5) / 9.5
    downside_return = (10.0 - 9.0) / 10.0
    recovery_confirmation = math.log1p(positive_return / downside_return)
    column = "intraday_liquidity_reliability_recovery_5m_l4_c2_r3"

    assert features[column].iloc[-1] == pytest.approx(
        low_reliability_premium * capacity_gate * recovery_confirmation
    )


def test_liquidity_reliability_recovery_balance_penalizes_extreme_recovery() -> None:
    closes = [10.0, 9.0, 9.5, 9.75]
    turnovers = [100.0, 80.0, 120.0, 150.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("liquidity_reliability_recovery_balance",),
            liquidity_reliability_recovery_balance_specs=((4, 2, 3),),
        ),
    )

    log_turnover = pd.Series([math.log1p(value) for value in turnovers])
    long_values = log_turnover.iloc[0:4]
    recent_capacity = log_turnover.iloc[2:4].mean()
    long_mean = long_values.mean()
    long_std = long_values.std()
    low_reliability_score = math.log1p(math.exp(-abs(long_std - long_mean))) + max(
        long_std - long_mean,
        0.0,
    )
    relative_capacity = recent_capacity / long_mean
    capacity_balance = 2.0 * relative_capacity / (1.0 + relative_capacity**2)
    capacity_quality = math.log1p(recent_capacity) * capacity_balance
    positive_return = (9.5 - 9.0) / 9.0 + (9.75 - 9.5) / 9.5
    downside_return = (10.0 - 9.0) / 10.0
    recovery_ratio = positive_return / downside_return
    recovery_balance = 2.0 * recovery_ratio / (1.0 + recovery_ratio**2)
    column = "intraday_liquidity_reliability_recovery_balance_5m_l4_c2_r3"

    assert features[column].iloc[-1] == pytest.approx(
        low_reliability_score * capacity_quality * recovery_balance
    )


def test_volatility_state_change_compares_short_and_long_realized_volatility() -> None:
    closes = [100.0, 101.0, 100.5, 102.0, 101.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
            }
            for i, close in enumerate(closes)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("volatility_state_change",),
            volatility_state_change_specs=((2, 4),),
        ),
    )

    returns = pd.Series(closes).pct_change()
    short_volatility = returns.rolling(2, min_periods=2).std()
    long_volatility = returns.rolling(4, min_periods=4).std()
    relative_column = "intraday_volatility_state_change_5m_s2_l4"
    trend_column = "intraday_volatility_state_trend_5m_s2_l4"

    assert features[relative_column].iloc[-1] == pytest.approx(
        short_volatility.iloc[-1] / long_volatility.iloc[-1] - 1.0
    )
    assert features[trend_column].iloc[-1] == pytest.approx(
        (short_volatility.iloc[-1] - short_volatility.iloc[-3])
        / long_volatility.iloc[-1]
    )


def test_volume_distribution_shape_uses_rolling_window_only() -> None:
    volumes = [10.0, 20.0, 30.0, 40.0, 100.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": 10.0 + i,
                "volume": volume,
            }
            for i, volume in enumerate(volumes)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("volume_distribution_shape",),
            volume_distribution_windows=(4,),
        ),
    )

    volume = pd.Series(volumes)
    rolling_mean = volume.rolling(4, min_periods=4).mean()
    rolling_std = volume.rolling(4, min_periods=4).std()
    zscore = (volume - rolling_mean) / rolling_std
    expected_burstiness = zscore.abs().iloc[-1]
    front_volume = sum(volumes[0:2])
    back_volume = sum(volumes[2:4])
    expected_back_loaded = back_volume / (front_volume + back_volume) - 0.5
    expected_concentration = sum(
        (value / sum(volumes[1:5])) ** 2 for value in volumes[1:5]
    )

    assert features["intraday_volume_burstiness_5m_w4"].iloc[-1] == pytest.approx(
        expected_burstiness
    )
    assert features["intraday_volume_back_loaded_5m_w4"].iloc[0] == pytest.approx(
        expected_back_loaded
    )
    assert features["intraday_volume_concentration_5m_w4"].iloc[-1] == pytest.approx(
        expected_concentration
    )


def test_microstructure_recovery_speed_rewards_recovery_and_pressure_relief() -> None:
    closes = [10.0, 9.0, 8.8, 9.2, 9.4]
    turnovers = [100.0, 400.0, 200.0, 100.0, 50.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("microstructure_recovery_speed",),
            microstructure_recovery_windows=(4,),
            microstructure_recovery_acceleration_specs=((2, 4),),
        ),
    )

    returns = pd.Series(closes).pct_change()
    positive_return = returns.clip(lower=0.0)
    downside_return = returns.clip(upper=0.0).abs()
    recovery_ratio = (
        positive_return.rolling(4, min_periods=4).sum().iloc[-1]
        / downside_return.rolling(4, min_periods=4).sum().iloc[-1]
    )
    expected_pressure_relief = ((400.0 + 200.0) - (0.0 + 0.0)) / (
        400.0 + 200.0
    )
    expected = math.log1p(recovery_ratio) * (1.0 + expected_pressure_relief)
    column = "intraday_microstructure_recovery_speed_5m_w4"

    assert features[column].iloc[-1] == pytest.approx(expected)


def test_sell_pressure_absorption_uses_downside_turnover_per_loss() -> None:
    closes = [10.0, 9.0, 9.5, 9.0]
    turnovers = [1000.0, 2000.0, 1500.0, 3000.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("sell_pressure_absorption",),
            sell_pressure_absorption_windows=(3,),
        ),
    )

    column = "intraday_sell_pressure_absorption_5m_w3"
    expected_downside_return = 0.1 + 0.0 + (9.5 - 9.0) / 9.5
    expected_downside_turnover = 2000.0 + 0.0 + 3000.0

    assert features[column].iloc[-1] == pytest.approx(
        expected_downside_turnover / expected_downside_return
    )


def test_downside_turnover_decay_compares_previous_and_recent_sell_pressure() -> None:
    closes = [10.0, 9.0, 8.8, 8.7, 8.6]
    turnovers = [100.0, 500.0, 300.0, 100.0, 50.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("downside_turnover_decay",),
            downside_turnover_decay_windows=(4,),
        ),
    )

    column = "intraday_downside_turnover_decay_5m_w4"
    expected_previous = 500.0 + 300.0
    expected_recent = 100.0 + 50.0

    assert features[column].iloc[-1] == pytest.approx(
        (expected_previous - expected_recent) / (expected_previous + expected_recent)
    )


def test_sell_pressure_recovery_requires_price_recovery_with_turnover_confirmation() -> None:
    closes = [10.0, 9.0, 9.5, 9.25]
    turnovers = [1000.0, 2000.0, 1500.0, 500.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("sell_pressure_recovery",),
            sell_pressure_recovery_windows=(3,),
        ),
    )

    column = "intraday_sell_pressure_recovery_5m_w3"
    expected_positive_return = (9.5 - 9.0) / 9.0
    expected_downside_return = 0.1 + (9.5 - 9.25) / 9.5
    expected_upside_turnover_share = 1500.0 / (2000.0 + 1500.0 + 500.0)

    assert features[column].iloc[-1] == pytest.approx(
        (expected_positive_return / expected_downside_return)
        * expected_upside_turnover_share
    )


def test_sell_pressure_exhaustion_requires_recovery_and_downside_turnover_decay() -> None:
    closes = [10.0, 9.0, 8.8, 8.9, 9.1]
    turnovers = [100.0, 500.0, 300.0, 100.0, 50.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("sell_pressure_exhaustion",),
            sell_pressure_exhaustion_windows=(4,),
        ),
    )

    column = "intraday_sell_pressure_exhaustion_5m_w4"
    expected_positive_return = (8.9 - 8.8) / 8.8 + (9.1 - 8.9) / 8.9
    expected_downside_return = (10.0 - 9.0) / 10.0 + (9.0 - 8.8) / 9.0
    expected_recovery = expected_positive_return / expected_downside_return
    expected_upside_turnover_share = (100.0 + 50.0) / (500.0 + 300.0 + 100.0 + 50.0)
    expected_downside_decay = ((500.0 + 300.0) - 0.0) / (500.0 + 300.0)

    assert features[column].iloc[-1] == pytest.approx(
        math.log1p(expected_recovery)
        * expected_upside_turnover_share
        * expected_downside_decay
    )


def test_sell_pressure_exhaustion_persistence_penalizes_short_window_bounces() -> None:
    closes = [10.0, 9.0, 8.8, 8.9, 9.1, 9.0]
    turnovers = [100.0, 500.0, 300.0, 100.0, 50.0, 40.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("sell_pressure_exhaustion", "sell_pressure_exhaustion_persistence"),
            sell_pressure_exhaustion_windows=(2, 3, 4),
            sell_pressure_exhaustion_persistence_specs=((4, 2, 3),),
        ),
    )

    values = features.iloc[-1]
    column = "intraday_sell_pressure_exhaustion_persistence_5m_l4_s2_m3"

    assert values[column] == pytest.approx(
        values["intraday_sell_pressure_exhaustion_5m_w4"]
        - 0.5
        * (
            values["intraday_sell_pressure_exhaustion_5m_w2"]
            + values["intraday_sell_pressure_exhaustion_5m_w3"]
        )
    )


def test_same_slot_intraday_memory_uses_lagged_same_time_residuals() -> None:
    rows = []
    prices = {
        "a": {
            "2024-01-01": (10.0, 11.0),
            "2024-01-02": (10.0, 12.0),
            "2024-01-03": (10.0, 13.0),
        },
        "b": {
            "2024-01-01": (20.0, 21.0),
            "2024-01-02": (20.0, 21.0),
            "2024-01-03": (20.0, 21.0),
        },
    }
    for instrument_id, by_date in prices.items():
        for trade_date, (open_close, slot_close) in by_date.items():
            rows.extend(
                [
                    {
                        "instrument_id": instrument_id,
                        "bar_end_time": f"{trade_date}T09:35:00+08:00",
                        "trade_date": trade_date,
                        "close_price": open_close,
                    },
                    {
                        "instrument_id": instrument_id,
                        "bar_end_time": f"{trade_date}T09:40:00+08:00",
                        "trade_date": trade_date,
                        "close_price": slot_close,
                    },
                ]
            )
    bars = pd.DataFrame(rows)

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("same_slot_intraday_memory",),
            same_slot_memory_windows=(2,),
        ),
    )

    values = features.set_index(["instrument_id", "timestamp"])
    column = "intraday_same_slot_residual_return_5m_d2"
    first_day_residual = 0.10 - 0.075
    second_day_residual = 0.20 - 0.125

    assert values.loc[("a", "2024-01-03T09:40:00+08:00"), column] == pytest.approx(
        (first_day_residual + second_day_residual) / 2.0
    )


def test_overnight_intraday_tug_of_war_separates_gap_recovery_and_fade() -> None:
    rows = [
        {
            "instrument_id": "a",
            "bar_end_time": "2024-01-01T15:00:00+08:00",
            "trade_date": "2024-01-01",
            "open_price": 10.0,
            "close_price": 10.0,
        },
        {
            "instrument_id": "a",
            "bar_end_time": "2024-01-02T09:35:00+08:00",
            "trade_date": "2024-01-02",
            "open_price": 9.0,
            "close_price": 9.9,
        },
    ]

    features = build_intraday_feature_matrix(
        pd.DataFrame(rows),
        IntradayFeatureConfig(factor_groups=("overnight_intraday_tug_of_war",)),
    )

    values = features.set_index("timestamp").loc["2024-01-02T09:35:00+08:00"]

    assert values["intraday_overnight_gap_5m"] == pytest.approx(-0.1)
    assert values["intraday_overnight_gap_down_recovery_5m"] == pytest.approx(0.01)
    assert values["intraday_overnight_gap_up_fade_5m"] == pytest.approx(0.0)
    assert values["intraday_overnight_intraday_disagreement_5m"] == pytest.approx(0.01)


def test_weak_tape_overnight_gap_conditions_gap_risk_on_market_state() -> None:
    rows = [
        {
            "instrument_id": "a",
            "bar_end_time": "2024-01-01T15:00:00+08:00",
            "trade_date": "2024-01-01",
            "open_price": 10.0,
            "close_price": 10.0,
        },
        {
            "instrument_id": "b",
            "bar_end_time": "2024-01-01T15:00:00+08:00",
            "trade_date": "2024-01-01",
            "open_price": 20.0,
            "close_price": 20.0,
        },
        {
            "instrument_id": "a",
            "bar_end_time": "2024-01-02T09:35:00+08:00",
            "trade_date": "2024-01-02",
            "open_price": 11.0,
            "close_price": 9.9,
        },
        {
            "instrument_id": "b",
            "bar_end_time": "2024-01-02T09:35:00+08:00",
            "trade_date": "2024-01-02",
            "open_price": 19.0,
            "close_price": 19.5,
        },
    ]

    features = build_intraday_feature_matrix(
        pd.DataFrame(rows),
        IntradayFeatureConfig(
            factor_groups=("weak_tape_overnight_gap",),
            weak_tape_gap_windows=(1,),
        ),
    )

    values = features.set_index(["instrument_id", "timestamp"])
    market_downside = 0.0175
    weak_tape_score = 0.5 + math.log1p(100.0 * market_downside)

    assert values.loc[
        ("a", "2024-01-02T09:35:00+08:00"),
        "intraday_weak_tape_gap_up_risk_5m_w1",
    ] == pytest.approx(0.1 * weak_tape_score)
    assert values.loc[
        ("a", "2024-01-02T09:35:00+08:00"),
        "intraday_weak_tape_gap_up_fade_risk_5m_w1",
    ] == pytest.approx(0.1 * 0.1 * (1.0 + weak_tape_score))
    assert values.loc[
        ("b", "2024-01-02T09:35:00+08:00"),
        "intraday_weak_tape_gap_down_recovery_risk_5m_w1",
    ] == pytest.approx(0.05 * ((19.5 / 19.0) - 1.0) * weak_tape_score)


def test_sell_pressure_quality_state_penalizes_false_absorption() -> None:
    closes = [10.0, 9.0, 9.5, 9.25]
    turnovers = [1000.0, 2000.0, 1500.0, 500.0]
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": close,
                "turnover": turnover,
            }
            for i, (close, turnover) in enumerate(zip(closes, turnovers))
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("sell_pressure_quality_state",),
            sell_pressure_quality_windows=(3,),
        ),
    )

    values = features.iloc[-1]
    downside_return = 0.1 + (9.5 - 9.25) / 9.5
    positive_return = (9.5 - 9.0) / 9.0
    absorption_score = math.log1p((2000.0 + 500.0) / downside_return)
    recovery_ratio = positive_return / downside_return
    recovery_balance = 2.0 * recovery_ratio / (1.0 + recovery_ratio**2)
    tape_pressure = 2.0 * ((0.5 + 0.0 + 0.5) / 3.0)
    tape_quality = 1.0 - tape_pressure

    assert values["intraday_sell_pressure_absorption_quality_5m_w3"] == pytest.approx(
        absorption_score * recovery_balance * tape_quality
    )
    assert values["intraday_false_absorption_risk_5m_w3"] == pytest.approx(
        absorption_score * (1.0 - recovery_balance) * (1.0 + tape_pressure)
    )


def test_market_downside_beta_uses_cross_sectional_down_market_returns() -> None:
    rows = [
        {"instrument_id": "a", "bar_end_time": "t0", "close_price": 10.0},
        {"instrument_id": "b", "bar_end_time": "t0", "close_price": 20.0},
        {"instrument_id": "a", "bar_end_time": "t1", "close_price": 9.0},
        {"instrument_id": "b", "bar_end_time": "t1", "close_price": 19.0},
        {"instrument_id": "a", "bar_end_time": "t2", "close_price": 9.5},
        {"instrument_id": "b", "bar_end_time": "t2", "close_price": 19.2},
    ]
    bars = pd.DataFrame(
        rows
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("market_downside_beta",),
            market_downside_beta_windows=(1,),
        ),
    )
    column = "intraday_market_downside_beta_5m_w1"
    values = features.set_index("instrument_id")[column]

    assert values.loc["a"] == pytest.approx(4.0 / 3.0)
    assert values.loc["b"] == pytest.approx(2.0 / 3.0)


def test_limit_pressure_resilience_uses_limit_pressure_state() -> None:
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "a",
                "bar_end_time": "t0",
                "close_price": 10.0,
                "limit_up_open": False,
                "limit_down_open": False,
            },
            {
                "instrument_id": "b",
                "bar_end_time": "t0",
                "close_price": 20.0,
                "limit_up_open": False,
                "limit_down_open": False,
            },
            {
                "instrument_id": "a",
                "bar_end_time": "t1",
                "close_price": 9.0,
                "limit_up_open": False,
                "limit_down_open": True,
            },
            {
                "instrument_id": "b",
                "bar_end_time": "t1",
                "close_price": 19.0,
                "limit_up_open": False,
                "limit_down_open": True,
            },
            {
                "instrument_id": "a",
                "bar_end_time": "t2",
                "close_price": 9.5,
                "limit_up_open": False,
                "limit_down_open": False,
            },
            {
                "instrument_id": "b",
                "bar_end_time": "t2",
                "close_price": 19.2,
                "limit_up_open": True,
                "limit_down_open": False,
            },
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("limit_pressure_resilience",),
            limit_pressure_resilience_windows=(1,),
        ),
    )

    column = "intraday_limit_pressure_resilience_5m_w1"
    values = features.set_index(["instrument_id", "timestamp"])[column]

    assert values.loc[("a", "t1")] == pytest.approx(-0.1)
    assert values.loc[("b", "t1")] == pytest.approx(-0.05)


def test_breadth_resilience_uses_weak_market_breadth_state() -> None:
    rows = [
        {"instrument_id": "a", "bar_end_time": "t0", "close_price": 10.0},
        {"instrument_id": "b", "bar_end_time": "t0", "close_price": 20.0},
        {"instrument_id": "c", "bar_end_time": "t0", "close_price": 30.0},
        {"instrument_id": "a", "bar_end_time": "t1", "close_price": 9.0},
        {"instrument_id": "b", "bar_end_time": "t1", "close_price": 19.0},
        {"instrument_id": "c", "bar_end_time": "t1", "close_price": 31.5},
        {"instrument_id": "a", "bar_end_time": "t2", "close_price": 9.45},
        {"instrument_id": "b", "bar_end_time": "t2", "close_price": 19.95},
        {"instrument_id": "c", "bar_end_time": "t2", "close_price": 33.0},
    ]
    bars = pd.DataFrame(rows)

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("breadth_resilience",),
            breadth_resilience_windows=(1,),
        ),
    )

    column = "intraday_breadth_resilience_5m_w1"
    values = features.set_index(["instrument_id", "timestamp"])[column]

    assert values.loc[("a", "t1")] == pytest.approx(-0.1)
    assert values.loc[("b", "t1")] == pytest.approx(-0.05)
    assert values.loc[("c", "t1")] == pytest.approx(0.05)
    assert ("a", "t2") not in values.index


def test_breadth_shock_residual_resilience_uses_relative_returns() -> None:
    rows = []
    closes = {
        "a": [100.0, 101.0, 102.01],
        "b": [100.0, 101.0, 99.99],
        "c": [100.0, 99.0, 98.01],
    }
    for instrument_id, values in closes.items():
        for index, close in enumerate(values):
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "bar_end_time": f"t{index}",
                    "close_price": close,
                }
            )
    bars = pd.DataFrame(rows)

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("breadth_shock_residual_resilience",),
            breadth_shock_residual_resilience_windows=(1,),
        ),
    )

    column = "intraday_breadth_shock_residual_resilience_5m_w1"
    values = features.set_index(["instrument_id", "timestamp"])[column]

    assert values.loc[("a", "t2")] == pytest.approx(0.02)
    assert values.loc[("b", "t2")] == pytest.approx(0.0)
    assert values.loc[("c", "t2")] == pytest.approx(0.0)


def test_market_state_features_broadcast_cross_sectional_risk_state() -> None:
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "a",
                "bar_end_time": "t0",
                "close_price": 10.0,
                "limit_up_open": False,
                "limit_down_open": False,
            },
            {
                "instrument_id": "b",
                "bar_end_time": "t0",
                "close_price": 20.0,
                "limit_up_open": False,
                "limit_down_open": False,
            },
            {
                "instrument_id": "a",
                "bar_end_time": "t1",
                "close_price": 9.0,
                "limit_up_open": False,
                "limit_down_open": True,
            },
            {
                "instrument_id": "b",
                "bar_end_time": "t1",
                "close_price": 19.0,
                "limit_up_open": False,
                "limit_down_open": True,
            },
            {
                "instrument_id": "a",
                "bar_end_time": "t2",
                "close_price": 9.45,
                "limit_up_open": False,
                "limit_down_open": False,
            },
            {
                "instrument_id": "b",
                "bar_end_time": "t2",
                "close_price": 19.95,
                "limit_up_open": True,
                "limit_down_open": False,
            },
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("market_state",),
            market_state_windows=(2,),
        ),
    )

    values = features.set_index(["instrument_id", "timestamp"])
    expected_downside_t1 = 0.075

    assert values.loc[("a", "t1"), "market_state_downside_5m"] == pytest.approx(
        expected_downside_t1
    )
    assert values.loc[("b", "t1"), "market_state_downside_5m"] == pytest.approx(
        expected_downside_t1
    )
    assert values.loc[("a", "t1"), "market_state_limit_pressure_5m"] == pytest.approx(
        1.0
    )
    assert values.loc[("a", "t2"), "market_state_limit_pressure_mean_5m_w2"] == (
        pytest.approx(0.5)
    )
    assert values.loc[("b", "t2"), "market_state_limit_pressure_mean_5m_w2"] == (
        pytest.approx(0.5)
    )
    assert values.loc[("a", "t2"), "market_state_weak_breadth_mean_5m_w2"] == (
        pytest.approx(0.25)
    )


def test_event_shock_proxy_combines_market_events_with_stock_recovery() -> None:
    rows = []
    closes = {
        "a": [100.0, 90.0, 99.0, 99.0],
        "b": [100.0, 95.0, 95.0, 104.5],
    }
    opens = {
        "a": [100.0, 80.0, 100.0, 99.0],
        "b": [100.0, 100.0, 94.0, 100.0],
    }
    turnovers = {
        "a": [100.0, 200.0, 500.0, 200.0],
        "b": [100.0, 300.0, 100.0, 500.0],
    }
    for instrument_id in ("a", "b"):
        for index, close in enumerate(closes[instrument_id]):
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "bar_end_time": f"t{index}",
                    "open_price": opens[instrument_id][index],
                    "close_price": close,
                    "turnover": turnovers[instrument_id][index],
                    "limit_up_open": False,
                    "limit_down_open": index == 1,
                }
            )
    bars = pd.DataFrame(rows)

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("event_shock_proxy",),
            event_shock_windows=(2,),
        ),
    )

    values = features.set_index(["instrument_id", "timestamp"])

    sync_column = "intraday_event_sync_down_resilience_5m_w2"
    limit_column = "intraday_event_limit_diffusion_resilience_5m_w2"
    assert values.loc[("a", "t2"), sync_column] == pytest.approx(-0.025)
    assert values.loc[("b", "t2"), sync_column] == pytest.approx(0.025)
    assert values.loc[("a", "t2"), limit_column] == pytest.approx(-0.025)
    assert values.loc[("b", "t2"), limit_column] == pytest.approx(0.025)

    open_jump_column = "intraday_event_open_jump_recovery_quality_5m_w2"
    a_t1_open_jump = abs(80.0 / 100.0 - 1.0)
    a_t2_open_jump = abs(100.0 / 90.0 - 1.0)
    a_t1_intraday = 90.0 / 80.0 - 1.0
    a_t2_intraday = 99.0 / 100.0 - 1.0
    expected_open_jump_recovery = (
        a_t1_intraday * a_t1_open_jump + a_t2_intraday * a_t2_open_jump
    ) / (a_t1_open_jump + a_t2_open_jump)
    assert values.loc[("a", "t2"), open_jump_column] == pytest.approx(
        expected_open_jump_recovery
    )

    turnover_column = "intraday_event_turnover_dislocation_recovery_5m_w2"
    a_logs = [math.log1p(value) for value in turnovers["a"]]
    a_t2_mean = (a_logs[0] + a_logs[1]) / 2.0
    a_t2_std = pd.Series(a_logs[0:2]).std()
    a_t2_dislocation = abs(a_logs[2] - a_t2_mean) / a_t2_std
    a_t3_mean = (a_logs[1] + a_logs[2]) / 2.0
    a_t3_std = pd.Series(a_logs[1:3]).std()
    a_t3_dislocation = abs(a_logs[3] - a_t3_mean) / a_t3_std
    expected_turnover_recovery = (
        0.10 * a_t2_dislocation + 0.0 * a_t3_dislocation
    ) / (a_t2_dislocation + a_t3_dislocation)
    assert values.loc[("a", "t3"), turnover_column] == pytest.approx(
        expected_turnover_recovery
    )


def test_daily_moving_average_features_use_completed_prior_sessions() -> None:
    rows = []
    closes = {
        "2024-01-01": (10.0, 10.5),
        "2024-01-02": (11.0, 11.5),
        "2024-01-03": (12.0, 12.5),
        "2024-01-04": (20.0, 20.5),
        "2024-01-05": (21.0, 21.5),
    }
    for trade_date, day_closes in closes.items():
        for index, close in enumerate(day_closes):
            rows.append(
                {
                    "instrument_id": "inst-1",
                    "bar_end_time": f"{trade_date}T09:{35 + index * 5}:00+08:00",
                    "trade_date": trade_date,
                    "close_price": close,
                }
            )
    bars = pd.DataFrame(rows)

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=("daily_moving_average",),
            daily_moving_average_windows=(2, 3),
            daily_moving_average_pairs=((2, 3),),
        ),
    )

    values = features.set_index(["timestamp"])
    deviation = "intraday_daily_ma_deviation_5m_d3"
    spread = "intraday_daily_ma_spread_5m_s2_l3"
    slope = "intraday_daily_ma_slope_5m_d3"

    assert values.loc["2024-01-04T09:35:00+08:00", deviation] == pytest.approx(
        12.5 / ((10.5 + 11.5 + 12.5) / 3.0) - 1.0
    )
    assert values.loc["2024-01-04T09:40:00+08:00", deviation] == pytest.approx(
        values.loc["2024-01-04T09:35:00+08:00", deviation]
    )
    assert values.loc["2024-01-04T09:35:00+08:00", spread] == pytest.approx(
        ((11.5 + 12.5) / 2.0) / ((10.5 + 11.5 + 12.5) / 3.0) - 1.0
    )
    assert values.loc["2024-01-05T09:35:00+08:00", slope] == pytest.approx(
        ((11.5 + 12.5 + 20.5) / 3.0) / ((10.5 + 11.5 + 12.5) / 3.0) - 1.0
    )
    assert values.loc["2024-01-04T09:35:00+08:00", deviation] != pytest.approx(
        20.5 / ((11.5 + 12.5 + 20.5) / 3.0) - 1.0
    )


def test_intraday_feature_config_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown factor groups"):
        IntradayFeatureConfig(factor_groups=("not_a_factor",))
