from __future__ import annotations

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
        ),
        reversal_lookback_bars=(1,),
        momentum_lookback_bars=(2,),
        volatility_windows=(3,),
        price_position_windows=(3,),
        range_volatility_windows=(3,),
        efficiency_windows=(3,),
        volume_windows=(3,),
        turnover_windows=(3,),
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
        market_state_windows=(3,),
        market_downside_beta_windows=(3,),
        breadth_resilience_windows=(3,),
        limit_pressure_resilience_windows=(3,),
    )

    features = build_intraday_feature_matrix(bars, config)

    assert "intraday_reversal_5m_lb1" in features
    assert "intraday_momentum_5m_lb2" in features
    assert "intraday_volatility_5m_w3" in features
    assert "intraday_range_position_5m_w3" in features
    assert "intraday_range_volatility_5m_w3" in features
    assert "intraday_efficiency_ratio_5m_w3" in features
    assert "intraday_volume_ratio_5m_w3" in features
    assert "intraday_turnover_zscore_5m_w3" in features
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
    assert "intraday_limit_pressure_resilience_5m_w3" in features
    assert "intraday_return_turnover_corr_5m_w3" in features
    assert "intraday_negative_return_persistence_5m_w3" in features
    assert "intraday_sell_pressure_absorption_5m_w3" in features
    assert "intraday_downside_turnover_decay_5m_w4" in features
    assert "intraday_sell_pressure_recovery_5m_w3" in features
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
    assert "intraday_range_position_5m_w48" in features
    assert "intraday_range_volatility_5m_w48" in features
    assert "intraday_efficiency_ratio_5m_w48" in features
    assert "intraday_turnover_ratio_5m_w48" in features
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
    assert "intraday_limit_pressure_resilience_5m_w48" in features
    assert "intraday_return_turnover_corr_5m_w48" in features
    assert "intraday_negative_return_persistence_5m_w48" in features
    assert "intraday_sell_pressure_absorption_5m_w48" in features
    assert "intraday_downside_turnover_decay_5m_w48" in features
    assert "intraday_sell_pressure_recovery_5m_w48" in features
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


def test_intraday_feature_config_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown factor groups"):
        IntradayFeatureConfig(factor_groups=("not_a_factor",))
