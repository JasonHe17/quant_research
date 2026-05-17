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
            "return_turnover_correlation",
            "negative_return_persistence",
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
    assert "intraday_return_turnover_corr_5m_w3" in features
    assert "intraday_negative_return_persistence_5m_w3" in features
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
    assert "intraday_return_turnover_corr_5m_w48" in features
    assert "intraday_negative_return_persistence_5m_w48" in features
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


def test_intraday_feature_config_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown factor groups"):
        IntradayFeatureConfig(factor_groups=("not_a_factor",))
