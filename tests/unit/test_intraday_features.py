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
    assert features.loc[0, "intraday_bar_return_5m"] == pytest.approx(0.1)
    assert features["intraday_reversal_5m_lb1"].notna().sum() == 5
    assert features["intraday_range_position_5m_w3"].iloc[-1] == pytest.approx(0.5)
    assert features["intraday_efficiency_ratio_5m_w3"].iloc[-1] == pytest.approx(1.0)
    assert features["intraday_vwap_deviation_5m_w3"].notna().sum() == 4


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
    assert not features.empty


def test_intraday_feature_config_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown factor groups"):
        IntradayFeatureConfig(factor_groups=("not_a_factor",))
