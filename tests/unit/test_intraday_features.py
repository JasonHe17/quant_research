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
                "close_price": float(10 + i + 1),
                "volume": float(100 + i * 10),
                "turnover": float(1000 + i * 100),
            }
            for i in range(6)
        ]
    )
    config = IntradayFeatureConfig(
        factor_groups=(
            "reversal",
            "momentum",
            "volatility",
            "volume",
            "turnover",
            "bar_return",
            "liquidity_impact",
        ),
        reversal_lookback_bars=(1,),
        momentum_lookback_bars=(2,),
        volatility_windows=(3,),
        volume_windows=(3,),
        turnover_windows=(3,),
    )

    features = build_intraday_feature_matrix(bars, config)

    assert "intraday_reversal_5m_lb1" in features
    assert "intraday_momentum_5m_lb2" in features
    assert "intraday_volatility_5m_w3" in features
    assert "intraday_volume_ratio_5m_w3" in features
    assert "intraday_turnover_zscore_5m_w3" in features
    assert "intraday_bar_return_5m" in features
    assert "intraday_amihud_5m" in features
    assert features.loc[0, "intraday_bar_return_5m"] == pytest.approx(0.1)
    assert features["intraday_reversal_5m_lb1"].notna().sum() == 5


def test_build_intraday_feature_matrix_supports_all_group_alias() -> None:
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "open_price": float(10 + i),
                "close_price": float(11 + i),
                "volume": float(100 + i),
                "turnover": float(1000 + i),
            }
            for i in range(50)
        ]
    )

    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(factor_groups=("all",)),
    )

    assert "intraday_momentum_5m_lb12" in features
    assert "intraday_turnover_ratio_5m_w48" in features
    assert not features.empty


def test_intraday_feature_config_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown factor groups"):
        IntradayFeatureConfig(factor_groups=("not_a_factor",))
