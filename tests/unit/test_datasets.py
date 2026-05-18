from __future__ import annotations

import pandas as pd
import pytest

from quant_research.datasets import (
    ForwardReturnLabelConfig,
    add_cross_sectional_label_rank,
    build_alpha_feature_matrix,
    build_forward_return_labels,
    build_multi_horizon_forward_return_labels,
    join_alpha_features_and_labels,
)
from quant_research.factors import FactorResult


def test_build_alpha_feature_matrix_pivots_factor_results() -> None:
    factor_a = FactorResult(
        factor_name="intraday_reversal_5m_lb1",
        frame=pd.DataFrame(
            [
                {
                    "instrument_id": "inst-1",
                    "timestamp": "2024-01-02T09:35:00+08:00",
                    "factor_value": 0.02,
                },
                {
                    "instrument_id": "inst-2",
                    "timestamp": "2024-01-02T09:35:00+08:00",
                    "factor_value": -0.01,
                },
            ]
        ),
    )
    factor_b = pd.DataFrame(
        [
            {
                "factor_name": "volume_spike_5m",
                "instrument_id": "inst-1",
                "bar_end_time": "2024-01-02T09:35:00+08:00",
                "factor_value": 1.5,
            }
        ]
    )

    matrix = build_alpha_feature_matrix([factor_a, factor_b])

    assert list(matrix.columns) == [
        "timestamp",
        "instrument_id",
        "intraday_reversal_5m_lb1",
        "volume_spike_5m",
    ]
    assert matrix.loc[0, "intraday_reversal_5m_lb1"] == pytest.approx(0.02)
    assert pd.isna(matrix.loc[1, "volume_spike_5m"])


def test_build_alpha_feature_matrix_rejects_duplicate_observations() -> None:
    factor_frame = pd.DataFrame(
        [
            {
                "factor_name": "alpha",
                "instrument_id": "inst-1",
                "timestamp": "2024-01-02T09:35:00+08:00",
                "factor_value": 0.1,
            },
            {
                "factor_name": "alpha",
                "instrument_id": "inst-1",
                "timestamp": "2024-01-02T09:35:00+08:00",
                "factor_value": 0.2,
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate factor observations"):
        build_alpha_feature_matrix([factor_frame])


def test_build_forward_return_labels_uses_entry_lag_and_horizon() -> None:
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": float(10 + i),
            }
            for i in range(5)
        ]
    )
    config = ForwardReturnLabelConfig(
        name="fwd_ret_2b",
        entry_lag_bars=1,
        horizon_bars=2,
    )

    labels = build_forward_return_labels(bars, config)

    assert labels["timestamp"].tolist() == ["t0", "t1"]
    assert labels.loc[0, "entry_timestamp"] == "t1"
    assert labels.loc[0, "exit_timestamp"] == "t3"
    assert labels.loc[0, "fwd_ret_2b"] == pytest.approx(13.0 / 11.0 - 1.0)


def test_build_multi_horizon_forward_return_labels_shares_entry() -> None:
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "inst-1",
                "bar_end_time": f"t{i}",
                "close_price": float(10 + i),
            }
            for i in range(6)
        ]
    )

    labels = build_multi_horizon_forward_return_labels(
        bars,
        (
            ForwardReturnLabelConfig(
                name="forward_return_1b",
                entry_lag_bars=1,
                horizon_bars=1,
            ),
            ForwardReturnLabelConfig(
                name="forward_return_3b",
                entry_lag_bars=1,
                horizon_bars=3,
            ),
        ),
    )

    assert labels["timestamp"].tolist() == ["t0", "t1"]
    assert labels.loc[0, "entry_timestamp"] == "t1"
    assert labels.loc[0, "forward_return_1b"] == pytest.approx(12.0 / 11.0 - 1.0)
    assert labels.loc[0, "forward_return_3b"] == pytest.approx(14.0 / 11.0 - 1.0)
    assert labels.loc[0, "forward_return_1b_exit_timestamp"] == "t2"
    assert labels.loc[0, "forward_return_3b_exit_timestamp"] == "t4"


def test_join_features_labels_and_add_cross_sectional_rank() -> None:
    features = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "inst-1",
                "alpha": 0.2,
            },
            {
                "timestamp": "t0",
                "instrument_id": "inst-2",
                "alpha": 0.1,
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "inst-1",
                "forward_return": 0.03,
            },
            {
                "timestamp": "t0",
                "instrument_id": "inst-2",
                "forward_return": -0.01,
            },
        ]
    )

    ranked = add_cross_sectional_label_rank(labels)
    dataset = join_alpha_features_and_labels(features, ranked)

    assert dataset.shape == (2, 5)
    assert dataset.loc[0, "label_rank"] == pytest.approx(0.5)
    assert dataset.loc[1, "label_rank"] == pytest.approx(1.0)
