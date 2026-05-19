from __future__ import annotations

import pandas as pd
import pytest

from quant_research.models import (
    evaluate_cross_sectional_predictions,
    infer_feature_columns,
    time_split,
)


def test_infer_feature_columns_excludes_labels_and_execution_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["t0"],
            "instrument_id": ["inst-1"],
            "alpha_a": [0.1],
            "alpha_b": [1.0],
            "forward_return": [0.02],
            "forward_return_rank": [0.5],
            "entry_price": [10.0],
            "exit_price": [11.0],
            "entry_timestamp": ["t1"],
            "exit_timestamp": ["t2"],
            "entry_tradable_bar": [True],
            "entry_limit_up_open": [False],
            "entry_limit_down_open": [False],
            "tradable_bar": [True],
            "buyable_bar": [True],
            "sellable_bar": [True],
            "suspended_bar": [False],
            "limit_up_open": [False],
            "limit_down_open": [False],
            "is_st": [False],
            "previous_close": [9.9],
        }
    )

    assert infer_feature_columns(frame) == ("alpha_a", "alpha_b")


def test_time_split_uses_timestamp_boundaries() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "2024-01-01T09:35:00+08:00", "instrument_id": "a"},
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "a"},
            {"timestamp": "2024-01-03T09:35:00+08:00", "instrument_id": "a"},
            {"timestamp": "2024-01-04T09:35:00+08:00", "instrument_id": "a"},
        ]
    )

    splits = time_split(
        frame,
        train_end="2024-01-02T09:35:00+08:00",
        valid_start="2024-01-03T09:35:00+08:00",
        valid_end="2024-01-03T09:35:00+08:00",
        test_start="2024-01-04T09:35:00+08:00",
    )

    assert splits["train"]["timestamp"].tolist() == [
        "2024-01-01T09:35:00+08:00",
        "2024-01-02T09:35:00+08:00",
    ]
    assert splits["valid"]["timestamp"].tolist() == ["2024-01-03T09:35:00+08:00"]
    assert splits["test"]["timestamp"].tolist() == ["2024-01-04T09:35:00+08:00"]


def test_time_split_purges_training_labels_that_overlap_evaluation() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T09:35:00+08:00",
                "exit_timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "exit_timestamp": "2024-01-03T09:35:00+08:00",
                "instrument_id": "a",
            },
            {
                "timestamp": "2024-01-03T09:35:00+08:00",
                "exit_timestamp": "2024-01-04T09:35:00+08:00",
                "instrument_id": "a",
            },
        ]
    )

    splits = time_split(
        frame,
        train_end="2024-01-02T09:35:00+08:00",
        valid_start="2024-01-03T09:35:00+08:00",
        test_start="2024-01-03T09:35:00+08:00",
    )

    assert splits["train"]["timestamp"].tolist() == ["2024-01-01T09:35:00+08:00"]
    assert splits["valid"]["timestamp"].tolist() == ["2024-01-03T09:35:00+08:00"]
    assert splits["test"]["timestamp"].tolist() == ["2024-01-03T09:35:00+08:00"]


def test_evaluate_cross_sectional_predictions_reports_ic_and_top_spread() -> None:
    predictions = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "score": 0.9,
                "forward_return": 0.03,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "score": 0.1,
                "forward_return": -0.01,
            },
            {
                "timestamp": "t1",
                "instrument_id": "a",
                "score": 0.8,
                "forward_return": 0.02,
            },
            {
                "timestamp": "t1",
                "instrument_id": "b",
                "score": 0.2,
                "forward_return": 0.00,
            },
        ]
    )

    metrics, by_timestamp = evaluate_cross_sectional_predictions(
        predictions,
        top_n=1,
    )

    assert metrics["timestamp_count"] == 2
    assert metrics["sample_count"] == 4
    assert metrics["top_minus_bottom_label"] == pytest.approx(0.03)
    assert by_timestamp["spearman_rank_ic"].tolist() == pytest.approx([1.0, 1.0])
