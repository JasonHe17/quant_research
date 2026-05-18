from __future__ import annotations

import pandas as pd
import pytest

from examples.analyze_candidate_policy_regime import (
    _summarize_composite,
    _timestamp_diagnostics,
)
from quant_research.portfolio import CandidateFactor


def test_candidate_policy_regime_timestamp_diagnostics_orients_factors() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "score": 0.9,
                "alpha_a": 0.1,
                "alpha_b": 0.9,
                "forward_return": 0.03,
                "forward_return_240b": 0.30,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "score": 0.2,
                "alpha_a": 0.5,
                "alpha_b": 0.2,
                "forward_return": 0.00,
                "forward_return_240b": 0.00,
            },
            {
                "timestamp": "t0",
                "instrument_id": "c",
                "score": 0.1,
                "alpha_a": 0.9,
                "alpha_b": 0.1,
                "forward_return": -0.02,
                "forward_return_240b": -0.20,
            },
        ]
    )
    candidates = (
        CandidateFactor("alpha_a", -1, -0.02),
        CandidateFactor("alpha_b", 1, 0.01),
    )

    diagnostics = _timestamp_diagnostics(
        frame,
        candidates=candidates,
        weights={"alpha_a": 0.7, "alpha_b": 0.3},
        top_n=1,
        label_column="forward_return_240b",
    )

    assert diagnostics["composite"]["score_top_minus_bottom_label"] == pytest.approx(0.5)
    factor_rows = {row["feature"]: row for row in diagnostics["factor_rows"]}
    assert factor_rows["alpha_a"]["top_minus_bottom_label"] == pytest.approx(0.5)
    assert factor_rows["alpha_b"]["top_minus_bottom_label"] == pytest.approx(0.5)
    exposure_rows = {row["feature"]: row for row in diagnostics["exposure_rows"]}
    assert (
        exposure_rows["alpha_a"]["top_score_abs_contribution_share"]
        + exposure_rows["alpha_b"]["top_score_abs_contribution_share"]
    ) == pytest.approx(1.0)


def test_candidate_policy_regime_composite_summary_reports_market_rates() -> None:
    rows = [
        {
            "sample_count": 3,
            "score_rank_ic": 1.0,
            "score_top_n_mean_label": 0.03,
            "score_bottom_n_mean_label": -0.02,
            "score_top_minus_bottom_label": 0.05,
        },
        {
            "sample_count": 3,
            "score_rank_ic": -0.5,
            "score_top_n_mean_label": 0.01,
            "score_bottom_n_mean_label": 0.02,
            "score_top_minus_bottom_label": -0.01,
        },
    ]
    frame = pd.DataFrame(
        {
            "forward_return": [0.01, 0.02, -0.01],
            "forward_return_240b": [0.10, 0.20, -0.10],
            "entry_tradable_bar": [True, True, False],
            "entry_limit_up_open": [False, True, False],
            "entry_limit_down_open": [False, False, True],
        }
    )

    summary = _summarize_composite(
        "2024-01",
        rows,
        frame,
        label_column="forward_return_240b",
    )

    assert summary["month"] == "2024-01"
    assert summary["label_column"] == "forward_return_240b"
    assert summary["market_mean_label"] == pytest.approx((0.10 + 0.20 - 0.10) / 3)
    assert summary["score_rank_ic_mean"] == pytest.approx(0.25)
    assert summary["entry_tradable_rate"] == pytest.approx(2 / 3)
