from __future__ import annotations

import argparse

import pandas as pd
import pytest

from examples.run_score_overlay_validation import (
    Scenario,
    _backtest_job,
    _combine_overlay_components,
    _daily_first_plus_condition_decisions,
    _entry_exclusion_mask,
    _optimizer_risk_penalty_bps,
    _use_daily_first_primary_component_for_condition,
)


def test_daily_first_plus_condition_decisions_keep_open_and_active_tail() -> None:
    scores = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "instrument_id": instrument_id,
                "score": score,
            }
            for timestamp in (
                "2024-01-02T09:35:00+08:00",
                "2024-01-02T09:40:00+08:00",
                "2024-01-02T14:35:00+08:00",
                "2024-01-03T09:35:00+08:00",
                "2024-01-03T14:35:00+08:00",
            )
            for instrument_id, score in (("a", 1.0), ("b", 0.0))
        ]
    )
    condition = {
        "2024-01-02T14:35:00+08:00": True,
        "2024-01-03T14:35:00+08:00": True,
    }

    filtered = _daily_first_plus_condition_decisions(scores, condition=condition)

    assert filtered["timestamp"].drop_duplicates().tolist() == [
        "2024-01-02T09:35:00+08:00",
        "2024-01-02T14:35:00+08:00",
        "2024-01-03T09:35:00+08:00",
        "2024-01-03T14:35:00+08:00",
    ]
    assert len(filtered) == 8


def test_daily_first_plus_condition_requires_parseable_timestamps() -> None:
    scores = pd.DataFrame(
        [{"timestamp": "not-a-time", "instrument_id": "a", "score": 1.0}]
    )

    with pytest.raises(ValueError, match="parseable timestamps"):
        _daily_first_plus_condition_decisions(scores, condition={"not-a-time": True})


def test_daily_first_plus_condition_sorts_before_selecting_session_first() -> None:
    scores = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "instrument_id": "a",
                "score": 1.0,
            }
            for timestamp in (
                "2024-01-02T14:35:00+08:00",
                "2024-01-02T09:35:00+08:00",
                "2024-01-02T09:40:00+08:00",
            )
        ]
    )

    filtered = _daily_first_plus_condition_decisions(
        scores,
        condition={"2024-01-02T14:35:00+08:00": True},
    )

    assert filtered["timestamp"].tolist() == [
        "2024-01-02T14:35:00+08:00",
        "2024-01-02T09:35:00+08:00",
    ]


def test_daily_first_primary_component_anchors_active_condition_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:35:00+08:00",
                "instrument_id": "a",
                "primary_component": -0.40,
                "satellite_component": 0.10,
            },
            {
                "timestamp": "2024-01-02T14:35:00+08:00",
                "instrument_id": "b",
                "primary_component": 0.40,
                "satellite_component": 0.20,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
                "primary_component": 0.25,
                "satellite_component": 0.30,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "b",
                "primary_component": -0.25,
                "satellite_component": 0.40,
            },
            {
                "timestamp": "2024-01-02T09:40:00+08:00",
                "instrument_id": "a",
                "primary_component": 0.10,
                "satellite_component": 0.50,
            },
        ]
    )

    anchored = _use_daily_first_primary_component_for_condition(
        frame,
        condition={"2024-01-02T14:35:00+08:00": True},
    )

    tail = anchored[anchored["timestamp"].eq("2024-01-02T14:35:00+08:00")]
    assert tail.set_index("instrument_id")["primary_component"].to_dict() == {
        "a": 0.25,
        "b": -0.25,
    }
    intraday = anchored[
        anchored["timestamp"].eq("2024-01-02T09:40:00+08:00")
    ].iloc[0]
    assert intraday["primary_component"] == pytest.approx(0.10)


def test_downside_penalty_only_deducts_weak_satellite_tail() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
                "primary_component": 0.40,
                "satellite_component": -0.40,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "b",
                "primary_component": 0.30,
                "satellite_component": 0.10,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "c",
                "primary_component": 0.20,
                "satellite_component": 0.40,
            },
        ]
    )

    score = _combine_overlay_components(
        frame,
        effective_weight=pd.Series(0.5, index=frame.index),
        overlay_mode="downside_penalty",
        downside_penalty_quantile=0.5,
    )

    assert score.iloc[0] < frame["primary_component"].iloc[0]
    assert score.iloc[1] == pytest.approx(frame["primary_component"].iloc[1])
    assert score.iloc[2] == pytest.approx(frame["primary_component"].iloc[2])


def test_downside_penalty_rejects_invalid_quantile() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "primary_component": 0.0,
                "satellite_component": 0.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="downside_penalty_quantile"):
        _combine_overlay_components(
            frame,
            effective_weight=pd.Series(1.0, index=frame.index),
            overlay_mode="downside_penalty",
            downside_penalty_quantile=1.0,
        )


def test_entry_exclusion_mask_blocks_only_active_weak_satellite_tail() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
                "satellite_component": -0.40,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "b",
                "satellite_component": 0.10,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "c",
                "satellite_component": 0.40,
            },
            {
                "timestamp": "2024-01-02T09:40:00+08:00",
                "instrument_id": "a",
                "satellite_component": -0.40,
            },
        ]
    )

    mask = _entry_exclusion_mask(
        frame,
        effective_weight=pd.Series([0.1, 0.1, 0.1, 0.0], index=frame.index),
        entry_exclusion_quantile=0.5,
    )

    assert mask.tolist() == [False, True, True, True]


def test_optimizer_risk_penalty_bps_penalizes_weak_satellite_tail() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
                "satellite_component": -0.40,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "b",
                "satellite_component": 0.10,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "c",
                "satellite_component": 0.40,
            },
        ]
    )

    penalty = _optimizer_risk_penalty_bps(
        frame,
        effective_weight=pd.Series(100.0, index=frame.index),
        risk_penalty_quantile=0.5,
    )

    assert penalty.iloc[0] == pytest.approx(50.0)
    assert penalty.iloc[1] == pytest.approx(0.0)
    assert penalty.iloc[2] == pytest.approx(0.0)


def test_optimizer_risk_penalty_mode_keeps_primary_score() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "primary_component": 0.25,
                "satellite_component": -0.25,
            }
        ]
    )

    score = _combine_overlay_components(
        frame,
        effective_weight=pd.Series(100.0, index=frame.index),
        overlay_mode="optimizer_risk_penalty",
        downside_penalty_quantile=0.2,
    )

    assert score.iloc[0] == pytest.approx(0.25)


def test_backtest_job_forwards_turnover_budget_controls(tmp_path) -> None:
    args = argparse.Namespace(
        catalog_path="catalog.duckdb",
        top_n=50,
        initial_cash=1_000_000.0,
        min_commission=5.0,
        lot_size=100,
        trade_policy="cost_aware_optimizer",
        rebalance_every_n_bars=48,
        policy_min_hold_bars=0,
        policy_no_trade_weight_band=0.002,
        policy_partial_rebalance_rate=0.5,
        policy_gross_exposure_scale=1.0,
        policy_gross_exposure_scale_path=None,
        policy_entry_rank=50,
        policy_exit_rank=150,
        policy_max_entries_per_rebalance=10,
        policy_max_exits_per_rebalance=10,
        policy_max_gross_turnover_per_rebalance=0.5,
        policy_total_gross_turnover_budget=155.0,
        policy_turnover_budget_pacing=0.1,
        policy_turnover_budget_period="path",
        policy_drawdown_brake_reduced_scale=0.5,
        policy_cost_pressure_threshold_bps=1000.0,
        policy_cost_pressure_reduced_scale=1.0,
        policy_cost_pressure_max_gross_turnover_per_rebalance=0.01,
        min_trade_weight=0.0005,
        limit_up_bps=980.0,
        limit_down_bps=980.0,
        data_access_mode="fast_parquet",
        streaming_chunk="month",
        streaming_chunk_padding_days=10,
        optimizer_score_to_edge_bps=100.0,
        optimizer_min_net_edge_bps=0.0,
        optimizer_risk_penalty_multiplier=1.0,
        optimizer_weighting="utility",
        exclude_st=True,
        policy="cost_aware_optimizer_budget155_daily",
    )
    scenario = Scenario(
        name="full_base",
        start="2023-01-01T00:00:00+08:00",
        end="2025-12-31T23:59:59+08:00",
        partition_glob="score_*.parquet",
        commission_bps=3.0,
        slippage_bps=1.0,
        sell_stamp_tax_bps=5.0,
        estimated_cost_bps=13.0,
        description="test",
    )

    job = _backtest_job(
        args,
        output_dir=tmp_path,
        method="vc_opt_risk_w25",
        scenario=scenario,
    )

    assert (
        job.command[job.command.index("--policy-max-gross-turnover-per-rebalance") + 1]
        == "0.5"
    )
    assert (
        job.command[job.command.index("--policy-total-gross-turnover-budget") + 1]
        == "155.0"
    )
    assert (
        job.command[job.command.index("--policy-cost-pressure-threshold-bps") + 1]
        == "1000.0"
    )
    assert (
        job.command[job.command.index("--policy-cost-pressure-reduced-scale") + 1]
        == "1.0"
    )
    assert (
        job.command[
            job.command.index(
                "--policy-cost-pressure-max-gross-turnover-per-rebalance"
            )
            + 1
        ]
        == "0.01"
    )
    assert job.command[job.command.index("--policy-turnover-budget-period") + 1] == "path"
