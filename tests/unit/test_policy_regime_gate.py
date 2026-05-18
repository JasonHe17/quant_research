from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.build_policy_regime_gate import build_policy_regime_gate


def test_build_policy_regime_gate_writes_lagged_schedule(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    score_dir = tmp_path / "scores"
    output_dir = tmp_path / "gate"
    dataset_dir.mkdir()
    score_dir.mkdir()
    dataset = pd.DataFrame(
        [
            _dataset_row("t0", "a", -0.03),
            _dataset_row("t0", "b", 0.01),
            _dataset_row("t0", "c", 0.02),
            _dataset_row("t1", "a", 0.03),
            _dataset_row("t1", "b", 0.00),
            _dataset_row("t1", "c", -0.01),
            _dataset_row("t2", "a", 0.02),
            _dataset_row("t2", "b", 0.00),
            _dataset_row("t2", "c", -0.01),
        ]
    )
    scores = pd.DataFrame(
        [
            _score_row("t0", "a", 0.9),
            _score_row("t0", "b", 0.2),
            _score_row("t0", "c", 0.1),
            _score_row("t1", "a", 0.9),
            _score_row("t1", "b", 0.2),
            _score_row("t1", "c", 0.1),
            _score_row("t2", "a", 0.9),
            _score_row("t2", "b", 0.2),
            _score_row("t2", "c", 0.1),
        ]
    )
    dataset.to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)
    scores.to_parquet(score_dir / "score_2024_01.parquet", index=False)

    summary = build_policy_regime_gate(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            scores_path=str(score_dir),
            output_dir=str(output_dir),
            label_column="forward_return",
            top_n=1,
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            state_confirmation_windows=1,
            max_scale_change_per_window=None,
            max_scale_increase_per_window=None,
            max_scale_decrease_per_window=None,
            scale_change_deadband=0.0,
            gate_mode="threshold",
            full_scale=1.0,
            reduced_scale=0.5,
            blocked_scale=0.0,
            warmup_scale=1.0,
            budget_min_scale=0.25,
            budget_max_scale=1.0,
            budget_top_return_floor=-0.001,
            budget_top_return_ceiling=0.001,
            budget_spread_floor=-0.001,
            budget_spread_ceiling=0.001,
            budget_rank_ic_floor=-0.05,
            budget_rank_ic_ceiling=0.05,
            min_top_return=0.0,
            min_spread=0.0,
            min_rank_ic=0.0,
            block_top_return=-0.01,
            block_spread=-0.01,
            block_rank_ic=-0.1,
            partition_start=None,
            partition_end=None,
            max_partitions=None,
        )
    )

    schedule = pd.read_csv(output_dir / "gross_exposure_schedule.csv")
    assert summary["schedule_count"] == 3
    assert schedule["gate_reason"].tolist() == [
        "warmup",
        "blocked_exposure",
        "full_exposure",
    ]
    assert schedule["gross_exposure_scale"].tolist() == [1.0, 0.0, 1.0]
    assert "raw_reason_counts" in summary


def _dataset_row(timestamp: str, instrument_id: str, forward_return: float) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "instrument_id": instrument_id,
        "forward_return": forward_return,
    }


def _score_row(timestamp: str, instrument_id: str, score: float) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "instrument_id": instrument_id,
        "score": score,
    }
