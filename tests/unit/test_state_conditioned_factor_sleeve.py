from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.build_state_conditioned_factor_sleeve import (
    build_state_conditioned_factor_sleeve,
)


def test_state_conditioned_factor_sleeve_uses_lagged_broad_tape(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T09:35:00+08:00",
                "instrument_id": "a",
                "market_state_return_5m": 0.01,
                "market_state_breadth_5m": 0.6,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
                "market_state_return_5m": -0.01,
                "market_state_breadth_5m": 0.7,
            },
            {
                "timestamp": "2024-01-03T09:35:00+08:00",
                "instrument_id": "a",
                "market_state_return_5m": 0.01,
                "market_state_breadth_5m": 0.4,
            },
        ]
    ).to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)

    summary = build_state_conditioned_factor_sleeve(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            target_features=["d10"],
            return_column="market_state_return_5m",
            breadth_column="market_state_breadth_5m",
            rule="return_and_breadth",
            full_scale=1.0,
            reduced_scale=0.5,
            blocked_scale=0.0,
            warmup_scale=0.0,
            base_schedule=None,
            combine_mode="min",
            partition_start=None,
            partition_end=None,
            output_dir=str(tmp_path / "out"),
        )
    )
    schedule = pd.read_csv(summary["artifacts"]["sleeve_schedule"])

    assert schedule["sleeve_state"].tolist() == ["warmup", "full", "blocked"]
    assert schedule["weight_scale"].tolist() == [0.0, 1.0, 0.0]


def test_state_conditioned_factor_sleeve_combines_with_base_schedule(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    timestamp = "2024-01-01T09:35:00+08:00"
    pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "instrument_id": "a",
                "market_state_return_5m": 0.01,
                "market_state_breadth_5m": 0.6,
            }
        ]
    ).to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)
    base_path = tmp_path / "base.csv"
    pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "feature": "base_factor",
                "weight_scale": 0.8,
                "shrink_reason": "base",
            }
        ]
    ).to_csv(base_path, index=False)

    summary = build_state_conditioned_factor_sleeve(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            target_features=["d10"],
            return_column="market_state_return_5m",
            breadth_column="market_state_breadth_5m",
            rule="return_positive",
            full_scale=1.0,
            reduced_scale=0.5,
            blocked_scale=0.0,
            warmup_scale=0.0,
            base_schedule=str(base_path),
            combine_mode="min",
            partition_start=None,
            partition_end=None,
            output_dir=str(tmp_path / "out"),
        )
    )
    schedule = pd.read_csv(summary["artifacts"]["schedule"])

    assert set(schedule["feature"]) == {"base_factor", "d10"}
    assert schedule.loc[schedule["feature"] == "base_factor", "weight_scale"].item() == 0.8
    assert schedule.loc[schedule["feature"] == "d10", "weight_scale"].item() == 0.0
