from __future__ import annotations

from pathlib import Path

import pandas as pd

from examples.build_factor_risk_gate import build_factor_risk_gate


def test_build_factor_risk_gate_uses_lagged_thresholds_and_base_schedule(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    pd.DataFrame(
        [
            {"timestamp": f"t{index}", "instrument_id": "a", "risk": float(value)}
            for index, value in enumerate([1, 1, 1, 1, 1, 10])
        ]
    ).to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)
    base_path = tmp_path / "base.csv"
    pd.DataFrame(
        [
            {"timestamp": f"t{index}", "gross_exposure_scale": 0.8}
            for index in range(6)
        ]
    ).to_csv(base_path, index=False)
    args = type(
        "Args",
        (),
        {
            "dataset_dir": str(dataset_dir),
            "feature": "risk",
            "output_dir": str(tmp_path / "gate"),
            "aggregate": "mean",
            "aggregate_quantile": 0.75,
            "lookback_windows": 3,
            "min_periods": 3,
            "high_quantile": 0.5,
            "extreme_quantile": 0.9,
            "full_scale": 1.0,
            "reduced_scale": 0.5,
            "blocked_scale": 0.0,
            "warmup_scale": 1.0,
            "state_confirmation_windows": 1,
            "max_scale_change_per_window": None,
            "max_scale_increase_per_window": None,
            "max_scale_decrease_per_window": None,
            "scale_change_deadband": 0.0,
            "base_schedule": str(base_path),
            "combine_mode": "min",
            "partition_start": None,
            "partition_end": None,
            "max_partitions": None,
        },
    )()

    summary = build_factor_risk_gate(args)
    schedule = pd.read_csv(summary["artifacts"]["schedule"])

    assert schedule.loc[2, "risk_state"] == "warmup"
    assert schedule.loc[5, "risk_state"] == "blocked"
    assert schedule.loc[5, "factor_gross_exposure_scale"] == 0.0
    assert schedule.loc[0, "gross_exposure_scale"] == 0.8
    assert schedule.loc[5, "gross_exposure_scale"] == 0.0


def test_build_factor_risk_gate_can_smooth_scale_changes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    pd.DataFrame(
        [
            {"timestamp": f"t{index}", "instrument_id": "a", "risk": float(value)}
            for index, value in enumerate([1, 1, 1, 10, 10, 20])
        ]
    ).to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)
    args = type(
        "Args",
        (),
        {
            "dataset_dir": str(dataset_dir),
            "feature": "risk",
            "output_dir": str(tmp_path / "gate"),
            "aggregate": "mean",
            "aggregate_quantile": 0.75,
            "lookback_windows": 3,
            "min_periods": 3,
            "high_quantile": 0.5,
            "extreme_quantile": 0.9,
            "full_scale": 1.0,
            "reduced_scale": 0.5,
            "blocked_scale": 0.0,
            "warmup_scale": 1.0,
            "state_confirmation_windows": 1,
            "max_scale_change_per_window": 0.25,
            "max_scale_increase_per_window": None,
            "max_scale_decrease_per_window": None,
            "scale_change_deadband": 0.0,
            "base_schedule": None,
            "combine_mode": "min",
            "partition_start": None,
            "partition_end": None,
            "max_partitions": None,
        },
    )()

    summary = build_factor_risk_gate(args)
    schedule = pd.read_csv(summary["artifacts"]["schedule"])

    assert schedule.loc[3, "raw_risk_state"] == "blocked"
    assert schedule.loc[3, "raw_gross_exposure_scale"] == 0.0
    assert schedule.loc[3, "risk_state"] == "step_limited_down"
    assert schedule.loc[3, "gross_exposure_scale"] == 0.75
    assert schedule.loc[4, "gross_exposure_scale"] == 0.5
    assert schedule.loc[5, "gross_exposure_scale"] == 0.25
