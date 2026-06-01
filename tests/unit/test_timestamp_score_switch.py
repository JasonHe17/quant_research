from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.build_timestamp_score_switch import build_timestamp_score_switch


def test_timestamp_score_switch_uses_challenger_only_on_active_timestamps(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    challenger_dir = tmp_path / "challenger"
    baseline_dir.mkdir()
    challenger_dir.mkdir()
    pd.DataFrame(
        [
            {"timestamp": "2024-01-01T09:35:00+08:00", "instrument_id": "a", "score": 1.0},
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "a", "score": 1.0},
        ]
    ).to_parquet(baseline_dir / "score_2024_01.parquet", index=False)
    pd.DataFrame(
        [
            {"timestamp": "2024-01-01T09:35:00+08:00", "instrument_id": "b", "score": 2.0},
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "b", "score": 2.0},
        ]
    ).to_parquet(challenger_dir / "score_2024_01.parquet", index=False)
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T09:35:00+08:00",
                "feature": "d10",
                "weight_scale": 0.0,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "feature": "d10",
                "weight_scale": 1.0,
            },
        ]
    ).to_csv(schedule_path, index=False)

    summary = build_timestamp_score_switch(
        argparse.Namespace(
            baseline_score_dir=str(baseline_dir),
            challenger_score_dir=str(challenger_dir),
            schedule=str(schedule_path),
            schedule_feature="d10",
            active_threshold=0.0,
            output_dir=str(tmp_path / "out"),
            method_name="decorrelated",
            partition_start=None,
            partition_end=None,
            resume_existing=False,
        )
    )
    output = pd.read_parquet(tmp_path / "out" / "scores" / "decorrelated" / "score_2024_01.parquet")

    assert summary["schedule"]["active_timestamp_count"] == 1
    assert output["instrument_id"].tolist() == ["a", "b"]
    assert output["signal_source"].tolist() == ["baseline", "challenger"]
