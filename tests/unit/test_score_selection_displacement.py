from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from examples.analyze_score_selection_displacement import (
    selection_displacement_by_timestamp,
)


def test_selection_displacement_reports_added_minus_removed_label(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    baseline_dir = tmp_path / "baseline"
    challenger_dir = tmp_path / "challenger"
    dataset_dir.mkdir()
    baseline_dir.mkdir()
    challenger_dir.mkdir()
    timestamp = "2024-01-02T09:35:00+08:00"
    pd.DataFrame(
        [
            {"timestamp": timestamp, "instrument_id": "a", "forward_return": 0.01},
            {"timestamp": timestamp, "instrument_id": "b", "forward_return": 0.02},
            {"timestamp": timestamp, "instrument_id": "c", "forward_return": 0.03},
        ]
    ).to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)
    pd.DataFrame(
        [
            {"timestamp": timestamp, "instrument_id": "a", "score": 3.0},
            {"timestamp": timestamp, "instrument_id": "b", "score": 2.0},
            {"timestamp": timestamp, "instrument_id": "c", "score": 1.0},
        ]
    ).to_parquet(baseline_dir / "score_2024_01.parquet", index=False)
    pd.DataFrame(
        [
            {"timestamp": timestamp, "instrument_id": "c", "score": 3.0},
            {"timestamp": timestamp, "instrument_id": "a", "score": 2.0},
            {"timestamp": timestamp, "instrument_id": "b", "score": 1.0},
        ]
    ).to_parquet(challenger_dir / "score_2024_01.parquet", index=False)
    regime_path = tmp_path / "regime.csv"
    pd.DataFrame(
        [{"timestamp": timestamp, "feature": "alpha", "regime_state": "stress"}]
    ).to_csv(regime_path, index=False)

    result = selection_displacement_by_timestamp(
        baseline_score_dir=baseline_dir,
        challenger_score_dir=challenger_dir,
        dataset_dir=dataset_dir,
        regime_schedule=regime_path,
        top_n=2,
        label_column="forward_return",
    )

    row = result.iloc[0]
    assert row["regime_state"] == "stress"
    assert row["overlap_count"] == 1
    assert row["added_count"] == 1
    assert row["removed_count"] == 1
    assert row["added_label_mean"] == pytest.approx(0.03)
    assert row["removed_label_mean"] == pytest.approx(0.02)
    assert row["replacement_label_delta"] == pytest.approx(0.01)
    assert row["top_label_delta"] == pytest.approx(0.005)
