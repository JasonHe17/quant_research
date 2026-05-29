from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    path = Path("examples/build_state_conditioned_score_switch.py")
    spec = importlib.util.spec_from_file_location("build_state_conditioned_score_switch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_daily_state_schedule_uses_lagged_expanding_threshold(tmp_path) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    rows = []
    for day, value in [
        ("2024-01-01", 0.1),
        ("2024-01-02", 0.2),
        ("2024-01-03", 0.3),
        ("2024-01-04", 0.05),
    ]:
        rows.append(
            {
                "timestamp": f"{day}T09:35:00+08:00",
                "market_state_downside_mean_5m_w48": value,
            }
        )
    pd.DataFrame(rows).to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)

    schedule = module._daily_state_schedule(
        dataset_dir,
        state_column="market_state_downside_mean_5m_w48",
        activation_quantile=0.5,
        min_history_days=2,
        active_when="gte",
        start="2024-01-01T00:00:00+08:00",
        end="2024-01-31T23:59:59+08:00",
    )

    assert schedule["active"].tolist() == [False, False, True, True]


def test_write_switched_scores_uses_challenger_on_active_dates(tmp_path) -> None:
    module = _load_module()
    baseline_dir = tmp_path / "baseline"
    challenger_dir = tmp_path / "challenger"
    score_dir = tmp_path / "scores"
    baseline_dir.mkdir()
    challenger_dir.mkdir()
    score_dir.mkdir()
    pd.DataFrame(
        [
            {"timestamp": "2024-01-01T09:35:00+08:00", "instrument_id": "a", "score": 1.0},
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "a", "score": 1.0},
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "b", "score": 0.5},
        ]
    ).to_parquet(baseline_dir / "score_2024_01.parquet", index=False)
    pd.DataFrame(
        [
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "b", "score": 2.0},
        ]
    ).to_parquet(challenger_dir / "score_2024_01.parquet", index=False)
    schedule = pd.DataFrame(
        [
            {"trade_date": "2024-01-01", "active": False},
            {"trade_date": "2024-01-02", "active": True},
        ]
    )

    rows = module._write_switched_scores(
        baseline_score_dir=baseline_dir,
        challenger_score_dir=challenger_dir,
        score_dir=score_dir,
        schedule=schedule,
        start="2024-01-01T00:00:00+08:00",
        end="2024-01-31T23:59:59+08:00",
        resume_existing=False,
    )

    output = pd.read_parquet(score_dir / "score_2024_01.parquet")
    assert rows == {"2024_01": 2}
    assert output["instrument_id"].tolist() == ["a", "b"]
    assert output["score"].tolist() == [1.0, 2.0]
    assert output["signal_source"].tolist() == ["baseline", "challenger"]
