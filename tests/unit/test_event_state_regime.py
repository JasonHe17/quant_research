from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.analyze_event_state_regime import analyze_event_state_regime


def test_event_state_regime_analysis_outputs_state_tables(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    score_dir = tmp_path / "scores"
    output_dir = tmp_path / "out"
    dataset_dir.mkdir()
    score_dir.mkdir()
    rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2024-01-01 09:35", periods=8, freq="5min")
    for timestamp_index, timestamp in enumerate(timestamps):
        for instrument_index in range(6):
            instrument = f"inst{instrument_index}"
            is_late = timestamp_index >= 4
            rows.append(
                {
                    "timestamp": timestamp,
                    "instrument_id": instrument,
                    "forward_return": 0.001 * (instrument_index - 2) - 0.002 * is_late,
                    "entry_tradable_bar": True,
                    "entry_limit_up_open": is_late and instrument_index == 0,
                    "entry_limit_down_open": is_late and instrument_index in {1, 2},
                    "intraday_event_sync_down_resilience_5m_w48": float(timestamp_index),
                    "intraday_event_limit_diffusion_resilience_5m_w48": float(
                        timestamp_index + instrument_index
                    ),
                    "intraday_event_turnover_dislocation_recovery_5m_w48": float(
                        instrument_index
                    ),
                    "intraday_event_open_jump_recovery_quality_5m_w48": float(
                        timestamp_index - instrument_index
                    ),
                }
            )
            score_rows.append(
                {
                    "timestamp": timestamp,
                    "instrument_id": instrument,
                    "score": float(instrument_index),
                }
            )
    pd.DataFrame(rows).to_parquet(dataset_dir / "dataset_2024_01.parquet", index=False)
    pd.DataFrame(score_rows).to_parquet(score_dir / "score_2024_01.parquet", index=False)

    summary = analyze_event_state_regime(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            validation_dir=None,
            score_dir=str(score_dir),
            output_dir=str(output_dir),
            scenario="full_base",
            method="decorrelated",
            policy="partial_rebalance_daily",
            label_column="forward_return",
            top_n=2,
            event_feature_columns=[
                "intraday_event_sync_down_resilience_5m_w48",
                "intraday_event_limit_diffusion_resilience_5m_w48",
                "intraday_event_turnover_dislocation_recovery_5m_w48",
                "intraday_event_open_jump_recovery_quality_5m_w48",
            ],
            months=["2024_01"],
            lookback_windows=3,
            min_periods=2,
            high_z=0.5,
            extreme_z=1.0,
            max_z_score=6.0,
            stabilization_windows=2,
            report_months=2,
        )
    )

    assert summary["timestamp_count"] == len(timestamps)
    assert (output_dir / "timestamp_event_states.csv").exists()
    assert (output_dir / "event_state_performance.csv").exists()
    assert (output_dir / "event_state_regime_report.md").exists()
    state_table = pd.read_csv(output_dir / "event_state_performance.csv")
    assert set(state_table["event_state"])
    assert "score_top_n_mean_label" in state_table.columns
