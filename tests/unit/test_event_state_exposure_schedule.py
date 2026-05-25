from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.build_event_state_exposure_schedule import (
    build_event_state_exposure_schedule,
)


def test_event_state_exposure_schedule_uses_lagged_states_and_base_schedule(
    tmp_path: Path,
) -> None:
    states_path = tmp_path / "states.csv"
    base_path = tmp_path / "base.csv"
    output_dir = tmp_path / "gate"
    timestamps = pd.date_range("2024-01-01 09:35", periods=4, freq="5min")
    pd.DataFrame(
        [
            {"timestamp": timestamps[index], "event_state": state}
            for index, state in enumerate(
                [
                    "calm",
                    "limit_diffusion",
                    "limit_diffusion_extreme",
                    "shock_extreme",
                ]
            )
        ]
    ).to_csv(states_path, index=False)
    pd.DataFrame(
        [
            {"timestamp": timestamps[0], "gross_exposure_scale": 0.8},
            {"timestamp": timestamps[1], "gross_exposure_scale": 0.8},
            {"timestamp": timestamps[2], "gross_exposure_scale": 0.8},
            {"timestamp": timestamps[3], "gross_exposure_scale": 0.8},
        ]
    ).to_csv(base_path, index=False)

    summary = build_event_state_exposure_schedule(
        argparse.Namespace(
            event_states_path=str(states_path),
            output_dir=str(output_dir),
            state_column="event_state",
            lag_windows=1,
            full_scale=1.0,
            reduced_scale=0.5,
            blocked_scale=0.0,
            warmup_scale=1.0,
            reduced_states=["limit_diffusion"],
            blocked_states=["limit_diffusion_extreme"],
            warmup_states=["warmup"],
            base_schedule=str(base_path),
            combine_mode="min",
        )
    )

    schedule = pd.read_csv(output_dir / "gross_exposure_schedule.csv")
    assert summary["schedule_count"] == 4
    assert pd.isna(schedule.loc[0, "effective_event_state"])
    assert schedule["effective_event_state"].iloc[1:].tolist() == [
        "calm",
        "limit_diffusion",
        "limit_diffusion_extreme",
    ]
    assert schedule["event_state_gross_exposure_scale"].tolist() == [1.0, 1.0, 0.5, 0.0]
    assert schedule["gross_exposure_scale"].tolist() == [0.8, 0.8, 0.5, 0.0]
    assert schedule["event_state_gate_reason"].tolist() == [
        "lagged_state_missing",
        "full_event_state",
        "reduced_event_state",
        "blocked_event_state",
    ]
