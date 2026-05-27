from __future__ import annotations

import pandas as pd

from examples.build_state_conditioned_factor_health_schedule import (
    build_state_conditioned_schedule,
)


def test_build_state_conditioned_schedule_selects_stress_memory() -> None:
    normal = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "normal_weight_scale": 1.0},
            {"timestamp": "t1", "feature": "alpha_a", "normal_weight_scale": 0.8},
        ]
    )
    stress = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "stress_weight_scale": 0.5},
            {"timestamp": "t1", "feature": "alpha_a", "stress_weight_scale": 0.4},
        ]
    )
    regime = pd.DataFrame(
        [
            {"timestamp": "t0", "regime_selector_scale": 1.0, "regime_weight": 0.0},
            {"timestamp": "t1", "regime_selector_scale": 0.5, "regime_weight": 1.0},
        ]
    )

    schedule = build_state_conditioned_schedule(
        normal,
        stress,
        regime,
        mode="select",
    )

    by_timestamp = schedule.set_index("timestamp")["weight_scale"]
    assert by_timestamp["t0"] == 1.0
    assert by_timestamp["t1"] == 0.4


def test_build_state_conditioned_schedule_blends_regime_weight() -> None:
    normal = pd.DataFrame(
        [{"timestamp": "t0", "feature": "alpha_a", "normal_weight_scale": 1.0}]
    )
    stress = pd.DataFrame(
        [{"timestamp": "t0", "feature": "alpha_a", "stress_weight_scale": 0.5}]
    )
    regime = pd.DataFrame(
        [{"timestamp": "t0", "regime_selector_scale": 0.5, "regime_weight": 0.5}]
    )

    schedule = build_state_conditioned_schedule(
        normal,
        stress,
        regime,
        mode="blend",
    )

    assert schedule.loc[0, "weight_scale"] == 0.75
    assert schedule.loc[0, "state_conditioned_mode"] == "blend"
