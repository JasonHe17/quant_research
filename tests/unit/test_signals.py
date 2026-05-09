from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.signals import SignalGenerator, SignalSpec


def test_signal_generator_identity_method() -> None:
    result = SignalGenerator().generate(
        _factors(),
        SignalSpec(name="identity-signal", factor_name="factor-a"),
    )

    assert result.frame["signal"].tolist() == [1.0, 2.0]
    assert result.frame["signal_name"].unique().tolist() == ["identity-signal"]
    assert result.diagnostics.loc[0, "instrument_count"] == 2


def test_signal_generator_rank_method() -> None:
    result = SignalGenerator().generate(
        _factors(),
        SignalSpec(
            name="rank-signal",
            factor_name="factor-a",
            method="rank",
            parameters={"ascending": False, "percentile": True},
        ),
    )

    assert result.frame["signal"].tolist() == [1.0, 0.5]


def test_signal_generator_threshold_method() -> None:
    result = SignalGenerator().generate(
        _factors(),
        SignalSpec(
            name="threshold-signal",
            factor_name="factor-a",
            method="threshold",
            parameters={"threshold": 1.5, "long_value": 1.0, "short_value": -1.0},
        ),
    )

    assert result.frame["signal"].tolist() == [-1.0, 1.0]


def test_signal_generator_persists_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore.from_path(tmp_path)
    result = SignalGenerator(artifact_store=store).generate(
        _factors(),
        SignalSpec(name="persisted", factor_name="factor-a"),
        persist=True,
    )

    assert set(result.artifacts) == {"signals", "diagnostics"}
    assert store.read_signal_artifact("persisted", "signals").equals(result.frame)


def test_signal_generator_validates_inputs() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        SignalGenerator().generate(
            pd.DataFrame([{"instrument_id": "inst-1"}]),
            SignalSpec(name="bad", factor_name="factor-a"),
        )

    with pytest.raises(ValueError, match="artifact_store"):
        SignalGenerator().generate(
            _factors(),
            SignalSpec(name="bad", factor_name="factor-a"),
            persist=True,
        )


def test_signal_spec_validates_inputs() -> None:
    with pytest.raises(ValueError, match="signal name"):
        SignalSpec(name="", factor_name="factor-a")

    with pytest.raises(ValueError, match="method"):
        SignalSpec(name="bad", factor_name="factor-a", method="unknown")


def _factors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "instrument_id": "inst-1",
                "factor_value": 1.0,
            },
            {
                "timestamp": "2024-01-01",
                "instrument_id": "inst-2",
                "factor_value": 2.0,
            },
        ]
    )
