from __future__ import annotations

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.metrics import MetricResult, MetricsEngine, MetricsReport
from quant_research.metrics.performance import total_return
from quant_research.metrics.risk import max_drawdown


def test_total_return() -> None:
    assert total_return(100.0, 125.0) == 0.25


def test_max_drawdown() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 130.0]) == -0.25


def test_metrics_engine_builds_equity_curve_report() -> None:
    report = MetricsEngine().from_equity_curve(
        _equity_curve(),
        name="smoke-report",
        metadata={"run_id": "run-1"},
    )

    values = {metric.name: metric.value for metric in report.metrics}
    assert values["total_return"] == pytest.approx(0.2)
    assert values["max_drawdown"] == pytest.approx(-1.0 / 12.0)
    assert report.metadata["run_id"] == "run-1"


def test_metrics_engine_persists_report(tmp_path) -> None:
    store = ArtifactStore.from_path(tmp_path)
    report = MetricsEngine(artifact_store=store).from_equity_curve(
        _equity_curve(),
        name="persisted-report",
        persist=True,
    )
    payload = store.read_metrics_report("persisted-report")

    assert set(report.artifacts) == {"metrics_report"}
    assert payload["name"] == "persisted-report"
    assert payload["metrics"][0]["name"] == "total_return"


def test_metrics_engine_validates_inputs() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        MetricsEngine().from_equity_curve(
            pd.DataFrame([{"timestamp": "2024-01-01"}]),
            name="bad",
        )

    with pytest.raises(ValueError, match="artifact_store"):
        MetricsEngine().from_equity_curve(
            _equity_curve(),
            name="bad",
            persist=True,
        )


def test_metric_models_validate_names() -> None:
    with pytest.raises(ValueError, match="metric name"):
        MetricResult("", 1.0)

    with pytest.raises(ValueError, match="report name"):
        MetricsReport("", ())


def _equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": "2024-01-01", "equity": 100.0},
            {"timestamp": "2024-01-02", "equity": 120.0},
            {"timestamp": "2024-01-03", "equity": 110.0},
            {"timestamp": "2024-01-04", "equity": 120.0},
        ]
    )
