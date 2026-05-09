"""Metrics engine scaffolding."""

from __future__ import annotations

import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.metrics.models import MetricResult, MetricsReport
from quant_research.metrics.performance import total_return
from quant_research.metrics.risk import max_drawdown


class MetricsEngine:
    """Computes standard metrics and persists reports."""

    def __init__(self, *, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def from_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        *,
        name: str,
        metadata: dict[str, object] | None = None,
        persist: bool = False,
    ) -> MetricsReport:
        _require_columns(equity_curve, ("timestamp", "equity"))
        if equity_curve.empty:
            raise ValueError("equity_curve must not be empty")
        values = [float(value) for value in equity_curve["equity"].tolist()]
        report = MetricsReport(
            name=name,
            metrics=(
                MetricResult("total_return", total_return(values[0], values[-1])),
                MetricResult("max_drawdown", max_drawdown(values)),
            ),
            metadata=dict(metadata or {}),
        )
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            return report.with_artifacts(
                self.artifact_store.write_metrics_report(report)
            )
        return report


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
