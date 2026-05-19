"""Signal generation scaffolding."""

from __future__ import annotations

import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.signals.models import SignalResult, SignalSpec


class SignalGenerator:
    """Transforms factor values into standard signal tables."""

    def __init__(self, *, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def generate(
        self,
        factors: pd.DataFrame,
        spec: SignalSpec,
        *,
        persist: bool = False,
    ) -> SignalResult:
        _require_columns(factors, ("timestamp", "instrument_id", "factor_value"))
        frame = _signals_from_factors(factors, spec=spec)
        diagnostics = _diagnostics(frame)
        result = SignalResult(spec=spec, frame=frame, diagnostics=diagnostics)
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            return result.with_artifacts(self.artifact_store.write_signal(result))
        return result


def _signals_from_factors(factors: pd.DataFrame, *, spec: SignalSpec) -> pd.DataFrame:
    if factors.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "instrument_id",
                "signal_name",
                "factor_name",
                "signal",
            ]
        )
    ordered = factors.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True)
    if spec.method == "identity":
        signals = ordered["factor_value"].astype(float)
    elif spec.method == "rank":
        ascending = bool(spec.parameters.get("ascending", True))
        percentile = bool(spec.parameters.get("percentile", True))
        signals = ordered.groupby("timestamp", sort=False)["factor_value"].rank(
            method="average",
            ascending=ascending,
            pct=percentile,
        )
    else:
        threshold = float(spec.parameters.get("threshold", 0.0))
        long_value = float(spec.parameters.get("long_value", 1.0))
        short_value = float(spec.parameters.get("short_value", 0.0))
        factor_values = ordered["factor_value"].astype(float)
        signals = pd.Series(short_value, index=ordered.index)
        signals.loc[factor_values >= threshold] = long_value
    output = ordered.loc[:, ["timestamp", "instrument_id"]].copy()
    output["signal_name"] = spec.name
    output["factor_name"] = spec.factor_name
    output["signal"] = signals.astype(float)
    return output


def _diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["timestamp", "instrument_count", "min_signal", "max_signal"]
        )
    diagnostics = (
        frame.groupby("timestamp", sort=True)
        .agg(
            instrument_count=("instrument_id", "size"),
            min_signal=("signal", "min"),
            max_signal=("signal", "max"),
        )
        .reset_index()
    )
    diagnostics["min_signal"] = diagnostics["min_signal"].astype(float)
    diagnostics["max_signal"] = diagnostics["max_signal"].astype(float)
    return diagnostics


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
