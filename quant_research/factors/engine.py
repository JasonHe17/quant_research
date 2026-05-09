"""Factor engine scaffold."""

from __future__ import annotations

import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.factors.base import Factor, FactorContext, FactorResult


class FactorEngine:
    """Orchestrates factor calculations."""

    def __init__(self, *, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def compute(
        self,
        factor: Factor,
        context: FactorContext,
        *,
        persist: bool = False,
    ) -> FactorResult:
        frame = factor.compute(context)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("factor.compute() must return a pandas DataFrame")
        normalized = _normalize_factor_frame(frame, factor_name=factor.name)
        result = FactorResult(
            factor_name=factor.name,
            frame=normalized,
            metadata={
                "inputs": factor.inputs,
                "start": context.start,
                "end": context.end,
                "symbols": context.symbols,
                "market": context.market,
                "asset_type": context.asset_type,
                "frequency": context.frequency,
                "snapshot": context.snapshot,
            },
        )
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            self.artifact_store.write_factor(result)
        return result


def _normalize_factor_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.DataFrame:
    normalized = frame.copy()
    if "factor_name" not in normalized.columns:
        normalized.insert(0, "factor_name", factor_name)
    elif (normalized["factor_name"] != factor_name).any():
        raise ValueError("factor_name column contains values for a different factor")
    return normalized
