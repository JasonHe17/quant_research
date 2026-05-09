"""Factor engine scaffold."""

from __future__ import annotations

from quant_research.factors.base import Factor, FactorContext


class FactorEngine:
    """Orchestrates factor calculations."""

    def compute(self, factor: Factor, context: FactorContext) -> object:
        return factor.compute(context)
