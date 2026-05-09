from __future__ import annotations

from quant_research.metrics.performance import total_return
from quant_research.metrics.risk import max_drawdown


def test_total_return() -> None:
    assert total_return(100.0, 125.0) == 0.25


def test_max_drawdown() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 130.0]) == -0.25
