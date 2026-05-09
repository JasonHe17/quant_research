"""Risk metric placeholders."""

from __future__ import annotations


def max_drawdown(equity_curve: list[float]) -> float:
    peak = None
    worst = 0.0
    for value in equity_curve:
        peak = value if peak is None else max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return worst
