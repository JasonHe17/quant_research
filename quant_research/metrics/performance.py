"""Performance metric placeholders."""

from __future__ import annotations


def total_return(start_value: float, end_value: float) -> float:
    if start_value == 0:
        raise ValueError("start_value must be non-zero")
    return end_value / start_value - 1.0
