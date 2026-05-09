"""Backtest event model placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RebalanceEvent:
    """A scheduled rebalance marker."""

    timestamp: str
    reason: str = "scheduled"
