"""Execution assumption placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    """Execution assumption set."""

    name: str
    price_field: str = "close_price"
    slippage_bps: float = 0.0
    commission_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.commission_bps < 0:
            raise ValueError("commission_bps must be non-negative")
