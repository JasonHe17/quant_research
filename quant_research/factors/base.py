"""Base factor interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class FactorContext(Protocol):
    """Minimal context expected by factor calculations."""

    start: str
    end: str


@dataclass(frozen=True, slots=True)
class Factor:
    """Declarative factor definition."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, context: FactorContext) -> object:
        raise NotImplementedError("factor computation is implemented by subclasses")
