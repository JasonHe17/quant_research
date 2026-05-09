"""Execution assumption placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    """Execution assumption set."""

    name: str
