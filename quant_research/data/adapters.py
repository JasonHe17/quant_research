"""Data-layer adapter placeholders."""

from __future__ import annotations


class QuantDbAdapter:
    """Adapter boundary for future quantdb.sdk integration."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded
