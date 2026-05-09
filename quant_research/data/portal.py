"""Research-facing data portal boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DataPortalConfig:
    """Configuration for research data access."""

    canonical_root: Path
    catalog_path: Path
    cache_root: Path | None = None
    snapshot: str | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        canonical_root: str | Path,
        catalog_path: str | Path,
        cache_root: str | Path | None = None,
        snapshot: str | None = None,
    ) -> "DataPortalConfig":
        return cls(
            canonical_root=Path(canonical_root),
            catalog_path=Path(catalog_path),
            cache_root=Path(cache_root) if cache_root is not None else None,
            snapshot=snapshot,
        )


class DataPortal:
    """Thin research facade over stable quantdb data interfaces."""

    def __init__(
        self,
        *,
        canonical_root: str | Path,
        catalog_path: str | Path,
        cache_root: str | Path | None = None,
        snapshot: str | None = None,
    ) -> None:
        self.config = DataPortalConfig.from_paths(
            canonical_root=canonical_root,
            catalog_path=catalog_path,
            cache_root=cache_root,
            snapshot=snapshot,
        )

    def resolve_instruments(
        self,
        symbols: list[str],
        *,
        market: str | None = None,
        asset_type: str | None = None,
    ) -> Any:
        """Resolve research symbols through the data-layer SDK."""

        raise NotImplementedError("DataPortal v0 SDK adapter is not implemented yet")

    def get_trading_calendar(
        self,
        market: str,
        start: str,
        end: str,
    ) -> Any:
        """Return a research-friendly trading calendar frame."""

        raise NotImplementedError("DataPortal v0 SDK adapter is not implemented yet")

    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str,
        adjustment: str,
        fields: list[str] | None = None,
    ) -> Any:
        """Return raw or adjusted bar data for research workflows."""

        raise NotImplementedError("DataPortal v0 SDK adapter is not implemented yet")

    def get_fundamentals_asof(
        self,
        symbols: list[str],
        dataset: str,
        as_of: str,
        fields: list[str] | None = None,
    ) -> Any:
        """Return point-in-time fundamentals visible at ``as_of``."""

        raise NotImplementedError("DataPortal v0 SDK adapter is not implemented yet")

    def get_market_indicators(
        self,
        names: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        frequency: str = "1m",
    ) -> Any:
        """Return market indicator data for research workflows."""

        raise NotImplementedError("DataPortal v0 SDK adapter is not implemented yet")
