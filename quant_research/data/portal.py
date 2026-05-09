"""Research-facing data portal boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from quant_research.data.adapters import QuantDbAdapter, select_fields
from quant_research.data.cache import (
    CachePolicy,
    DataFrameCache,
    catalog_reference_for_path,
)


@dataclass(frozen=True, slots=True)
class DataPortalConfig:
    """Configuration for research data access."""

    canonical_root: Path
    catalog_path: Path
    cache_root: Path | None = None
    snapshot: str | None = None
    quant_dataset_root: Path | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        canonical_root: str | Path,
        catalog_path: str | Path,
        cache_root: str | Path | None = None,
        snapshot: str | None = None,
        quant_dataset_root: str | Path | None = None,
    ) -> "DataPortalConfig":
        canonical = Path(canonical_root).resolve()
        catalog = Path(catalog_path).resolve()
        inferred_dataset_root = (
            Path(quant_dataset_root).resolve()
            if quant_dataset_root is not None
            else _infer_quant_dataset_root(canonical)
        )
        return cls(
            canonical_root=canonical,
            catalog_path=catalog,
            cache_root=Path(cache_root).resolve() if cache_root is not None else None,
            snapshot=snapshot,
            quant_dataset_root=inferred_dataset_root,
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
        quant_dataset_root: str | Path | None = None,
        adapter: QuantDbAdapter | None = None,
    ) -> None:
        self.config = DataPortalConfig.from_paths(
            canonical_root=canonical_root,
            catalog_path=catalog_path,
            cache_root=cache_root,
            snapshot=snapshot,
            quant_dataset_root=quant_dataset_root,
        )
        if adapter is None:
            if self.config.quant_dataset_root is None:
                raise ValueError(
                    "quant_dataset_root could not be inferred; pass it explicitly"
                )
            adapter = QuantDbAdapter(
                quant_dataset_root=self.config.quant_dataset_root,
            )
        self._adapter = adapter
        self._cache = (
            DataFrameCache(policy=CachePolicy(root=self.config.cache_root))
            if self.config.cache_root is not None and self.config.snapshot is not None
            else None
        )

    def resolve_instruments(
        self,
        symbols: list[str],
        *,
        market: str | None = None,
        asset_type: str | None = None,
    ) -> Any:
        """Resolve research symbols through the data-layer SDK."""

        return self._adapter.resolve_instruments(
            symbols, market=market, asset_type=asset_type
        )

    def get_trading_calendar(
        self,
        market: str,
        start: str,
        end: str,
        *,
        cache: bool = True,
    ) -> Any:
        """Return a research-friendly trading calendar frame."""

        return self._cached_frame(
            dataset="trading_calendar",
            parameters={"market": market, "start": start, "end": end},
            enabled=cache,
            compute=lambda: self._adapter.get_trading_calendar(
                market=market, start=start, end=end
            ),
        )

    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str,
        adjustment: str,
        fields: list[str] | None = None,
        *,
        market: str | None = None,
        asset_type: str | None = None,
        cache: bool = True,
    ) -> Any:
        """Return raw or adjusted bar data for research workflows."""

        frame = self._cached_frame(
            dataset="bars",
            parameters={
                "symbols": symbols,
                "start": start,
                "end": end,
                "frequency": frequency,
                "adjustment": adjustment,
                "market": market,
                "asset_type": asset_type,
            },
            enabled=cache,
            compute=lambda: self._adapter.get_bars(
                symbols,
                start=start,
                end=end,
                frequency=frequency,
                adjustment=adjustment,
                market=market,
                asset_type=asset_type,
            ),
        )
        return select_fields(frame, fields)

    def get_fundamentals_asof(
        self,
        symbols: list[str],
        dataset: str,
        as_of: str,
        fields: list[str] | None = None,
        *,
        market: str | None = None,
        asset_type: str | None = None,
        cache: bool = True,
    ) -> Any:
        """Return point-in-time fundamentals visible at ``as_of``."""

        frame = self._cached_frame(
            dataset="fundamentals_asof",
            parameters={
                "symbols": symbols,
                "dataset": dataset,
                "as_of": as_of,
                "market": market,
                "asset_type": asset_type,
            },
            enabled=cache,
            compute=lambda: self._adapter.get_fundamentals_asof(
                symbols,
                dataset=dataset,
                as_of=as_of,
                market=market,
                asset_type=asset_type,
            ),
        )
        return select_fields(frame, fields)

    def get_market_indicators(
        self,
        names: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        frequency: str = "1m",
        *,
        market: str = "CN",
        indicator_type: str | None = None,
        fields: list[str] | None = None,
        cache: bool = True,
    ) -> Any:
        """Return market indicator data for research workflows."""

        frame = self._cached_frame(
            dataset="market_indicators",
            parameters={
                "names": names,
                "start": start,
                "end": end,
                "frequency": frequency,
                "market": market,
                "indicator_type": indicator_type,
            },
            enabled=cache,
            compute=lambda: self._adapter.get_market_indicators(
                names=names,
                start=start,
                end=end,
                frequency=frequency,
                market=market,
                indicator_type=indicator_type,
            ),
        )
        return select_fields(frame, fields)

    def list_available_datasets(
        self,
        *,
        domain: str | None = None,
        fields: list[str] | None = None,
    ) -> Any:
        """Return the stable dataset inventory exposed by quantdb."""

        frame = self._adapter.list_available_datasets(domain=domain)
        return select_fields(frame, fields)

    def _cached_frame(
        self,
        *,
        dataset: str,
        parameters: dict[str, object],
        enabled: bool,
        compute: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        if not enabled or self._cache is None or self.config.snapshot is None:
            return compute()
        return self._cache.get_or_compute(
            dataset=dataset,
            parameters=parameters,
            snapshot=self.config.snapshot,
            catalog_reference=catalog_reference_for_path(self.config.catalog_path),
            compute=compute,
        )


def _infer_quant_dataset_root(canonical_root: Path) -> Path | None:
    if canonical_root.name == "canonical_store":
        return canonical_root.parent
    return None
