from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research import DataPortal, DataPortalConfig
from quant_research.data import CacheManifest, CachePolicy
from quant_research.data.adapters import QuantDbAdapter
from quant_research.experiments import ExperimentConfig


def test_data_portal_exposes_v0_methods() -> None:
    portal = DataPortal(
        canonical_root="../quant_dataset/canonical_store",
        catalog_path="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
        cache_root="/ssd/quant_cache",
        snapshot="2026-05-09",
    )

    assert isinstance(portal.config, DataPortalConfig)
    for method_name in (
        "resolve_instruments",
        "get_trading_calendar",
        "get_bars",
        "get_fundamentals_asof",
        "get_market_indicators",
        "list_available_datasets",
    ):
        assert callable(getattr(portal, method_name))


def test_data_portal_returns_dataframes_through_quantdb_adapter(tmp_path: Path) -> None:
    sdk = _FakeQuantDbSdk()
    adapter = QuantDbAdapter(quant_dataset_root=tmp_path, sdk_module=sdk)
    portal = DataPortal(
        canonical_root=tmp_path / "canonical_store",
        catalog_path=tmp_path / "canonical_store/catalog/quant_research.duckdb",
        cache_root=tmp_path / "cache",
        snapshot="2026-05-09",
        adapter=adapter,
    )

    instruments = portal.resolve_instruments(["600000.SH"], market="CN")
    bars = portal.get_bars(
        ["600000.SH"],
        start="2024-01-01T09:31:00+08:00",
        end="2024-01-01T09:31:00+08:00",
        frequency="1m",
        adjustment="raw",
        market="CN",
        fields=["instrument_id", "close_price"],
    )

    assert isinstance(instruments, pd.DataFrame)
    assert instruments.loc[0, "instrument_id"] == "inst-600000"
    assert list(bars.columns) == ["instrument_id", "close_price"]
    assert bars.loc[0, "close_price"] == 10.5

    datasets = portal.list_available_datasets()
    assert datasets.loc[0, "dataset_name"] == "minute_bars"

    cached_bars = portal.get_bars(
        ["600000.SH"],
        start="2024-01-01T09:31:00+08:00",
        end="2024-01-01T09:31:00+08:00",
        frequency="1m",
        adjustment="raw",
        market="CN",
        fields=["instrument_id", "close_price"],
    )
    assert sdk.minute_bar_calls == 1
    assert cached_bars.equals(bars)


def test_data_portal_requires_market_for_symbol_resolution(tmp_path: Path) -> None:
    adapter = QuantDbAdapter(quant_dataset_root=tmp_path, sdk_module=_FakeQuantDbSdk())
    portal = DataPortal(
        canonical_root=tmp_path / "canonical_store",
        catalog_path=tmp_path / "canonical_store/catalog/quant_research.duckdb",
        adapter=adapter,
    )

    with pytest.raises(ValueError, match="market"):
        portal.resolve_instruments(["600000.SH"])


def test_cache_policy_uses_snapshot_scoped_cache_roots() -> None:
    policy = CachePolicy(root=Path("/ssd/quant_cache"))
    assert policy.snapshot_root("2026-05-09") == Path(
        "/ssd/quant_cache/snapshots/2026-05-09"
    )


def test_cache_manifest_records_rebuild_inputs() -> None:
    manifest = CacheManifest.create(
        dataset="minute_bars",
        parameters={"symbols": ["600000.SH"]},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
        artifact_path="/ssd/quant_cache/snapshots/2026-05-09/market/sample.parquet",
        row_count=10,
    )
    payload = manifest.to_dict()

    assert payload["dataset"] == "minute_bars"
    assert payload["snapshot"] == "2026-05-09"
    assert payload["catalog_reference"] == "catalog-sha256:abc"
    assert payload["row_count"] == 10


def test_experiment_config_requires_data_snapshot() -> None:
    config = ExperimentConfig(name="smoke", data_snapshot="2026-05-09")
    assert config.data_snapshot == "2026-05-09"

    with pytest.raises(ValueError, match="data_snapshot"):
        ExperimentConfig(name="bad", data_snapshot="")


@dataclass(frozen=True, slots=True)
class _FakeInstrument:
    instrument_id: str
    canonical_code: str
    market: str
    asset_type: str
    display_name: str
    valid_from: object | None = None
    valid_to: object | None = None
    timezone: str = "Asia/Shanghai"


class _FakeQuantDbSdk:
    def __init__(self) -> None:
        self.minute_bar_calls = 0

    def resolve_instrument(
        self,
        alias_code: str,
        *,
        market: str,
        asset_type: str | None = None,
    ) -> _FakeInstrument:
        return _FakeInstrument(
            instrument_id="inst-600000",
            canonical_code=alias_code,
            market=market,
            asset_type=asset_type or "equity",
            display_name="浦发银行",
        )

    def get_minute_bars(self, instrument_id: str, **_: object) -> tuple[object, ...]:
        self.minute_bar_calls += 1
        return (
            SimpleNamespace(
                instrument_id=instrument_id,
                canonical_code="600000.SH",
                close_price=10.5,
            ),
        )

    def get_adjusted_bars(self, instrument_id: str, **_: object) -> tuple[object, ...]:
        return self.get_minute_bars(instrument_id)

    def list_available_datasets(
        self, *, domain: str | None = None
    ) -> tuple[object, ...]:
        if domain not in (None, "market"):
            return ()
        return (
            SimpleNamespace(
                dataset_name="minute_bars",
                domain="market",
                point_in_time=False,
            ),
        )
