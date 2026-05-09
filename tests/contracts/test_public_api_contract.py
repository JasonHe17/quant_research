from __future__ import annotations

from pathlib import Path

import pytest

from quant_research import DataPortal, DataPortalConfig
from quant_research.data import CacheManifest, CachePolicy
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
    ):
        assert callable(getattr(portal, method_name))


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
