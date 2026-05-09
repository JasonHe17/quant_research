# Quant Research

Research framework for the `quant_trade` workspace.

This package owns research-facing workflows:

- data access ergonomics through `DataPortal`
- local hot-data cache manifests
- universe definitions
- factor computation orchestration
- signal generation
- experiments and reproducibility
- backtest orchestration
- portfolio construction
- metrics and research artifacts

The canonical data platform lives in the sibling `quant_dataset` repository.
Research code should depend on stable `quantdb` public interfaces, especially
`quantdb.sdk`, and must not import bootstrap, raw sync, test, or private storage
internals from `quant_dataset`.

## Intended Workspace Layout

```text
quant_trade/
  quant_dataset/
  quant_research/
```

## Development

```bash
python -m pytest
```

## DataPortal v0

```python
from quant_research import DataPortal

data = DataPortal(
    canonical_root="../quant_dataset/canonical_store",
    catalog_path="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    cache_root="/ssd/quant_cache",
    snapshot="2026-05-09",
)

datasets = data.list_available_datasets()
calendar = data.get_trading_calendar("CN", "2024-01-01", "2024-01-31")
bars = data.get_bars(
    ["600000.SH"],
    start="2024-01-02T09:31:00+08:00",
    end="2024-01-02T15:00:00+08:00",
    frequency="1m",
    adjustment="raw",
    market="CN",
)
```

`DataPortal` returns Pandas DataFrames and delegates data reads to the sibling
`quant_dataset` repository through stable `quantdb.sdk` interfaces.

## Cache Manifests

Local cache files are rebuildable acceleration artifacts. Cache manifests record
the request parameters and data snapshot needed to decide whether an artifact can
be reused.

```python
from quant_research.data import CacheManifest, CacheManifestStore

store = CacheManifestStore(root="/ssd/quant_cache")
manifest = CacheManifest.create(
    dataset="minute_bars",
    parameters={"symbols": ["600000.SH"], "frequency": "1m"},
    snapshot="2026-05-09",
    catalog_reference="catalog-sha256:...",
    artifact_path="/ssd/quant_cache/snapshots/2026-05-09/market/sample.parquet",
    row_count=1000,
)

path = store.write(manifest)
```

## Current Scope

This repository has the first `DataPortal v0` adapter. The next implementation
targets are cache-backed `DataPortal` reads, minimal factor interfaces, and
experiment run metadata.
