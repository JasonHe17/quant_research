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

When both `cache_root` and `snapshot` are set, data reads use the local
DataFrame cache by default. Pass `cache=False` to bypass cache for one call.

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

## Minimal Factors

Factors receive a `FactorContext` and return a Pandas DataFrame. The
`FactorEngine` normalizes the result with a `factor_name` column and can persist
factor outputs through `ArtifactStore`.

```python
from quant_research.artifacts import ArtifactStore
from quant_research.factors import Factor, FactorContext, FactorEngine


class CloseReturn(Factor):
    def compute(self, context: FactorContext):
        bars = context.data.get_bars(
            list(context.symbols),
            start=context.start,
            end=context.end,
            frequency=context.frequency,
            adjustment="raw",
            market=context.market,
        )
        frame = bars[["instrument_id", "bar_end_time", "close_price"]].copy()
        frame["factor_value"] = frame["close_price"].pct_change().fillna(0.0)
        return frame[["instrument_id", "bar_end_time", "factor_value"]]


context = FactorContext(
    data=data,
    start="2024-01-02T09:31:00+08:00",
    end="2024-01-02T15:00:00+08:00",
    symbols=("600000.SH",),
    market="CN",
    snapshot="2026-05-09",
)
engine = FactorEngine(artifact_store=ArtifactStore.from_path("research_store"))
result = engine.compute(CloseReturn("close_return", ("close_price",)), context)
```

## Current Scope

This repository has the first `DataPortal v0` adapter. The next implementation
target is experiment run metadata.
