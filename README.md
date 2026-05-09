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

## Experiment Runs

Experiment runs capture the data snapshot, parameters, artifacts, cache
manifests, and metrics needed to reproduce a research result.

```python
from quant_research.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentRunStore,
)

run_store = ExperimentRunStore(root="research_store")
runner = ExperimentRunner(run_store=run_store)
config = ExperimentConfig(
    name="close-return-smoke",
    data_snapshot="2026-05-09",
    parameters={"symbols": ["600000.SH"]},
)

run = runner.create_run(config)
completed = runner.complete_run(
    run,
    artifacts={"factor": "research_store/factors/close_return.pkl"},
    metrics={"total_return": 0.12},
    cache_manifest_ids=("manifest-id",),
)
```

## Minimal Backtests

Backtests are organized as an orchestration boundary. The framework validates
standard output tables, computes basic metrics, and can persist artifacts. The
actual simulator is injected by the caller.

```python
import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.backtest import BacktestConfig, BacktestEngine, BacktestFrames


def simulator(config: BacktestConfig) -> BacktestFrames:
    return BacktestFrames(
        trades=pd.DataFrame([
            {
                "timestamp": config.start,
                "instrument_id": "inst-600000",
                "quantity": 10,
                "price": 10.0,
            }
        ]),
        positions=pd.DataFrame([
            {
                "timestamp": config.start,
                "instrument_id": "inst-600000",
                "quantity": 10,
                "market_value": 100.0,
            }
        ]),
        equity_curve=pd.DataFrame([
            {"timestamp": config.start, "equity": 100.0},
            {"timestamp": config.end, "equity": 112.0},
        ]),
    )


config = BacktestConfig(
    name="close-return-backtest",
    start="2024-01-02",
    end="2024-01-31",
    data_snapshot="2026-05-09",
)
engine = BacktestEngine(artifact_store=ArtifactStore.from_path("research_store"))
result = engine.run(config, simulator, persist=True)
```

## Portfolio Construction

Portfolio construction turns signal tables into target weights and rebalance
orders. The v0 constructor provides contract-level equal and signal weighting;
full optimization models should be added later behind this boundary.

```python
import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.portfolio import PortfolioConfig, PortfolioConstructor

signals = pd.DataFrame([
    {"timestamp": "2024-01-31", "instrument_id": "inst-600000", "signal": 1.0},
    {"timestamp": "2024-01-31", "instrument_id": "inst-000001", "signal": 2.0},
])

constructor = PortfolioConstructor(
    artifact_store=ArtifactStore.from_path("research_store")
)
result = constructor.build(
    signals,
    PortfolioConfig(name="monthly-signal", weighting="signal", max_weight=0.6),
    persist=True,
)
```

## Signal Generation

Signal generation converts factor value tables into the standard
`timestamp, instrument_id, signal` shape consumed by portfolio construction.
The v0 generator supports `identity`, `rank`, and `threshold` methods.

```python
from quant_research.signals import SignalGenerator, SignalSpec

signal_result = SignalGenerator(
    artifact_store=ArtifactStore.from_path("research_store")
).generate(
    result.frame,
    SignalSpec(
        name="ranked-close-return",
        factor_name="close_return",
        method="rank",
        parameters={"ascending": False},
    ),
    persist=True,
)
```

## Universes

Universes define the security membership that downstream factors, signals,
portfolios, and backtests consume. The v0 builder supports static symbol lists
and optional resolution through `DataPortal`.

```python
from quant_research.universe import UniverseBuilder, UniverseSpec, active_on

universe = UniverseBuilder(
    artifact_store=ArtifactStore.from_path("research_store")
).build(
    UniverseSpec(
        name="cn-core",
        symbols=("600000.SH", "000001.SZ"),
        market="CN",
        asset_type="equity",
        start="2024-01-01",
        end="2024-12-31",
    ),
    data=data,
    persist=True,
)

active_members = active_on(universe, "2024-06-30")
```

## Current Scope

This repository has the first `DataPortal v0` adapter. The next implementation
target is metrics/reporting scaffolding.
