# Quant Research Framework Pipeline v0

## Status

Proposed and covered by smoke integration tests.

## Purpose

This framework is a research orchestration layer. It should define stable
interfaces for research workflows without embedding a specific alpha model,
portfolio optimizer, execution simulator, or live-trading adapter.

The canonical data platform remains the sibling `quant_dataset` repository. This
package consumes that data through `DataPortal` and stable `quantdb.sdk`
interfaces.

## Pipeline Shape

```text
DataPortal
  -> UniverseBuilder
  -> FactorEngine
  -> SignalGenerator
  -> PortfolioConstructor
  -> BacktestEngine
  -> MetricsEngine
  -> ExperimentRunner
  -> ArtifactStore
```

## Standard Tables

Universe members:

```text
symbol, instrument_id, market, asset_type, effective_from, effective_to
```

Factor output:

```text
factor_name, instrument_id, timestamp/bar_end_time, factor_value
```

Signal output:

```text
timestamp, instrument_id, signal_name, factor_name, signal
```

Portfolio targets:

```text
timestamp, instrument_id, target_weight
```

Rebalance orders:

```text
timestamp, instrument_id, current_weight, target_weight, delta_weight
```

Backtest outputs:

```text
trades: timestamp, instrument_id, quantity, price
positions: timestamp, instrument_id, quantity, market_value
equity_curve: timestamp, equity
diagnostics: implementation-specific diagnostics
```

Metrics reports:

```text
name, metrics, metadata, artifacts
```

Experiment runs:

```text
run_id, config, status, artifacts, metrics, cache_manifest_ids
```

## Artifact Boundaries

Research outputs are written under a caller-provided artifact root:

```text
research_store/
  universes/
  factors/
  signals/
  portfolios/
  backtests/
  reports/
  experiments/
```

Artifacts are research outputs. They must not be written into
`quant_dataset/canonical_store`.

## Deferred Scope

The v0 framework intentionally does not include:

- alpha model implementations
- large factor libraries
- portfolio optimizers
- execution or fill models beyond output contracts
- live trading adapters
- schedulers or daemons
- dashboards or notebooks

These should be added behind the established boundaries.
