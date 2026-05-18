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

## Frequency And Horizon Separation

The framework uses 5-minute bars as the default observation and execution grid
because that is the smallest practical step under current compute and data
constraints. That grid is not the research horizon.

The system should keep these concepts separate:

- observation frequency: how often market state is refreshed;
- feature frequency: the native sampling interval of a feature, such as 5m or
  1d;
- execution frequency: how often the strategy may submit target changes;
- forecast horizon: the return interval used to score a signal;
- holding policy: the rule that decides whether to keep or replace a position
  after costs and risk penalties.

The baseline dataset builder now supports multiple forward-return horizons in
one materialization run. For example:

```bash
conda run -n quant python examples/build_baseline_a_alpha_dataset.py \
  --catalog-path ../quant_dataset/canonical_store/catalog/quant_research.duckdb \
  --start 2023-01-03T09:35:00+08:00 \
  --end 2025-12-31T15:00:00+08:00 \
  --output-dir runs/framework_v1_acceptance/multi_horizon/alpha_dataset \
  --factor-groups reversal volatility \
  --horizon-bars 48 240 960
```

If only one horizon is supplied, the label keeps the base name. If multiple
horizons are supplied, labels are named `<label-name>_<horizon>b`, for example
`forward_return_48b`, `forward_return_240b`, and `forward_return_960b`, with
matching rank columns. This lets evaluation compare short, medium, and long
holding periods before deciding whether a signal belongs in a fast or slow
policy.

## Framework Benchmark And Policy Validation

`examples/run_framework_v1_benchmark.py` still keeps the original Baseline A
backtests as regression checks for data, execution, and cost plumbing. Those
checks are not the final strategy selection mechanism.

For strategy-level validation, the benchmark can optionally run candidate policy
validation when an admission report is available:

```bash
conda run -n quant python examples/run_framework_v1_benchmark.py \
  --output-dir runs/framework_v1_acceptance/standard \
  --candidate-admission-report \
    runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json
```

For an end-to-end benchmark, use `--auto-factor-admission`. In that mode the
benchmark first completes dataset, factor evaluation, and regression backtests;
writes an intermediate benchmark summary; runs
`examples/analyze_framework_v1_acceptance.py` to create
`factor_admission/factor_admission_report.json`; and then runs
`examples/run_candidate_policy_validation.py` with that generated admission
report.

```bash
conda run -n quant python examples/run_framework_v1_benchmark.py \
  --output-dir runs/framework_v1_acceptance/standard \
  --auto-factor-admission
```

The policy validation stage writes
`candidate_policy_validation/validation_summary.json` and surfaces the
`policy_leaderboard` in the benchmark summary. The leaderboard compares policy
families across produced scenarios, so the framework can choose between slower
holding policies and faster cost-aware optimizers from measured return, cost,
and risk tradeoffs instead of from a fixed manual holding period.

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
