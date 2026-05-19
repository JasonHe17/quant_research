# Framework v1 Acceptance Plan

This document defines the standard acceptance suite for the research framework.
It validates the framework, not alpha profitability. A losing baseline can pass
framework acceptance if data coverage, artifact generation, execution
diagnostics, costs, and scenario behavior are coherent.

## Research Basis

- Backtesting must explicitly account for transaction costs, survivorship bias,
  look-ahead risk, and implementation assumptions. See CFA Institute,
  "Backtesting and Simulation":
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2025/backtesting-and-simulation
- Backtest overfitting risk increases with repeated search and weak validation.
  Bailey, Borwein, Lopez de Prado, and Zhu propose estimating the probability of
  backtest overfitting:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Factor searches need multiple-testing discipline. Harvey, Liu, and Zhu show
  that the traditional single-test threshold is too lenient for large factor
  searches:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314
- Model/process validation should include conceptual soundness, ongoing
  monitoring, and outcomes analysis. See Federal Reserve SR 11-7:
  https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm

## Standard Suite

Run:

```bash
conda run -n quant python examples/run_framework_v1_benchmark.py \
  --output-dir runs/framework_v1_acceptance/standard
```

Default settings:

- Window: `2023-01-03T09:35:00+08:00` to
  `2025-12-31T15:00:00+08:00`.
- Profile: `standard`.
- Universe: full CN main-board universe unless `--max-symbols` is supplied for a
  smoke run.
- Dataset partitioning: monthly, with 30 calendar days of warmup padding.
- Factors: all currently implemented intraday factor groups.
- Labels: one-bar delayed entry. The benchmark default is a 48 five-minute-bar
  forward return for backward compatibility. Multi-horizon research runs should
  pass `--label-horizon-bars 48 240 960`; when multiple horizons are supplied,
  labels are written as `forward_return_48b`, `forward_return_240b`, and
  `forward_return_960b`, and the current single-factor gates use the first
  horizon as the primary label.
- Factor evaluation: six worker processes by default; reduce with
  `--evaluation-workers` on memory-constrained machines.
- Backtest streaming: fast parquet runs use monthly chunks by default, with
  10 calendar days of boundary padding for lookback and next-bar continuity.
  Increase memory headroom before switching `--streaming-chunk year`.
- Backtest scheduling: after dataset and factor evaluation complete, backtest
  scenarios run through a resource-aware scheduler. `--backtest-workers`
  defaults to `6`, while `--backtest-memory-budget-gb 0` auto-detects available
  memory and uses a conservative fraction. Tune `--full-backtest-memory-gb` and
  `--yearly-backtest-memory-gb` when a machine has known memory headroom. A
  backtest whose estimate exceeds the configured budget is rejected before
  launch.
- Costs: 3 bps commission, 1 bp slippage, 5 bps sell stamp tax, 5 CNY minimum
  commission, 100-share board lot, T+1 selling.
- Tradability: ST exclusion, suspension filter, open-limit buy/sell blocks.

The standard profile runs these backtest scenarios:

- `full_base`: full-window baseline with production-like costs.
- `year_<YYYY>_base`: one scenario per calendar year in the acceptance window.
- `full_high_cost`: full-window transaction-cost stress using
  `--cost-stress-multiplier` (default `2.0`).

The `robust` profile additionally runs:

- `full_zero_cost`: diagnostic upper bound without explicit transaction costs.
- `full_trade_filter_stress`: higher minimum-trade-weight stress.

Use `--profile quick` only for development smoke checks.

## Candidate Policy Validation

Baseline A backtests are plumbing regression checks. For strategy-level review,
run candidate policy validation from an admission report:

```bash
conda run -n quant python examples/run_framework_v1_benchmark.py \
  --output-dir runs/framework_v1_acceptance/standard \
  --auto-factor-admission \
  --resume-existing
```

or, when the admission report already exists:

```bash
conda run -n quant python examples/run_candidate_policy_validation.py \
  --dataset-dir runs/framework_v1_acceptance/standard/alpha_dataset \
  --label-column forward_return \
  --admission-report runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json \
  --factor-correlation runs/framework_v1_acceptance/standard/factor_evaluation/feature_correlation.csv \
  --output-dir runs/framework_v1_acceptance/standard/candidate_policy_validation \
  --profile standard \
  --methods decorrelated equal ic_weighted \
  --primary-method decorrelated \
  --policy partial_rebalance_daily \
  --resume-existing
```

This wrapper keeps scenario construction serial by default
(`--scenario-workers 1`) and runs score-backtest subprocesses with
`--backtest-workers 6 --backtest-memory-budget-gb 30.0`. The primary gate is
currently `decorrelated + partial_rebalance_daily`; the leaderboard should still
be reviewed because higher-return policies can be less cost-robust.

For factor-promotion work, use the baseline hierarchy in
`docs/validation/factor_development_standard.md` rather than comparing only to a
naive control. Every strategy-level review should state the naive/control
anchor, the active/default baseline, and the research frontier baseline when
one exists. As of the 2026-05-19 daily moving-average review,
`score_budget_gate_v1` remains the active/default baseline, while the fixed
`high_dispersion_current` ribbon-dispersion gate is the research frontier
challenger. Future daily-MA-family candidates must report marginal contribution
against that frontier as well as against the active baseline.

## Required Outputs

The benchmark writes:

- `commands.json`: exact subprocess commands used by the suite.
- `benchmark_summary.json`: consolidated machine-readable acceptance result.
- `alpha_dataset/`: supervised feature/label partitions and manifests.
- `factor_evaluation/`: single-factor diagnostics.
- `backtests/<scenario>/`: trades, equity curve, final positions, and summary
  for each execution scenario.
- `candidate_policy_validation/`: optional combination-method and policy
  comparison outputs when candidate policy validation is enabled.
- `logs/`: stdout/stderr for every stage.

Interrupted runs can continue with `--resume-existing`; completed stages are
detected by their summary JSON files and skipped.

## Failure Gates

The suite fails when:

- Dataset rows or label rows are empty.
- Factor evaluation produces no feature summary.
- A required backtest scenario is missing.
- Any scenario emits non-finite metrics, non-positive final equity, or invalid
  drawdown outside `[-1, 0]`.
- `full_base` has no signals, no execution rows, or no trades.

## Warning Gates

The suite warns, but does not fail, when:

- The validation window spans fewer than three calendar years.
- `--max-symbols` is used or the full-base universe has fewer than 100
  instruments.
- Cost-stress or zero-cost consistency checks are inconclusive.

Warnings are acceptable for smoke runs. Production acceptance should use the
default standard profile without `--max-symbols`.

## Interpretation

Do not promote a new factor or framework change from a single aggregate return.
Review at least:

- Dataset coverage and entry-tradability filter counts.
- Feature count and top factor IC summaries.
- Full-window and yearly backtest metrics.
- Execution constraint counts: non-tradable rows, limit blocks, positive targets,
  trade counts, transaction costs, and turnover.
- Cost sensitivity between `full_base`, `full_high_cost`, and, in robust mode,
  `full_zero_cost`.
- Candidate policy validation leaderboard, monthly summaries, and yearly slices
  when the task is factor promotion or compatibility revalidation.

The acceptance suite establishes that the framework is operational and stable
enough for factor research. It is not a production trading approval process.
