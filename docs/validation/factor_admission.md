# Factor Admission Plan

This document defines the first standard gate between single-factor diagnostics
and downstream portfolio/backtest work. It is deliberately stricter than a
smoke test and deliberately weaker than production deployment approval.

## Inputs

Run the Framework v1 standard acceptance suite first:

```bash
conda run -n quant python examples/run_framework_v1_benchmark.py \
  --output-dir runs/framework_v1_acceptance/standard \
  --resume-existing \
  --enforce-gates
```

Then generate the factor admission report:

```bash
conda run -n quant python examples/analyze_framework_v1_acceptance.py \
  --benchmark-summary runs/framework_v1_acceptance/standard/benchmark_summary.json \
  --output-dir runs/framework_v1_acceptance/standard/factor_admission
```

The analysis reads:

- `benchmark_summary.json` for acceptance status and backtest context.
- `factor_evaluation/summary.json` for feature coverage and row counts.
- `factor_evaluation/single_factor_by_timestamp.csv` for timestamp-level IC,
  top-minus-bottom spread, and turnover stability.

## Default Gates

A `candidate` factor must pass all hard and soft checks:

- Coverage at least `0.95`.
- At least `1000` evaluated timestamps.
- Absolute mean Spearman rank IC at least `0.001`.
- Absolute IC t-stat at least `2.0`.
- Directional IC hit rate at least `0.52`.
- At least three observed calendar years.
- At least two stable years with yearly IC matching the selected direction.
- Positive cost-adjusted top-minus-bottom spread.
- Top-N turnover no higher than `0.95`.

The default cost proxy is `13 bps`, representing a conservative round-trip
estimate for 3 bps commission, 1 bp slippage on each side, and 5 bps sell stamp
tax.

## Status Semantics

- `candidate`: eligible for the next research step, usually portfolio-level
  composition or model inclusion tests.
- `watchlist`: statistically usable, but fails at least one soft economic or
  stability gate. Do not promote without targeted analysis.
- `reject`: fails at least one hard statistical or coverage gate. Keep only for
  diagnostics or redesign.

Negative-IC factors are not automatically rejected. The report marks them as
`invert` when the inverse direction passes the same directional checks.

## Outputs

The report directory contains:

- `factor_admission_report.json`: complete machine-readable report.
- `factor_admission_table.csv`: flat factor table for notebooks and review.
- `factor_admission_report.md`: human-readable summary.

Use `--enforce-candidates` in automation when the pipeline should fail if no
factor reaches `candidate` status.
