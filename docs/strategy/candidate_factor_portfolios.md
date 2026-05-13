# Candidate Factor Portfolio Experiments

This document defines the first portfolio-layer experiment for factors that pass
the standard admission gates.

## Inputs

Run these prerequisites first:

```bash
conda run -n quant python examples/run_framework_v1_benchmark.py \
  --output-dir runs/framework_v1_acceptance/standard \
  --resume-existing \
  --enforce-gates

conda run -n quant python examples/analyze_framework_v1_acceptance.py \
  --benchmark-summary runs/framework_v1_acceptance/standard/benchmark_summary.json \
  --output-dir runs/framework_v1_acceptance/standard/factor_admission \
  --enforce-candidates
```

The portfolio experiment then consumes:

- `alpha_dataset/dataset_*.parquet`
- `factor_admission/factor_admission_report.json`
- `factor_evaluation/feature_correlation.csv`

## Score Construction

Candidate features are transformed partition by partition:

1. Rank each candidate factor cross-sectionally at each timestamp.
2. Center percentile ranks around zero.
3. Apply the admission direction: `long` keeps the rank sign, `invert` flips it.
4. Combine oriented ranks into a single `score`.

The standard methods are:

- `equal`: equal weight across admitted candidates.
- `ic_weighted`: weights proportional to absolute admitted IC.
- `decorrelated`: non-negative ridge-adjusted inverse-correlation weights using
  the factor correlation matrix.

## Smoke Run

Use one partition before running larger windows:

```bash
conda run -n quant python examples/run_candidate_factor_portfolios.py \
  --dataset-dir runs/framework_v1_acceptance/standard/alpha_dataset \
  --admission-report runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json \
  --factor-correlation runs/framework_v1_acceptance/standard/factor_evaluation/feature_correlation.csv \
  --output-dir runs/candidate_factor_portfolios/smoke_2023_01 \
  --max-partitions 1 \
  --run-backtests \
  --start 2023-01-03T09:35:00+08:00 \
  --end 2023-01-31T15:00:00+08:00 \
  --top-n 50 \
  --data-access-mode fast_parquet \
  --streaming-chunk month
```

## Outputs

The script writes:

- `scores/<method>/score_<partition>.parquet`
- `backtests/<method>/summary.json` when `--run-backtests` is set
- `summary.json` with candidate features, method weights, score row counts, and
  optional backtest summaries

The score backtest supports fast parquet monthly streaming. Keep
`--hold-rank-buffer` disabled for streaming runs until stateful buffered holding
is added.
