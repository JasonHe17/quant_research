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

Use `--partition-start` and `--partition-end` to isolate a specific month or
quarter without creating a temporary dataset directory.

## Outputs

The script writes:

- `scores/<method>/score_<partition>.parquet`
- `backtests/<method>/summary.json` when `--run-backtests` is set
- `summary.json` with candidate features, method weights, score row counts, and
  optional backtest summaries

The score backtest supports fast parquet monthly streaming. Keep
`--hold-rank-buffer` disabled for streaming runs until stateful buffered holding
is added.

Use the production-oriented policy path for turnover-control experiments:

```bash
conda run -n quant python examples/run_tree_score_backtest.py \
  --predictions-path runs/candidate_factor_portfolios/q1_2023/scores/decorrelated/*.parquet \
  --start 2023-01-03T09:35:00+08:00 \
  --end 2023-03-31T15:00:00+08:00 \
  --top-n 50 \
  --trade-policy rank_buffer_drop \
  --rebalance-every-n-bars 48 \
  --policy-entry-rank 50 \
  --policy-exit-rank 150 \
  --policy-max-entries-per-rebalance 10 \
  --policy-max-exits-per-rebalance 10 \
  --policy-no-trade-weight-band 0.002 \
  --policy-partial-rebalance-rate 1.0 \
  --data-access-mode fast_parquet \
  --streaming-chunk month \
  --output-dir runs/candidate_factor_portfolios/policy_q1_2023/decorrelated
```

The old `naive_top_n` path remains available only as a baseline comparison.
When a policy uses `policy_exit_rank` or `hold_rank_buffer` above `top_n`, the
score loader must read through the larger rank so the policy can distinguish
buffered holds from true exits.

Initial Q1 2023 policy comparison:

| Method | Policy | Return | Max drawdown | Gross turnover | Trades | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| decorrelated | naive top-50 every bar | 2.49% | -7.16% | 118.49 | 6,348 | 81,779 |
| decorrelated | daily rank buffer, entry 50 / exit 150, drop 10 | 7.92% | -6.00% | 47.42 | 1,148 | 33,462 |
| equal | naive top-50 every bar | 1.09% | -8.44% | 118.94 | 6,380 | 81,839 |
| equal | daily rank buffer, entry 50 / exit 150, drop 10 | 9.04% | -5.10% | 53.68 | 1,101 | 37,853 |
| ic_weighted | naive top-50 every bar | 0.32% | -8.16% | 118.77 | 6,367 | 80,795 |
| ic_weighted | daily rank buffer, entry 50 / exit 150, drop 10 | 7.38% | -4.81% | 53.45 | 1,089 | 37,442 |

Initial 2023 full-year policy comparison:

| Method | Policy | Return | Max drawdown | Gross turnover | Trades | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| decorrelated | naive top-50 every bar | -15.12% | -26.55% | 495.76 | 26,754 | 310,553 |
| decorrelated | daily rank buffer, entry 50 / exit 150, drop 10 | 8.41% | -12.15% | 158.63 | 4,801 | 112,619 |
| equal | naive top-50 every bar | -25.09% | -34.10% | 495.89 | 26,768 | 299,109 |
| equal | daily rank buffer, entry 50 / exit 150, drop 10 | -2.29% | -17.41% | 198.35 | 4,619 | 133,317 |
| ic_weighted | naive top-50 every bar | -20.75% | -29.20% | 496.39 | 26,908 | 304,434 |
| ic_weighted | daily rank buffer, entry 50 / exit 150, drop 10 | -2.17% | -17.75% | 202.63 | 4,534 | 134,450 |

The initial result supports making `rank_buffer_drop` with a lower rebalance
frequency the default candidate for the next broader policy experiment. It is
not yet a promotion result; it still needs method-by-method full-year coverage,
multi-year validation, and cost stress checks. The 2023 full-year result also
shows that the trading policy is not enough by itself: the decorrelated
combination turns positive, while equal and IC-weighted remain slightly negative
after costs.
