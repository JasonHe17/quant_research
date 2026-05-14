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

For systematic policy comparison, prefer the portfolio orchestration entrypoint:

```bash
conda run -n quant python examples/run_candidate_factor_portfolios.py \
  --output-dir runs/candidate_factor_portfolios/policy_set_q1_2023_decorrelated \
  --methods decorrelated \
  --partition-start 2023_01 \
  --partition-end 2023_03 \
  --run-backtests \
  --start 2023-01-03T09:35:00+08:00 \
  --end 2023-03-31T15:00:00+08:00 \
  --top-n 50 \
  --backtest-policy-set comparison \
  --policy-no-trade-weight-band 0.002 \
  --policy-set-drop-count 10 \
  --policy-set-exit-rank 150 \
  --policy-set-rebalance-every-n-bars 48 \
  --policy-set-partial-rebalance-rate 0.5 \
  --backtest-workers 2 \
  --backtest-memory-estimate-gb 5 \
  --resume-existing
```

The `comparison` set writes nested results under
`backtests/<method>/<policy_name>/`, writes subprocess logs under `logs/`, and
writes a flat `backtest_summary.csv` for ranking and review. Use
`--backtest-workers` together with `--backtest-memory-budget-gb` and
`--backtest-memory-estimate-gb` to run independent score backtests concurrently
without exceeding local memory. `--resume-existing` skips policy runs whose
`summary.json` already exists and is the default operational mode for long
promotion-grade sweeps.

The standard comparison set currently covers:

- `naive_top_n_every_bar`: research baseline.
- `top_k_drop_daily`: daily rank-buffer policy with `exit_rank=top_n`.
- `entry_exit_buffer_every_bar`: entry/exit rank buffer without slower rebalance.
- `entry_exit_buffer_daily`: entry/exit rank buffer with daily rebalance.
- `partial_rebalance_daily`: daily entry/exit buffer with partial movement toward
  target weights.

The old `naive_top_n` path remains available only as a baseline comparison.
When a policy uses `policy_exit_rank` or `hold_rank_buffer` above `top_n`, the
score loader must read through the larger rank so the policy can distinguish
buffered holds from true exits.

Streaming score backtests build sparse execution frames by default. Each chunk
keeps only instruments that are either current holdings at the start of the
chunk or appear in that chunk's shifted target weights. The simulator still
sees every bar needed to maintain and exit live positions, but inactive
non-target market rows are not materialized into execution rows. Summary
`execution_constraint_counts` therefore describe the simulated relevant
execution universe, while `bar_count` and `instrument_count` still describe the
full signal-window market coverage.

Sparse execution was regression-checked on the decorrelated daily
`entry_rank=50 / exit_rank=150 / drop=10` policy with identical portfolio
metrics to the dense frame:

| Window | Dense execution rows | Sparse execution rows | Return | Max drawdown | Gross turnover | Trades | Cost | Sparse runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 2023 | 13,133,280 | 904,224 | 7.92% | -6.00% | 47.42 | 1,148 | 33,462 | 0:30 |
| 2023 full year | 57,729,216 | 7,591,248 | 8.41% | -12.15% | 158.63 | 4,801 | 112,619 | 2:25 |

Q1 2023 decorrelated policy-set smoke result:

| Policy | Rebalance bars | Exit rank | Partial rate | Return | Max drawdown | Gross turnover | Trades | Cost | Execution rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive_top_n_every_bar | 1 | - | 1.0 | 2.49% | -7.16% | 118.49 | 6,348 | 81,779 | 5,788,896 |
| top_k_drop_daily | 48 | 50 | 1.0 | 8.17% | -5.65% | 59.68 | 1,069 | 41,754 | 861,072 |
| entry_exit_buffer_every_bar | 1 | 150 | 1.0 | 4.19% | -7.10% | 109.87 | 5,735 | 76,532 | 4,124,448 |
| entry_exit_buffer_daily | 48 | 150 | 1.0 | 7.92% | -6.00% | 47.42 | 1,148 | 33,462 | 904,224 |
| partial_rebalance_daily | 48 | 150 | 0.5 | 4.76% | -5.82% | 11.89 | 2,274 | 15,809 | 256,320 |

This smoke run confirms that the policy-set runner can compare the main policy
families from one command. It does not promote a policy by itself. The next
promotion-grade run must cover all combination methods and at least one full
year, then repeat the leading policies under zero/base/stressed transaction
costs.

Promotion-grade 2023 base-cost sweep:

```bash
conda run -n quant python examples/run_candidate_factor_portfolios.py \
  --output-dir runs/candidate_factor_portfolios/policy_set_year_2023_base \
  --methods equal ic_weighted decorrelated \
  --partition-start 2023_01 \
  --partition-end 2023_12 \
  --run-backtests \
  --start 2023-01-03T09:35:00+08:00 \
  --end 2023-12-31T15:00:00+08:00 \
  --top-n 50 \
  --commission-bps 3 \
  --slippage-bps 1 \
  --sell-stamp-tax-bps 5 \
  --min-commission 5 \
  --backtest-policy-set comparison \
  --policy-no-trade-weight-band 0.002 \
  --policy-set-drop-count 10 \
  --policy-set-exit-rank 150 \
  --policy-set-rebalance-every-n-bars 48 \
  --policy-set-partial-rebalance-rate 0.5 \
  --backtest-workers 2 \
  --backtest-memory-estimate-gb 5 \
  --resume-existing
```

For follow-up stress runs, use `--backtest-policies` to restrict the fixed
comparison set to named candidates, for example
`--backtest-policies top_k_drop_daily partial_rebalance_daily`. The filter is
recorded in `summary.json` so partial sweeps remain auditable.

2023 base-cost policy-set result:

| Method | Policy | Return | Max drawdown | Gross turnover | Trades | Cost | Execution rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decorrelated | top_k_drop_daily | 9.42% | -13.39% | 218.71 | 4,584 | 151,785 | 7,318,512 |
| decorrelated | entry_exit_buffer_daily | 8.41% | -12.15% | 158.63 | 4,801 | 112,619 | 7,591,248 |
| decorrelated | partial_rebalance_daily | 8.39% | -13.88% | 43.85 | 9,568 | 63,934 | 1,117,344 |
| ic_weighted | partial_rebalance_daily | 7.04% | -12.30% | 43.59 | 9,445 | 63,311 | 1,117,344 |
| equal | partial_rebalance_daily | 6.13% | -13.11% | 43.91 | 9,573 | 63,949 | 1,117,344 |
| ic_weighted | entry_exit_buffer_daily | -2.17% | -17.75% | 202.63 | 4,534 | 134,450 | 8,598,336 |
| equal | entry_exit_buffer_daily | -2.29% | -17.41% | 198.35 | 4,619 | 133,317 | 8,674,080 |
| equal | top_k_drop_daily | -3.35% | -20.77% | 251.51 | 4,432 | 166,584 | 8,442,096 |
| ic_weighted | top_k_drop_daily | -4.94% | -17.32% | 245.11 | 4,398 | 161,933 | 8,331,264 |
| decorrelated | entry_exit_buffer_every_bar | -12.66% | -24.99% | 465.36 | 25,230 | 297,546 | 23,812,560 |
| decorrelated | naive_top_n_every_bar | -15.12% | -26.55% | 495.76 | 26,754 | 310,553 | 30,256,656 |
| equal | entry_exit_buffer_every_bar | -19.50% | -29.23% | 442.05 | 19,532 | 273,821 | 26,926,512 |
| ic_weighted | naive_top_n_every_bar | -20.75% | -29.20% | 496.39 | 26,908 | 304,434 | 38,960,784 |
| equal | naive_top_n_every_bar | -25.09% | -34.10% | 495.89 | 26,768 | 299,109 | 40,299,216 |
| ic_weighted | entry_exit_buffer_every_bar | -25.26% | -32.98% | 493.01 | 26,728 | 296,200 | 32,329,584 |

The 2023 sweep narrows the next cost-stress candidates. Decorrelated remains
the strongest score combination, but the policy choice changes the cost and
capacity profile materially. `top_k_drop_daily` has the highest 2023 base-cost
return but also the highest turnover among the daily policies. The
`entry_exit_buffer_daily` variant gives similar decorrelated return with lower
turnover and shallower drawdown. `partial_rebalance_daily` has lower return for
decorrelated, but it is the only policy family that stays positive across all
three score-combination methods while cutting gross turnover to roughly 44.
Every-bar policies remain research-only because costs and turnover dominate.

2023 candidate cost-stress result:

Cost profiles:

| Profile | Commission bps | Slippage bps | Sell tax bps | Min commission |
| --- | ---: | ---: | ---: | ---: |
| zero | 0 | 0 | 0 | 0 |
| base | 3 | 1 | 5 | 5 |
| stressed | 6 | 2 | 10 | 5 |

| Method | Policy | Zero return | Base return | Stressed return | Base turnover | Stressed cost | Execution rows | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decorrelated | top_k_drop_daily | 26.38% | 9.42% | -5.13% | 218.71 | 280,914 | 7,318,512 | Reject for promotion; cost fragile |
| decorrelated | entry_exit_buffer_daily | 20.56% | 8.41% | -2.16% | 158.63 | 211,509 | 7,591,248 | Reject for promotion; cost fragile |
| decorrelated | partial_rebalance_daily | 15.02% | 8.39% | 6.60% | 43.85 | 81,318 | 1,117,344 | Promote to broader validation |
| equal | partial_rebalance_daily | 12.86% | 6.13% | 4.25% | 43.91 | 81,003 | 1,117,344 | Promote as robustness check |
| ic_weighted | partial_rebalance_daily | 13.65% | 7.04% | 5.23% | 43.59 | 80,592 | 1,117,344 | Promote as robustness check |

The stress result changes the policy priority. The two higher-return
decorrelated daily full-rebalance variants depend too much on favorable cost
assumptions: zero-to-base drag is 12.15-16.95 percentage points, and both turn
negative under the stressed profile. `partial_rebalance_daily` gives up some
zero-cost return but keeps turnover near 44, uses about one seventh of the
heavy daily execution rows, and remains positive under stressed costs across
all three score-combination methods. Use `decorrelated/partial_rebalance_daily`
as the default candidate for the next broader validation phase, with equal and
IC-weighted partial variants retained as method-robustness controls.

Multi-year validation runner:

```bash
conda run -n quant python examples/run_candidate_policy_validation.py \
  --output-dir runs/candidate_factor_portfolios/partial_rebalance_validation_standard \
  --profile standard \
  --methods decorrelated equal ic_weighted \
  --primary-method decorrelated \
  --policy partial_rebalance_daily \
  --resume-existing
```

The standard profile infers full years from
`runs/framework_v1_acceptance/standard/alpha_dataset`, runs `full_base`,
calendar-year base slices, and `full_high_cost`, then writes
`validation_summary.csv` and `validation_summary.json`. Use `--profile robust`
to add a full-window zero-cost diagnostic.

2023-2025 standard validation result:

| Scenario | Method | Return | Max drawdown | Gross turnover | Trades | Cost | Execution rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_base | decorrelated | 34.97% | -32.79% | 128.79 | 29,052 | 193,134 | 3,414,336 |
| full_base | equal | 21.88% | -34.47% | 128.59 | 29,007 | 190,769 | 3,414,432 |
| full_base | ic_weighted | 25.40% | -32.44% | 127.66 | 28,620 | 189,347 | 3,414,864 |
| year_2023_base | decorrelated | 8.39% | -13.88% | 43.86 | 9,568 | 63,934 | 1,123,104 |
| year_2023_base | equal | 6.13% | -13.11% | 43.92 | 9,573 | 63,949 | 1,123,104 |
| year_2023_base | ic_weighted | 7.04% | -12.30% | 43.60 | 9,445 | 63,311 | 1,123,104 |
| year_2024_base | decorrelated | -11.55% | -32.13% | 43.92 | 9,561 | 60,526 | 1,131,744 |
| year_2024_base | equal | -10.49% | -31.04% | 43.85 | 9,493 | 60,441 | 1,131,744 |
| year_2024_base | ic_weighted | -10.26% | -28.83% | 43.84 | 9,593 | 61,204 | 1,131,744 |
| year_2025_base | decorrelated | 26.39% | -13.49% | 43.40 | 9,411 | 64,394 | 1,120,272 |
| year_2025_base | equal | 30.24% | -16.88% | 43.97 | 9,751 | 66,460 | 1,119,024 |
| year_2025_base | ic_weighted | 16.05% | -12.56% | 43.44 | 9,310 | 62,858 | 1,119,504 |
| full_high_cost | decorrelated | 28.04% | -33.77% | 128.31 | 28,936 | 241,880 | 3,414,336 |
| full_high_cost | equal | 15.55% | -35.51% | 128.12 | 28,821 | 235,878 | 3,414,432 |
| full_high_cost | ic_weighted | 19.09% | -33.38% | 127.38 | 28,521 | 235,979 | 3,414,864 |

Validation status is `warn`: full-window base and high-cost checks pass, and
turnover remains below the 160 full-window gate, but every method loses money
in the 2024 annual slice. The policy is therefore not production-promotable
yet. Keep `decorrelated/partial_rebalance_daily` as the current research
baseline because it is cost-resilient and operationally tractable, but the next
development task must explain the 2024 regime failure before expanding capital
or adding more factor combinations.

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

Initial 2023 decorrelated cost stress:

| Cost profile | Commission bps | Slippage bps | Sell tax bps | Return | Max drawdown | Gross turnover | Trades | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero cost | 0 | 0 | 0 | 20.56% | -10.63% | 158.44 | 4,849 | 0 |
| base cost | 3 | 1 | 5 | 8.41% | -12.15% | 158.63 | 4,801 | 112,619 |
| stressed cost | 6 | 2 | 10 | -2.16% | -16.93% | 159.05 | 4,767 | 211,509 |

The zero-cost and base-cost gap confirms that costs absorb roughly 12
percentage points of 2023 return. The stressed-cost run turns negative, so the
policy has positive net alpha under current assumptions but does not yet have a
large cost safety margin.

Initial Q1 2023 decorrelated parameter grid:

| Exit rank | Drop | Return | Max drawdown | Gross turnover | Trades | Cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 5 | 7.21% | -6.21% | 46.97 | 598 | 33,076 |
| 100 | 10 | 9.18% | -5.91% | 53.89 | 1,102 | 38,010 |
| 100 | 20 | 9.77% | -5.44% | 60.60 | 2,173 | 42,714 |
| 150 | 5 | 8.20% | -5.86% | 41.23 | 617 | 29,292 |
| 150 | 10 | 7.92% | -6.00% | 47.42 | 1,148 | 33,462 |
| 150 | 20 | 9.60% | -5.10% | 54.10 | 2,228 | 38,228 |
| 200 | 5 | 9.40% | -5.85% | 38.23 | 640 | 27,380 |
| 200 | 10 | 8.35% | -6.18% | 42.36 | 1,160 | 30,147 |
| 200 | 20 | 8.47% | -5.26% | 49.20 | 2,243 | 34,772 |

The grid suggests the policy improvement is not a one-parameter artifact. Wider
exit buffers reduce turnover, and larger drop budgets can improve Q1 return but
raise trade count and cost. Use `exit_rank=150 or 200` and `drop=5 or 10` as the
next conservative search region before expanding to multi-year runs.
