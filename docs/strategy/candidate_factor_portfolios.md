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
`validation_summary.csv`, `validation_monthly_summary.csv`, and
`validation_summary.json`. Use `--profile robust` to add a full-window
zero-cost diagnostic.

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

Primary 2024 monthly failure diagnostic:

| Month | Return | Max drawdown | Trades | Cost | Gross traded notional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024-01 | -15.65% | -16.64% | 841 | 6,032 | 5,311,813 |
| 2024-06 | -11.46% | -12.39% | 739 | 4,596 | 2,572,969 |
| 2024-12 | -2.51% | -9.59% | 877 | 5,610 | 3,497,147 |
| 2024-08 | -2.32% | -6.58% | 868 | 5,315 | 2,778,417 |
| 2024-02 | -1.89% | -18.68% | 620 | 3,815 | 2,045,211 |

The loss is concentrated in January and June, while monthly transaction costs
are too small to explain those moves by themselves. Treat this as a signal or
regime failure rather than a pure cost-control failure. The next diagnostic
should compare 2024 candidate-factor legs, market regime labels, and exposure
concentration before changing the trading policy again.

2024 regime diagnostic:

```bash
conda run -n quant python examples/analyze_candidate_policy_regime.py \
  --validation-dir runs/candidate_factor_portfolios/partial_rebalance_validation_standard \
  --output-dir runs/candidate_factor_portfolios/partial_rebalance_validation_standard/regime_diagnostics_2024 \
  --scenario year_2024_base \
  --method decorrelated \
  --policy partial_rebalance_daily \
  --year 2024
```

The diagnostic writes `composite_monthly.csv`, `factor_legs_monthly.csv`,
`top_score_exposure_monthly.csv`, and `regime_failure_report.md`. The 2024
failure is not primarily a transaction-cost or tradability problem:

| Month | Portfolio return | Score IC | Score top-minus-bottom | Top-score label | Market label | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2024-01 | -15.65% | 0.0652 | 0.67% | -0.70% | -0.82% | Score ranks help, but selected basket is still negative |
| 2024-06 | -11.46% | 0.0373 | 0.19% | -0.54% | -0.43% | Weak positive ranking inside a negative tape |
| 2024-02 | -1.89% | -0.0536 | -0.18% | 0.05% | 0.47% | Direct score/factor inversion |

Worst-month factor-leg diagnostics show two distinct issues. In January and
June, the volatility legs still have positive top-minus-bottom spreads, but the
top leg itself has negative forward returns, so the long-only portfolio loses
despite positive relative ranking. In February, the volatility legs invert:
`intraday_volatility_5m_w6`, `w12`, and `w24` all have negative directional IC.
Top-score exposure is also concentrated: `intraday_volatility_5m_w24` explains
roughly 58-59% of absolute top-score contribution in the main loss months, with
`intraday_amihud_5m` contributing another 18-21%.

Next framework work should therefore add a portfolio risk/regime layer before
more factor-combination sweeps:

- A policy-level market/regime gate now has first plumbing through
  `--policy-gross-exposure-scale` and
  `--policy-gross-exposure-scale-path`. Build the time-varying schedule with
  `examples/build_policy_regime_gate.py`; it shifts score-health diagnostics by
  the label maturation horizon before rolling, so forward-return labels that
  are not yet observable cannot control the current target book.
- Factor exposure caps or shrinkage so one volatility horizon cannot dominate
  the selected basket.
- A rolling leg-health diagnostic that downweights factors after recent
  directional IC inversion, especially short-horizon volatility legs.

Example lagged regime-gate build and replay:

```bash
conda run -n quant python examples/build_policy_regime_gate.py \
  --dataset-dir runs/framework_v1_acceptance/standard/alpha_dataset \
  --scores-path runs/candidate_factor_portfolios/policy_set_q1_2023_decorrelated/scores/decorrelated \
  --output-dir runs/candidate_factor_portfolios/policy_set_q1_2023_decorrelated/regime_gate/decorrelated \
  --top-n 50 \
  --lookback-windows 20 \
  --min-periods 5 \
  --label-lag-windows 48 \
  --state-confirmation-windows 2 \
  --max-scale-change-per-window 0.25

conda run -n quant python examples/run_tree_score_backtest.py \
  --predictions-path runs/candidate_factor_portfolios/policy_set_q1_2023_decorrelated/scores/decorrelated/*.parquet \
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
  --policy-partial-rebalance-rate 0.5 \
  --policy-gross-exposure-scale-path runs/candidate_factor_portfolios/policy_set_q1_2023_decorrelated/regime_gate/decorrelated/gross_exposure_schedule.csv \
  --data-access-mode fast_parquet \
  --streaming-chunk month \
  --output-dir runs/candidate_factor_portfolios/policy_set_q1_2023_decorrelated/regime_gated_backtest/decorrelated
```

Use `examples/compare_policy_backtests.py` when a gated replay needs overall
and monthly attribution against the ungated policy:

```bash
conda run -n quant python examples/compare_policy_backtests.py \
  --baseline-dir runs/candidate_factor_portfolios/partial_rebalance_validation_standard/regime_gated_backtest_2024_h1/decorrelated/partial_rebalance_daily_baseline \
  --candidate-dir runs/candidate_factor_portfolios/partial_rebalance_validation_standard/regime_gated_backtest_2024_h1/decorrelated/partial_rebalance_daily_budget_deadband_gate \
  --output-dir runs/candidate_factor_portfolios/partial_rebalance_validation_standard/regime_gated_backtest_2024_h1/decorrelated/comparison_budget_deadband_vs_baseline \
  --baseline-name baseline_partial_rebalance_daily \
  --candidate-name budget_deadband_gate \
  --start 2024-01-02T09:35:00+08:00 \
  --end 2024-06-28T15:00:00+08:00
```

Initial Q1 2023 lag-48 regime-gate replay:

| Variant | Return | Max drawdown | Gross turnover | Trades | Avg target gross | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline `partial_rebalance_daily` | 4.76% | -5.82% | 11.89 | 2,274 | - | Current default candidate policy |
| Lag-48 rolling gate | 4.21% | -3.99% | 17.04 | 3,102 | 0.60 | Lower drawdown, but lower return and higher turnover |
| Lag-48 gate + 2-signal confirmation + 0.25 scale step | 3.92% | -4.08% | 16.81 | 3,064 | 0.60 | Step limit slightly reduces turnover, but does not fix the trade-noise problem |
| Lag-48 continuous budget gate + 0.10 scale step | 3.62% | -4.45% | 13.94 | 2,892 | 0.67 | Lower turnover than binary gates, still worse than baseline |
| Budget gate + 0.03 deadband + asymmetric 0.20 down / 0.05 up steps | 3.63% | -4.43% | 13.37 | 2,789 | 0.64 | Further reduces trade noise, but still does not beat baseline |

The first gated replay is useful as a framework check, not a promotion result.
It confirms that a matured-label schedule can reduce drawdown, but the raw
full/reduced/blocked scale changes add trading. The first smoothing pass reduces
turnover only marginally and lowers return further. The continuous risk-budget
gate reduces the trade-noise problem versus binary gates, but still loses too
much return and remains above baseline turnover. Adding a deadband plus slower
re-risking lowers turnover further, but does not solve the net-return problem.
Do not enable any gate as a default until it shows positive net value in
multi-year acceptance. The next work should compare these gates on 2024 loss
months where the regime failure was originally observed; if they only help
there, they should remain targeted risk overlays rather than default policy.

H1 2024 targeted replay confirms that interpretation:

| Variant | Return | Max drawdown | Gross turnover | Trades | Avg target gross |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline `partial_rebalance_daily` | -24.06% | -32.13% | 22.07 | 4,621 | 0.96 |
| Budget + deadband gate | -16.09% | -18.10% | 17.45 | 4,262 | 0.44 |

Monthly attribution:

| Month | Baseline return | Gated return | Delta | Read |
| --- | ---: | ---: | ---: | --- |
| 2024-01 | -15.65% | -10.85% | +4.80% | Helps original negative-market loss month |
| 2024-02 | -1.89% | -2.24% | -0.35% | Does not fix score/factor inversion |
| 2024-03 | 3.95% | 3.73% | -0.22% | Slight drag |
| 2024-04 | -0.00% | -3.21% | -3.20% | Clear false positive / opportunity cost |
| 2024-05 | -0.29% | 0.70% | +0.99% | Helps modestly |
| 2024-06 | -11.46% | -4.78% | +6.69% | Helps original negative top-leg month |

The gate is therefore useful as a targeted risk overlay for negative top-leg or
negative-market regimes, but it does not solve factor inversion months and it
can create false positives. Promotion requires a second-stage guard such as
market-state confirmation, factor-leg health, or an optimizer-level turnover
budget.

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

## Calibrated Optimizer Combination-Layer Sweep

After promoting the calibrated expected-edge optimizer with regime gross
exposure gating and a `0.10` per-rebalance gross-turnover budget, the next
combination-layer check reused the same production-like trading framework and
changed only the score-combination method. This keeps the comparison focused on
factor combination quality rather than policy differences.

Full-window quick comparison output:

```text
runs/candidate_factor_portfolios/promoted_combination_methods_full_base_quick
```

| Method | Return | Max drawdown | Gross turnover | Trades | Cost | Avg target gross | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| equal | 36.40% | -7.89% | 166.00 | 14,357 | 150,913 | 0.395 | Highest return, but above 160 turnover gate |
| decorrelated | 32.21% | -8.49% | 156.37 | 13,663 | 141,100 | 0.397 | Current promoted default; passes gate |
| ic_weighted | 16.15% | -9.69% | 157.91 | 14,260 | 138,315 | 0.369 | Weaker risk-adjusted result |

The quick sweep changes the candidate ranking under the new framework:
`equal` now beats `decorrelated` on full-window return and drawdown, while
`ic_weighted` is not competitive. Because `equal` breaches the turnover gate,
it was expanded to standard validation before any default change.

Standard validation for `equal`, budget `0.10`:

```text
runs/candidate_factor_portfolios/promoted_combination_methods_equal_standard_budget010
```

Wrapper status: `warn`, with zero failed checks and one turnover warning.

| Scenario | Return | Max drawdown | Gross turnover | Trades | Cost | Avg target gross | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2023-2025 full | 36.40% | -7.89% | 166.00 | 14,357 | 150,913 | 0.395 | Stronger than current default, but above turnover gate |
| 2023-2025 high cost | 23.47% | -8.71% | 165.79 | 14,289 | 254,659 | 0.395 | Positive under doubled costs |
| 2023 | 8.93% | -6.44% | 59.91 | 4,134 | 46,322 | 0.477 | Positive yearly slice |
| 2024 | 16.69% | -7.23% | 48.33 | 4,987 | 44,267 | 0.317 | Stronger than current default |
| 2025 | 7.07% | -4.89% | 57.09 | 5,225 | 48,206 | 0.377 | Positive yearly slice |

Narrow budget check for `equal`, budget `0.095`:

```text
runs/candidate_factor_portfolios/promoted_combination_methods_equal_standard_budget0095
```

Wrapper status: `warn`, with zero failed checks and one turnover warning.

| Scenario | Return | Max drawdown | Gross turnover | Trades | Cost | Avg target gross | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2023-2025 full | 37.18% | -9.08% | 162.63 | 14,368 | 148,917 | 0.394 | Still above turnover gate |
| 2023-2025 high cost | 24.42% | -9.83% | 162.37 | 14,341 | 250,294 | 0.394 | Positive under doubled costs |
| 2023 | 6.56% | -7.04% | 59.55 | 4,236 | 46,114 | 0.478 | Lower return than budget 0.10 |
| 2024 | 15.31% | -7.14% | 47.26 | 4,963 | 43,419 | 0.320 | Still strong |
| 2025 | 6.91% | -7.02% | 55.83 | 5,223 | 47,682 | 0.378 | Positive yearly slice |

Decision: keep `decorrelated` as the promoted default because it is the only
combination method that currently passes the standard gate without warnings.
Promote `equal` to the leading research candidate, not to the default. The
`0.095` check shows that simply tightening the single per-rebalance turnover
budget does not reliably bring full-window turnover below 160 and can hurt
individual yearly slices. The next framework task should therefore improve the
trade optimizer or turnover budget allocation itself, rather than continue a
blind scalar threshold sweep.

Path-level turnover-budget quick check for `equal`:

```text
runs/candidate_factor_portfolios/equal_path_turnover_budget_quick_budget155_v5
```

| Scenario | Return | Max drawdown | Gross turnover | Planned turnover | Trades | Cost | Avg target gross | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2023-2025 full | 33.86% | -7.89% | 148.11 | 155.00 | 12,736 | 134,011 | 0.358 | Passes turnover gate; budget exhausted from 2025-09 |

The path-level budget is more useful than another scalar per-rebalance budget
sweep: it allows the optimizer to spend up to the normal per-rebalance cap while
keeping the whole validation path inside the turnover envelope. Next step is a
standard validation comparing `equal` budget `155` against the promoted
`decorrelated` default.

Standard validation after path-level turnover budget:

```text
runs/candidate_factor_portfolios/equal_path_turnover_budget_standard_budget155
runs/candidate_factor_portfolios/decorrelated_promoted_standard_after_path_budget
```

Both wrappers completed with `overall_status=pass`, zero warnings, and zero
failed checks.

| Method | Control | Full return | Full drawdown | Full turnover | High-cost return | 2023 | 2024 | 2025 | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| equal | Path budget 155 | 33.86% | -7.89% | 148.11 | 22.21% | 8.93% | 16.69% | 7.07% | Best standard result; budget exhausted from 2025-09 |
| decorrelated | Per-rebalance budget 0.10 | 32.21% | -8.49% | 156.37 | 20.31% | 7.97% | 12.18% | 5.10% | Current promoted default remains valid |

Decision: `equal` with a path-level gross-turnover budget of `155` becomes the
leading promoted candidate, but not yet a live default. The full-path budget is
acceptable for research validation, while production needs a replenishing or
rolling turnover budget horizon so the strategy cannot spend the entire long-run
budget before the end of an operating period.

Rolling turnover-budget smoke:

```text
runs/candidate_factor_portfolios/equal_monthly_budget_quick_smoke
```

| Method | Period | Budget | Return | Max drawdown | Gross turnover | Period count | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| equal | month | 6 | 37.27% | -7.24% | 163.53 | 36 | Replenishes across all months; too loose for 160 full-turnover gate |

The rolling ledger fixes the production problem of path-budget exhaustion, but
budget size now needs calibration. The next sweep should focus on monthly
budgets around `4.0-4.5` or annual budgets around `52`.

Rolling budget calibration:

```text
runs/candidate_factor_portfolios/equal_monthly_budget_quick_budget43
runs/candidate_factor_portfolios/equal_annual_budget_quick_budget52
runs/candidate_factor_portfolios/equal_annual_budget_standard_budget52
```

Quick full-window calibration:

| Method | Period | Budget | Return | Max drawdown | Gross turnover | Planned turnover | Period count | Status | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| equal | month | 4.3 | 21.95% | -10.49% | 146.41 | 143.86 | 36 | pass | Too restrictive; monthly local caps suppress useful turnover and lose too much return |
| equal | year | 52 | 33.44% | -7.85% | 149.42 | 154.30 | 3 | pass | Preserves the path-budget return profile while replenishing annually |

Standard validation for `equal`, annual budget `52`:

| Scenario | Return | Max drawdown | Gross turnover | Planned turnover | Trades | Cost | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2023-2025 full | 33.44% | -7.85% | 149.42 | 154.30 | 13,008 | 135,304 | Passes full-window turnover gate |
| 2023-2025 high cost | 22.05% | -8.72% | 149.47 | 154.30 | 12,939 | 229,027 | Positive under doubled costs |
| 2023 | 8.51% | -6.44% | 50.69 | 52.00 | 3,504 | 39,394 | Budget spent, yearly slice positive |
| 2024 | 16.69% | -7.23% | 48.33 | 50.08 | 4,987 | 44,267 | Same strong 2024 profile as path budget |
| 2025 | 6.83% | -4.89% | 49.13 | 52.00 | 4,476 | 41,337 | Budget spent without late-year path exhaustion |

Decision: promote `equal` with annual gross-turnover budget `52` as the current
combination-layer candidate. It is slightly lower return than full-path budget
`155` (`33.44%` vs. `33.86%`) but removes the production flaw where a
multi-year path budget can be depleted before the end of the operating horizon.
Annual replenishment is also a better match for how a live strategy would be
monitored: it preserves a clear yearly turnover envelope of about `52`, keeps
the full validation turnover below `160`, and avoids the excessive local
constraint of a monthly `4.3` budget. Do not promote the monthly ledger until a
separate experiment shows that monthly pacing improves live risk without
destroying return.

## Downside-Volatility Gross-Exposure Gate

The next candidate control uses
`intraday_downside_volatility_5m_w48` as a market-wide risk gate rather than as
a standalone alpha. The schedule is built from the cross-sectional mean factor
value, with high/extreme thresholds computed from lagged rolling history only.
It is then combined with the existing regime gate by taking the lower
gross-exposure scale.

```text
runs/candidate_factor_portfolios/downside_volatility_w48_risk_gate_v1
runs/candidate_factor_portfolios/downside_volatility_w48_risk_gate_v1_promoted_standard
```

Gate state counts: `full=26472`, `reduced=3978`, `blocked=4301`,
`warmup=48`.

Standard validation completed with `overall_status=pass`, zero warnings, and
zero failed checks.

| Scenario | Return | Max drawdown | Gross turnover | Planned turnover | Trades | Cost | Avg target gross | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2023-2025 full | 38.77% | -6.96% | 149.96 | 156.58 | 13,433 | 139,397 | 0.381 | Passes turnover gate and improves promoted decorrelated baseline |
| 2023-2025 high cost | 27.31% | -8.33% | 149.70 | 156.58 | 13,433 | 235,246 | 0.381 | Positive under doubled costs |
| 2023 | 10.44% | -6.96% | 56.56 | 58.62 | 3,976 | 43,685 | 0.459 | Positive yearly slice |
| 2024 | 15.68% | -7.21% | 41.94 | 44.07 | 4,505 | 39,086 | 0.316 | Positive yearly slice |
| 2025 | 7.05% | -5.54% | 51.25 | 54.58 | 4,952 | 44,583 | 0.362 | Positive yearly slice |

Baseline comparison against
`runs/candidate_factor_portfolios/decorrelated_promoted_standard_after_path_budget`:

| Scenario | Baseline return | Gate return | Baseline drawdown | Gate drawdown | Baseline turnover | Gate turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-2025 full | 32.21% | 38.77% | -8.49% | -6.96% | 156.37 | 149.96 |
| 2023-2025 high cost | 20.31% | 27.31% | -9.28% | -8.33% | 156.19 | 149.70 |
| 2023 | 7.97% | 10.44% | -7.47% | -6.96% | 57.95 | 56.56 |
| 2024 | 12.18% | 15.68% | -7.23% | -7.21% | 44.05 | 41.94 |
| 2025 | 5.10% | 7.05% | -7.10% | -5.54% | 54.25 | 51.25 |

Decision: keep the factor as a risk-control candidate and promote the gate to
the next integration target for the portfolio layer. This is not evidence that
`intraday_downside_volatility_5m_w48` should be traded as a standalone alpha:
the isolated sleeve lost `34.96%` in the same 2023-2025 window after the
single-factor score path was fixed.
