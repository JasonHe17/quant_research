# Fixed-Framework Candidate Baseline - 2026-05-31

This report rebuilds the candidate portfolio baseline after registry v66 and
the fixed-framework priority revalidation.

## Evidence

- Previous standard benchmark:
  `runs/framework_v1_acceptance/standard/candidate_policy_validation/validation_summary.json`
- Seed retest:
  `runs/candidate_factor_portfolios/fixed_framework_seed_amihud_eod_lb1_2026_05_31_standard/validation_summary.json`
- Alpha-only v66 baseline:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/validation_summary.json`
- Alpha plus EOD lb1 overlay:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_plus_eod_lb1_v66_nohealth_2026_05_31_standard/validation_summary.json`

All runs use:

- dataset:
  `runs/framework_v1_acceptance/standard/alpha_dataset`
- admission report:
  `runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json`
- correlation matrix:
  `runs/framework_v1_acceptance/standard/factor_evaluation/feature_correlation.csv`
- method: `decorrelated`
- policy: `partial_rebalance_daily`
- admission statuses: `candidate`, `watchlist`
- registry status: `candidate`
- profile: `standard`

The two full candidate-pool runs use `factor_health_mode=off`. The previous
monitor mode is diagnostic only and does not change scores; disabling it avoids
rebuilding a large diagnostic schedule while preserving the portfolio score and
backtest policy.

## Result Table

| run | features | validation | full base | high cost | worst year | max drawdown | turnover | decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| previous standard benchmark | 24 | pass | 25.23% | 19.65% | 1.00% | -29.96% | 73.09 | superseded by registry v66 |
| seed: Amihud + EOD lb1 | 2 | warn | 54.39% | 47.60% | -1.79% | -34.07% | 71.95 | useful reference, but too concentrated |
| alpha-only v66 | 20 | warn | 27.00% | 21.19% | -4.59% | -29.52% | 73.61 | new candidate baseline |
| alpha + EOD lb1 v66 | 21 | warn | 25.91% | 20.08% | -4.63% | -28.71% | 73.90 | do not use as default baseline |

## Readout

The registry cleanup improves the alpha-only full-period result versus the
previous standard benchmark:

- full-base return improves from `25.23%` to `27.00%`
- high-cost return improves from `19.65%` to `21.19%`
- turnover remains comparable: `73.09` to `73.61`

The cost is worse yearly stability. The previous standard benchmark had no
negative yearly slice; alpha-only v66 has a negative 2024 slice of `-4.59%`.
The weak months are still concentrated around January and June 2024.

Adding `intraday_eod_reversal_5m_lb1_tail6` to the full alpha pool does not
improve the baseline:

- full-base return drops from `27.00%` to `25.91%`
- high-cost return drops from `21.19%` to `20.08%`
- worst year is effectively unchanged: `-4.59%` to `-4.63%`
- drawdown improves modestly: `-29.52%` to `-28.71%`

Therefore EOD lb1 should remain a candidate for a dedicated event-overlay test,
not a default member of the ordinary alpha-rank pool.

## Candidate Weights

In the alpha-only v66 baseline, the largest decorrelated weights are:

| feature | weight |
|---|---:|
| `intraday_weak_tape_gap_up_risk_5m_w48` | 29.33% |
| `intraday_overnight_gap_5m` | 28.22% |
| `intraday_amihud_5m` | 6.24% |
| `intraday_sell_pressure_exhaustion_persistence_5m_l96_s24_m48` | 6.17% |
| `intraday_volume_confirmed_momentum_5m_w48` | 5.06% |

When EOD lb1 is added, it receives a `4.58%` decorrelated weight, but the
portfolio-level result gets worse. This supports keeping it outside the default
alpha baseline.

## Decision

Use the alpha-only v66 run as the current candidate baseline:

`runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/validation_summary.json`

Do not replace the accepted framework benchmark yet. This candidate baseline is
better in full-window return and cost stress, but it introduces a negative 2024
slice. The next step should target that 2024 weakness before promoting this run
to the standard acceptance benchmark.

## Next Steps

1. Run 2024 attribution on the alpha-only v66 baseline, focusing on January and
   June 2024.
2. Test whether an event-overlay gate can use EOD lb1 without degrading the
   alpha-only baseline.
3. Use alpha-only v66, not alpha+EOD, as the baseline for the next incremental
   factor tests.
