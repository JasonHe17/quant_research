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
- Repaired alpha-rank research benchmark:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- Repaired benchmark attribution:
  `docs/validation/fixed_framework_alpha_rank_repaired_benchmark_attribution_2026_05_31.md`
- Repaired benchmark 2025 degradation attribution:
  `docs/validation/fixed_framework_alpha_rank_repaired_benchmark_2025_degradation_attribution_2026_05_31.md`
- Drawdown overlay screen:
  `docs/validation/fixed_framework_alpha_rank_drawdown_overlay_screen_2026_05_31.md`
- State overlay screen:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_screen_2026_05_31.md`
- State overlay robustness:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_robustness_2026_06_01.md`
- State overlay schedule attribution:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_schedule_attribution_2026_06_01.md`

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

The full candidate-pool runs use `factor_health_mode=off`. The previous monitor
mode is diagnostic only and does not change scores; disabling it avoids
rebuilding a large diagnostic schedule while preserving the portfolio score and
backtest policy. The repaired benchmark uses an external lagged factor-weight
schedule for `intraday_overnight_gap_5m` plus a row contribution cap.

## Result Table

| run | features | validation | full base | high cost | worst year | max drawdown | turnover | decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| previous standard benchmark | 24 | pass | 25.23% | 19.65% | 1.00% | -29.96% | 73.09 | superseded by registry v66 |
| seed: Amihud + EOD lb1 | 2 | warn | 54.39% | 47.60% | -1.79% | -34.07% | 71.95 | useful reference, but too concentrated |
| alpha-only v66 | 20 | warn | 27.00% | 21.19% | -4.59% | -29.52% | 73.61 | superseded by repaired research benchmark |
| alpha + EOD lb1 v66 | 21 | warn | 25.91% | 20.08% | -4.63% | -28.71% | 73.90 | do not use as default baseline |
| overnight-gap gate + contribution cap25 | 20 | pass | 27.97% | 22.15% | 0.90% | -31.70% | 73.23 | current alpha-rank research benchmark |
| state budget min90 l120 | 20 | pass | 30.32% | 24.39% | 0.86% | -29.87% | 71.02 | current state-aware frontier candidate |

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

The repaired alpha-rank benchmark applies the lagged deep25 health gate only to
`intraday_overnight_gap_5m` and adds a row contribution cap of `25%`. This fixes
the 2024 yearly slice while improving full-window and high-cost returns:

- full-base return improves from `27.00%` to `27.97%`;
- high-cost return improves from `21.19%` to `22.15%`;
- worst year improves from `-4.59%` to `0.90%`;
- 2025 improves from `18.09%` to `19.71%`;
- full-base max drawdown worsens from `-28.53%` to `-30.77%`.

The drawdown tradeoff should be monitored, but the repaired benchmark clears
the standard validation gates and removes the 2024 stability blocker.
Attribution shows that the repair works by suppressing the overnight-gap leg
and lowering row contribution concentration, while the remaining weak months
are dominated by `intraday_weak_tape_gap_up_risk_5m_w48`.

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

Use the repaired alpha-rank run as the no-overlay control benchmark for
alpha-rank portfolio iteration:

`runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`

Use `state budget min90 l120` as the current state-aware frontier candidate:

`runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/validation_summary.json`

Keep the alpha-only v66 run as the naive/control comparison for attribution and
ablation work. Replacement of the accepted framework benchmark should still be
handled through a separate default-change review.

## Next Steps

1. Prepare a research-benchmark replacement note for `budget_min90_l120`, while
   leaving the production/default benchmark unchanged.
2. Use the repaired benchmark and `budget_min90_l120` state-aware overlay as
   frontier comparisons for the next incremental alpha-rank factor tests.
