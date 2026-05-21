# Legacy Factor Revalidation Review - 2026-05-20

This review records the first full legacy-factor revalidation run after factor
health monitoring was separated from alpha score shrinkage.

## Evidence

- Summary:
  `runs/legacy_factor_revalidation/current/legacy_factor_revalidation_summary.json`
- Table:
  `runs/legacy_factor_revalidation/current/legacy_factor_revalidation_summary.csv`
- Mode: `factor_health_mode=monitor`
- Profile: `standard`
- Methods: `decorrelated`, `equal`, `ic_weighted`
- Backtest policies: `partial_rebalance_daily`,
  `cost_aware_optimizer_daily`
- Admission statuses allowed for portfolio validation: `candidate`,
  `watchlist`

The run completed with 22 result rows. Twenty factors completed portfolio
validation. Two factors were intentionally filtered before portfolio validation:
one was rejected by the new admission gate and one had no matching rows in the
shared admission report.

All completed factor directories contain non-empty validation summary, monthly
summary, factor contribution, factor health summary, and factor health
attribution files.

## Decision Summary

| bucket | count | factors |
|---|---:|---|
| Confirmed | 8 | `intraday_volatility_5m_w24`, `intraday_volatility_5m_w12`, `intraday_volatility_5m_w6`, `intraday_amihud_5m`, `intraday_range_volatility_5m_w48`, `intraday_efficiency_ratio_5m_w48`, `intraday_downside_volatility_5m_w48`, `intraday_sell_pressure_absorption_5m_w48` |
| Upgrade review | 2 | `intraday_volume_confirmed_momentum_5m_w48`, `intraday_downside_turnover_decay_5m_w48` |
| Horizon or policy review | 10 | `intraday_turnover_ratio_5m_w48`, `intraday_turnover_zscore_5m_w48`, `intraday_volume_ratio_5m_w48`, `intraday_momentum_5m_lb12`, `intraday_return_skewness_5m_w48`, `intraday_signed_turnover_imbalance_5m_w48`, `intraday_risk_adjusted_momentum_5m_w48`, `intraday_return_turnover_corr_5m_w48`, `intraday_negative_return_persistence_5m_w48`, `intraday_sell_pressure_recovery_5m_w48` |
| Deprecation or separate data review | 2 | `intraday_limit_pressure_resilience_5m_w48`, `intraday_daily_moving_average_state_5m` |

The strongest confirmed factor is
`intraday_sell_pressure_absorption_5m_w48`: best full-period return `81.54%`
and high-cost return `64.79%`, with the best policy
`cost_aware_optimizer_daily`. The strongest partial-rebalance confirmed factors
are `intraday_volatility_5m_w6`, `intraday_amihud_5m`, and
`intraday_efficiency_ratio_5m_w48`.

## Full Result Table

| factor | action | new admission | validation | full | high cost | policy |
|---|---|---|---:|---:|---:|---|
| `intraday_volatility_5m_w24` | `confirmed` | `candidate` | `completed` | 23.04% | 17.92% | `partial_rebalance_daily` |
| `intraday_volatility_5m_w12` | `confirmed` | `candidate` | `completed` | 25.22% | 19.95% | `partial_rebalance_daily` |
| `intraday_volatility_5m_w6` | `confirmed` | `candidate` | `completed` | 42.47% | 36.53% | `partial_rebalance_daily` |
| `intraday_amihud_5m` | `confirmed` | `candidate` | `completed` | 40.97% | 35.25% | `partial_rebalance_daily` |
| `intraday_turnover_ratio_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 19.25% | 14.10% | `partial_rebalance_daily` |
| `intraday_turnover_zscore_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | -2.88% | -6.00% | `partial_rebalance_daily` |
| `intraday_volume_ratio_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 25.74% | 20.51% | `partial_rebalance_daily` |
| `intraday_momentum_5m_lb12` | `horizon_or_policy_review` | `watchlist` | `completed` | -2.98% | -7.05% | `partial_rebalance_daily` |
| `intraday_range_volatility_5m_w48` | `confirmed` | `candidate` | `completed` | 13.07% | 8.60% | `partial_rebalance_daily` |
| `intraday_efficiency_ratio_5m_w48` | `confirmed` | `candidate` | `completed` | 30.55% | 24.69% | `partial_rebalance_daily` |
| `intraday_downside_volatility_5m_w48` | `confirmed` | `candidate` | `completed` | 6.20% | 2.16% | `partial_rebalance_daily` |
| `intraday_return_skewness_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 5.28% | 1.70% | `partial_rebalance_daily` |
| `intraday_signed_turnover_imbalance_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 9.98% | 4.91% | `partial_rebalance_daily` |
| `intraday_risk_adjusted_momentum_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 10.84% | 6.30% | `partial_rebalance_daily` |
| `intraday_volume_confirmed_momentum_5m_w48` | `upgrade_to_candidate` | `candidate` | `completed` | 8.15% | 3.76% | `partial_rebalance_daily` |
| `intraday_return_turnover_corr_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 8.54% | 3.63% | `partial_rebalance_daily` |
| `intraday_negative_return_persistence_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 3.19% | -1.17% | `partial_rebalance_daily` |
| `intraday_limit_pressure_resilience_5m_w48` | `deprecated_review` | `reject` | `admission_filtered` |  |  |  |
| `intraday_sell_pressure_absorption_5m_w48` | `confirmed` | `candidate` | `completed` | 81.54% | 64.79% | `cost_aware_optimizer_daily` |
| `intraday_downside_turnover_decay_5m_w48` | `upgrade_to_candidate` | `candidate` | `completed` | 2.25% | -1.96% | `partial_rebalance_daily` |
| `intraday_sell_pressure_recovery_5m_w48` | `horizon_or_policy_review` | `watchlist` | `completed` | 11.78% | 7.01% | `partial_rebalance_daily` |
| `intraday_daily_moving_average_state_5m` | `deprecated_review` | `missing` | `admission_filtered` |  |  |  |

## Health-Monitor Readout

The health monitor stayed in diagnostic mode. Actual `weight_scale` remained
`1.0`; recommended scales are therefore only diagnostics.

Confirmed factors still show non-trivial impaired periods, but the best factors
kept enough edge after costs:

| factor | full-base health | recommended scale | note |
|---|---:|---:|---|
| `intraday_sell_pressure_absorption_5m_w48` | 0.555 | 0.667 | Best full and high-cost result; low watch count relative to many watchlist factors |
| `intraday_volatility_5m_w6` | 0.604 | 0.704 | Strongest volatility variant and validation pass |
| `intraday_amihud_5m` | 0.519 | 0.639 | Strong return but warning-level stability profile |
| `intraday_efficiency_ratio_5m_w48` | 0.519 | 0.640 | Validation pass despite moderate health score |

Weak or fragile factors share either poor realized returns, high-cost failure,
or low health:

| factor | full-base health | recommended scale | issue |
|---|---:|---:|---|
| `intraday_turnover_zscore_5m_w48` | 0.448 | 0.587 | Full and high-cost returns both negative |
| `intraday_momentum_5m_lb12` | 0.368 | 0.527 | Full and high-cost returns both negative |
| `intraday_downside_turnover_decay_5m_w48` | 0.427 | 0.571 | Candidate admission, but high-cost return is negative |

## Compact Core Validation

The first post-revalidation compact core used the four strongest and most
complementary confirmed factors:

- `intraday_sell_pressure_absorption_5m_w48`
- `intraday_volatility_5m_w6`
- `intraday_amihud_5m`
- `intraday_efficiency_ratio_5m_w48`

Evidence:

- Summary:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_summary.json`
- Tables:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_summary.csv`
- Factor health:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_factor_health_summary.csv`
- Contribution concentration:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_factor_contribution_summary.csv`

The run completed. Validation status is `warn`, with zero failed checks and one
warning: `primary_yearly_base_positive_returns` because `year_2024_base` is
negative.

| method | policy | full base | full high cost | gross turnover | max drawdown | transaction cost |
|---|---|---:|---:|---:|---:|---:|
| `decorrelated` | `partial_rebalance_daily` | 42.92% | 36.39% | 111.60 | -32.34% | 156,322 |
| `ic_weighted` | `partial_rebalance_daily` | 38.75% | 33.16% | 111.17 | -31.32% | 155,788 |
| `equal` | `partial_rebalance_daily` | 34.11% | 28.25% | 111.19 | -30.23% | 153,915 |
| `ic_weighted` | `cost_aware_optimizer_daily` | 37.84% | -4.06% | 458.55 | -22.09% | 345,592 |
| `decorrelated` | `cost_aware_optimizer_daily` | 29.54% | -10.31% | 475.71 | -23.98% | 345,703 |
| `equal` | `cost_aware_optimizer_daily` | 21.78% | -20.35% | 521.78 | -23.21% | 364,662 |

Primary method and policy:

- Use `decorrelated` + `partial_rebalance_daily` as the compact-core baseline.
  It has the best full-base and high-cost return, and it passes the turnover
  control check.
- Do not use `cost_aware_optimizer_daily` as the default for this combination.
  It improves drawdown but increases realized turnover materially and fails
  high-cost robustness for all three weighting methods.

Yearly stability for the primary compact core:

| scenario | return | max drawdown | transaction cost |
|---|---:|---:|---:|
| `year_2023_base` | 14.97% | -10.77% | 52,717 |
| `year_2024_base` | -5.96% | -28.63% | 51,724 |
| `year_2025_base` | 31.85% | -16.43% | 49,996 |

Health monitor remained diagnostic in the compact-core run. Full-base average
health scores were `0.604` for volatility, `0.555` for sell-pressure
absorption, `0.519` for Amihud, and `0.519` for efficiency ratio. Recommended
scales ranged from `0.639` to `0.704`, while actual weight scales stayed at
`1.0`.

Contribution concentration needs monitoring. In full-base results, average
largest absolute contribution share was `0.533` for `decorrelated`, `0.480` for
`ic_weighted`, and `0.283` for `equal`. The compact core is therefore validated
as a baseline, but future additions should be judged by whether they reduce
2024 fragility and contribution concentration without damaging high-cost
returns.

Follow-up attribution is recorded in
`docs/validation/compact_core_2024_attribution_2026_05_20.md`. The first-pass
finding is that 2024 weakness is concentrated in January and June, not in
abnormal turnover or transaction costs. The dominant risk is sell-pressure
absorption contribution concentration during months when its realized spread
turns negative.

## Required Follow-Ups

1. Keep confirmed factors in the active candidate pool.
   Prioritize `intraday_sell_pressure_absorption_5m_w48`,
   `intraday_volatility_5m_w6`, `intraday_amihud_5m`, and
   `intraday_efficiency_ratio_5m_w48` as the validated compact-core baseline.

2. Promote only one upgrade candidate immediately:
   `intraday_volume_confirmed_momentum_5m_w48` is eligible for candidate review
   but not for compact-core inclusion after the incremental test in
   `docs/validation/compact_core_plus_volume_confirmed_momentum_2026_05_20.md`.
   `intraday_downside_turnover_decay_5m_w48` must stay under cost-fragility
   review until it passes high-cost validation or a lower-turnover policy is
   shown to fix the weakness.

3. Do not add more raw turnover or short-lookback momentum variants until a
   horizon or execution-policy change is tested. Existing turnover/momentum
   watchlist factors often retain some gross signal but lose robustness through
   health deterioration, negative high-cost performance, or both.

4. Run a focused policy search for the watchlist factors with positive high-cost
   returns, especially `intraday_volume_ratio_5m_w48`,
   `intraday_turnover_ratio_5m_w48`, `intraday_sell_pressure_recovery_5m_w48`,
   and `intraday_risk_adjusted_momentum_5m_w48`. The goal is to check whether
   they add diversification under stricter turnover or entry/exit buffers.

5. Use the completed 2024 attribution report before broadening the candidate
   pool. New factors must reduce January/June fragility or contribution
   concentration without breaking high-cost robustness.

6. Do not broaden the compact core by simply adding the next positive
   high-cost watchlist factor. The first incremental test reduced contribution
   concentration only slightly and diluted full-window returns. The completed
   generic overlay test also failed to solve the target failure mode:
   contribution caps diluted full-window and high-cost return, while health
   shrink left `year_2024_base` negative. See
   `docs/validation/compact_core_overlay_experiments_2026_05_21.md`.

   The first targeted sell-pressure regime guard was more informative but still
   not adoptable. It improved `year_2024_base` to `-1.96%`, but reduced
   full-base return to `33.52%` and full high-cost return to `27.88%`. See
   `docs/validation/compact_core_sell_pressure_regime_guard_2026_05_21.md`.

7. Treat `intraday_limit_pressure_resilience_5m_w48` as rejected in this data
   snapshot because coverage is only about `21.64%`, below the `95%` hard gate.
   Retry only after the feature coverage problem is fixed.

8. Revalidate `intraday_daily_moving_average_state_5m` separately. It is marked
   `missing` because the shared admission report did not contain the required
   daily moving-average feature rows. This is a data/batch mismatch, not a
   performance conclusion.

## Next Research Direction

The next factor-development round should not start from broad factor discovery.
It should start from two controlled branches:

- Strengthen the sell-pressure absorption family. It is the clearest winner and
  should be tested with variants that separate absorption from rebound, limit
  pressure, and downside liquidity states.
- Use the validated compact volatility-liquidity core as the baseline, then
  test whether positive high-cost watchlist factors contribute after
  correlation, concentration, and 2024 stability checks.

Health-based shrinkage has now been tested as a compact-core overlay. It
remains a controlled score-construction branch because it preserved full-window
and high-cost primary returns, but it did not solve 2024 and it increased
sell-pressure contribution concentration. The current evidence supports using
health monitoring for attribution and candidate triage first, not as an
automatic alpha transformation.

The targeted sell-pressure guard evidence narrows the next factor-development
task: keep the sell-pressure state variable, but require stricter combined
failure conditions and avoid broad exposure cuts in profitable months.
