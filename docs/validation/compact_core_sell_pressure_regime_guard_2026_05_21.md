# Compact Core Sell-Pressure Regime Guard - 2026-05-21

This report validates a sell-pressure-specific gross-exposure guard for the
four-factor compact core. The guard is built from lagged standalone
`intraday_sell_pressure_absorption_5m_w48` score health, not from the combined
compact-core score.

## Evidence

- Baseline compact core:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_summary.json`
- Guard schedule:
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v1_2026_05_21/gate_threshold_mild_slow/gross_exposure_schedule.csv`
- Guard schedule summary:
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v1_2026_05_21/gate_threshold_mild_slow/summary.json`
- Standard guard validation:
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_mild_slow_2026_05_21_standard/validation_summary.json`
- Quick rejected variants:
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v1_2026_05_21/quick_2024_threshold_20/summary.json`,
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v1_2026_05_21/quick_2024_threshold_strict/summary.json`,
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v1_2026_05_21/quick_2024_budget_20/summary.json`,
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v1_2026_05_21/quick_2024_threshold_deep_only/summary.json`,
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v1_2026_05_21/quick_2024_threshold_mild_slow/summary.json`
- Narrow retry schedules:
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v2_2026_05_21/summary.json`,
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v3_2026_05_21/summary.json`
- Narrow retry standard validation:
  `runs/candidate_factor_portfolios/compact_core_sell_pressure_regime_guard_v2_rank_080_2026_05_21_standard/validation_summary.json`

The standard guard validation completed with validation status `warn`, zero
failed checks, and one warning: `primary_yearly_base_positive_returns` because
`year_2024_base` remains negative.

## Guard Construction

The selected `gate_threshold_mild_slow` schedule uses:

- Standalone sell-pressure scores:
  `runs/legacy_factor_revalidation/current/factors/intraday_sell_pressure_absorption_5m_w48/full_base/scores/decorrelated`
- Dataset:
  `runs/legacy_factor_revalidation/current/shared_benchmark/alpha_dataset`
- Label: `forward_return_48b`
- `lookback_windows=20`, `min_periods=5`, `label_lag_windows=48`
- `state_confirmation_windows=24`
- `full_scale=1.0`, `reduced_scale=0.75`, `blocked_scale=0.5`
- Reduced exposure when lagged rolling top return, spread, or rank IC is below
  zero.
- Blocked exposure when lagged rolling top return or spread is below `-0.003`,
  or rank IC is below `-0.08`.
- Scale step limit: `0.1` per window for both increases and decreases.

The schedule is intentionally milder than the first quick variants. Its
full-window mean gross-exposure scale is `0.740`, with minimum scale `0.5`.

## Quick 2024 Screen

The first quick screen used existing compact-core 2024 scores and changed only
the gross-exposure schedule.

| run | 2024 return | max drawdown | gross turnover | avg target exposure | risk reductions |
|---|---:|---:|---:|---:|---:|
| baseline | -5.96% | -28.63% | 40.21 | 0.898 | 0 |
| threshold 20 | -15.90% | -24.82% | 45.76 | 0.566 | 3,371 |
| threshold strict | -18.88% | -22.61% | 55.45 | 0.437 | 4,590 |
| budget 20 | -13.32% | -21.98% | 45.94 | 0.550 | 3,508 |
| threshold deep only | -3.96% | -24.23% | 43.31 | 0.714 | 2,669 |
| threshold mild slow | -1.96% | -21.98% | 40.64 | 0.673 | 2,412 |

The first three variants were rejected immediately. They reduced January and
June losses but over-cut exposure in other months and increased resize/risk
reduction activity. The `mild_slow` variant was selected for standard
validation because it materially improved 2024 without increasing turnover
much in the quick screen.

## Standard Validation

Primary method and policy: `decorrelated` + `partial_rebalance_daily`.

| scenario | baseline return | guard return | delta | baseline drawdown | guard drawdown | baseline turnover | guard turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_base` | 42.92% | 33.52% | -9.40 pp | -32.34% | -27.55% | 111.60 | 116.50 |
| `full_high_cost` | 36.39% | 27.88% | -8.51 pp | -32.48% | -27.67% | 111.46 | 116.28 |
| `year_2023_base` | 14.97% | 11.36% | -3.61 pp | -10.77% | -11.29% | 39.76 | 40.08 |
| `year_2024_base` | -5.96% | -1.96% | 4.00 pp | -28.63% | -21.98% | 40.21 | 40.64 |
| `year_2025_base` | 31.85% | 23.08% | -8.77 pp | -16.43% | -13.63% | 35.85 | 37.98 |

The guard improves the exact target failure mode: 2024 return improves by
`4.00 pp` and drawdown improves by `6.65 pp`. It also keeps primary turnover
inside the standard control limit.

The cost is too high for adoption. Full-window return falls by `9.40 pp`, high
cost return falls by `8.51 pp`, and 2025 return falls by `8.77 pp`.

## 2024 Monthly Readout

| month | baseline | guard | delta |
|---|---:|---:|---:|
| `2024-01` | -11.64% | -7.74% | 3.90 pp |
| `2024-02` | -2.03% | -4.42% | -2.39 pp |
| `2024-03` | 2.89% | 2.65% | -0.24 pp |
| `2024-04` | -0.60% | 0.43% | 1.03 pp |
| `2024-05` | -1.52% | 0.40% | 1.92 pp |
| `2024-06` | -11.38% | -6.29% | 5.09 pp |
| `2024-07` | -0.24% | -1.85% | -1.61 pp |
| `2024-08` | -2.15% | -2.16% | -0.01 pp |
| `2024-09` | 17.62% | 14.59% | -3.03 pp |
| `2024-10` | 4.34% | 5.30% | 0.96 pp |
| `2024-11` | 5.64% | 2.33% | -3.31 pp |
| `2024-12` | -3.83% | -3.33% | 0.50 pp |

The signal is directionally useful: January and June improve materially. The
remaining issue is false positives in months where the compact core should stay
fully exposed, especially September and November.

## Cost-Aware Readout

| scenario | baseline cost-aware | guard cost-aware | delta |
|---|---:|---:|---:|
| `full_base` | 29.54% | 23.56% | -5.98 pp |
| `full_high_cost` | -10.31% | -4.58% | 5.73 pp |
| `year_2024_base` | -0.40% | 1.99% | 2.39 pp |

The guard improves the cost-aware high-cost branch and turns 2024 cost-aware
positive. It still does not make `cost_aware_optimizer_daily` viable as the
default because full high-cost remains negative.

## Narrow Retry

The next retry tested narrower schedules from the same standalone sell-pressure
observations.

`v2` required lagged rolling top return and spread to weaken together before
reducing exposure. The best variant, `gate_combined_rank_080`, also required
weak rank IC for the combined condition. Its full-window mean schedule scale is
`0.938`, versus `0.740` for `gate_threshold_mild_slow`.

`v3` added a narrow top-return-only branch to catch January 2024, where top
return was weak but spread stayed near or above zero.

### Quick 2024 Screen

| run | 2024 return | max drawdown | gross turnover | avg target exposure |
|---|---:|---:|---:|---:|
| baseline | -5.96% | -28.63% | 40.21 | 0.898 |
| v1 `threshold_mild_slow` | -1.96% | -21.98% | 40.64 | 0.673 |
| v2 `combined_mild_080` | -4.96% | -27.30% | 40.79 | 0.848 |
| v2 `combined_mild_085` | -5.37% | -28.10% | 39.71 | 0.863 |
| v2 `combined_deep_080` | -4.85% | -27.75% | 41.76 | 0.852 |
| v2 `combined_rank_080` | -3.94% | -26.88% | 40.78 | 0.853 |
| v3 `top_or_combined_075` | -7.02% | -26.94% | 41.89 | 0.835 |
| v3 `top_rank_or_combined_075` | -7.16% | -27.85% | 41.99 | 0.844 |

The v2 combined-rank variant is the only narrow retry worth standard
validation. The v3 top-return branch improved June but damaged the annual
result, confirming that standalone top-return weakness is not reliable enough
as an exposure trigger.

### Standard Validation For v2 `combined_rank_080`

Primary method and policy: `decorrelated` + `partial_rebalance_daily`.

| scenario | baseline return | v2 return | delta | baseline drawdown | v2 drawdown | baseline turnover | v2 turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_base` | 42.92% | 40.87% | -2.05 pp | -32.34% | -30.85% | 111.60 | 115.22 |
| `full_high_cost` | 36.39% | 34.91% | -1.48 pp | -32.48% | -30.93% | 111.46 | 115.21 |
| `year_2023_base` | 14.97% | 16.10% | 1.13 pp | -10.77% | -10.90% | 39.76 | 40.93 |
| `year_2024_base` | -5.96% | -3.94% | 2.02 pp | -28.63% | -26.88% | 40.21 | 40.78 |
| `year_2025_base` | 31.85% | 28.89% | -2.96 pp | -16.43% | -16.35% | 35.85 | 36.66 |

The v2 schedule is much less destructive than v1. It improves 2024 by
`2.02 pp`, full-window drawdown by `1.49 pp`, and full high-cost drawdown by
`1.55 pp`, while giving up `2.05 pp` full-base return and `1.48 pp` high-cost
return.

The remaining problem is that it does not solve the main January failure mode.

| month | baseline | v1 guard | v2 guard |
|---|---:|---:|---:|
| `2024-01` | -11.64% | -7.74% | -11.80% |
| `2024-02` | -2.03% | -4.42% | -3.07% |
| `2024-06` | -11.38% | -6.29% | -9.12% |
| `2024-09` | 17.62% | 14.59% | 17.54% |
| `2024-11` | 5.64% | 2.33% | 4.78% |

The combined condition avoids most September and November false positives but
misses January. It primarily helps June.

Cost-aware readout:

| scenario | baseline cost-aware | v2 cost-aware | delta |
|---|---:|---:|---:|
| `full_base` | 29.54% | 29.15% | -0.39 pp |
| `full_high_cost` | -10.31% | -3.05% | 7.26 pp |
| `year_2024_base` | -0.40% | 1.70% | 2.10 pp |

This is useful risk-control evidence, especially for high-cost robustness, but
not enough for adoption as the compact-core default.

## Decision

Do not adopt `gate_threshold_mild_slow` or v2 `combined_rank_080` as the
compact-core production overlay.

Reasons:

1. v1 improves January and June but gives back too much full-window and 2025
   return.
2. v2 is less costly, but `year_2024_base` remains negative at `-3.94%`.
3. v2 does not fix January 2024: `-11.80%` versus baseline `-11.64%`.
4. v3 confirms that top-return-only exposure cuts are not reliable.

Keep the result as positive research evidence. A sell-pressure-specific regime
state is more useful than generic contribution caps or generic health shrink,
but timestamp-level gross-exposure overlays are not yet solving the full target
problem.

## Next Step

Stop tuning this timestamp-level sell-pressure exposure overlay for now. The
next step should move back to factor development or a higher-level regime
definition:

- Develop a complementary factor or factor variant that targets January-style
  sell-pressure stress without relying on top-return-only exposure cuts.
- Prioritize candidates such as sell-pressure recovery, limit-pressure
  resilience, downside-liquidity exhaustion, or a market-stress interaction.
- If the guard is revisited, use a day-level or rebalance-level state and make
  January/June protection versus September/November false positives an explicit
  validation table.
- Keep `factor_health_mode=monitor` as the production default until a targeted
  factor or guard improves 2024 without materially reducing full-window and
  high-cost primary returns.
