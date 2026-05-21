# Compact Core Plus Volume-Confirmed Momentum - 2026-05-20

This report validates `intraday_volume_confirmed_momentum_5m_w48` as the first
incremental addition to the compact volatility-liquidity core.

## Evidence

- Baseline:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_summary.json`
- Incremental run:
  `runs/candidate_factor_portfolios/compact_core_plus_volume_confirmed_momentum_2026_05_20_standard/validation_summary.json`
- Incremental tables:
  `runs/candidate_factor_portfolios/compact_core_plus_volume_confirmed_momentum_2026_05_20_standard/validation_summary.csv`
- Incremental health attribution:
  `runs/candidate_factor_portfolios/compact_core_plus_volume_confirmed_momentum_2026_05_20_standard/validation_factor_health_attribution.csv`
- Prior 2024 attribution:
  `docs/validation/compact_core_2024_attribution_2026_05_20.md`

The run completed with validation status `warn`, zero failed checks, and one
warning: `primary_yearly_base_positive_returns` because `year_2024_base`
remains negative.

## Primary Comparison

Primary method and policy: `decorrelated` + `partial_rebalance_daily`.

| scenario | baseline return | plus-volume return | delta | baseline drawdown | plus-volume drawdown |
|---|---:|---:|---:|---:|---:|
| `full_base` | 42.92% | 35.58% | -7.34 pp | -32.34% | -31.49% |
| `full_high_cost` | 36.39% | 29.43% | -6.96 pp | -32.48% | -31.60% |
| `year_2023_base` | 14.97% | 12.33% | -2.64 pp | -10.77% | -10.68% |
| `year_2024_base` | -5.96% | -5.18% | 0.79 pp | -28.63% | -27.54% |
| `year_2025_base` | 31.85% | 31.58% | -0.28 pp | -16.43% | -16.49% |

The addition gives a small 2024 improvement and a small drawdown improvement,
but it materially reduces full-window and high-cost returns. It does not meet
the bar for joining the compact core.

## 2024 Failure-Mode Check

The goal was to reduce January and June fragility. Results were mixed:

| month | baseline return | plus-volume return | delta |
|---|---:|---:|---:|
| `2024-01` | -11.64% | -12.29% | -0.65 pp |
| `2024-06` | -11.38% | -10.71% | 0.67 pp |
| `2024-09` | 17.62% | 17.52% | -0.10 pp |

The factor improves June slightly but worsens January. It does not solve the
dominant stress months identified in the 2024 attribution report.

## Contribution And Health

The addition reduces measured concentration but does not change the main risk:
sell-pressure absorption remains the largest contribution feature in all
`11,033` observed bars of the 2024 primary run.

| scenario | method | baseline largest share | plus-volume largest share | baseline top-two share | plus-volume top-two share |
|---|---|---:|---:|---:|---:|
| `full_base` | `decorrelated` | 0.533 | 0.525 | 0.906 | 0.890 |
| `year_2024_base` | `decorrelated` | 0.534 | 0.524 | 0.906 | 0.889 |
| `year_2024_base` | `equal` | 0.285 | 0.259 | 0.552 | 0.499 |
| `year_2024_base` | `ic_weighted` | 0.482 | 0.473 | 0.757 | 0.742 |

The added factor's health is weaker than the four core factors:

| feature | full-base average health | full-base recommended scale | impaired count |
|---|---:|---:|---:|
| `intraday_volume_confirmed_momentum_5m_w48` | 0.386 | 0.541 | 15,450 |
| `intraday_amihud_5m` | 0.519 | 0.639 | 12,055 |
| `intraday_efficiency_ratio_5m_w48` | 0.519 | 0.640 | 12,190 |
| `intraday_sell_pressure_absorption_5m_w48` | 0.555 | 0.667 | 13,454 |
| `intraday_volatility_5m_w6` | 0.604 | 0.704 | 10,448 |

## Decision

Do not add `intraday_volume_confirmed_momentum_5m_w48` to the compact-core
baseline. Keep it as a `candidate`, but mark the compact-core incremental test
as failed for core inclusion.

Reasons:

1. Full-base return falls from `42.92%` to `35.58%`.
2. Full high-cost return falls from `36.39%` to `29.43%`.
3. 2024 improves only slightly and remains negative.
4. January 2024 gets worse.
5. Sell-pressure absorption remains the dominant contribution driver.

## Next Step

The next experiment should target the actual failure mode directly rather than
adding another weak diversifier. Test a controlled sell-pressure contribution
cap or health-gated overlay in monitor-derived diagnostics. The acceptance bar
should be:

- Improve or neutralize January and June 2024.
- Keep full-base and high-cost returns close to or above the four-factor compact
  core.
- Reduce sell-pressure contribution concentration in `year_2024_base`.
- Keep the production framework in `factor_health_mode=monitor` unless the
  overlay passes full-window, high-cost, and yearly robustness checks.
