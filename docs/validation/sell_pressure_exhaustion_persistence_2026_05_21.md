# Sell-Pressure Exhaustion Persistence - 2026-05-21

This report records the development and validation of
`intraday_sell_pressure_exhaustion_persistence_5m_l96_s24_m48`.

The factor is designed to test persistent seller exhaustion rather than a
generic complementary signal. It keeps slow sell-pressure exhaustion over a
96-bar window and penalizes short-window relief rallies:

`exhaustion_96 - 0.5 * (exhaustion_24 + exhaustion_48)`

where each exhaustion component is:

`log1p(max(recovery_ratio, 0)) * upside_turnover_participation * max(downside_turnover_decay, 0)`

## Hypothesis

Names should be stronger long-only candidates when earlier downside turnover
was heavy, recent downside turnover has faded, price has recovered relative to
recent downside damage, and upside turnover participates in the recovery.

The persistence transform rejects the weaker interpretation that any short
rebound after selling is bullish. A strong slow exhaustion reading that is not
dominated by 24- or 48-bar relief rallies is more consistent with durable
supply exhaustion than with crowded intraday bounce chasing.

## Evidence

- Raw exhaustion dataset:
  `runs/factor_research/sell_pressure_exhaustion_2026_05_21/alpha_dataset`
- Raw exhaustion evaluation:
  `runs/factor_research/sell_pressure_exhaustion_2026_05_21/factor_evaluation`
- Official persistence dataset:
  `runs/factor_research/sell_pressure_exhaustion_2026_05_21/persistence_official_alpha_dataset`
- Official persistence admission:
  `runs/factor_research/sell_pressure_exhaustion_2026_05_21/persistence_official_factor_admission/factor_admission_report.json`
- Single-factor policy validation:
  `runs/factor_research/sell_pressure_exhaustion_2026_05_21/persistence_policy_validation_standard/validation_summary.json`
- Compact-core plus persistence validation:
  `runs/factor_research/sell_pressure_exhaustion_2026_05_21/compact_core_plus_persistence_policy_validation_standard/validation_summary.json`

## Admission

The raw exhaustion family was not sufficient:

| feature | admission | direction | rank IC | hit rate | cost-adjusted spread | note |
|---|---|---|---:|---:|---:|---|
| `intraday_sell_pressure_exhaustion_5m_w96` | `reject` | `long` | 0.00669 | 51.74% | 0.55 bps | Failed hit-rate gate |
| `intraday_sell_pressure_exhaustion_5m_w48` | `reject` | `long` | 0.00038 | 49.31% | -9.48 bps | Weak and unstable |
| `intraday_sell_pressure_exhaustion_5m_w24` | `watchlist` | `invert` | -0.00262 | 52.93% | -1.71 bps | Short-window bounce behaved negatively after costs |

The persistence transform passed all admission gates:

| feature | admission | direction | rank IC | t-stat | hit rate | spread | cost-adjusted spread | stable years | turnover |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `intraday_sell_pressure_exhaustion_persistence_5m_l96_s24_m48` | `candidate` | `long` | 0.00812 | 19.87 | 56.03% | 11.16 bps | 7.71 bps | 3/3 | 26.53% |

Yearly rank IC remained positive: 2023 `0.00766`, 2024 `0.00739`, and
2025 `0.00930`.

## Standalone Policy Validation

The single-factor primary policy was positive over the full window and under
high costs, but it was not year-stable:

| scenario | policy | return | max drawdown | gross turnover |
|---|---|---:|---:|---:|
| `full_base` | `partial_rebalance_daily` | 16.22% | -34.53% | 97.99 |
| `full_high_cost` | `partial_rebalance_daily` | 11.64% | -35.17% | 97.37 |
| `year_2023_base` | `partial_rebalance_daily` | -4.72% | -16.20% | 33.72 |
| `year_2024_base` | `partial_rebalance_daily` | -9.90% | -29.88% | 40.40 |
| `year_2025_base` | `partial_rebalance_daily` | 21.46% | -14.79% | 39.01 |

`cost_aware_optimizer_daily` is not suitable for this standalone factor. It
lost money in the full window and all annual slices, with high turnover and
large drawdowns.

## Compact-Core Incremental Test

Baseline compact core:

- `intraday_sell_pressure_absorption_5m_w48`
- `intraday_volatility_5m_w6`
- `intraday_amihud_5m`
- `intraday_efficiency_ratio_5m_w48`

Incremental test added
`intraday_sell_pressure_exhaustion_persistence_5m_l96_s24_m48` with
`decorrelated` + `partial_rebalance_daily`.

| scenario | baseline return | plus-persistence return | delta | baseline drawdown | plus-persistence drawdown |
|---|---:|---:|---:|---:|---:|
| `full_base` | 42.92% | 31.36% | -11.56 pp | -32.34% | -30.08% |
| `full_high_cost` | 36.39% | 26.15% | -10.24 pp | -32.48% | -30.13% |
| `year_2023_base` | 14.97% | 8.68% | -6.29 pp | -10.77% | -10.19% |
| `year_2024_base` | -5.96% | -6.50% | -0.53 pp | -28.63% | -28.36% |
| `year_2025_base` | 31.85% | 33.89% | 2.04 pp | -16.43% | -16.24% |

The added factor reduced contribution concentration:

| scenario | baseline largest share | plus-persistence largest share | baseline top-two share | plus-persistence top-two share |
|---|---:|---:|---:|---:|
| `full_base` | 0.533 | 0.491 | 0.906 | 0.846 |
| `year_2024_base` | 0.534 | 0.492 | 0.906 | 0.847 |

However, the reduction in concentration did not translate into enough economic
benefit. The factor improved January and June 2024, but worsened March,
September, and December, leaving the full 2024 slice slightly worse than the
baseline.

| month | baseline return | plus-persistence return | delta |
|---|---:|---:|---:|
| `2024-01` | -14.45% | -13.64% | 0.81 pp |
| `2024-03` | 8.84% | 6.62% | -2.23 pp |
| `2024-06` | -10.03% | -8.03% | 2.00 pp |
| `2024-09` | 16.20% | 14.30% | -1.90 pp |
| `2024-12` | -2.34% | -3.38% | -1.05 pp |

The new factor also has the weakest monitor health in the five-factor run:
full-base average health `0.443`, recommended scale `0.583`, and impaired
count `13,701`.

## Decision

Keep `intraday_sell_pressure_exhaustion_persistence_5m_l96_s24_m48` as a
`candidate`, but do not add it to the compact-core default.

Reasons:

1. The factor has a coherent microstructure explanation and passes formal
   single-factor admission with stable yearly IC.
2. Standalone primary validation is positive in the full window and high-cost
   stress, but 2023 and 2024 are negative.
3. Compact-core inclusion reduces full-base return from 42.92% to 31.36% and
   high-cost return from 36.39% to 26.15%.
4. It improves drawdown and contribution concentration, but not enough to
   offset the return drag.
5. The 2024 failure mode is not solved; the factor helps January and June but
   gives back too much in other months.

## Next Step

Do not keep adding sell-pressure variants as default alpha components unless
they improve the compact-core economics. The next factor-development direction
should target a genuinely different mechanism: either market-state resilience,
limit-pressure resilience, or execution/liquidity stability that protects weak
tape without diluting strong months.
