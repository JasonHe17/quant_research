# Compact Core 2024 Weakness Attribution - 2026-05-20

This report attributes the `year_2024_base` warning in the compact-core
validation run.

## Evidence

- Validation summary:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_summary.json`
- Monthly summary:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_monthly_summary.csv`
- Health attribution:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_factor_health_attribution.csv`
- Contribution diagnostics:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/year_2024_base/scores/decorrelated/diagnostics/`

Scope:

- Scenario: `year_2024_base`
- Primary method: `decorrelated`
- Primary policy: `partial_rebalance_daily`
- Factors:
  `intraday_sell_pressure_absorption_5m_w48`,
  `intraday_volatility_5m_w6`, `intraday_amihud_5m`,
  `intraday_efficiency_ratio_5m_w48`

## Summary

The 2024 weakness is not primarily an execution-cost or worker/runtime issue.
Turnover and transaction costs are close to the 2023 and 2025 yearly slices:

| scenario | return | max drawdown | gross turnover | trades | transaction cost |
|---|---:|---:|---:|---:|---:|
| `year_2023_base` | 14.97% | -10.77% | 39.76 | 7,527 | 52,717 |
| `year_2024_base` | -5.96% | -28.63% | 40.21 | 7,928 | 51,724 |
| `year_2025_base` | 31.85% | -16.43% | 35.85 | 7,038 | 49,996 |

The loss is concentrated in two stress months:

| excluded months | compound return | months kept |
|---|---:|---:|
| none | -5.96% | 12 |
| `2024-01` | 6.42% | 11 |
| `2024-06` | 6.12% | 11 |
| `2024-01`, `2024-06` | 20.09% | 10 |

January and June are therefore the first failure modes to solve. The rest of
the year is positive in aggregate.

## Monthly Path

| month | return | max drawdown | trades | transaction cost | cost bps on notional |
|---|---:|---:|---:|---:|---:|
| `2024-01` | -11.64% | -12.92% | 580 | 4,436 | 9.89 |
| `2024-02` | -2.03% | -16.27% | 592 | 3,708 | 17.34 |
| `2024-03` | 2.89% | -4.26% | 696 | 4,562 | 14.80 |
| `2024-04` | -0.60% | -11.51% | 663 | 4,275 | 15.58 |
| `2024-05` | -1.52% | -4.50% | 639 | 4,178 | 14.90 |
| `2024-06` | -11.38% | -12.48% | 648 | 4,122 | 16.31 |
| `2024-07` | -0.24% | -7.94% | 781 | 4,897 | 17.28 |
| `2024-08` | -2.15% | -6.32% | 700 | 4,410 | 16.95 |
| `2024-09` | 17.62% | -16.24% | 533 | 3,364 | 16.89 |
| `2024-10` | 4.34% | -12.42% | 649 | 4,182 | 15.72 |
| `2024-11` | 5.64% | -7.90% | 740 | 4,885 | 14.45 |
| `2024-12` | -3.83% | -12.03% | 707 | 4,705 | 14.11 |

## Policy Readout

All methods and policies lose money in January and June. The issue is therefore
not unique to the `decorrelated` scorer or the `partial_rebalance_daily`
execution policy.

| method | policy | 2024 return | return excluding Jan and Jun | Jan | Jun |
|---|---|---:|---:|---:|---:|
| `ic_weighted` | `cost_aware_optimizer_daily` | 0.61% | 23.79% | -11.02% | -8.66% |
| `decorrelated` | `cost_aware_optimizer_daily` | -0.40% | 23.56% | -10.43% | -10.01% |
| `equal` | `cost_aware_optimizer_daily` | -0.61% | 21.28% | -11.41% | -7.49% |
| `equal` | `partial_rebalance_daily` | -2.42% | 25.38% | -13.25% | -10.29% |
| `ic_weighted` | `partial_rebalance_daily` | -3.36% | 24.06% | -12.91% | -10.56% |
| `decorrelated` | `partial_rebalance_daily` | -5.96% | 20.09% | -11.64% | -11.38% |

`cost_aware_optimizer_daily` reduces the 2024 base loss, but it should not be
promoted to default from this evidence alone. In the full-window high-cost
stress test, it failed high-cost robustness for all three weighting methods.

## Signal Attribution

The primary score is highly concentrated. For the 2024 primary run,
`intraday_sell_pressure_absorption_5m_w48` is the largest contribution feature
in all `11,033` observed bars. Its average largest absolute contribution share
is `0.534`, and the average top-two contribution share is about `0.906`.

Loss months differ most clearly in the sell-pressure absorption factor:

| feature | loss months | avg health in winning months | avg health in loss months | largest contribution count in wins | largest contribution count in losses | avg rank IC in wins | avg rank IC in losses | avg spread in wins | avg spread in losses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `intraday_sell_pressure_absorption_5m_w48` | 8 | 0.619 | 0.491 | 3,646 | 7,387 | 0.0852 | 0.0356 | 0.00503 | -0.00120 |
| `intraday_amihud_5m` | 8 | 0.558 | 0.482 | 0 | 0 | 0.0335 | 0.0137 | 0.00197 | 0.00150 |
| `intraday_efficiency_ratio_5m_w48` | 8 | 0.520 | 0.516 | 0 | 0 | 0.0112 | 0.0172 | 0.00435 | 0.00633 |
| `intraday_volatility_5m_w6` | 8 | 0.552 | 0.567 | 0 | 0 | 0.0291 | 0.0303 | 0.00598 | 0.00467 |

The sell-pressure factor keeps dominating the score while its realized
top-minus-bottom label spread deteriorates in loss months. Amihud also weakens,
but it is not the contribution driver. Volatility and efficiency ratio do not
explain the loss-month split as directly.

Selected stress-month diagnostics:

| month | month return | sell-pressure health | sell-pressure spread | combined top-score mean label | negative top-score label share | largest contribution share |
|---|---:|---:|---:|---:|---:|---:|
| `2024-01` | -11.64% | 0.536 | -0.00172 | -0.00297 | 51.1% | 0.534 |
| `2024-06` | -11.38% | 0.409 | -0.00355 | -0.00495 | 62.3% | 0.533 |
| `2024-09` | 17.62% | 0.521 | 0.00353 | 0.00257 | 48.8% | 0.528 |

This supports a signal-state explanation: the compact core is profitable when
the dominant sell-pressure absorption signal has positive realized spread, but
the portfolio remains too exposed when that spread turns negative.

## Decisions

1. Keep `decorrelated` + `partial_rebalance_daily` as the compact-core baseline.
   The 2024 issue is real, but it is concentrated and not caused by abnormal
   turnover or transaction costs.

2. Do not switch the default to `cost_aware_optimizer_daily`. It helps the 2024
   base slice, but full-window high-cost robustness remains unacceptable.

3. Treat sell-pressure contribution concentration as the main compact-core risk.
   Any next factor must be evaluated by whether it reduces Jan/Jun losses or
   lowers contribution concentration without breaking full-window high-cost
   return.

4. The first incremental addition test is complete:
   `intraday_volume_confirmed_momentum_5m_w48` should not be added to the
   compact-core baseline. It modestly reduced contribution concentration and
   improved `year_2024_base` by only `0.79 pp`, but full-base return fell from
   `42.92%` to `35.58%` and full high-cost return fell from `36.39%` to
   `29.43%`. See
   `docs/validation/compact_core_plus_volume_confirmed_momentum_2026_05_20.md`.

5. The generic overlay experiment is complete. Contribution caps at `0.45` and
   `0.50` reduced measured concentration but diluted full-window and high-cost
   return too much. Health shrink preserved full-window robustness but left
   `year_2024_base` negative at `-5.79%` and increased the 2024 largest
   contribution share from `0.534` to `0.657`. See
   `docs/validation/compact_core_overlay_experiments_2026_05_21.md`.

6. Keep health-based shrinkage as a controlled experiment, not a framework
   default. The next experiment should target sell-pressure concentration with
   a factor-specific regime guard or variant based on lagged sell-pressure
   spread/health state, while the production framework remains in monitor mode.

7. The first factor-specific regime guard is also complete. It confirmed that
   lagged sell-pressure state can reduce the January and June failure modes,
   but the current rule over-cuts exposure in profitable regimes. It improved
   `year_2024_base` from `-5.96%` to `-1.96%`, while reducing full-base return
   from `42.92%` to `33.52%`. See
   `docs/validation/compact_core_sell_pressure_regime_guard_2026_05_21.md`.
