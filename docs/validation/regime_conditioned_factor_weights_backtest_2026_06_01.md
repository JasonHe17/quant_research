# Regime-Conditioned Factor Weights Backtest - 2026-06-01

This note records the first full-sample backtest of the regime-conditioned
decorrelated factor-weight challenger.

## Artifacts

- Score and dynamic weights:
  `runs/candidate_factor_portfolios/regime_conditioned_factor_weights_2026_06_01_standard/`
- Comparable 0.002 no-trade-band full-window backtests:
  `runs/candidate_factor_portfolios/regime_conditioned_factor_weights_2026_06_01_standard_band002/`
- High-cost and yearly slices:
  `runs/candidate_factor_portfolios/regime_conditioned_factor_weights_2026_06_01_standard_band002_slices/`
- Conservative blend retry:
  `runs/candidate_factor_portfolios/regime_conditioned_factor_weights_blend20_2026_06_01_standard/`
- Local fragile-sleeve shrink retry:
  `runs/candidate_factor_portfolios/regime_stress_fragile_sleeve_shrink50_2026_06_01_standard/`
- Local fragile-sleeve shrink selection-displacement diagnostics:
  `runs/candidate_factor_portfolios/regime_stress_fragile_sleeve_shrink50_2026_06_01_standard/selection_displacement/`
- Comparison tables:
  `runs/candidate_factor_portfolios/regime_conditioned_factor_weights_2026_06_01_standard_band002/benchmark_comparison.csv`
  and
  `runs/candidate_factor_portfolios/regime_conditioned_factor_weights_2026_06_01_standard_band002/stress_month_comparison.csv`

## Setup

- Candidate set: registry-filtered alpha-rank candidates, `19` features.
- Method: `decorrelated`.
- Dynamic weight mode: `state_conditioned_decorrelated`.
- Correlation lookback: `48` score bars, minimum `24`.
- Regime selector: lagged high cross-sectional volatility or weak breadth.
- State lookback: `240` bars, minimum `48`.
- Score transform: cross-sectional rank.
- Row-level factor contribution cap: `25%`.
- Primary policy: `partial_rebalance_daily`, no-trade weight band `0.002`.

Dynamic-weight schedule coverage:

| state | timestamp count |
| --- | ---: |
| normal | 18,933 |
| stress | 15,818 |
| warmup | 48 |

## Benchmark Comparison

| run | full base | high cost | 2023 | 2024 | 2025 | max DD | turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regime-conditioned weights | 14.33% | 9.39% | 0.58% | -7.01% | 4.09% | -27.95% | 118.80 |
| 20% regime / 80% fixed blend | 21.92% | not run | not run | not run | not run | -28.26% | 119.44 |
| stress fragile-sleeve shrink 50% | 25.73% | not run | not run | not run | not run | -27.68% | 121.18 |
| no-overlay control | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | -30.77% | 121.27 |
| state-aware frontier | 30.32% | 24.39% | 1.47% | 0.86% | 18.28% | -28.92% | 117.63 |

The challenger reduces full-window drawdown versus the no-overlay control, but
it gives up too much return and fails the 2024 and 2025 yearly-slice tests.
It also underperforms the state-aware frontier on full return, high-cost return,
2024 return, 2025 return, and drawdown.

## Stress Months

| run | 2024-01 return | 2024-01 DD | 2024-06 return | 2024-06 DD |
| --- | ---: | ---: | ---: | ---: |
| regime-conditioned weights | -9.69% | -12.26% | -9.49% | -10.44% |
| no-overlay control | -11.98% | -13.69% | -8.19% | -9.65% |
| state-aware frontier | -10.85% | -12.63% | -7.55% | -8.85% |

The mechanism helps January 2024 relative to both benchmarks, but worsens June
2024. This is not enough to justify promotion because the full-period and
yearly opportunity cost is large.

## Conservative Blend Retry

The follow-up blended the state-conditioned schedule with the fixed
decorrelated weights:

`final_weight = 0.80 * fixed_decorrelated_weight + 0.20 * regime_weight`.

This repaired much of the full replacement damage, lifting full-window
`partial_rebalance_daily` return from `14.33%` to `21.92%`. It still failed the
research benchmark gate:

- full-window return remains `6.05pp` below the no-overlay control and
  `8.40pp` below the state-aware frontier;
- max drawdown is acceptable at `-28.26%`, but the return give-up is too large;
- `cost_aware_optimizer_daily` remains unusable in this score layer, with
  `-23.13%` full-window return and `653.23` gross turnover.

Blend stress-month behavior:

| run | 2024-01 return | 2024-01 DD | 2024-06 return | 2024-06 DD |
| --- | ---: | ---: | ---: | ---: |
| 20% regime / 80% fixed blend | -11.12% | -12.09% | -8.43% | -10.05% |
| no-overlay control | -11.98% | -13.69% | -8.19% | -9.65% |
| state-aware frontier | -10.85% | -12.63% | -7.55% | -8.85% |

Because the blended schedule still underperforms the control on full return and
does not improve the June 2024 stress month, high-cost and yearly-slice
validation were not run for the blend retry.

## Local Fragile-Sleeve Shrink Retry

To avoid overfitting, this retry used one pre-registered, low-degree rule rather
than a grid search:

- state: the same lagged observable `stress` regime from the dynamic-weight
  schedule;
- action: shrink only volatility / liquidity-pressure / sell-pressure sleeves;
- stress scale: fixed at `0.5`;
- normal and warmup scale: fixed at `1.0`;
- no search over scale, feature subsets, or regime thresholds.

Shrunk features:

- `intraday_adverse_selection_5m_w48`
- `intraday_downside_volatility_5m_w48`
- `intraday_price_efficiency_cost_5m`
- `intraday_range_volatility_5m_w48`
- `intraday_sell_pressure_absorption_5m_w48`
- `intraday_volatility_5m_w6`
- `intraday_volatility_5m_w12`
- `intraday_volatility_5m_w24`
- `intraday_vpin_5m_w48`

The shrink schedule covers `34,799` timestamps and `15,818` stress timestamps.
Primary full-window `partial_rebalance_daily` result:

| run | full base | max DD | turnover | trades | transaction cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| stress fragile-sleeve shrink 50% | 25.73% | -27.68% | 121.18 | 24,302 | 165,757 |
| no-overlay control | 27.97% | -30.77% | 121.27 | 24,030 | 163,936 |
| state-aware frontier | 30.32% | -28.92% | 117.63 | 25,427 | 170,445 |

Stress-month behavior:

| run | 2024-01 return | 2024-01 DD | 2024-06 return | 2024-06 DD |
| --- | ---: | ---: | ---: | ---: |
| stress fragile-sleeve shrink 50% | -11.41% | -12.48% | -8.62% | -10.20% |
| no-overlay control | -11.98% | -13.69% | -8.19% | -9.65% |
| state-aware frontier | -10.85% | -12.63% | -7.55% | -8.85% |

This version improves full-window drawdown but gives up full-window return and
does not improve June 2024. Since the primary gate fails, high-cost and
yearly-slice validation were not run, and no parameter sweep was attempted.

Selection-displacement diagnostics versus the repaired no-overlay control also
fail the pre-backtest state gate:

| state | timestamps | overlap | top-label delta | added minus removed label |
| --- | ---: | ---: | ---: | ---: |
| stress | 15,818 | 46.30% | -0.0106pp | -0.0197pp |
| normal | 18,933 | 51.30% | -0.0040pp | -0.0081pp |
| warmup | 48 | 58.75% | +0.0290pp | +0.0704pp |
| overall | 34,799 | 49.04% | -0.0069pp | -0.0136pp |

The important failure is in the intended enabled state: during stress, the
candidate score stream adds names whose realized forward label is lower than
the names removed from the repaired control basket. Disabled-state behavior is
also not baseline-preserving under this implementation, because normal-state
top-50 overlap is only `51.30%` and replacement quality is negative. This is
exactly why future conditional sleeves should use score-level switching when
the intended disabled behavior is "use the baseline unchanged."

## Overfit Guardrail

Do not tune this family further by scanning stress scales, sleeve subsets, or
state thresholds against the same 2023-2025 full-window result. Any next
candidate should be evaluated in this order:

1. pre-register the state definition, active sleeve, and transformation;
2. run enabled-state selection-displacement diagnostics versus the no-overlay
   control before a full backtest;
3. require positive added-minus-removed label quality in enabled states and no
   silent change in disabled states;
4. only then run full, high-cost, yearly, and stress-month validations against
   both the no-overlay control and state-aware frontier.

## Decision

Do not promote `state_conditioned_decorrelated` factor weights, the 20% blend,
or the stress fragile-sleeve shrink overlay.

Keep the implementation as a research harness because it is useful for future
state-conditioned allocator experiments, but do not replace the fixed
decorrelated alpha-rank control or the state-aware frontier. Any retry should
start with enabled-state selection-displacement evidence and should not use the
full-window backtest as the parameter-selection objective.
