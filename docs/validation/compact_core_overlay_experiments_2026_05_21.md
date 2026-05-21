# Compact Core Overlay Experiments - 2026-05-21

This report validates generic score-construction overlays for the four-factor
compact core after the 2024 weakness attribution.

## Evidence

- Baseline:
  `runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/validation_summary.json`
- Contribution cap 0.45:
  `runs/candidate_factor_portfolios/compact_core_contribution_cap_045_2026_05_21_standard/validation_summary.json`
- Contribution cap 0.50:
  `runs/candidate_factor_portfolios/compact_core_contribution_cap_050_2026_05_21_standard/validation_summary.json`
- Health shrink:
  `runs/candidate_factor_portfolios/compact_core_factor_health_shrink_2026_05_21_standard/validation_summary.json`
- Prior attribution:
  `docs/validation/compact_core_2024_attribution_2026_05_20.md`

All four runs completed with validation status `warn`, zero failed checks, and
one warning: `primary_yearly_base_positive_returns` because `year_2024_base`
remains negative.

Scope:

- Factors: `intraday_sell_pressure_absorption_5m_w48`,
  `intraday_volatility_5m_w6`, `intraday_amihud_5m`,
  `intraday_efficiency_ratio_5m_w48`
- Primary method: `decorrelated`
- Primary policy: `partial_rebalance_daily`
- Profile: `standard`

## Primary Comparison

| run | full base | full high cost | 2023 | 2024 | 2025 | max drawdown, full base |
|---|---:|---:|---:|---:|---:|---:|
| baseline monitor | 42.92% | 36.39% | 14.97% | -5.96% | 31.85% | -32.34% |
| contribution cap 0.45 | 34.42% | 28.85% | 7.99% | -4.21% | 30.56% | -29.68% |
| contribution cap 0.50 | 32.05% | 26.39% | 9.11% | -6.19% | 27.83% | -29.45% |
| health shrink | 42.79% | 37.14% | 11.00% | -5.79% | 42.35% | -31.84% |

The contribution caps reduce drawdown and measured concentration, but they
materially dilute full-window and high-cost return. The 0.45 cap improves 2024
by only `1.75 pp` while giving up `8.50 pp` of full-base return and `7.54 pp`
of high-cost return. The 0.50 cap is worse than the baseline in 2024 and also
dilutes full-window returns.

Health shrink is better than the caps on full-window robustness. It is nearly
flat to baseline in full-base return, improves full high-cost return, and helps
2025. It still does not solve the target failure mode: `year_2024_base` remains
negative at `-5.79%`, only `0.17 pp` better than baseline.

## 2024 Stress Months

Primary method and policy, isolated `year_2024_base` scenario:

| month | baseline | cap 0.45 | cap 0.50 | health shrink |
|---|---:|---:|---:|---:|
| `2024-01` | -11.64% | -11.83% | -11.86% | -11.49% |
| `2024-02` | -2.03% | -0.44% | -2.11% | -2.38% |
| `2024-03` | 2.89% | 3.02% | 3.73% | 3.68% |
| `2024-04` | -0.60% | -0.19% | -0.42% | -0.73% |
| `2024-05` | -1.52% | 0.08% | -0.15% | -0.37% |
| `2024-06` | -11.38% | -11.44% | -11.31% | -10.48% |
| `2024-07` | -0.24% | 0.27% | 0.27% | -0.34% |
| `2024-08` | -2.15% | -2.86% | -2.89% | -2.05% |
| `2024-09` | 17.62% | 17.20% | 17.05% | 15.37% |
| `2024-10` | 4.34% | 4.37% | 4.28% | 5.99% |
| `2024-11` | 5.64% | 4.57% | 4.93% | 4.82% |
| `2024-12` | -3.83% | -3.89% | -4.71% | -5.07% |

None of the generic overlays neutralizes January and June. Health shrink
slightly improves both stress months, but not enough to change the annual
failure state.

## Cost-Aware Readout

| run | full base | full high cost | 2024 |
|---|---:|---:|---:|
| baseline monitor | 29.54% | -10.31% | -0.40% |
| contribution cap 0.45 | -1.21% | -41.57% | -7.65% |
| contribution cap 0.50 | 5.11% | -29.32% | -5.09% |
| health shrink | 31.51% | -2.99% | 0.83% |

The contribution caps are especially poor under `cost_aware_optimizer_daily`.
Health shrink improves the cost-aware branch, but full high-cost remains
negative, so this is not a default-policy replacement.

## Contribution And Health

| run | scenario | avg largest contribution | max largest contribution | avg top-two contribution |
|---|---|---:|---:|---:|
| baseline monitor | `full_base` | 0.533 | 1.000 | 0.906 |
| baseline monitor | `year_2024_base` | 0.534 | 0.667 | 0.906 |
| contribution cap 0.45 | `full_base` | 0.447 | 0.550 | 0.881 |
| contribution cap 0.45 | `year_2024_base` | 0.447 | 0.457 | 0.881 |
| contribution cap 0.50 | `full_base` | 0.482 | 0.500 | 0.901 |
| contribution cap 0.50 | `year_2024_base` | 0.482 | 0.500 | 0.901 |
| health shrink | `full_base` | 0.633 | 1.000 | 0.902 |
| health shrink | `year_2024_base` | 0.657 | 0.887 | 0.901 |

The caps reduce concentration by construction. The problem is that the return
cost is too high. Health shrink does not reduce contribution concentration; it
raises the average largest-contribution share in 2024 from `0.534` to `0.657`.

Health shrink did apply real score scaling. In the full-base run, average
weight scales were `0.667` for sell-pressure absorption, `0.704` for
volatility, `0.639` for Amihud, and `0.640` for efficiency ratio. Minimum
scales reached `0.25` for all four factors.

In `year_2024_base`, sell-pressure absorption had average weight scale `0.655`
and `5,710` lagged shrink observations. The overlay therefore acted on the
dominant factor, but it did not remove the 2024 loss.

## Decision

1. Reject generic contribution caps for compact-core production use. They
   reduce concentration, but the full-window and high-cost return cost is too
   large.

2. Do not switch the production framework from `factor_health_mode=monitor` to
   `factor_health_mode=shrink`. Shrink is useful evidence and may remain a
   controlled score-construction branch, but it does not solve 2024 and it
   worsens contribution concentration.

3. Keep the compact-core baseline as `decorrelated` +
   `partial_rebalance_daily` with health monitoring only.

4. The next research branch should be targeted rather than generic: build a
   sell-pressure-specific regime guard or factor variant that reacts to lagged
   sell-pressure spread/health state, and validate it against January and June
   2024 before broad factor discovery resumes.

## Follow-Up

The first targeted sell-pressure regime guard is complete. It improved the
2024 primary slice from `-5.96%` to `-1.96%`, but full-base return fell to
`33.52%` and full high-cost return fell to `27.88%`. Keep it as useful
research evidence, not a production overlay. See
`docs/validation/compact_core_sell_pressure_regime_guard_2026_05_21.md`.
