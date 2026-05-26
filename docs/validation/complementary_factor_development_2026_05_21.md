# Complementary Factor Development - 2026-05-21

Status note, 2026-05-26: this is a historical development review. Phrases such
as "current leading equal annual-budget-52" refer to the comparison baseline
used during the 2026-05-21 review, not to the latest research frontier. For
current baseline hierarchy and frontier selection, use
`docs/validation/factor_development_standard.md`.

This report records three targeted factor-development branches after the
compact-core and sell-pressure persistence reviews:

- `intraday_breadth_shock_residual_resilience_5m_w{24,48,96}`
- `intraday_turnover_stability_5m_w{24,48,96}`
- `intraday_liquidity_reliability_5m_w{24,48,96}`
- `intraday_liquidity_reliability_recovery_5m_l{48,96}_c{12,24}_r{24,48}`
- `intraday_liquidity_reliability_recovery_balance_5m_l{48,96}_c{12,24}_r{24,48}`

The objective was not to add a superficially complementary feature. Each branch
was tested as a potentially tradable mechanism with admission and policy-level
evidence.

## Breadth-Shock Residual Resilience

Formula:

`rolling_sum((return - market_median_return) * stress_weight) / rolling_sum(stress_weight)`

where `stress_weight = max(0.5 - up_rate, 0) + max(lagged_up_rate_mean - up_rate, 0)`.

The hypothesis was that stocks with positive residual returns during weak
breadth shocks may be more robust long-only holdings. The evidence rejected that
interpretation.

Artifacts:

- Dataset: `runs/factor_research/breadth_shock_residual_resilience_2026_05_21/alpha_dataset`
- Evaluation: `runs/factor_research/breadth_shock_residual_resilience_2026_05_21/factor_evaluation`
- Admission: `runs/factor_research/breadth_shock_residual_resilience_2026_05_21/factor_admission/factor_admission_report.json`

Admission result:

| feature | status | direction | rank IC | hit rate | stable years | cost-adjusted spread |
|---|---|---|---:|---:|---:|---:|
| `intraday_breadth_shock_residual_resilience_5m_w24` | `reject` | `invert` | -0.00252 | 49.70% | 2 | 7.65 bps |
| `intraday_breadth_shock_residual_resilience_5m_w48` | `reject` | `invert` | -0.00237 | 49.31% | 1 | 17.63 bps |
| `intraday_breadth_shock_residual_resilience_5m_w96` | `reject` | `invert` | -0.00343 | 49.21% | 1 | 41.38 bps |

The direction was unstable across years. The strongest rejection signal was the
2025 reversal versus 2023-2024: the factor looked like names that resisted weak
breadth shocks and then caught down later. No portfolio validation was run.

Decision: keep as `reject`; do not retry as another plain breadth-resilience
variant.

## Turnover Stability

Formula:

`mean(log1p(turnover)) / std(log1p(turnover))`

The hypothesis was that stable turnover supply represents stable liquidity
participation and should improve execution quality. The factor did not produce a
clean linear alpha.

Artifacts:

- Research memory: `runs/factor_research_memory/intraday_turnover_stability_5m_w48/factor_research_memory_check.json`
- Dataset: `runs/factor_research/turnover_stability_2026_05_21/alpha_dataset`
- Evaluation: `runs/factor_research/turnover_stability_2026_05_21/factor_evaluation`
- Admission: `runs/factor_research/turnover_stability_2026_05_21/factor_admission/factor_admission_report.json`
- Full-base policy evidence: `runs/factor_research/turnover_stability_2026_05_21/policy_validation_standard/full_base/backtest_summary.csv`

Admission result:

| feature | status | direction | rank IC | hit rate | stable years | failed checks |
|---|---|---|---:|---:|---:|---|
| `intraday_turnover_stability_5m_w24` | `watchlist` | `invert` | -0.01519 | 56.25% | 3 | `cost_adjusted_spread` |
| `intraday_turnover_stability_5m_w48` | `watchlist` | `invert` | -0.00813 | 53.80% | 3 | `cost_adjusted_spread` |
| `intraday_turnover_stability_5m_w96` | `watchlist` | `invert` | -0.00736 | 53.32% | 3 | `cost_adjusted_spread` |

The quantile and top-N evidence was non-linear: rank IC selected the inverted
direction, while top-N and quintile results did not support a clean monotonic
long/short ordering.

Full-base policy evidence:

| method | policy | return | max drawdown | gross turnover |
|---|---|---:|---:|---:|
| `decorrelated` | `partial_rebalance_daily` | 45.16% | -34.36% | 119.61 |
| `equal` | `partial_rebalance_daily` | 48.30% | -32.97% | 118.63 |
| `ic_weighted` | `partial_rebalance_daily` | 46.40% | -33.06% | 119.08 |
| `decorrelated` | `cost_aware_optimizer_daily` | -76.11% | -77.42% | 324.87 |
| `equal` | `cost_aware_optimizer_daily` | -75.37% | -77.09% | 265.08 |
| `ic_weighted` | `cost_aware_optimizer_daily` | -75.47% | -77.22% | 281.30 |

Decision: keep as `watchlist` for future non-linear liquidity-state research,
but do not use the current ratio expression in the compact core or as a
standalone tradable factor.

## Liquidity Reliability

Formula:

`mean(log1p(turnover)) - std(log1p(turnover))`

This was designed as a conservative lower bound on liquidity supply. The
original long-side hypothesis failed: high reliability underperformed. The
admitted signal was the inverted side, which is better interpreted as a
low-reliability or smaller-name risk premium rather than execution quality.

Artifacts:

- Research memory: `runs/factor_research_memory/intraday_liquidity_reliability_5m_w48/factor_research_memory_check.json`
- Dataset: `runs/factor_research/liquidity_reliability_2026_05_21/alpha_dataset`
- Evaluation: `runs/factor_research/liquidity_reliability_2026_05_21/factor_evaluation`
- Admission: `runs/factor_research/liquidity_reliability_2026_05_21/factor_admission/factor_admission_report.json`
- Quick policy validation: `runs/factor_research/liquidity_reliability_2026_05_21/policy_validation_quick/validation_summary.json`

Admission result:

| feature | status | direction | rank IC | t-stat | hit rate | stable years | cost-adjusted spread |
|---|---|---|---:|---:|---:|---:|---:|
| `intraday_liquidity_reliability_5m_w24` | `candidate` | `invert` | -0.05998 | -65.08 | 66.01% | 3 | 11.56 bps |
| `intraday_liquidity_reliability_5m_w48` | `candidate` | `invert` | -0.05608 | -60.27 | 64.68% | 3 | 7.48 bps |
| `intraday_liquidity_reliability_5m_w96` | `candidate` | `invert` | -0.05333 | -56.91 | 63.84% | 3 | 4.20 bps |

The windows are highly correlated, so `w24` is the primary research member.
The factor correlation between `w24` and `w48` is `0.964`, and between `w48`
and `w96` is `0.975`.

Quick full-base policy evidence:

| method | policy | return | max drawdown | gross turnover |
|---|---|---:|---:|---:|
| `decorrelated` | `partial_rebalance_daily` | 46.01% | -34.49% | 119.25 |
| `decorrelated` | `cost_aware_optimizer_daily` | -33.91% | -43.53% | 248.78 |

Decision: keep `intraday_liquidity_reliability_5m_w24` on `watchlist`, not
`candidate`, despite formal admission. The inverted signal is statistically
strong, but the strategy evidence says it is not cost robust. It should not be
added to the compact core until a capacity-aware transform or portfolio policy
passes cost-aware validation and annual stability checks.

## Liquidity-Reliability Recovery

Formula:

`-(mean(log1p(turnover)) - std(log1p(turnover))) * capacity_gate * log1p(recovery_ratio)`

where `capacity_gate` uses recent log-turnover capacity relative to its longer
history and `recovery_ratio` is rolling positive return divided by rolling
downside return.

The hypothesis was that the low-liquidity reliability premium becomes tradable
only after recent capacity and price recovery confirm that sellers are
exhausted. This first version improved IC, but still failed the cost-adjusted
spread gate.

Artifacts:

- Research memory: `runs/factor_research_memory/intraday_liquidity_reliability_recovery_5m_l48_c12_r24/factor_research_memory_check.json`
- Dataset: `runs/factor_research/liquidity_reliability_recovery_2026_05_21/alpha_dataset`
- Evaluation: `runs/factor_research/liquidity_reliability_recovery_2026_05_21/factor_evaluation`
- Admission: `runs/factor_research/liquidity_reliability_recovery_2026_05_21/factor_admission/factor_admission_report.json`

Admission result:

| feature | status | direction | rank IC | t-stat | hit rate | stable years | cost-adjusted spread |
|---|---|---|---:|---:|---:|---:|---:|
| `intraday_liquidity_reliability_recovery_5m_l48_c12_r24` | `watchlist` | `long` | 0.02631 | 33.78 | 60.12% | 3 | -11.05 bps |
| `intraday_liquidity_reliability_recovery_5m_l96_c24_r48` | `watchlist` | `long` | 0.03413 | 40.29 | 61.83% | 3 | -14.66 bps |

The key diagnostic was the quantile shape: the second through fourth quintiles
were positive, but the top quintile was slightly negative. That is consistent
with a game-theoretic crowding interpretation: moderate recovery after impaired
liquidity can mark seller exhaustion, while extreme recovery can mark a crowded
rebound, adverse selection, or short-horizon overextension.

Decision: keep as `watchlist` and do not portfolio-test this monotonic recovery
version. It is useful mainly because it identified the non-linear shape that
the next factor must model explicitly.

## Liquidity-Reliability Recovery Balance

Formula:

`softplus(std(log1p(turnover)) - mean(log1p(turnover))) * capacity_balance * recovery_balance`

where:

- `capacity_balance = 2 * relative_capacity / (1 + relative_capacity^2)`
- `recovery_balance = 2 * recovery_ratio / (1 + recovery_ratio^2)`

The financial interpretation is deliberately non-linear. Low liquidity
reliability still represents a risk premium, but it is tradable only when recent
turnover capacity is not absent or excessively spiking and when recovery is
credible but not overextended. The balance terms are smooth saturating functions
rather than framework-level alpha clipping.

Artifacts:

- Research memory: `runs/factor_research_memory/intraday_liquidity_reliability_recovery_balance_5m_l48_c12_r24/factor_research_memory_check.json`
- Dataset: `runs/factor_research/liquidity_reliability_recovery_balance_2026_05_21/alpha_dataset`
- Evaluation: `runs/factor_research/liquidity_reliability_recovery_balance_2026_05_21/factor_evaluation`
- Admission: `runs/factor_research/liquidity_reliability_recovery_balance_2026_05_21/factor_admission/factor_admission_report.json`
- Standard policy validation: `runs/factor_research/liquidity_reliability_recovery_balance_2026_05_21/policy_validation_standard/validation_summary.json`

Admission result:

| feature | status | direction | rank IC | t-stat | hit rate | stable years | cost-adjusted spread | top-N turnover |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `intraday_liquidity_reliability_recovery_balance_5m_l48_c12_r24` | `candidate` | `long` | 0.05908 | 65.11 | 65.42% | 3 | 19.33 bps | 7.30% |
| `intraday_liquidity_reliability_recovery_balance_5m_l96_c24_r48` | `candidate` | `long` | 0.05618 | 61.08 | 64.44% | 3 | 16.26 bps | 4.30% |

The quintile profile became monotonic: for the 48-bar member, mean forward
return increased from -7.01 bps in quintile 1 to 9.39 bps in quintile 5. The
96-bar member showed the same shape, from -6.26 bps to 8.72 bps.

Standard policy evidence:

| scenario | policy | return | max drawdown | gross turnover |
|---|---|---:|---:|---:|
| full base | `partial_rebalance_daily` | 55.60% | -32.31% | 117.02 |
| full base | `cost_aware_optimizer_daily` | 99.75% | -21.95% | 164.36 |
| full high cost | `partial_rebalance_daily` | 48.36% | -32.45% | 116.69 |
| full high cost | `cost_aware_optimizer_daily` | 83.36% | -22.12% | 163.73 |
| 2023 base | `partial_rebalance_daily` | 13.12% | -8.55% | 38.45 |
| 2023 base | `cost_aware_optimizer_daily` | 20.91% | -7.32% | 46.21 |
| 2024 base | `partial_rebalance_daily` | 0.22% | -26.75% | 39.52 |
| 2024 base | `cost_aware_optimizer_daily` | 11.45% | -22.76% | 51.67 |
| 2025 base | `partial_rebalance_daily` | 43.29% | -15.43% | 39.24 |
| 2025 base | `cost_aware_optimizer_daily` | 47.40% | -12.80% | 63.32 |

Initial decision: register as `candidate`, not `promoted`. It is the first
branch in this complementary-factor run that passes admission, cost-aware
full-base, high-cost stress, and positive annual primary-return checks. The
remaining risk is the weak 2024 partial-rebalance slice and elevated drawdown,
so promotion should wait for candidate review, combination with the existing
compact core, and a drawdown/risk gate review.

Follow-up integration review on 2026-05-22 changed the lifecycle decision to
`watchlist`: direct inclusion in the then-leading equal annual-budget-52
portfolio was strongly dilutive. Adding both balance windows returned 4.49%
with -25.11% max drawdown, and adding only the l48 window returned 2.80% with
max drawdown of -22.85%, versus the then-leading baseline 33.44% with -7.85% max
drawdown. See
`docs/validation/liquidity_reliability_recovery_balance_integration_2026_05_22.md`.

## Overall Decision

Do not add any of these factors directly to the default compact-core strategy
yet.

The useful results are:

1. Plain market breadth resilience remains unreliable.
2. Stable turnover as a ratio is non-linear and cost fragile.
3. Low liquidity reliability carries a strong risk premium, but direct trading
   fails cost-aware constraints.
4. Monotonic low-liquidity recovery is still cost fragile because the top bucket
   behaves like overextended recovery.
5. Balanced low-liquidity recovery is a viable standalone research signal, but
   it is not a viable default `candidate` after portfolio integration review.
   It should remain on `watchlist` until a portfolio-level allocator or gate
   proves incremental value versus the then-leading baseline.

The next step should not be another local recovery-balance variant. Future
factor work should require immediate incremental validation against the leading
portfolio, so strong standalone IC does not get mistaken for portfolio value.
