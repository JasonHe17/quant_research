# Factor Research Batch 2026-05-17

This note records a controlled single-factor expansion for `intraday_negative_return_persistence_5m_w48`.

## Hypothesis

The factor measures the rolling 48-bar share of 5-minute close-to-close returns that are negative. The expected use is inverted: persistent negative prints indicate weak intraday tape quality and should penalize long-only selection.

The design intentionally differs from the 2026-05-16 risk and momentum watchlist factors:

- It ignores return magnitude, unlike downside volatility and risk-adjusted momentum.
- It ignores distribution shape beyond sign frequency, unlike return skewness.
- It ignores turnover participation, unlike signed turnover imbalance and return-turnover correlation.

## Pre-Development Memory Check

Artifact: `runs/factor_research_memory/intraday_negative_return_persistence_5m_w48/factor_research_memory_check.json`.

Status was `warn`, with no blocking rejected/deprecated match. Warnings were expected because the factor is in the risk family and shares the 48-bar close-return input with existing watchlist factors.

## Implementation

- Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`.
- Factor group: `negative_return_persistence`.
- Feature column: `intraday_negative_return_persistence_5m_w48`.
- Inputs: `instrument_id`, `bar_end_time`, `close_price`.
- Registry: `configs/factors/factor_registry.json` version 4.

## Standard Single-Factor Validation

Artifacts:

| artifact | path |
| --- | --- |
| Alpha dataset | `runs/framework_v1_acceptance/factor_batch_2026_05_17_negative_return_persistence/alpha_dataset` |
| Factor evaluation | `runs/framework_v1_acceptance/factor_batch_2026_05_17_negative_return_persistence/factor_evaluation` |
| Admission report | `runs/framework_v1_acceptance/factor_batch_2026_05_17_negative_return_persistence/factor_admission/factor_admission_report.json` |
| Candidate review | `runs/factor_candidate_reviews/intraday_negative_return_persistence_5m_w48/factor_candidate_review.json` |

Dataset coverage: 36 monthly partitions from 2023-01-03 09:35 +08:00 through 2025-12-31 15:00 +08:00, 103,351,780 joined feature-label rows, ST exclusion enabled, 48-bar forward-return label, one-bar entry lag, and price-limit-aware entry filtering.

## Admission Result

| factor_id | status | direction | rank_ic | t_stat | hit_rate | stable_years | cost_adj_spread | top_n_turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| intraday_negative_return_persistence_5m_w48 | watchlist | invert | -0.029490 | -37.96 | 0.5935 | 3 | -0.002031 | 0.1808 |

Yearly rank IC was stable and negative in all observed years: 2023 `-0.037845`, 2024 `-0.032422`, 2025 `-0.018239`.

## Risk-Gate Follow-Up

Because the factor failed standalone economic admission but has a clean risk interpretation, it was tested as a lagged gross-exposure gate on the promoted standard decorrelated policy. The same cost-aware optimizer, 48-bar rebalance cadence, path turnover budget, ST exclusion, price-limit filter, and scenario validation gates were used. Backtests used the framework's scenario parallelism with `--scenario-workers 2`.

Artifacts:

| artifact | path |
| --- | --- |
| Standalone gate schedule | `runs/candidate_factor_portfolios/negative_return_persistence_w48_risk_gate_v1/summary.json` |
| Standalone gate validation | `runs/candidate_factor_portfolios/negative_return_persistence_w48_risk_gate_v1_promoted_standard/validation_summary.json` |
| Combined gate schedule | `runs/candidate_factor_portfolios/negative_return_persistence_w48_plus_downside_volatility_gate_v1/summary.json` |
| Combined gate validation | `runs/candidate_factor_portfolios/negative_return_persistence_w48_plus_downside_volatility_gate_v1_promoted_standard/validation_summary.json` |
| Drawdown-guard schedule | `runs/candidate_factor_portfolios/negative_return_persistence_w48_plus_downside_volatility_gate_v2_drawdown_guard/summary.json` |
| Drawdown-guard validation | `runs/candidate_factor_portfolios/negative_return_persistence_w48_plus_downside_volatility_gate_v2_drawdown_guard_promoted_standard/validation_summary.json` |
| Budget-deadband recheck | `runs/candidate_factor_portfolios/budget_deadband_regime_gate_promoted_standard_recheck/validation_summary.json` |
| Policy drawdown brake validation | `runs/candidate_factor_portfolios/negative_return_persistence_w48_plus_downside_volatility_gate_v1_drawdown_brake_7pct_promoted_standard/validation_summary.json` |

The standalone negative-return-persistence gate produced 34,799 schedule observations: 26,655 `full`, 4,713 `reduced`, 3,383 `blocked`, and 48 `warmup`.

| policy | full return | full max DD | gross turnover | high-cost return | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ungated promoted baseline | 32.21% | -8.49% | 156.37 | 20.31% | 7.97% | 12.18% | 5.10% |
| Downside-volatility gate | 38.77% | -6.96% | 149.96 | 27.31% | 10.44% | 15.68% | 7.05% |
| Negative-return-persistence gate | 36.16% | -7.75% | 151.49 | 24.02% | 6.59% | 12.24% | 9.14% |
| Negative-return-persistence + downside-volatility gate | 40.73% | -8.00% | 147.87 | 29.19% | 9.12% | 16.47% | 9.15% |
| Drawdown-guard combined gate | 39.26% | -8.31% | 146.87 | 27.88% | 8.78% | 16.50% | 8.34% |
| Budget-deadband regime gate recheck | 32.21% | -8.49% | 156.37 | 20.31% | 7.97% | 12.18% | 5.10% |
| V1 gate + policy drawdown brake, -7% to 0.5x | 40.73% | -8.00% | 147.87 | 29.19% | 9.12% | 16.47% | 9.15% |
| V1 gate + policy drawdown brake, -7% to 0.5x, week chunks | 41.75% | -6.49% | 147.29 | 28.46% | 9.89% | 16.47% | 9.15% |

Interpretation: the new gate is not better than the downside-volatility gate by itself. As a second-stage overlay combined by minimum gross-exposure scale, it adds return in the full and high-cost scenarios and improves 2024-2025 results, but gives up drawdown control and underperforms downside volatility in 2023. This is useful interaction evidence, not enough to promote the factor as standalone alpha.

A more conservative drawdown-guard variant used earlier thresholds and heavier reduction: `high_quantile=0.75`, `extreme_quantile=0.90`, `reduced_scale=0.35`, `blocked_scale=0.0`. It passed all validation checks, but increased blocked observations from 3,383 to 5,094 without improving drawdown. Full-window return fell from 40.73% to 39.26%, max drawdown worsened from -8.00% to -8.31%, high-cost return fell from 29.19% to 27.88%, and 2023 max drawdown worsened to -8.31%. This rules out simple threshold tightening as the next drawdown-control path.

A score-health budget-deadband regime gate was also rechecked as a different mechanism. Directly combining it with the v1 factor gate by `min` had no effect because the factor gate was already never above the regime gate on aligned timestamps. Used alone, the budget-deadband schedule passed validation but reproduced the ungated promoted baseline exactly: 32.21% full-window return, -8.49% max drawdown, 20.31% high-cost return, and 156.37 gross turnover. This rules out the existing budget-deadband schedule as a replacement drawdown controller for the current promoted policy.

A policy-native drawdown brake was then added to the score backtest path and validation CLI. It computes realized path drawdown from prior streaming chunks and caps gross exposure when the drawdown is below a configured threshold. The first validation used the v1 combined gate plus `policy_drawdown_brake_threshold=-0.07` and `policy_drawdown_brake_reduced_scale=0.5`. It passed all validation checks, but produced exactly the same path as the v1 gate because the current implementation updates the brake at monthly streaming-chunk boundaries and the relevant drawdown happens inside the month. This validates the interface but not the risk-control objective; the next retry must make the brake decision at rebalance timestamps or daily chunks.

To probe that limitation, the streaming backtest path was extended to support `week` chunks. A 2023-2025 standard validation with `--streaming-chunk week --streaming-chunk-padding-days 0` passed all checks and improved the combined-gate drawdown to -6.49% while keeping gross turnover at 147.29. The same setup also passed the high-cost stress test with 28.46% return and -7.64% max drawdown. The day-chunk path was also prototyped, but it was too slow to use as a practical validation route.

## Final Governance Decision

The factor is moved to `watchlist` with `decision_reason=cost_fragile`. It is statistically strong but not economically admissible as a standalone trading alpha because the inverted top-minus-bottom spread is negative after the standard 13 bps cost proxy.

The risk-gate follow-up changes the next step but not the standalone decision: keep `watchlist`, record `portfolio_validation_status=risk_gate_incremental_watchlist`, and only retry with a rebalance-level drawdown brake or an added liquidity/execution condition. Simple quantile-threshold tightening, the existing budget-deadband regime schedule, and the monthly-chunk drawdown brake were tested and rejected.
