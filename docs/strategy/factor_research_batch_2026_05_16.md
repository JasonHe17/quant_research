# Factor Research Batch 2026-05-16

This note records the second controlled factor expansion. The batch adds
point-in-time 5-minute OHLCV transforms and a screening run before any formal
promotion decision.

## Candidate Set

All candidates use live-available 5-minute bars and keep the current A-share
research assumptions: long-only portfolio use, T+1-safe labels, ST-aware
universe filtering, and price-limit-aware entry filtering.

| factor_id | family | hypothesis |
| --- | --- | --- |
| intraday_downside_volatility_5m_w48 | volatility | Downside-only realized volatility should be a cleaner long-only risk penalty than symmetric volatility. |
| intraday_return_skewness_5m_w48 | risk | Intraday return skewness can separate unstable burst-like moves from smoother accumulation. |
| intraday_money_flow_5m_w48 | volume | Closes near intrabar highs on sustained volume should proxy accumulation more directly than raw volume expansion. |
| intraday_signed_turnover_imbalance_5m_w48 | turnover | Turnover attached to upticks or downticks should be more informative than raw turnover shocks. |
| intraday_risk_adjusted_momentum_5m_w48 | momentum | Volatility-scaled continuation may separate durable demand from noisy price jumps. |
| intraday_volume_confirmed_momentum_5m_w48 | momentum | Continuation confirmed by volume expansion should be more robust than raw momentum or raw volume alone. |
| intraday_gap_5m | event | Short interval gaps capture discrete repricing and temporary order-book imbalance. |
| intraday_return_turnover_corr_5m_w48 | turnover | Return-turnover correlation distinguishes directional participation from indiscriminate turnover. |

## Implementation Boundary

- Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`.
- Dataset CLI: `examples/build_baseline_a_alpha_dataset.py --factor-groups all`.
- Registry: `configs/factors/factor_registry.json` version 3.
- Inputs: standard 5-minute OHLCV bars plus turnover where required.
- Initial status: `candidate` with `pending_single_factor_review`.

## Smoke And Screening Evidence

Two non-promotional checks were run:

| run | scope | purpose | output |
| --- | --- | --- | --- |
| `runs/factor_smoke/new_intraday_factors_2026_05_16` | 2024-01-02 to 2024-01-05, 3 symbols | Plumbing and artifact smoke test | 384 rows, 38 features |
| `runs/factor_smoke/new_intraday_factors_2024_q1_m30` | 2024 Q1, 30 symbols | Pre-admission screening | 78,824 rows, 38 features |

The Q1 screening report is
`runs/factor_smoke/new_intraday_factors_2024_q1_m30/factor_admission_screening/factor_admission_report.md`.
It is not a substitute for the standard 2023-2025 admission suite.

## Q1 Screening Result

| factor_id | screening_status | direction | rank_ic | t_stat | hit_rate | cost_adj_spread | turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| intraday_downside_volatility_5m_w48 | candidate | invert | -0.053147 | -9.04 | 0.5885 | 0.004550 | 0.0405 |
| intraday_return_turnover_corr_5m_w48 | candidate | invert | -0.040434 | -8.91 | 0.5972 | 0.003524 | 0.0760 |
| intraday_risk_adjusted_momentum_5m_w48 | candidate | invert | -0.035621 | -6.82 | 0.5643 | 0.000099 | 0.1556 |
| intraday_return_skewness_5m_w48 | candidate | invert | -0.031260 | -7.37 | 0.5980 | 0.001373 | 0.0735 |
| intraday_signed_turnover_imbalance_5m_w48 | candidate | invert | -0.023837 | -4.86 | 0.5318 | 0.000574 | 0.1104 |
| intraday_money_flow_5m_w48 | reject | long | 0.002278 | 0.47 | 0.5163 | 0.001260 | 0.1057 |
| intraday_volume_confirmed_momentum_5m_w48 | reject | invert | -0.002526 | -0.56 | 0.5113 | -0.000326 | 0.4858 |
| intraday_gap_5m | reject | invert | -0.003512 | -0.97 | 0.5160 | -0.000837 | 0.8009 |

Screening priority for formal standard admission:

1. `intraday_downside_volatility_5m_w48`
2. `intraday_return_turnover_corr_5m_w48`
3. `intraday_risk_adjusted_momentum_5m_w48`
4. `intraday_return_skewness_5m_w48`
5. `intraday_signed_turnover_imbalance_5m_w48`

The three rejected screening features should stay registered as candidates only
until the standard run confirms or rejects them; do not promote them based on
the Q1 screen.

## Standard Admission Run

The full all-feature Framework v1 rebuild was attempted first, but the dataset
stage was killed by the OS after five partitions. To keep the evidence
point-in-time and avoid mixing incomplete artifacts with prior batches, the
formal admission run was narrowed to the eight new factor groups from this
batch only.

| artifact | path |
| --- | --- |
| New-factor alpha dataset | `runs/framework_v1_acceptance/factor_batch_2026_05_16_new_factors/alpha_dataset` |
| Single-factor evaluation | `runs/framework_v1_acceptance/factor_batch_2026_05_16_new_factors/factor_evaluation` |
| Admission report | `runs/framework_v1_acceptance/factor_batch_2026_05_16_new_factors/factor_admission/factor_admission_report.json` |

Dataset coverage: 36 monthly partitions from 2023-01-03 09:35 +08:00 through
2025-12-31 15:00 +08:00, 103,495,412 rows before factor-level missing-value
filtering, ST exclusion enabled, 48-bar forward-return label, one-bar entry lag,
and 9.8% limit-aware entry filters.

## Standard Admission Result

| factor_id | status | direction | rank_ic | t_stat | hit_rate | stable_years | cost_adj_spread | top_n_turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| intraday_downside_volatility_5m_w48 | candidate | invert | -0.062040 | -73.25 | 0.6995 | 3 | 0.010894 | 0.0343 |
| intraday_volume_confirmed_momentum_5m_w48 | candidate | invert | -0.006572 | -12.18 | 0.5278 | 3 | 0.001751 | 0.5493 |
| intraday_return_skewness_5m_w48 | watchlist | invert | -0.015701 | -24.28 | 0.5862 | 3 | -0.002461 | 0.0759 |
| intraday_return_turnover_corr_5m_w48 | watchlist | invert | -0.012260 | -17.56 | 0.5527 | 2 | -0.002644 | 0.1080 |
| intraday_risk_adjusted_momentum_5m_w48 | watchlist | invert | -0.009718 | -12.46 | 0.5341 | 2 | -0.000514 | 0.2267 |
| intraday_signed_turnover_imbalance_5m_w48 | watchlist | invert | -0.005775 | -8.82 | 0.5247 | 2 | -0.001447 | 0.1719 |
| intraday_money_flow_5m_w48 | reject | long | 0.001778 | 3.03 | 0.5071 | 1 | 0.000256 | 0.1603 |
| intraday_gap_5m | reject | invert | -0.000072 | -0.46 | 0.5049 | 2 | -0.001200 | 0.8479 |

The final registry decision is stricter than the single-factor admission table:
`intraday_volume_confirmed_momentum_5m_w48` is moved to watchlist because the
portfolio validation below was negative, and because its standalone evidence is
weak and turnover-heavy.

## Candidate Reviews And Portfolio Validation

Candidate review reports were generated for the two single-factor candidates
and for the two strongest watchlist features:

| factor_id | review |
| --- | --- |
| intraday_downside_volatility_5m_w48 | `runs/factor_candidate_reviews/intraday_downside_volatility_5m_w48/factor_candidate_review.md` |
| intraday_volume_confirmed_momentum_5m_w48 | `runs/factor_candidate_reviews/intraday_volume_confirmed_momentum_5m_w48/factor_candidate_review.md` |
| intraday_return_skewness_5m_w48 | `runs/factor_candidate_reviews/intraday_return_skewness_5m_w48/factor_candidate_review.md` |
| intraday_return_turnover_corr_5m_w48 | `runs/factor_candidate_reviews/intraday_return_turnover_corr_5m_w48/factor_candidate_review.md` |

The registered candidate portfolio used equal weights across
`intraday_downside_volatility_5m_w48` and
`intraday_volume_confirmed_momentum_5m_w48`, with registry enforcement and the
standard cost-aware optimizer.

| metric | value |
| --- | ---: |
| total_return | -0.113221 |
| final_equity | 886,779.26 |
| max_drawdown | -0.275034 |
| gross_turnover | 154.5274 |
| planned_gross_turnover | 156.0000 |
| total_transaction_cost | 114,580.94 |
| trade_count | 12,014 |
| signal_count | 8,949 |

Portfolio artifact:
`runs/candidate_factor_portfolios/factor_batch_2026_05_16_new_factors_backtest/summary.json`.

An isolated downside-volatility-only portfolio attempt was also run, but the
current single-factor score construction degenerated to zero-valued score files
and the backtest aborted with no executable shifted signals. Treat that as a
portfolio-validation framework limitation for single-factor score construction,
not as promotion evidence.

## Final Governance Decision

| factor_id | registry_status | decision |
| --- | --- | --- |
| intraday_downside_volatility_5m_w48 | candidate | Strong standalone inverted risk penalty; keep for targeted portfolio review, do not promote from the negative equal-combo portfolio. |
| intraday_volume_confirmed_momentum_5m_w48 | watchlist | Passed single-factor gates only weakly and failed portfolio validation when combined with downside volatility. |
| intraday_return_skewness_5m_w48 | watchlist | Statistically strong but negative after transaction-cost spread. |
| intraday_return_turnover_corr_5m_w48 | watchlist | Statistically useful but cost-fragile and only two stable annual slices. |
| intraday_risk_adjusted_momentum_5m_w48 | watchlist | Inverted relation with negative after-cost spread. |
| intraday_signed_turnover_imbalance_5m_w48 | watchlist | Inverted relation with negative after-cost spread. |
| intraday_money_flow_5m_w48 | reject | Failed coverage, directional hit-rate, and annual stability gates. |
| intraday_gap_5m | reject | Near-zero IC, failed t-stat and hit-rate gates, negative after-cost spread. |

No factor from this batch is promoted. The only active follow-up candidate is
`intraday_downside_volatility_5m_w48`, preferably as a risk penalty or gate
rather than as an equal-weight alpha sleeve.
