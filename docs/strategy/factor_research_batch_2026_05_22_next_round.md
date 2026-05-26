# Factor Research Batch - 2026-05-22 Next Round

Status note, 2026-05-26: this is a historical batch review. Its references to
the "current leading" equal annual-budget-52 branch, active/default replay, and
daily-MA frontier are preserved as the decision context for the 2026-05-22
research round. For current portfolio-frontier comparisons, use
`docs/validation/factor_development_standard.md`; optimizer-native work should
compare against the 2026-05-25 volume-concentration cost-pressure frontier
`vc_opt_risk_cp0010_w50`.

This batch starts the next factor-development round after the compact-core,
sell-pressure persistence, and liquidity-recovery balance reviews.

The batch was initially registered as `planned`, not `candidate`. The
research-memory checks found blocking similarity to prior failed ideas, so each
candidate must pass single-factor admission and immediate portfolio validation
before it can be considered for promotion. The latest weak-tape gap retry has
now completed single-factor admission and standalone portfolio validation; it
is a statistical `candidate`, but not a promoted basket replacement.

## Initial Planned Groups

| factor_id | group | features | material difference |
|---|---|---|---|
| `intraday_same_slot_residual_memory_5m` | `same_slot_intraday_memory` | `intraday_same_slot_residual_return_5m_d5`, `intraday_same_slot_residual_return_5m_d20` | Uses only lagged prior-session observations from the same intraday time slot, after subtracting the contemporaneous market-median return. This is not raw rolling momentum. |
| `intraday_overnight_intraday_tug_of_war_5m` | `overnight_intraday_tug_of_war` | `intraday_overnight_gap_5m`, `intraday_overnight_gap_down_recovery_5m`, `intraday_overnight_gap_up_fade_5m`, `intraday_overnight_intraday_disagreement_5m` | Separates previous-session close to current-session open from within-session response. This is not the rejected raw 5-minute gap feature. |
| `intraday_sell_pressure_quality_state_5m_w48` | `sell_pressure_quality_state` | `intraday_sell_pressure_absorption_quality_5m_w48`, `intraday_false_absorption_risk_5m_w48` | Decomposes high apparent absorption into recovery-balanced quality and weak-breadth false-absorption risk, targeting the January/June 2024 failure mode. |

## Research Memory Readout

| factor_id | status | blocking match | decision |
|---|---|---|---|
| `intraday_same_slot_residual_memory_5m` | `blocked` | `intraday_range_position_5m_w48` | Proceed only because the transform uses lagged same-slot residual history rather than price position or raw intraday momentum. |
| `intraday_overnight_intraday_tug_of_war_5m` | `blocked` | `intraday_gap_5m` | Proceed only because it explicitly uses session-boundary overnight gap plus same-session response buckets. |
| `intraday_sell_pressure_quality_state_5m_w48` | `blocked` | `intraday_vwap_deviation_5m_w48` | Proceed only because it is a false-absorption state decomposition, not a VWAP stretch, monotonic recovery, or generic sell-pressure variant. |

Evidence:

- `runs/factor_research_memory/intraday_same_slot_residual_memory_5m/factor_research_memory_check.json`
- `runs/factor_research_memory/intraday_overnight_intraday_tug_of_war_5m/factor_research_memory_check.json`
- `runs/factor_research_memory/intraday_sell_pressure_quality_state_5m_w48/factor_research_memory_check.json`

## Implementation Notes

- Same-slot memory uses `shift(1)` before rolling over prior observations in
  the same `(instrument_id, intraday_slot)` group.
- Overnight tug-of-war uses completed prior-session close and current-session
  open, then updates the intraday response from current bar close.
- Sell-pressure quality uses rolling downside turnover per downside return,
  recovery balance, and weak-breadth pressure. It writes a quality score and a
  false-absorption risk score so admission can learn direction separately.

## Validation Plan

1. Build a new-factor-only dataset with:
   `--factor-groups same_slot_intraday_memory overnight_intraday_tug_of_war sell_pressure_quality_state`.
2. Run standard factor evaluation and admission against `forward_return_48b`.
3. For any admitted feature, run candidate review before portfolio validation.
4. Run an incremental portfolio test against both:
   - compact-core `decorrelated + partial_rebalance_daily`;
   - then-leading `equal` annual-budget-52 branch.
5. Report a dedicated 2024-01 and 2024-06 table. A factor that improves only
   full-window IC but worsens those months should remain watchlist or reject.

Promotion requires positive full-window and high-cost incremental contribution,
not just standalone IC.

## Current Results

Full new-factor-only dataset build completed for 2023-01 through 2025-12:

- Dataset rows: `103351780`
- Feature rows: `109255584`
- Dataset: `runs/factor_research/next_round_2026_05_22/alpha_dataset`
- Evaluation: `runs/factor_research/next_round_2026_05_22/factor_evaluation/summary.json`
- Admission: `runs/factor_research/next_round_2026_05_22/factor_admission/factor_admission_report.json`

Single-factor admission admitted three features:

| feature | status | direction | rank IC | t-stat | cost-adjusted spread | note |
|---|---:|---:|---:|---:|---:|---|
| `intraday_false_absorption_risk_5m_w48` | `candidate` | `invert` | -0.019502 | -34.399 | 0.003751 | Strongest standalone candidate in this batch. |
| `intraday_overnight_gap_down_recovery_5m` | `candidate` | `invert` | -0.016688 | -40.056 | 0.003658 | Useful only with explicit inverted direction. |
| `intraday_overnight_gap_5m` | `candidate` | `long` | 0.015189 | 33.747 | 0.003060 | Low turnover, but research-memory similarity to the old rejected gap needs portfolio review. |
| `intraday_sell_pressure_absorption_quality_5m_w48` | `watchlist` | `invert` | -0.022298 | -28.616 | -0.002901 | Statistically strong but cost-fragile and directionally conflicted. |
| `intraday_same_slot_residual_return_5m_d5` | `reject` | `long` | 0.000049 | 0.117 | -0.001551 | Too weak and too costly. |
| `intraday_same_slot_residual_return_5m_d20` | `reject` | `invert` | -0.002713 | -6.479 | -0.000585 | Failed coverage, hit-rate, and cost-adjusted spread. |
| `intraday_overnight_gap_up_fade_5m` | `reject` | `invert` | -0.002583 | -7.487 | 0.003474 | Failed hit-rate and stable-year gates. |
| `intraday_overnight_intraday_disagreement_5m` | `reject` | `invert` | -0.001990 | -5.124 | 0.000484 | Failed hit-rate gate. |

Candidate reviews were rendered and all three admitted features are ready for
portfolio review:

- `runs/factor_candidate_reviews/intraday_overnight_gap_5m/factor_candidate_review.json`
- `runs/factor_candidate_reviews/intraday_overnight_gap_down_recovery_5m/factor_candidate_review.json`
- `runs/factor_candidate_reviews/intraday_false_absorption_risk_5m_w48/factor_candidate_review.json`

The registry was split from three planned feature groups into feature-level
entries where admission status is mixed. After the combined weak-tape
deduplication review, registry version `32` validates with `42` entries,
`15` candidates, `16` watchlist entries, and `11` rejects.

Quick portfolio validation completed:

- Output: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_quick`
- Summary: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_quick/validation_summary.json`
- Status: `pass`
- Result count: `21`

Primary-policy evidence from `decorrelated/partial_rebalance_daily` is positive
but risky:

- Total return: `0.170905`
- Max drawdown: `-0.366323`
- Gross turnover: `122.461764`
- Transaction cost: `162006.49`

The equal-weighted primary policy was similar: total return `0.175797`, max
drawdown `-0.361537`, gross turnover `121.968677`. High-turnover policies in
the same quick run were mostly negative, so the useful implementation path is
the partial-rebalance daily policy, not every-bar churn.

Standard primary-policy validation completed with `warn` status:

- Output: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_standard_primary`
- Summary: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_standard_primary/validation_summary.json`
- Failed checks: `0`
- Warning checks: `1`
- Warning: `primary_yearly_base_positive_returns` because `year_2023_base` was negative.

Primary `decorrelated/partial_rebalance_daily` evidence:

| slice | total return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| `full_base` | 0.170905 | -0.366323 | 122.461764 | 162006.49 |
| `full_high_cost` | 0.115674 | -0.374905 | 122.380906 | 201900.74 |
| `year_2023_base` | -0.036031 | -0.159554 | 41.814504 | 53722.90 |
| `year_2024_base` | 0.005794 | -0.277838 | 41.460586 | 54704.11 |
| `year_2025_base` | 0.204876 | -0.168166 | 40.295499 | 55107.94 |

Key `full_base` stress months:

| method | month | return | max drawdown |
|---|---:|---:|---:|
| `decorrelated` | 2024-01 | -0.143815 | -0.148978 |
| `decorrelated` | 2024-06 | -0.091364 | -0.106269 |
| `decorrelated` | 2024-09 | 0.183514 | -0.170762 |
| `decorrelated` | 2024-11 | 0.078560 | -0.085273 |
| `equal` | 2024-01 | -0.147629 | -0.153534 |
| `equal` | 2024-06 | -0.092737 | -0.109457 |
| `equal` | 2024-09 | 0.185949 | -0.171071 |
| `equal` | 2024-11 | 0.082191 | -0.086454 |

Conclusion: keep the three admitted features as `candidate`, but do not promote
the basket. The full-window and high-cost returns are positive, while 2023 is
negative, 2024 is only marginally positive, and the full-window drawdown remains
too deep. The next useful experiment is a risk-control or allocator test aimed
at January/June 2024 stress without giving back the full-window and high-cost
edge.

## 2024 Health-Shrink Control

A focused 2024 control test changed only `--factor-health-mode` from `monitor`
to `shrink` for the same three candidates and the same
`partial_rebalance_daily` policy:

- Output: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_health_shrink_2024`
- Summary: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_health_shrink_2024/validation_summary.json`
- Status: `fail`
- Failed check: `primary_full_high_cost_positive_return`

Primary `decorrelated/partial_rebalance_daily` comparison:

| 2024 slice | monitor return | shrink return | monitor max DD | shrink max DD |
|---|---:|---:|---:|---:|
| `base` | 0.005794 | 0.004499 | -0.277838 | -0.277680 |
| `high_cost` | not separately run in standard annual slice | -0.011141 | not separately run in standard annual slice | -0.279601 |

Stress-month comparison shows no useful January improvement and only limited
June relief:

| method | month | monitor return | shrink return | monitor max DD | shrink max DD |
|---|---:|---:|---:|---:|---:|
| `decorrelated` | 2024-01 | -0.143815 | -0.144852 | -0.148978 | -0.147457 |
| `decorrelated` | 2024-06 | -0.091364 | -0.089841 | -0.106269 | -0.098642 |
| `decorrelated` | 2024-09 | 0.183514 | 0.183594 | -0.170762 | -0.152359 |
| `decorrelated` | 2024-11 | 0.078560 | 0.040931 | -0.085273 | -0.070949 |
| `equal` | 2024-01 | -0.147629 | -0.147045 | -0.153534 | -0.150059 |
| `equal` | 2024-06 | -0.092737 | -0.078748 | -0.109457 | -0.086365 |
| `equal` | 2024-09 | 0.185949 | 0.173978 | -0.171071 | -0.155452 |
| `equal` | 2024-11 | 0.082191 | 0.029117 | -0.086454 | -0.067731 |

Decision: reject generic health shrink for this basket. It slightly reduces
some stress-month drawdowns, but base return deteriorates and high-cost 2024
turns negative. The next risk-control attempt should use a distinct regime
state or allocator, not this generic lagged health scale.

## Downside-Volatility State Gate

The next control used the existing `intraday_downside_volatility_5m_w48`
gross-exposure state gate. The gate dataset came from
`runs/framework_v1_acceptance/factor_batch_2026_05_16_new_factors/alpha_dataset`.

The default gate schedule had `34799` observations:

| state | count |
|---|---:|
| `full` | 26472 |
| `reduced` | 3978 |
| `blocked` | 4301 |
| `warmup` | 48 |

The mild gate (`reduced=0.75`, `blocked=0.5`) was worse than the default
setting in the 2024 slice:

| method | gate | base return | base max DD | high-cost return | high-cost max DD |
|---|---:|---:|---:|---:|---:|
| `decorrelated` | default | 0.017578 | -0.260830 | -0.001922 | -0.262984 |
| `equal` | default | 0.027859 | -0.255894 | 0.008020 | -0.257987 |
| `decorrelated` | mild | 0.011288 | -0.264643 | -0.006094 | -0.266419 |
| `equal` | mild | 0.015619 | -0.266563 | -0.000815 | -0.268184 |

Because the default gate was better in 2024, it was promoted to a full
standard validation with `equal` as the primary method:

- Output: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_downside_vol_gate_equal_primary_standard`
- Summary: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_downside_vol_gate_equal_primary_standard/validation_summary.json`
- Status: `warn`
- Failed checks: `0`
- Warning checks: `2`
- Warnings: `primary_full_base_turnover_control`, `primary_yearly_base_positive_returns`

Primary `equal/partial_rebalance_daily` evidence:

| slice | total return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| `full_base` | 0.140937 | -0.351763 | 165.161003 | 218201.46 |
| `full_high_cost` | 0.070957 | -0.364070 | 164.975561 | 271726.64 |
| `year_2023_base` | -0.068918 | -0.164028 | 56.797085 | 73402.83 |
| `year_2024_base` | 0.027859 | -0.255894 | 54.702330 | 69530.54 |
| `year_2025_base` | 0.084683 | -0.119241 | 53.485240 | 72258.04 |

Decision: reject this downside-volatility gate as a replacement control for
the new three-factor basket. It improves the 2024 annual slice, but it worsens
2023, cuts the strong 2025 contribution, lowers full-window and high-cost
returns versus monitor mode, and adds a turnover warning. Keep the three
features as candidates, but do not promote the basket or this gate. The next
round should focus on either a higher-level allocator/regime state or
additional complementary factor design rather than blanket exposure scaling.

## Regime Diagnosis And Next Feature

Dedicated diagnostics were run on the ungated standard basket:

- 2023: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_standard_primary/regime_diagnostics_2023_decorrelated`
- 2024: `runs/candidate_factor_portfolios/next_round_2026_05_22_candidates_standard_primary/regime_diagnostics_2024_decorrelated`

The failure mode is mostly absolute long-only exposure in weak tape, not
execution blockage. Tradable rate stayed near `100%`. In 2024-01, score IC was
positive (`0.0339`) and top-minus-bottom label spread was positive (`0.25%`),
but the top-score bucket still had `-0.93%` mean forward label because the
market label was deeply negative (`-0.82%`). 2024-06 was similar but weaker:
score spread was nearly flat (`0.01%`) and the top-score bucket was `-0.58%`.

The same pattern appears in 2023. March and April had negative top-minus-bottom
score spreads, while August and October retained positive spreads but still had
negative top-score labels in a weak market. The most persistent weak leg in bad
months was `intraday_overnight_gap_5m`: its standalone top-label mean was
negative in all 2023 and 2024 worst-month tables.

Follow-up weak-tape factor family:

- `intraday_weak_tape_overnight_gap_risk_5m`
- Feature group: `weak_tape_overnight_gap`
- Candidate features:
  - `intraday_weak_tape_gap_up_risk_5m_w48`
  - `intraday_weak_tape_gap_down_recovery_risk_5m_w48`
- Rejected feature:
  - `intraday_weak_tape_gap_up_fade_risk_5m_w48`
- Expected direction: `invert`
- Research memory: `runs/factor_research_memory/intraday_weak_tape_gap_up_risk_5m_w48/factor_research_memory_check.json`

The research-memory check is blocked by prior raw-gap, gap-up-fade, breadth,
and market-downside failures. This is still allowed as a one-time retry because
those rejected entries explicitly permit a materially different event-state
split tied to breadth or market state. The new feature is not a plain gap or a
plain breadth factor; it activates overnight-gap risk only when rolling weak
breadth and market downside are elevated.

## Weak-Tape Gap Admission

The weak-tape gap dataset was built for 2023-01 through 2025-12:

- Dataset: `runs/factor_research/weak_tape_gap_2026_05_22/alpha_dataset`
- Evaluation: `runs/factor_research/weak_tape_gap_2026_05_22/factor_evaluation/summary.json`
- Admission: `runs/factor_research/weak_tape_gap_2026_05_22/factor_admission/factor_admission_report.json`
- Dataset rows: `103351780`

Admission accepted two inverted risk legs and rejected the plain fade-style leg:

| feature | status | direction | rank IC | t-stat | hit rate | cost-adjusted spread | stable years |
|---|---:|---:|---:|---:|---:|---:|---:|
| `intraday_weak_tape_gap_down_recovery_risk_5m_w48` | `candidate` | `invert` | -0.016688 | -40.056 | 0.6042 | 0.003658 | 3 |
| `intraday_weak_tape_gap_up_risk_5m_w48` | `candidate` | `invert` | -0.007222 | -21.010 | 0.5767 | 0.004058 | 3 |
| `intraday_weak_tape_gap_up_fade_risk_5m_w48` | `reject` | `invert` | -0.002583 | -7.487 | 0.5165 | 0.003474 | 1 |

Decision: standalone admission accepts the first two legs, but only the
gap-up weak-tape risk leg remains an active incremental registry feature after
the combined duplicate check. The rejected gap-up-fade leg confirms that the
old fade idea remains unstable unless it is conditioned more tightly.

## Weak-Tape Gap Portfolio Validation

Standalone quick validation passed:

- Output: `runs/candidate_factor_portfolios/weak_tape_gap_2026_05_22_quick`
- Status: `pass`
- Full-base return: `0.135099`
- Full-base max drawdown: `-0.323998`
- Gross turnover: `117.338852`

Standard primary validation finished with `warn`:

- Output: `runs/candidate_factor_portfolios/weak_tape_gap_2026_05_22_standard_primary`
- Summary: `runs/candidate_factor_portfolios/weak_tape_gap_2026_05_22_standard_primary/validation_summary.json`
- Failed checks: `0`
- Warning checks: `1`
- Warning: `primary_yearly_base_positive_returns` because both 2023 and 2024 were negative.

| slice | total return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| `full_base` | 0.135099 | -0.323998 | 117.338852 | 158825.87 |
| `full_high_cost` | 0.085429 | -0.332599 | 117.181124 | 197965.30 |
| `year_2023_base` | -0.007757 | -0.146121 | 39.865089 | 52372.48 |
| `year_2024_base` | -0.051085 | -0.255417 | 40.935157 | 52826.91 |
| `year_2025_base` | 0.187414 | -0.144263 | 39.655908 | 54073.45 |

The lagged factor-health monitor showed many impaired windows and recommended
average weight scales near `0.59` to `0.62`, so a full standard
`--factor-health-mode shrink` validation was run:

- Output: `runs/candidate_factor_portfolios/weak_tape_gap_2026_05_22_health_shrink_standard`
- Summary: `runs/candidate_factor_portfolios/weak_tape_gap_2026_05_22_health_shrink_standard/validation_summary.json`
- Status: `warn`

The shrink run produced the same realized path as monitor mode in this two-leg
rank basket: full-base return stayed `0.135099`, high-cost return stayed
`0.085429`, 2023 stayed `-0.007757`, and 2024 stayed `-0.051085`.

Decision: the weak-tape gap interaction is a valid single-factor candidate, but
it is not a standalone promoted portfolio. It should next be tested as an
additional leg or risk-state input inside the existing 2026-05-22 candidate
basket, not as a replacement basket by itself.

## Combined Deduplicated Basket

The combined dataset joined the original admitted 2026-05-22 candidates with
the weak-tape retry group:

- Dataset: `runs/factor_research/combined_gap_absorption_weak_tape_2026_05_22/alpha_dataset`
- Evaluation: `runs/factor_research/combined_gap_absorption_weak_tape_2026_05_22/factor_evaluation/summary.json`
- Admission: `runs/factor_research/combined_gap_absorption_weak_tape_2026_05_22/factor_admission/factor_admission_report.json`
- Admission counts: `5` candidates, `1` watchlist, `3` rejects.

The combined correlation check showed that two weak-tape columns should not be
double-counted:

| weak-tape column | existing column | rank correlation | decision |
|---|---|---:|---|
| `intraday_weak_tape_gap_down_recovery_risk_5m_w48` | `intraday_overnight_gap_down_recovery_5m` | 0.999776 | Treat as duplicate; keep the existing column in the basket. |
| `intraday_weak_tape_gap_up_fade_risk_5m_w48` | `intraday_overnight_gap_up_fade_5m` | 0.999996 | Already rejected and near duplicate; do not include. |

The deduplicated portfolio feature set is therefore:

- `intraday_overnight_gap_5m`
- `intraday_overnight_gap_down_recovery_5m`
- `intraday_false_absorption_risk_5m_w48`
- `intraday_weak_tape_gap_up_risk_5m_w48`

Quick validation passed:

- Output: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_quick_dedup`
- Summary: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_quick_dedup/validation_summary.json`
- Primary `decorrelated/partial_rebalance_daily`: return `0.183513`, max drawdown `-0.238106`, gross turnover `121.733902`.
- `ic_weighted` had higher full-base return (`0.302910`) but much deeper drawdown (`-0.328768`), so it was not selected as primary.

Standard validation passed with no warnings:

- Output: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_standard_dedup`
- Summary: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_standard_dedup/validation_summary.json`
- Failed checks: `0`
- Warning checks: `0`

Primary `decorrelated/partial_rebalance_daily` evidence:

| slice | total return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| `full_base` | 0.183513 | -0.238106 | 121.733902 | 165655.09 |
| `full_high_cost` | 0.130797 | -0.246723 | 121.582980 | 210241.33 |
| `year_2023_base` | 0.020504 | -0.118977 | 41.321188 | 53988.28 |
| `year_2024_base` | 0.061159 | -0.220512 | 42.789348 | 55578.96 |
| `year_2025_base` | 0.165858 | -0.121865 | 40.680401 | 53842.77 |

Decision: promote this result to the next review stage as a deduplicated
candidate basket, not as a standalone weak-tape basket. The only active
incremental weak-tape feature is `intraday_weak_tape_gap_up_risk_5m_w48`;
`intraday_weak_tape_gap_down_recovery_risk_5m_w48` remains an admission-valid
diagnostic but should not be counted separately from
`intraday_overnight_gap_down_recovery_5m` in portfolio construction.

## Promoted-Candidate Tracking Review

A unified candidate review was rendered for the active weak-tape incremental
leg:

- Output: `runs/factor_candidate_reviews/intraday_weak_tape_overnight_gap_risk_5m`
- Status: `ready_for_portfolio_review`

The promoted-candidate review then compared the deduplicated four-leg basket
against the prior three-leg 2026-05-22 basket:

- Output: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_promoted_candidate_review_v1`
- Review: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_promoted_candidate_review_v1/promoted_candidate_review.json`
- Decision: `accepted_for_promoted_candidate_tracking`
- Default strategy change: `false`
- Failed checks: `0`
- Warning checks: `2`

Primary comparison:

| slice | current return | prior return | delta return | current max DD | prior max DD | delta max DD |
|---|---:|---:|---:|---:|---:|---:|
| `full_base` | 18.35% | 17.09% | 1.26% | -23.81% | -36.63% | 12.82% |
| `full_high_cost` | 13.08% | 11.57% | 1.51% | -24.67% | -37.49% | 12.82% |
| `year_2023_base` | 2.05% | -3.60% | 5.65% | -11.90% | -15.96% | 4.06% |
| `year_2024_base` | 6.12% | 0.58% | 5.54% | -22.05% | -27.78% | 5.73% |
| `year_2025_base` | 16.59% | 20.49% | -3.90% | -12.19% | -16.82% | 4.63% |

Monthly concentration checks passed: full-base top-3 absolute monthly return
share was `33.07%`, and high-cost top-3 absolute monthly return share was
`33.79%`. Factor contribution checks also passed under the four-leg basket:
the full-base average largest absolute contribution share was `44.90%`, with
maximum largest contribution share `75.79%`.

Review warnings:

- The basket has only been compared against the prior 2026-05-22 three-leg
  candidate basket, not replayed against the broader active/default and
  daily-MA research-frontier stack in a unified dataset.
- The 2025 annual return gives back `3.90%` versus the prior three-leg basket,
  even though 2025 drawdown improves by `4.63%`.

Decision: accept the deduplicated four-leg basket for promoted-candidate
tracking, but do not change the active/default strategy. The next required
gate is either the next unseen-data batch or a unified active/default frontier
comparison.

## Existing-Artifact Frontier Comparison

The next review compared the current four-leg basket against the available
active/default and daily-MA research-frontier standard validation artifacts.
This is a score-level existing-artifact comparison, not a true unified replay,
because the active/default and daily-MA frontier artifacts use different factor
stacks and selected score construction.

- Output: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_frontier_comparison_v1`
- Review: `runs/candidate_factor_portfolios/combined_gap_absorption_weak_tape_2026_05_22_frontier_comparison_v1/frontier_comparison_report.json`
- Decision: `tracking_only_after_frontier_comparison`
- Default strategy change: `false`
- True unified replay required before default promotion: `true`
- Failed checks: `0`
- Warning checks: `3`

Frontier snapshot:

| artifact | full return | high-cost return | 2023 return | 2024 return | 2025 return | worst max DD | full turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| active/default score-budget gate v1 | 14.38% | 9.43% | -1.30% | 9.91% | 9.95% | -19.83% | 124.777 |
| daily-MA frontier high-dispersion current | 20.75% | 16.18% | -3.92% | 14.37% | 8.48% | -17.19% | 101.861 |
| prior 2026-05-22 three-leg basket | 17.09% | 11.57% | -3.60% | 0.58% | 20.49% | -37.49% | 122.462 |
| current 2026-05-22 four-leg dedup basket | 18.35% | 13.08% | 2.05% | 6.12% | 16.59% | -24.67% | 121.734 |

The four-leg basket still improves the prior 2026-05-22 three-leg basket:
full-base return improves by `1.26%`, high-cost return improves by `1.51%`,
2023 improves by `5.65%`, 2024 improves by `5.54%`, and drawdown improves in
every tested slice. The remaining prior-basket warning is the 2025 return
giveback of `-3.90%`, although 2025 drawdown improves by `4.63%`.

Against the active/default score-budget gate, the four-leg basket has higher
full-base, high-cost, 2023, and 2025 returns, but it has worse full-window
drawdown and lower 2024 return. Against the daily-MA frontier, it repairs the
negative 2023 slice and improves 2025 return, but trails full-window return,
high-cost return, and 2024 return, with worse full-window drawdown.

Decision: keep the weak-tape gap-up risk leg and the deduplicated four-leg
basket in promoted-candidate tracking, but do not change the default strategy.
The next promotion gate must be either a true unified replay or the next
unseen-data batch.

## True Unified Replay

The follow-up replay rebuilt one unified alpha dataset containing the daily-MA
frontier factors, overnight/gap-response factors, sell-pressure absorption risk,
and weak-tape gap state factors. All replay variants below therefore use the
same rows, labels, factor admission file, and feature correlation file.

- Dataset: `runs/factor_research/unified_daily_ma_gap_weak_tape_2026_05_23/alpha_dataset`
- Evaluation: `runs/factor_research/unified_daily_ma_gap_weak_tape_2026_05_23/factor_evaluation/summary.json`
- Admission: `runs/factor_research/unified_daily_ma_gap_weak_tape_2026_05_23/factor_admission/factor_admission_report.json`
- Active/default replay: `runs/candidate_factor_portfolios/unified_active_default_score_budget_gate_v1_2026_05_23_standard/validation_summary.json`
- Daily-MA frontier replay: `runs/candidate_factor_portfolios/unified_daily_ma_frontier_high_dispersion_2026_05_23_standard/validation_summary.json`
- Four-leg gap/weak-tape replay: `runs/candidate_factor_portfolios/unified_four_leg_gap_weak_tape_2026_05_23_standard/validation_summary.json`
- Daily-MA frontier plus gap/weak-tape replay: `runs/candidate_factor_portfolios/unified_daily_ma_frontier_plus_gap_weak_tape_2026_05_23_standard/validation_summary.json`

Unified replay snapshot:

| artifact | status | full return | high-cost return | 2023 return | 2024 return | 2025 return | worst max DD | full turnover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| active/default score-budget gate v1 | warn | 14.38% | 9.43% | -1.30% | 9.91% | 9.95% | -19.83% | 124.777 |
| daily-MA frontier high-dispersion | warn | 20.75% | 16.18% | -3.92% | 14.37% | 8.48% | -17.19% | 101.861 |
| four-leg gap/weak-tape basket | pass | 18.35% | 13.08% | 2.05% | 6.12% | 16.59% | -24.67% | 121.734 |
| daily-MA frontier + gap/weak-tape | warn | 20.04% | 15.21% | -1.20% | 15.96% | 14.79% | -14.60% | 104.531 |

The four-leg gap/weak-tape basket passed the standard validation on its own:
all 2023/2024/2025 annual slices were positive, full-base return was `18.35%`,
and high-cost return was `13.08%`. Its weakness is risk shape: worst drawdown
was `-24.67%`, worse than both the active/default replay and the daily-MA
frontier replay.

Adding the four gap/weak-tape legs to the daily-MA frontier improved the
frontier's 2023 slice from `-3.92%` to `-1.20%`, 2024 from `14.37%` to
`15.96%`, 2025 from `8.48%` to `14.79%`, and worst drawdown from `-17.19%`
to `-14.60%`. It did not dominate on full-window or high-cost return: full
return fell from `20.75%` to `20.04%`, and high-cost return fell from `16.18%`
to `15.21%`.

Decision: keep the weak-tape gap-up risk leg and the deduplicated four-leg
basket in promoted-candidate tracking. The true unified replay supports the
factor as a useful satellite or stability repair candidate, but not as a
default-strategy replacement yet. Default strategy change remains `false`.

## Controlled Overlay Replay

The next check treated the four-leg gap/weak-tape basket as a controlled
satellite over the daily-MA frontier score stream instead of giving all 15
features equal access to the allocator. Each input score stream was
rank-normalized cross-sectionally by timestamp, then blended as
`(1 - w) * daily_ma_frontier + w * gap_weak_tape_satellite`. The replay reused
the daily-MA frontier high-dispersion gross-exposure schedule.

- Unconditional overlay grid: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_grid_2026_05_23_standard/validation_summary.json`
- Unconditional fine grid: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_fine_grid_2026_05_23_standard/validation_summary.json`
- High-dispersion-only overlay grid: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_high_dispersion_grid_2026_05_23_standard/validation_summary.json`
- Overlay runner: `examples/run_score_overlay_validation.py`

Overlay replay snapshot:

| artifact | status | full return | high-cost return | 2023 return | 2024 return | 2025 return | worst max DD | full turnover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| daily-MA frontier high-dispersion | warn | 20.75% | 16.18% | -3.92% | 14.37% | 8.48% | -17.19% | 101.861 |
| unconditional 2.5% overlay | warn | 23.12% | 18.29% | -1.07% | 16.67% | 12.18% | -15.51% | 100.605 |
| unconditional 5% overlay | warn | 25.45% | 20.35% | -3.47% | 17.29% | 13.38% | -14.78% | 100.134 |
| high-dispersion-only 20% overlay | warn | 24.42% | 19.76% | -3.79% | 14.67% | 12.23% | -16.94% | 102.021 |

The unconditional 5% overlay is the strongest controlled allocator result so
far. Versus the daily-MA frontier replay, it improved full-window return from
`20.75%` to `25.45%`, high-cost return from `16.18%` to `20.35%`, worst
drawdown from `-17.19%` to `-14.78%`, and full-window turnover from `101.861`
to `100.134`. It also improved all three annual slices, but 2023 remained
negative at `-3.47%`.

The high-dispersion-only overlay confirmed that the satellite has portfolio
value, but it did not beat the unconditional 5% overlay on full return,
high-cost return, drawdown, or 2023 repair. The controlled overlay therefore
upgrades the weak-tape basket from passive tracking to a portfolio-construction
candidate, but not yet to a default replacement. The 2.5%-7.5% fine grid
confirmed 5% as the preferred return/risk tradeoff; the 2.5% overlay is a
stability fallback because it reduced the 2023 loss to `-1.07%` while still
improving full-window and high-cost returns versus the daily-MA frontier.
Required next gate: run an unseen-data or frozen promotion replay before
changing the default strategy. Default strategy change remains `false`.

## Frozen Overlay Promotion Replay

The frozen promotion replay fixed the overlay design before running the review:
5% satellite weight as the preferred challenger and 2.5% as the stability
fallback. No new weights, calendar rules, or regime patches were introduced.
The robust replay added a zero-cost diagnostic to the standard full/year/high
cost scenarios. The available unified overlay score partitions still end at
2025-12, so this is a frozen replay rather than a true unseen-data replay.

- Frozen replay: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_frozen_promotion_replay_2026_05_23_robust/validation_summary.json`
- Frozen promotion review: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_frozen_promotion_review_v1/promotion_review.json`
- Frozen promotion report: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_frozen_promotion_review_v1/promotion_review.md`
- Review runner: `examples/review_score_overlay_promotion.py`

Frozen replay snapshot:

| artifact | status | full return | high-cost return | zero-cost return | 2023 return | 2024 return | 2025 return | full max DD | full turnover |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active/default | warn | 14.38% | 9.43% | n/a | -1.30% | 9.91% | 9.95% | -19.16% | 124.777 |
| daily-MA frontier | warn | 20.75% | 16.18% | n/a | -3.92% | 14.37% | 8.48% | -16.13% | 101.861 |
| frozen 2.5% overlay | warn | 23.12% | 18.29% | 42.70% | -1.07% | 16.67% | 12.18% | -13.18% | 100.605 |
| frozen 5% overlay | warn | 25.45% | 20.35% | 46.17% | -3.47% | 17.29% | 13.38% | -13.56% | 100.134 |

Frozen promotion review decision:
`accepted_as_frozen_challenger_no_default_change`.

The 5% overlay passed the fixed-configuration comparison against the daily-MA
frontier: full return improved by `4.70pp`, high-cost return by `4.17pp`,
full-window max drawdown by `2.57pp`, and full-window turnover fell by `1.727`.
Monthly concentration checks also passed. In full-base, the candidate had
18/36 positive months, worst month `-5.55%`, and top-3 absolute monthly return
share `32.61%`; the candidate-versus-frontier delta top-3 share was `22.50%`.
Under high-cost, the candidate had 17/36 positive months, worst month `-5.60%`,
top-3 absolute monthly return share `32.29%`, and delta top-3 share `21.87%`.

The remaining blockers are explicit and unchanged: 2023 is still negative for
the 5% overlay (`-3.47%`), and there is no true post-2025 unseen replay in the
current unified overlay score set. The 2.5% overlay is the stability fallback
because it reduces the 2023 loss to `-1.07%`, but its full-window and high-cost
returns are lower than the 5% challenger. Keep the 5% overlay as the frozen
challenger, keep the default strategy unchanged, and wait for the next
post-2025 unified score batch before any default-change review.

## Post-2025 Unseen Replay

The initial unseen preflight found that the annual 2026 CN equity 5-minute
parquet was absent. The data blocker was resolved by extending the fast-parquet
loader to include the promoted BaoStock CN equity 5m update parquet for 2026
ranges:
`../quant_dataset/canonical_store/v1/market/records=minute_bar/market_baostock_cn_equity_update__5m__5m.parquet`
(`2025-12-26T09:35:00+08:00` to `2026-05-15T15:00:00+08:00`).

- Preflight report: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_unseen_preflight_2026_05_23/preflight.md`
- Preflight JSON: `runs/candidate_factor_portfolios/daily_ma_frontier_gap_weak_tape_overlay_unseen_preflight_2026_05_23/preflight.json`
- Unseen dataset: `runs/factor_research/unified_daily_ma_gap_weak_tape_unseen_2026_05_23/alpha_dataset`
- Unseen replay: `runs/candidate_factor_portfolios/unseen_2026_overlay_control_grid_2026_05_23/validation_summary.json`
- Unseen review: `runs/candidate_factor_portfolios/unseen_2026_overlay_control_grid_2026_05_23/unseen_review.md`

Result: the true 2026-YTD replay rejects the frozen 5% overlay as a default
change. On 2026-01 through 2026-05 data, primary-only returned `-0.28%`
full-base and `-0.68%` high-cost; the 5% frozen challenger returned `-0.75%`
full-base and `-1.17%` high-cost. The 2.5% fallback improved the primary-only
line with `0.24%` full-base and `1.95%` zero-cost, but still failed high-cost at
`-0.18%`. Keep default unchanged, reject the 5% default-change case, and treat
2.5% only as a watchlist input for a cost-aware/gated retry.

## 2026 Unseen Attribution

Attribution was added after the default-change rejection:

- Attribution report:
  `runs/candidate_factor_portfolios/unseen_2026_overlay_control_grid_2026_05_23/attribution_v1/attribution_report.md`
- Attribution JSON:
  `runs/candidate_factor_portfolios/unseen_2026_overlay_control_grid_2026_05_23/attribution_v1/attribution_summary.json`
- Attribution runner: `examples/analyze_unseen_overlay_attribution.py`

The attribution treats 2026-YTD as a high-instability tape and does not add
external event labels. It uses only existing replay artifacts: equity curves,
risk-state schedule, score partitions, and the unified 2026 alpha dataset.

Portfolio-level monthly evidence:

| method | Jan | Feb | Mar | Apr | May | full-base |
|---|---:|---:|---:|---:|---:|---:|
| primary-only 0% | -0.18% | 2.11% | -2.55% | -0.03% | 0.41% | -0.28% |
| fallback 2.5% | 1.67% | 2.06% | -3.70% | 0.01% | 0.30% | 0.24% |
| frozen challenger 5% | 1.09% | 1.31% | -3.03% | 0.03% | -0.09% | -0.75% |

Risk-state equity attribution shows that the 2.5% overlay helped the
production-cost path mainly by reducing the damage in `full` and `reduced`
states, but the improvement was too small to survive high-cost stress:

| method | blocked | full | reduced |
|---|---:|---:|---:|
| primary-only 0% | 1.02% | -0.15% | -1.01% |
| fallback 2.5% | 0.32% | 0.10% | -0.05% |
| frozen challenger 5% | 0.95% | -0.70% | -0.81% |

The 5% overlay did not fail because its top-score label quality collapsed. Its
average top-50 label spread versus the universe was `-0.017 bps`, better than
primary-only at `-0.338 bps` and 2.5% at `-0.227 bps`. The problem is portfolio
path and state fit: 5% replaced too much of the primary selection set in the
2026 tape. Average monthly top-50 overlap versus primary was only `42.70%` to
`62.77%` for 5%, versus `73.54%` to `81.50%` for 2.5%.

Decision remains unchanged:

- Reject the frozen 5% default-change case.
- Keep 2.5% as watchlist-only.
- Permit only one constrained retry: 2.5% overlay with explicit cost-aware or
  state-gated rules.
- Do not reopen broad overlay-weight search, and do not add another standalone
  gap-fade variant.

## Constrained 2.5% Full/Reduced Gate Retry

The allowed constrained retry was run with only one new rule: apply the 2.5%
gap/weak-tape satellite overlay only when the existing ribbon-dispersion risk
schedule was in `full` or `reduced`; keep `blocked` as primary-only. No new
factor, weight search, or calendar patch was introduced.

- Validation:
  `runs/candidate_factor_portfolios/unseen_2026_overlay_2p5_full_reduced_gate_2026_05_24/validation_summary.json`
- Attribution report:
  `runs/candidate_factor_portfolios/unseen_2026_overlay_2p5_full_reduced_gate_2026_05_24/attribution_v1/attribution_report.md`
- Attribution JSON:
  `runs/candidate_factor_portfolios/unseen_2026_overlay_2p5_full_reduced_gate_2026_05_24/attribution_v1/attribution_summary.json`

Result: reject the local overlay patch. The full/reduced-gated 2.5% overlay
was worse than primary-only and worse than the earlier unconditional 2.5%
watchlist line.

| method | full-base | high-cost | zero-cost | full DD | turnover |
|---|---:|---:|---:|---:|---:|
| primary-only 0% | -0.28% | -0.68% | 1.43% | -5.30% | 11.083 |
| unconditional 2.5% | 0.24% | -0.18% | 1.95% | -6.84% | 11.575 |
| full/reduced-gated 2.5% | -1.01% | -1.40% | 0.49% | -7.02% | 11.440 |

Monthly comparison shows the gated retry gave back too much in February and
March:

| method | Jan | Feb | Mar | Apr | May |
|---|---:|---:|---:|---:|---:|
| primary-only 0% | -0.18% | 2.11% | -2.55% | -0.03% | 0.41% |
| full/reduced-gated 2.5% | 1.36% | 0.70% | -3.34% | -0.03% | 0.36% |

State attribution confirms that the realized path damage came from `full`
state, not just transaction cost:

| method | blocked | full | reduced |
|---|---:|---:|---:|
| primary-only 0% | 1.02% | -0.15% | -1.01% |
| full/reduced-gated 2.5% | 1.64% | -1.75% | -0.74% |

Decision: close the local overlay-patch path. The 2026 high-instability tape
does not justify another small overlay-weight or simple state-gate tweak. The
next useful work should be a genuinely new robustness mechanism, a different
allocator, or a later unseen-data replay, not another gap/weak-tape overlay
variant.
