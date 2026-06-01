# Fixed-Framework Alpha-Rank State Overlay Schedule Attribution - 2026-06-01

This note attributes the promoted `budget_min90_l120` state-aware
gross-exposure schedule. The goal is to verify that the overlay reduces
exposure when the lagged score-basket quality is weak, rather than merely
adding turnover or fitting one month.

## Evidence

- Robustness decision:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_robustness_2026_06_01.md`
- Alpha-rank research benchmark replacement:
  `docs/validation/fixed_framework_alpha_rank_research_benchmark_replacement_2026_06_01.md`
- Promoted standard validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/validation_summary.json`
- Schedule:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_robustness_schedules_2026_06_01/budget_min90_l120/gross_exposure_schedule.csv`
- Attribution artifacts:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/schedule_attribution_2026_06_01/`

The attribution de-duplicates full-base equity curves by timestamp before
joining bar returns, because the streaming backtest output contains repeated
timestamps from chunk padding. State-bucket returns are diagnostic time-slice
attribution, not standalone counterfactual backtests.

## Overall Schedule

| metric | value |
| --- | ---: |
| timestamp rows | 34,799 |
| average scale | 0.9360 |
| floor-0.90 share | 50.96% |
| partial-scale share | 28.10% |
| full-1.00 share | 20.94% |
| warmup share | 0.31% |
| average health score | 0.3581 |
| average rolling top-label | 0.0868% |
| average rolling top-minus-bottom label | 0.3146% |
| average rolling rank IC | 0.0533 |

The schedule is intentionally conservative: it never drops below `0.90`, but it
spends about half the validation bars at that floor.

## Yearly Schedule

| year | avg scale | floor share | partial share | full share | avg health | avg top-label | avg spread | avg rank IC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 0.9413 | 44.84% | 31.08% | 24.08% | 0.4070 | 0.0580% | 0.3010% | 0.0580 |
| 2024 | 0.9294 | 57.43% | 27.54% | 15.03% | 0.2938 | 0.0965% | 0.3891% | 0.0458 |
| 2025 | 0.9374 | 50.59% | 25.68% | 23.73% | 0.3742 | 0.1054% | 0.2536% | 0.0560 |

The overlay reduces exposure most heavily in 2024, which is the year that
created the repaired benchmark's drawdown concern. This supports the intended
use: a portfolio-level quality state, not a generic permanent leverage cut.

## Scale Buckets

| scale bucket | bar share | avg scale | avg top-label | avg spread | avg rank IC | repaired compound | l120 compound | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| floor_0p90 | 50.96% | 0.9000 | -0.2145% | 0.1455% | 0.0377 | -4.22% | -2.62% | +1.60pp |
| partial_0p90_1p00 | 28.10% | 0.9536 | 0.3174% | 0.3678% | 0.0535 | 21.99% | 21.13% | -0.86pp |
| full_1p00 | 20.63% | 1.0000 | 0.5168% | 0.6602% | 0.0914 | 7.76% | 8.84% | +1.08pp |
| warmup | 0.31% | 1.0000 | n/a | n/a | n/a | 1.10% | 1.10% | +0.00pp |

The key sanity check passes: the floor bucket has negative rolling top-label
quality, while the full-exposure bucket has materially stronger top-label,
spread, and rank-IC quality. The repaired benchmark loses `-4.22%` in the floor
timestamps; the l120 path loses only `-2.62%` there.

The partial bucket is the main opportunity-cost area. It is not catastrophic,
but it explains why this overlay still gives back some rebound months.

## Month-Level Readout

Representative de-risked drawdown months show the intended behavior:

| month | avg scale | floor share | avg health | avg top-label | return delta vs repaired | DD delta vs repaired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-06 | 0.9046 | 83.22% | 0.0462 | -0.5223% | +0.64pp | +0.80pp |
| 2024-02 | 0.9181 | 77.36% | 0.1808 | 0.3090% | +1.08pp | +1.22pp |
| 2024-01 | 0.9304 | 62.03% | 0.3044 | -0.2646% | +1.12pp | +1.06pp |
| 2024-04 | 0.9316 | 49.48% | 0.3160 | 0.1838% | +0.78pp | +0.99pp |

The months with the largest gains versus the repaired benchmark are also
mostly the intended drawdown months:

| month | repaired return | l120 return | delta | repaired DD | l120 DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-01 | -11.98% | -10.85% | +1.12pp | -13.69% | -12.63% | +1.06pp |
| 2024-02 | 0.92% | 2.00% | +1.08pp | -15.46% | -14.24% | +1.22pp |
| 2024-04 | -1.59% | -0.81% | +0.78pp | -10.41% | -9.42% | +0.99pp |
| 2024-06 | -8.19% | -7.55% | +0.64pp | -9.65% | -8.85% | +0.80pp |
| 2024-05 | -2.83% | -2.16% | +0.68pp | -5.13% | -4.30% | +0.83pp |

Largest givebacks are mostly opportunity-cost months:

| month | repaired return | l120 return | delta | repaired DD | l120 DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-09 | 15.74% | 14.84% | -0.90pp | -14.79% | -14.07% | +0.72pp |
| 2024-11 | 5.15% | 4.42% | -0.73pp | -6.94% | -6.63% | +0.31pp |
| 2025-02 | 5.47% | 4.77% | -0.71pp | -3.66% | -3.49% | +0.17pp |
| 2025-04 | 0.56% | 0.01% | -0.56pp | -12.99% | -12.85% | +0.14pp |

The giveback profile is acceptable for a risk overlay: most return givebacks
still improve drawdown slightly, but they confirm that this should remain a
frontier candidate rather than a default replacement without a formal review.

## Correlation Check

Across months, more floor exposure is positively associated with drawdown
improvement versus the repaired benchmark:

| schedule metric | monthly return delta corr | monthly DD delta corr |
| --- | ---: | ---: |
| floor share | +0.163 | +0.471 |
| average de-risk amount | +0.109 | +0.455 |
| average scale | -0.109 | -0.455 |
| average rank IC | -0.021 | -0.479 |

This is a weak sample of only 36 months, so it is not treated as statistical
proof. It is still directionally consistent with the mechanism: more de-risking
is tied more strongly to drawdown improvement than to raw return improvement.

## Decision

Keep `budget_min90_l120` as the current state-aware alpha-rank frontier
candidate. The schedule attribution supports the intended mechanism:

1. low exposure is assigned to weaker rolling score-basket states;
2. 2024 receives the strongest de-risking;
3. floor-state timestamps reduce losses versus the repaired no-overlay path;
4. opportunity cost exists but is concentrated in rebound months.

Do not yet replace the production/default framework benchmark. The alpha-rank
research benchmark replacement note promotes this overlay only as the
state-aware frontier and explicitly leaves the production/default benchmark
unchanged.
