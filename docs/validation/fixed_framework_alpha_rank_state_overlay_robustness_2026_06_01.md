# Fixed-Framework Alpha-Rank State Overlay Robustness - 2026-06-01

This note records the local robustness check around the state-aware
gross-exposure overlay. The tested overlays all use the repaired alpha-rank
benchmark scores, top-50 score-basket quality, a `49` window label lag, and a
budget-style exposure schedule.

## Evidence

- Repaired benchmark:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- Original state overlay screen:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_screen_2026_05_31.md`
- Robustness schedules:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_robustness_schedules_2026_06_01/`
- Robustness full-base screens:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_robustness_*_full_base_screen/summary.json`
- Promoted `budget_min90_l120` standard validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/validation_summary.json`
- Monthly comparison:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/monthly_comparison_to_repaired_and_l96.csv`
- Schedule attribution:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_schedule_attribution_2026_06_01.md`
- Alpha-rank research benchmark replacement:
  `docs/validation/fixed_framework_alpha_rank_research_benchmark_replacement_2026_06_01.md`

## Schedule Grid

| schedule | lookback | min scale | budget range | mean scale | median | scale < 0.95 | scale < 0.90 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| budget_min90_l96 | 96 | 0.90 | base | 0.9344 | 0.9000 | 22,676 | 0 |
| budget_min90_l72 | 72 | 0.90 | base | 0.9329 | 0.9000 | 23,257 | 0 |
| budget_min90_l120 | 120 | 0.90 | base | 0.9360 | 0.9000 | 21,910 | 0 |
| budget_min88_l96 | 96 | 0.88 | base | 0.9213 | 0.8800 | 23,449 | 19,953 |
| budget_min92_l96 | 96 | 0.92 | base | 0.9475 | 0.9200 | 21,636 | 0 |
| budget_min90_l96_wide | 96 | 0.90 | wider | 0.9343 | 0.9014 | 22,676 | 0 |
| budget_min90_l96_tight | 96 | 0.90 | tighter | 0.9345 | 0.9000 | 22,676 | 0 |

The grid tests whether the result depends on one exact lookback, floor, or
budget boundary. The important failure case is `budget_min88_l96`: allowing the
schedule below `0.90` cuts exposure too aggressively and destroys the return
improvement.

## Full-Base Robustness

| run | full return | max DD | delta vs repaired | DD delta vs repaired | delta vs l96 | DD delta vs l96 | cost | trades | avg gross |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repaired benchmark | 27.97% | -30.77% | +0.00pp | +0.00pp | -1.88pp | -1.42pp | 163,936 | 24,030 | 0.9009 |
| state budget min90 l96 | 29.85% | -29.34% | +1.88pp | +1.42pp | +0.00pp | +0.00pp | 170,180 | 25,432 | 0.8659 |
| budget_min90_l72 | 28.99% | -29.76% | +1.02pp | +1.00pp | -0.86pp | -0.42pp | 170,736 | 25,561 | 0.8658 |
| budget_min90_l120 | 30.32% | -28.92% | +2.35pp | +1.85pp | +0.47pp | +0.43pp | 170,445 | 25,427 | 0.8650 |
| budget_min88_l96 | 23.70% | -30.72% | -4.27pp | +0.05pp | -6.15pp | -1.37pp | 167,431 | 25,191 | 0.8517 |
| budget_min92_l96 | 28.74% | -29.92% | +0.77pp | +0.85pp | -1.12pp | -0.58pp | 168,288 | 25,114 | 0.8725 |
| budget_min90_l96_wide | 28.19% | -29.53% | +0.22pp | +1.23pp | -1.66pp | -0.19pp | 169,320 | 25,369 | 0.8648 |
| budget_min90_l96_tight | 28.17% | -29.51% | +0.20pp | +1.26pp | -1.69pp | -0.16pp | 170,440 | 25,478 | 0.8663 |

The result survives local perturbation. Four nearby variants improve both
full-window return and drawdown versus the repaired benchmark:
`budget_min90_l72`, `budget_min90_l120`, `budget_min92_l96`, and the original
`budget_min90_l96`. The exact `l96` choice is therefore not a one-point
artifact.

The best full-base variant is `budget_min90_l120`, so it received full
standard validation.

## Standard Validation

| run | full base | high cost | 2023 | 2024 | 2025 | full DD | high-cost DD | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repaired benchmark | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | -30.77% | -31.70% | pass |
| state budget min90 l96 | 29.85% | 23.93% | 2.30% | 0.89% | 17.21% | -29.34% | -30.40% | pass |
| state budget min90 l120 | 30.32% | 24.39% | 1.47% | 0.86% | 18.28% | -28.92% | -29.87% | pass |

Compared with the repaired benchmark, `budget_min90_l120` improves:

- full-base return by `+2.35pp`;
- high-cost return by `+2.23pp`;
- full-base max drawdown by `+1.85pp`;
- high-cost max drawdown by `+1.83pp`.

Compared with `budget_min90_l96`, `budget_min90_l120` improves:

- full-base return by `+0.47pp`;
- high-cost return by `+0.46pp`;
- 2025 return by `+1.07pp`;
- full-base max drawdown by `+0.43pp`;
- high-cost max drawdown by `+0.53pp`.

The tradeoff is weaker early-year stability: `2023` falls from `2.30%` to
`1.47%`, and `2024` falls from `0.89%` to `0.86%`. Both remain positive, so
the standard validation status is still `pass`.

## Monthly Readout

Largest `budget_min90_l120` improvements versus `budget_min90_l96`:

| month | l96 return | l120 return | delta | l96 DD | l120 DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-02 | 1.08% | 2.00% | +0.92pp | -14.89% | -14.24% | +0.65pp |
| 2024-07 | 0.88% | 1.61% | +0.73pp | -6.36% | -6.15% | +0.21pp |
| 2024-01 | -11.36% | -10.85% | +0.51pp | -13.06% | -12.63% | +0.43pp |
| 2023-03 | -2.48% | -2.08% | +0.40pp | -4.70% | -4.36% | +0.33pp |

Largest givebacks versus `budget_min90_l96`:

| month | l96 return | l120 return | delta | l96 DD | l120 DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-11 | 5.10% | 4.42% | -0.68pp | -6.58% | -6.63% | -0.04pp |
| 2025-02 | 5.43% | 4.77% | -0.66pp | -3.50% | -3.49% | +0.01pp |
| 2025-07 | 3.25% | 2.69% | -0.56pp | -2.60% | -2.46% | +0.14pp |
| 2023-11 | 1.60% | 1.08% | -0.52pp | -2.36% | -2.46% | -0.09pp |

The l120 variant improves the original drawdown months more cleanly, especially
2024-01 and 2024-02. It gives back some rebound months, but the net result is
still better on full-window and high-cost return.

## Decision

Promote `state budget min90 l120` to the current state-aware alpha-rank
frontier candidate. Keep the repaired no-overlay benchmark as the control.

Do not replace the production/default framework benchmark yet. The state-aware
overlay now has local robustness support, but it is still selected on the fixed
2023-2025 validation window and needs a benchmark-change review before becoming
the default.

## Next Tests

1. Use `budget_min90_l120` as the frontier comparison for the next incremental
   alpha-rank factor tests, with the repaired no-overlay benchmark as the
   control.
2. Leave the production/default benchmark unchanged until a separate
   default-change review.
