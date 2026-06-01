# Fixed-Framework Alpha-Rank State Overlay Screen - 2026-05-31

This note records the first state-aware gross-exposure overlay screen against
the repaired alpha-rank research benchmark.

The overlay uses lagged realized score-basket quality, not live equity
drawdown. It builds a timestamp-level `gross_exposure_scale` schedule from the
repaired benchmark scores and the fixed forward-return labels, shifted by the
label horizon before rolling.

## Evidence

- Repaired benchmark:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- Drawdown overlay screen:
  `docs/validation/fixed_framework_alpha_rank_drawdown_overlay_screen_2026_05_31.md`
- State overlay observations:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_observations_2026_05_31/regime_gate_observations.csv`
- State overlay schedules:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_schedules_2026_05_31/`
- State overlay standard validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l96_2026_05_31_standard/validation_summary.json`
- Monthly comparison to repaired benchmark:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l96_2026_05_31_standard/monthly_comparison_to_repaired.csv`
- Robustness follow-up:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_robustness_2026_06_01.md`
- Alpha-rank research benchmark replacement:
  `docs/validation/fixed_framework_alpha_rank_research_benchmark_replacement_2026_06_01.md`

## Construction

The promoted state overlay candidate is `budget_min90_l96`:

- score source:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/full_base/scores/decorrelated`
- top basket: top 50 names by repaired benchmark score
- lookback: `96` score windows
- min periods: `48`
- label lag: `49` windows
- gate mode: budget
- scale range: `0.90` to `1.00`
- top-return budget range: `-0.10%` to `0.15%`
- spread budget range: `-0.10%` to `0.15%`
- rank-IC budget range: `-0.03` to `0.05`

The schedule has `34,799` timestamp rows. Mean scale is `0.9344`, median scale
is `0.9000`, and the scale never falls below `0.90`. This is deliberately much
less reactive than the drawdown brake, and it acts before equity drawdown is
observed.

## Full-Base Screen

| run | full return | max DD | return delta vs repaired | DD delta vs repaired | cost | trades | avg gross |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repaired benchmark | 27.97% | -30.77% | +0.00pp | +0.00pp | 163,936 | 24,030 | 0.9009 |
| state budget min90 l96 | 29.85% | -29.34% | +1.88pp | +1.42pp | 170,180 | 25,432 | 0.8659 |
| state threshold mild l96 | 28.66% | -29.41% | +0.68pp | +1.36pp | 170,438 | 25,433 | 0.8720 |
| state budget min85 l96 | 27.50% | -28.89% | -0.48pp | +1.88pp | 168,678 | 25,393 | 0.8361 |
| alpha-only v66 control | 27.00% | -28.53% | -0.97pp | +2.23pp | 164,533 | 24,043 | 0.9022 |

`budget_min90_l96` is the only tested state overlay that improves both return
and drawdown versus the repaired benchmark. The more aggressive `budget_min85`
improves drawdown more but gives back return, while the threshold version is
inferior to the budget version.

## Standard Validation

| run | full base | high cost | 2023 | 2024 | 2025 | full DD | high-cost DD | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alpha-only v66 | 27.00% | 21.19% | 1.45% | -4.59% | 18.09% | -28.53% | -29.52% | warn |
| repaired benchmark | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | -30.77% | -31.70% | pass |
| drawdown brake t25 s90 | 27.05% | 21.71% | 1.09% | 2.87% | 19.60% | -30.53% | -31.33% | pass |
| state budget min90 l96 | 29.85% | 23.93% | 2.30% | 0.89% | 17.21% | -29.34% | -30.40% | pass |

The state overlay passes all standard validation gates. Compared with the
repaired benchmark, it improves:

- full-base return by `+1.88pp`;
- high-cost return by `+1.78pp`;
- full-base max drawdown by `+1.42pp`;
- high-cost max drawdown by `+1.30pp`.

The tradeoffs are also clear:

- 2025 yearly return falls from `19.71%` to `17.21%`;
- 2024 remains positive but is effectively flat versus repaired
  (`0.89%` versus `0.90%`);
- full-base transaction cost rises from `163,936` to `170,180`;
- full-base trade count rises from `24,030` to `25,432`.

## Monthly Readout

The full-path monthly improvements are concentrated in the 2024 drawdown
months:

| month | repaired return | state return | delta | repaired DD | state DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-06 | -8.19% | -7.21% | +0.98pp | -9.65% | -8.65% | +1.01pp |
| 2024-04 | -1.59% | -0.89% | +0.71pp | -10.41% | -9.72% | +0.69pp |
| 2024-01 | -11.98% | -11.36% | +0.61pp | -13.69% | -13.06% | +0.63pp |
| 2024-05 | -2.83% | -2.28% | +0.55pp | -5.13% | -4.47% | +0.66pp |

The largest monthly givebacks are mostly rebound or 2025 opportunity-cost
months:

| month | repaired return | state return | delta | repaired DD | state DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-09 | 15.74% | 14.87% | -0.87pp | -14.79% | -14.27% | +0.51pp |
| 2024-07 | 1.69% | 0.88% | -0.81pp | -6.26% | -6.36% | -0.10pp |
| 2025-04 | 0.56% | -0.16% | -0.73pp | -12.99% | -13.17% | -0.18pp |
| 2024-03 | 5.04% | 4.39% | -0.65pp | -4.56% | -4.34% | +0.22pp |

The overlay does what the drawdown brake failed to do: it improves the bad
2024 path months before reacting to equity drawdown. The cost is that it trims
some profitable rebound and 2025 exposure.

## Decision

This first screen promoted `state budget min90 l96` to the state-aware
alpha-rank frontier candidate. The 2026-06-01 robustness follow-up supersedes
that exact parameter choice with `state budget min90 l120`, which improves the
full-window and high-cost result while preserving positive yearly slices.

Do not replace the production/default benchmark from this first screen alone.
The robustness follow-up addresses local parameter stability, but benchmark
replacement still needs a separate default-change review because the overlay is
selected on the fixed 2023-2025 validation window.

## Next Test

1. Use `budget_min90_l120` as the frontier comparison for incremental
   alpha-rank factor tests, and keep the repaired no-overlay benchmark as the
   control.
2. Leave production/default benchmark replacement behind a separate
   default-change review.
