# Fixed-Framework Alpha-Rank Research Benchmark Replacement - 2026-06-01

This note records the alpha-rank research benchmark update after the repaired
benchmark, state-overlay robustness check, and schedule attribution.

This is a research-frontier replacement only. It does not change the
production/default framework benchmark or the active/default allocator.

## Evidence

- Framework v1 standard benchmark replacement:
  `docs/validation/framework_v1_benchmark_replacement_2026_05_31.md`
- Candidate baseline rebuild:
  `docs/validation/fixed_framework_candidate_baseline_2026_05_31.md`
- Repaired no-overlay benchmark attribution:
  `docs/validation/fixed_framework_alpha_rank_repaired_benchmark_attribution_2026_05_31.md`
- State-overlay robustness:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_robustness_2026_06_01.md`
- State-overlay schedule attribution:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_schedule_attribution_2026_06_01.md`
- Repaired no-overlay validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- Promoted state-aware validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/validation_summary.json`
- First incremental alpha-rank factor screen:
  `docs/validation/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_screen_2026_06_01.md`
- d10 2025 generalization attribution:
  `docs/validation/fixed_framework_alpha_rank_daily_ma_d10dev_2025_generalization_attribution_2026_06_01.md`

## Benchmark Stack

| layer | artifact | role |
| --- | --- | --- |
| production/default framework benchmark | `runs/framework_v1_acceptance/standard/candidate_policy_validation/validation_summary.json` | Accepted standard framework validation. Unchanged by this note. |
| naive alpha-rank control | `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/validation_summary.json` | Registry-v66 alpha-only comparison; useful for ablation, not frontier. |
| no-overlay alpha-rank control | `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json` | Current control for alpha-rank factor tests. |
| state-aware alpha-rank frontier | `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/validation_summary.json` | Current research frontier for fixed-framework alpha-rank portfolio tests. |

New fixed-framework alpha-rank factor tests should report marginal contribution
against both the no-overlay control and the state-aware frontier when the
candidate can be composed with the state overlay.

## Standard Results

| run | status | full base | high cost | 2023 | 2024 | 2025 | full DD | high-cost DD | mean turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production/default standard | pass | 25.23% | 19.65% | 1.00% | 4.17% | 16.65% | -28.96% | -29.96% | 73.09 |
| alpha-only v66 | warn | 27.00% | 21.19% | 1.45% | -4.59% | 18.09% | -28.53% | -29.52% | 73.61 |
| repaired no-overlay control | pass | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | -30.77% | -31.70% | 73.23 |
| state budget min90 l120 | pass | 30.32% | 24.39% | 1.47% | 0.86% | 18.28% | -28.92% | -29.87% | 71.02 |

The state-aware frontier improves the repaired no-overlay control by:

- `+2.35pp` full-base return;
- `+2.23pp` high-cost return;
- `+1.85pp` full-base max drawdown;
- `+1.83pp` high-cost max drawdown;
- `-2.21` mean gross-turnover points.

The tradeoff is lower yearly return in each isolated year slice:

- 2023: `1.72%` to `1.47%`;
- 2024: `0.90%` to `0.86%`;
- 2025: `19.71%` to `18.28%`.

All yearly slices remain positive, so the candidate still clears the standard
validation gates.

## What Changed

The no-overlay control keeps the repaired score construction:

1. use only registry `candidate` factors with admission statuses `candidate`
   and `watchlist`;
2. restrict ordinary score construction to `evaluation_role=alpha_rank`;
3. use `decorrelated` weights, rank score transform, and equal
   admission-evidence mode;
4. apply the lagged deep25 factor-weight schedule only to
   `intraday_overnight_gap_5m`;
5. cap row-level factor contribution at `25%`.

The state-aware frontier adds one portfolio-level exposure schedule:

- schedule: `budget_min90_l120`;
- score basket: top 50 names by repaired benchmark score;
- lookback: `120` score windows;
- min periods: `60`;
- label lag: `49` windows;
- gross exposure scale range: `0.90` to `1.00`.

The overlay does not alter factor scores or factor admission. It only caps
gross exposure by timestamp.

## Attribution Readout

The schedule attribution supports the intended mechanism:

- floor-0.90 timestamps have negative rolling top-label quality: `-0.2145%`;
- full-1.00 timestamps have stronger rolling top-label quality: `+0.5168%`;
- 2024 receives the strongest de-risking, with average scale `0.9294` and
  floor share `57.43%`;
- floor-state timestamps reduce the repaired path loss from `-4.22%` to
  `-2.62%`.

The overlay is not free. It trims exposure in some rebound/opportunity months,
especially 2024-09, 2024-11, and 2025-02. That is acceptable for a research
frontier, but it is why this note does not promote the overlay to the
production/default benchmark.

## Decision

Promote `state budget min90 l120` to the current fixed-framework alpha-rank
research frontier benchmark.

Keep the repaired no-overlay benchmark as the alpha-rank control benchmark.
This matters because new alpha factors must show whether their contribution is
real alpha improvement, interaction with the state overlay, or merely exposure
timing.

Do not replace the production/default framework benchmark or active/default
allocator from this note. That requires a separate default-change review with
out-of-window evidence or a clearly accepted production validation protocol.

## Next Steps

1. Test a coarse state-conditioned daily-MA deviation sleeve before testing a
   broader daily-MA batch.
2. For any new factor that improves only one of the two benchmark layers,
   classify the result as interaction evidence rather than immediate promotion
   evidence.
3. Do not run another overlay grid until a new factor batch or a new
   out-of-window validation sample justifies it.
