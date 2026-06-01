# Fixed-Framework Alpha-Rank Process Migration Audit - 2026-06-01

This audit checks whether the documentation and governance artifacts now point
future fixed-framework alpha-rank factor work to the repaired process.

## Current Required Flow

New fixed-framework alpha-rank factor work must use this flow:

1. Register the hypothesis and check research memory.
2. Run single-factor admission with registry-aware roles.
3. Compare incremental portfolio behavior against both:
   - repaired no-overlay alpha-rank control:
     `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/`
   - state-aware alpha-rank frontier:
     `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/`
4. Report full-window base-cost, high-cost, yearly slices, drawdown, turnover,
   cost, and selection displacement.
5. For conditional candidates, use lagged observable state. Require positive
   enabled-state selection-displacement before standard validation.
6. Use score-level switching when the disabled state is intended to preserve
   the baseline score stream exactly. Do not treat factor-weight scaling inside
   a recomputed score stack as a baseline-preserving sleeve.
7. Do not tune calendar months, single-year patches, or d10 broad-tape
   thresholds.

## Files Reviewed

Primary entry points:

- `README.md`
- `docs/validation/factor_development_standard.md`
- `docs/validation/factor_admission.md`
- `docs/strategy/candidate_factor_portfolios.md`

Alpha-rank benchmark and d10 notes:

- `docs/validation/fixed_framework_alpha_rank_research_benchmark_replacement_2026_06_01.md`
- `docs/validation/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_screen_2026_06_01.md`
- `docs/validation/fixed_framework_alpha_rank_daily_ma_d10dev_2025_generalization_attribution_2026_06_01.md`
- `docs/validation/fixed_framework_alpha_rank_d10_state_sleeve_screen_2026_06_01.md`

Governance artifact:

- `configs/factors/factor_registry.json`

## Migration Fixes Applied

The audit found three migration gaps and repaired them:

1. `README.md` still presented the 2026-05-25 volume-concentration optimizer
   branch as the latest research frontier without distinguishing alpha-rank
   factor work. It now points alpha-rank work to the repaired no-overlay control
   plus `budget_min90_l120` state-aware frontier.
2. `docs/validation/factor_development_standard.md` did not make
   selection-displacement and score-level switching explicit requirements for
   conditional alpha-rank candidates. It now does.
3. `configs/factors/factor_registry.json` still allowed daily-MA retry as a
   generic incremental or gated state package. It now records the failed d10
   incremental/state-sleeve evidence and blocks broad-tape threshold retuning.

## Confirmed Current State

- The production/default framework benchmark is unchanged.
- The fixed-framework alpha-rank research benchmark is the two-layer stack:
  no-overlay control plus state-aware frontier.
- d10 is documented as non-promoted:
  - unconditional alpha-rank addition is blocked by yearly-slice damage;
  - lagged broad-tape score-switch sleeve fails full-base and drawdown versus
    the no-overlay control;
  - broad-tape threshold tuning is explicitly closed.
- `examples/build_timestamp_score_switch.py` is the intended harness for
  baseline-preserving conditional score sleeves.
- Factor-weight schedules are still allowed for explicit alpha transforms or
  risk gates, but not as proof that disabled states equal the baseline.

## Residual Historical Documents

Older strategy batch documents remain in the repository as historical logs.
They may mention previous frontiers, branches, or decisions. The current entry
documents now label these as historical and point new factor development to the
updated alpha-rank process.

When a historical document conflicts with this audit or with
`fixed_framework_alpha_rank_research_benchmark_replacement_2026_06_01.md`, the
newer fixed-framework alpha-rank process controls future work.
