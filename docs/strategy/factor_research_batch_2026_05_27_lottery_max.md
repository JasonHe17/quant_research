# Factor Research Batch - 2026-05-27 Lottery MAX Effect

This batch adds a cross-sectional MAX-effect signal for 5-minute A-share bars.

## Hypothesis

Stocks with unusually large recent single-bar gains attract lottery-demand
chasing and become temporarily overpriced. The proposed factor therefore
penalizes high rolling MAX returns and rewards names with lower recent upside
spikes inside the same timestamp cross-section.

The signal is intentionally distinct from time-series reversal:

- `intraday_reversal` uses cumulative return over a lookback window.
- `intraday_cross_sectional_reversal` demeans cumulative return by the market
  move.
- `lottery_max` uses only the largest single 5-minute return in the lookback
  window, then converts it to an inverted timestamp-level cross-sectional rank.

## Implementation

Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`

Factor group: `lottery_max`

Default windows: `24`, `48`, and `96` five-minute bars.

Emitted features:

| feature | interpretation |
|---|---|
| `intraday_lottery_max_5m_w24` | Inverted cross-sectional percentile of the rolling 24-bar maximum close return. |
| `intraday_lottery_max_5m_w48` | Inverted cross-sectional percentile of the rolling 48-bar maximum close return. |
| `intraday_lottery_max_5m_w96` | Inverted cross-sectional percentile of the rolling 96-bar maximum close return. |

For each instrument and timestamp:

```text
bar_return = close / previous_close - 1
rolling_max = rolling_max(bar_return, window)
score = 1 - cross_sectional_percentile(rolling_max at timestamp)
```

High raw MAX values receive low scores. The registered admission direction is
therefore `long` because the feature is already inverted.

## Validation Plan

The research-memory check flags this idea as similar to prior intraday reversal
work because it shares the reversal family, close-price inputs, and nearby
lookback horizons. That is a valid caution for cost-aware validation, but it is
not an exact rerun: the transform ranks the single largest positive bar rather
than cumulative return, median-demeaned return, or tail-only reversal.

Build a new-factor-only dataset:

```bash
conda run -n quant python examples/build_baseline_a_alpha_dataset.py \
  --start 2023-01-01T00:00:00+08:00 \
  --end 2025-12-31T23:59:59+08:00 \
  --factor-groups lottery_max \
  --output-dir runs/factor_research/lottery_max_2026_05_27/alpha_dataset \
  --workers 1
```

Then run the standard single-factor evaluation and admission review against the
48-bar forward return label. Portfolio review should only proceed for admitted
features and should test incremental value versus the current composite score.

## Portfolio Integration Follow-Up

The admitted MAX factors were first validated as standalone candidate scores
and then as overlays on the current research-frontier volume-concentration
optimizer score:

- Primary score:
  `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_cost_pressure_cap0010_standard/scores/vc_opt_risk_cp0010_w50`
- Satellite score:
  `runs/factor_research/lottery_max_2026_05_27/candidate_policy_validation/full_base/scores/equal`
- Execution policy:
  `cost_aware_optimizer_budget155_cost_pressure_cap0010_daily`
- Cost control: path gross-turnover budget 155, cost-pressure turnover cap
  0.01 after 1000 bps realized transaction-cost pressure.

Unconditional MAX overlays did not beat the research frontier. The best
unconditional weight, `frontier_lottery_w10`, improved drawdown but reduced
full-window and high-cost returns relative to `vc_opt_risk_cp0010_w50`.

Regime diagnostics showed that the lottery-MAX contribution is state dependent.
Restricting the overlay to `risk_state=full` failed: every tested weight
generated a negative 2024 base slice. Reversing the condition was useful. The
validated branch only applies the MAX satellite when the daily MA/ribbon
schedule is not full risk:

```text
risk_state in {reduced, blocked, warmup}
```

Artifacts:

- `runs/factor_research/lottery_max_2026_05_27/overlay_frontier_vc_cp0010_w50_lottery_full_risk_state_standard/validation_summary.json`
- `runs/factor_research/lottery_max_2026_05_27/overlay_frontier_vc_cp0010_w50_lottery_nonfull_risk_state_w10_standard/validation_summary.json`
- `runs/factor_research/lottery_max_2026_05_27/overlay_frontier_vc_cp0010_w50_lottery_nonfull_risk_state_grid_standard/validation_summary.json`

The focused non-full-risk grid passed with zero failures and zero warnings:

| method | full base | high cost | 2023 base | 2024 base | 2025 base | worst return | worst drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vc_opt_risk_cp0010_w50` baseline | 16.07% | 8.10% | 5.40% | 3.22% | 24.13% | 3.22% | -22.26% |
| `frontier_lottery_nonfullrisk_grid_w05` | 17.05% | 7.04% | 5.66% | 6.36% | 20.86% | 5.66% | -20.95% |
| `frontier_lottery_nonfullrisk_grid_w075` | 16.45% | 7.46% | 6.06% | 10.23% | 20.34% | 6.06% | -19.87% |
| `frontier_lottery_nonfullrisk_grid_w10` | 16.22% | 8.47% | 6.16% | 7.39% | 19.49% | 6.16% | -19.93% |
| `frontier_lottery_nonfullrisk_grid_w125` | 15.40% | 6.86% | 4.69% | 6.89% | 19.16% | 4.69% | -19.42% |

`w05` maximizes full-base return but sacrifices doubled-cost performance.
`w075` gives the best 2024 base slice and mean scenario return, but also stays
below the frontier in the doubled-cost stress. `w10` is the most robust
candidate because it improves full-base return, high-cost return, worst annual
return, and worst drawdown versus the current frontier baseline at the same
policy.

Current decision: keep lottery MAX as a conditional frontier overlay candidate,
not a default replacement. The preferred branch is
`frontier_lottery_nonfullrisk_grid_w10`. It should move next to capacity and
out-of-sample/live-simulation checks before allocator-registry promotion. Do
not continue tuning full-risk activation; that branch is rejected.

## Capacity Follow-Up

The preferred branch was tested under same-bar turnover participation caps and
compared with the underlying `vc_opt_risk_cp0010_w50` frontier baseline. The
stress uses full-base costs, `allow_same_bar_capacity`, and the same
cost-aware optimizer policy.

Artifacts:

- `runs/factor_research/lottery_max_2026_05_27/capacity_frontier_lottery_nonfullrisk_grid_w10_5pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/capacity_frontier_lottery_nonfullrisk_grid_w10_2pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/capacity_baseline_vc_opt_risk_cp0010_w50_5pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/capacity_baseline_vc_opt_risk_cp0010_w50_2pct_full_base/summary.json`

| method | cap | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|---:|
| `frontier_lottery_nonfullrisk_grid_w10` | 5% | 6.99% | -7.46% | 120.33 | 3,647 | 41.06% | 50.47% |
| `frontier_lottery_nonfullrisk_grid_w10` | 2% | 2.96% | -6.08% | 73.41 | 6,180 | 135.60% | 67.85% |
| `vc_opt_risk_cp0010_w50` baseline | 5% | 6.89% | -8.71% | 120.60 | 3,382 | 37.38% | 48.23% |
| `vc_opt_risk_cp0010_w50` baseline | 2% | 2.10% | -6.24% | 72.83 | 5,717 | 129.51% | 66.86% |

The conditional MAX overlay does not materially worsen capacity versus the
frontier baseline and keeps a small return advantage in both capped scenarios.
However, the absolute capacity diagnostics are not governance-clean: the 5%
cap already has unfilled capacity notional above 40% of traded notional, and
the 2% cap exceeds 100% of traded notional. This violates the previous
capacity-monitoring standard used for event-proxy allocators.

Updated decision: do not promote `frontier_lottery_nonfullrisk_grid_w10` yet.
Keep it as a research-frontier overlay candidate only. The next useful work is
not more MAX weight tuning; it is an execution/capacity variant, for example a
liquidity-aware entry filter, lower per-rebalance entries, or a capacity-aware
optimizer constraint applied to both the frontier baseline and MAX overlay.

## Execution-Capacity Probe

The next probe tested execution controls on the preferred conditional MAX
overlay. Lowering `max_entries_per_rebalance` and `max_exits_per_rebalance` to
5 was rejected because it hurt alpha before solving the capacity problem:
full-base return fell to 11.64% and max drawdown deepened to -10.95%.

A per-rebalance gross-turnover cap was more useful. Setting
`policy_max_gross_turnover_per_rebalance=0.005` from the start reduced realized
turnover and capacity pressure, but by itself it allowed a much higher average
target gross exposure and produced an unacceptable -26% drawdown. Adding a
static gross-exposure scale made the path more usable.

Artifacts:

- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_entries5_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_capacity_5pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_capacity_2pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross05_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross05_capacity_5pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross05_capacity_2pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross035_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross035_capacity_5pct_full_base/summary.json`

| branch | capacity cap | return | max drawdown | gross turnover | unfilled / traded |
|---|---:|---:|---:|---:|---:|
| base conditional MAX | none | 16.22% | -8.23% | 144.77 | 0.00% |
| entries/exits 5 | none | 11.64% | -10.95% | 152.80 | 0.00% |
| turnover cap 0.005 | none | 49.11% | -26.01% | 95.85 | 0.00% |
| turnover cap 0.005 | 5% | 46.83% | -25.89% | 90.58 | 9.69% |
| turnover cap 0.005 | 2% | 39.77% | -25.71% | 77.48 | 35.15% |
| turnover cap 0.005 + gross 0.50 | none | 24.03% | -15.24% | 27.83 | 0.00% |
| turnover cap 0.005 + gross 0.50 | 5% | 23.86% | -15.19% | 26.93 | 7.28% |
| turnover cap 0.005 + gross 0.50 | 2% | 23.17% | -14.99% | 24.47 | 28.22% |
| turnover cap 0.005 + gross 0.35 | none | 13.39% | -11.81% | 18.79 | 0.00% |
| turnover cap 0.005 + gross 0.35 | 5% | 13.09% | -11.72% | 18.40 | 4.68% |

The first capacity-clean branch is `turnover_cap005_gross035` under the 5%
same-bar participation stress. It passes the prior unfilled/traded warning
threshold, but it gives up too much return versus the current uncapped frontier
baseline. The more attractive `gross05` branch has a stronger return/drawdown
profile but still misses the 5% capacity threshold.

Decision: do not promote an execution variant yet. The promising next branch is
between gross 0.35 and 0.50, with the same 0.005 per-rebalance turnover cap.
Run a narrow gross-exposure grid, for example 0.40 and 0.45, under full-base and
5% capacity stress before spending time on full annual/high-cost validation.

### Narrow Gross-Exposure Grid

The narrow grid kept `policy_max_gross_turnover_per_rebalance=0.005` and tested
static gross-exposure scales between the capacity-clean 0.35 branch and the
higher-return 0.50 branch.

Artifacts:

- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross040_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross040_capacity_5pct_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross045_full_base/summary.json`
- `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross045_capacity_5pct_full_base/summary.json`

| gross scale | cap | return | max drawdown | gross turnover | limited events | unfilled / traded |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | none | 13.39% | -11.81% | 18.79 | 0 | 0.00% |
| 0.35 | 5% | 13.09% | -11.72% | 18.40 | 260 | 4.68% |
| 0.40 | none | 18.36% | -11.79% | 22.44 | 0 | 0.00% |
| 0.40 | 5% | 18.18% | -11.76% | 21.91 | 330 | 5.18% |
| 0.45 | none | 21.11% | -12.46% | 32.77 | 0 | 0.00% |
| 0.45 | 5% | 20.70% | -12.40% | 31.96 | 479 | 5.50% |
| 0.50 | none | 24.03% | -15.24% | 27.83 | 0 | 0.00% |
| 0.50 | 5% | 23.86% | -15.19% | 26.93 | 490 | 7.28% |

`gross040` is the best current compromise. It preserves a full-base return
above the research-frontier baseline while cutting turnover materially. Its 5%
capacity stress is only slightly above the prior 5% unfilled/traded warning
threshold, and drawdown remains close to the capacity-clean `gross035` branch.
`gross045` is a higher-return alternative, but the capacity miss is larger and
drawdown starts to rise. `gross050` remains too capacity-sensitive and has
larger drawdown.

Updated decision: carry `turnover_cap005_gross040` forward as the execution
candidate for standard validation. Treat `gross035` as the conservative
capacity-clean fallback and `gross045` as a watchlist upper-bound. The next
step should rerun `gross040` across annual base slices and high-cost full
window before any allocator-registry work.

### Standard validation of `turnover_cap005_gross040`

Path:

- `runs/factor_research/lottery_max_2026_05_27/overlay_frontier_vc_cp0010_w50_lottery_nonfull_risk_state_w10_turnover_cap005_gross040_standard/validation_summary.json`

The unified validation run completed with `overall_status=pass`, `failed=0`,
and `warnings=0`.

| branch | scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|---:|
| `tc005_g040_w10` | full base | 18.36% | -11.79% | 22.44 | 33,866 |
| `tc005_g040_w10` | high cost | 18.03% | -11.41% | 27.48 | 49,447 |
| `tc005_g040_w10` | 2023 base | 3.37% | -3.83% | 18.66 | 22,983 |
| `tc005_g040_w10` | 2024 base | 4.47% | -7.59% | 22.43 | 27,174 |
| `tc005_g040_w10` | 2025 base | 11.25% | -4.63% | 13.95 | 18,625 |
| baseline `vc_opt_risk_cp0010_w50` | full base | 16.07% | -8.32% | 146.23 | 138,758 |
| baseline `vc_opt_risk_cp0010_w50` | high cost | 8.10% | -13.44% | 85.28 | 147,920 |
| original overlay `grid_w10` | full base | 16.22% | -8.23% | 144.77 | 138,849 |
| original overlay `grid_w10` | high cost | 8.47% | -14.15% | 84.80 | 147,605 |

This execution variant is materially more cost-stable than the original
overlay: full-base return improves to 18.36%, high-cost return remains 18.03%,
and gross turnover falls from roughly 145 to 22. The tradeoff is concentration
and exposure timing: drawdown rises versus the baseline full-base profile, and
the annual slices are weaker in 2023 and 2024, while 2025 remains strong.

Decision: `turnover_cap005_gross040` passes standard validation as a
portfolio-construction candidate, not as a direct replacement for the existing
baseline. It should be tested next in the downstream combination/allocator
flow with explicit capacity guardrails. Keep `gross035` as the capacity-clean
fallback if allocator-level capacity diagnostics penalize the 0.40 exposure
scale.

### Allocator guardrail capacity diagnostics for `gross040`

Capacity monitor summary:

- `runs/factor_research/lottery_max_2026_05_27/capacity_diagnostics_turnover_cap005_gross040_summary.json`
- New 2% stress artifact: `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross040_capacity_2pct_full_base/summary.json`

| scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|
| no capacity cap | 18.36% | -11.79% | 22.44 | 0 | 0.00% | 0.00% |
| 5% same-bar cap | 18.18% | -11.76% | 21.91 | 330 | 5.18% | 47.59% |
| 2% same-bar cap | 17.69% | -11.71% | 20.34 | 1,247 | 21.23% | 54.18% |

Read: return and drawdown are stable under capacity clipping, so the alpha is
not purely coming from unfillable tiny-bar trades. However, the allocator
capacity monitor should not mark this branch as clean. The 5% same-bar cap is
slightly above the 5% unfilled/traded warning threshold, and the 2% stress is a
clear breach on both unfilled/traded and unfilled/desired. This is acceptable
for a research-size combination candidate, but not for registry promotion with
`capacity_checked` governance unless the allocator either uses the lower
`gross035` fallback or adds a stronger liquidity/capacity filter.

Next decision: do not add `gross040` to `candidate_allocator_registry.json` yet.
The current allocator registry schema also cannot faithfully represent this
score-level construction, because the branch is a conditional overlay of the
VC optimizer score and the lottery MAX satellite rather than a direct
registered-factor weighted basket. The next practical step is to test either
`gross035` as the registry-safe fallback, or build a reusable score-overlay
allocator definition before registry promotion.

### Fallback validation for `gross035`

Paths:

- Standard validation: `runs/factor_research/lottery_max_2026_05_27/overlay_frontier_vc_cp0010_w50_lottery_nonfull_risk_state_w10_turnover_cap005_gross035_standard/validation_summary.json`
- Capacity monitor summary: `runs/factor_research/lottery_max_2026_05_27/capacity_diagnostics_turnover_cap005_gross035_summary.json`
- New 2% stress artifact: `runs/factor_research/lottery_max_2026_05_27/execution_probe_nonfullrisk_w10_turnover_cap005_gross035_capacity_2pct_full_base/summary.json`

The unified validation run completed with `overall_status=pass`, `failed=0`,
and `warnings=0`.

| branch | scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|---:|
| `tc005_g035_w10` | full base | 13.39% | -11.81% | 18.79 | 29,517 |
| `tc005_g035_w10` | high cost | 17.91% | -9.34% | 30.41 | 55,093 |
| `tc005_g035_w10` | 2023 base | 1.95% | -3.70% | 16.64 | 21,369 |
| `tc005_g035_w10` | 2024 base | 5.48% | -6.60% | 16.26 | 20,994 |
| `tc005_g035_w10` | 2025 base | 8.74% | -3.73% | 15.52 | 21,889 |

| scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|
| no capacity cap | 13.39% | -11.81% | 18.79 | 0 | 0.00% | 0.00% |
| 5% same-bar cap | 13.09% | -11.72% | 18.40 | 260 | 4.68% | 48.60% |
| 2% same-bar cap | 12.82% | -11.68% | 17.13 | 1,075 | 20.16% | 53.51% |

Decision: `gross035` is not an attractive replacement for `gross040` or the
research-frontier baseline. It is 5% capacity-clean, but its full-base return is
below the baseline and its 2% same-bar stress still breaches the unfilled/traded
capacity guardrail. Keep it only as a conservative fallback reference. Do not
promote it to the allocator registry.

Updated next step: stop tuning gross exposure on this branch. Further progress
requires either a liquidity-aware/capacity-aware score filter before trade
selection, or framework work to express and validate score-level conditional
overlays with capacity monitors. The current MAX overlay has useful alpha, but
execution/capacity integration remains the blocker for allocator promotion.

### Framework extension status

The allocator framework now supports score-level conditional overlays without
forcing them into the registered-factor basket schema:

- `score.construction = "factor_basket"` remains the default and preserves the
  existing allocator registry behavior.
- `score.construction = "score_overlay"` can declare `primary_score_dir`,
  `satellite_score_dir`, `method_prefix`, `overlay_weights`, `overlay_mode`,
  `rank_normalize`, and an optional condition schedule.
- `examples/run_allocator_validation.py` dispatches score-overlay allocators to
  `examples/run_score_overlay_validation.py`, including dry-run command
  generation, standard/robust scenario expansion, and execution-policy pass
  through.
- `examples/run_score_overlay_validation.py` now has a reusable
  `run_score_overlay_validation()` function and `--dry-run` mode, so allocator
  governance can prepare or run overlay validation from a registry definition.
- Existing `governance.capacity_monitoring` validation can be attached to
  score-overlay allocators once their capacity diagnostic summary is available.

Validation:

- Unit tests: `conda run -n quant pytest tests/unit/test_allocator_registry.py tests/unit/test_allocator_validation.py tests/unit/test_score_overlay_validation.py -q`
- Registry compatibility check: `runs/allocator_registry_validation/score_overlay_schema_extension_check/allocator_registry_validation.json`

Decision: framework support is in place. Do not register the current MAX
`gross035` or `gross040` branches yet; use the new score-overlay allocator path
for the next candidate only after adding a liquidity/capacity-aware filter or
producing a capacity monitor that passes the intended guardrails.

### Score-overlay registry replay and entry-exclusion probe

Temporary registry artifact:

- `runs/factor_research/lottery_max_2026_05_27/lottery_max_score_overlay_gross040_allocator_registry.json`

The enhanced allocator framework can now replay the `gross040` score overlay
from registry. Registry validation produced `status=warn` with no errors and
two expected capacity warnings:

- 5% same-bar cap `unfilled/traded=5.18%`, above the 5% guardrail.
- 2% same-bar cap `unfilled/traded=21.23%`, a clear guardrail breach.

The replay command used `examples/run_allocator_validation.py` with the
temporary registry, `--profile standard`, and `--resume-existing`; it reproduced
the existing `frontier_lottery_nonfullrisk_tc005_g040_w10` standard validation.

Next, a more conservative score-overlay mode was tested:

- `overlay_mode=entry_exclusion`
- condition: non-full risk states (`reduced`, `blocked`, `warmup`)
- primary score: `vc_opt_risk_cp0010_w50`
- satellite score: lottery MAX equal score
- policy: same `turnover005_gross040` cost-aware optimizer

Quick full-base screen:

| branch | return | max drawdown | gross turnover | transaction cost | blocked score rows |
|---|---:|---:|---:|---:|---:|
| entry exclusion w05 | 21.31% | -12.22% | 24.86 | 36,955 | 1.68% |
| entry exclusion w10 | 19.96% | -12.22% | 25.13 | 37,201 | 3.35% |
| blend `gross040` | 18.36% | -11.79% | 22.44 | 33,866 | n/a |

The w05 branch was promoted to standard validation:

- Standard validation: `runs/factor_research/lottery_max_2026_05_27/overlay_frontier_vc_cp0010_w50_lottery_nonfull_risk_state_entry_exclusion_w05_standard/validation_summary.json`
- Capacity summary: `runs/factor_research/lottery_max_2026_05_27/capacity_diagnostics_entry_exclusion_w05_summary.json`

| branch | scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|---:|
| entry exclusion w05 | full base | 21.31% | -12.22% | 24.86 | 36,955 |
| entry exclusion w05 | high cost | 19.41% | -12.22% | 24.22 | 45,623 |
| entry exclusion w05 | 2023 base | 2.81% | -3.54% | 19.47 | 23,426 |
| entry exclusion w05 | 2024 base | 3.68% | -8.82% | 15.35 | 20,441 |
| entry exclusion w05 | 2025 base | 9.45% | -4.35% | 18.14 | 23,382 |

Capacity stress:

| scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|
| no capacity cap | 21.31% | -12.22% | 24.86 | 0 | 0.00% | 0.00% |
| 5% same-bar cap | 21.29% | -12.21% | 24.23 | 411 | 6.01% | 49.65% |

Decision: entry exclusion w05 is a stronger alpha variant than blend `gross040`
and passes standard validation, including high-cost stress. It is not a
capacity fix: 5% same-bar capacity is worse than blend `gross040` and breaches
the 5% unfilled/traded guardrail. Keep it as a high-alpha watchlist score
overlay, but do not promote it to the allocator registry. The next experiment
should target genuine capacity reduction, for example by adding a liquidity or
capacity eligibility input to score-overlay construction rather than relying on
lottery-tail exclusion alone.

### Lagged liquidity eligibility probe

To test whether a simple capacity screen can repair the entry-exclusion branch,
two filtered score directories were generated from
`frontier_lottery_nonfullrisk_entry_excl_w05`:

- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_q20`
- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_q30`

The filter uses canonical 5m bars to compute each stock's lagged rolling mean
turnover over the previous 48 bars. At each timestamp, stocks below the
cross-sectional turnover percentile threshold are marked `entry_eligible=false`.
This avoids same-bar turnover leakage; the current execution bar's turnover is
not used in the eligibility score.

Eligibility coverage:

| branch | retained rows / all rows | retained rows / original eligible rows |
|---|---:|---:|
| `liqturn48_q20` | 80.79% | 82.17% |
| `liqturn48_q30` | 70.80% | 72.01% |

Validation results:

| branch | scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|---:|
| entry exclusion w05 | full base | 21.31% | -12.22% | 24.86 | 36,955 |
| `liqturn48_q20` | full base | 21.30% | -11.86% | 68.23 | 58,416 |
| `liqturn48_q20` | high cost | 6.56% | -13.16% | 74.95 | 109,702 |
| `liqturn48_q20` | 2023 base | -1.22% | -4.47% | 36.82 | 27,631 |
| `liqturn48_q20` | 2024 base | 1.28% | -7.25% | 43.46 | 31,160 |
| `liqturn48_q20` | 2025 base | 6.33% | -3.14% | 40.79 | 30,331 |
| `liqturn48_q30` | full base | 1.95% | -13.03% | 113.05 | 77,757 |

Capacity stress for `liqturn48_q20`:

| scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|
| no capacity cap | 21.30% | -11.86% | 68.23 | 0 | 0.00% | 0.00% |
| 5% same-bar cap | 21.48% | -11.85% | 67.53 | 173 | 1.43% | 32.43% |

Decision: the liquidity filter does solve the 5% same-bar capacity guardrail,
but it creates a different problem: much higher realized turnover and weaker
robustness. The `q20` branch keeps full-sample alpha and becomes capacity-clean,
yet high-cost return falls to 6.56% and 2023 turns negative. The `q30` branch is
too destructive. Do not promote either filtered branch.

Next experiment: replace hard entry filtering with capacity-aware sizing or a
smooth liquidity penalty inside score construction. The aim is to keep the
entry-exclusion branch's lower turnover and year-by-year robustness while
reducing participation failures below the 5% capacity guardrail.

### Smooth liquidity score-penalty probe

The hard `entry_eligible` filter was replaced with a continuous penalty applied
to the score only:

```text
score_adj = score - penalty * max(0, 0.20 - liquidity_rank) / 0.20
```

where `liquidity_rank` is the same lagged 48-bar turnover percentile used in the
hard-filter probe. `entry_eligible` is left unchanged, so the branch avoids
hard exclusion of currently held or otherwise valid names.

Generated score directories:

- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_soft_q20_p002`
- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_soft_q20_p005`
- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_soft_q20_p010`

Full-base screen:

| branch | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| entry exclusion w05 | 21.31% | -12.22% | 24.86 | 36,955 |
| hard `liqturn48_q20` | 21.30% | -11.86% | 68.23 | 58,416 |
| soft `q20_p002` | 20.37% | -11.13% | 53.18 | 56,066 |
| soft `q20_p005` | 15.05% | -12.16% | 72.54 | 68,054 |
| soft `q20_p010` | 12.99% | -10.33% | 76.87 | 69,872 |

The only viable soft-penalty setting is `q20_p002`. Stronger penalties lower
alpha and increase turnover, so they are rejected.

`q20_p002` robustness:

| scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| full base | 20.37% | -11.13% | 53.18 | 56,066 |
| high cost | 13.18% | -11.50% | 58.62 | 85,716 |
| 2023 base | 1.92% | -3.03% | 34.02 | 30,748 |
| 2024 base | 4.11% | -8.72% | 26.82 | 26,828 |
| 2025 base | 7.48% | -3.90% | 27.34 | 26,422 |

Capacity diagnostics:

| branch | scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|---:|
| entry exclusion w05 | 5% cap | 21.29% | -12.21% | 24.23 | 411 | 6.01% | 49.65% |
| hard `liqturn48_q20` | 5% cap | 21.48% | -11.85% | 67.53 | 173 | 1.43% | 32.43% |
| soft `q20_p002` | 5% cap | 20.28% | -11.15% | 52.80 | 159 | 1.09% | 32.71% |
| soft `q20_p002` | 2% cap | 19.99% | -11.23% | 49.95 | 1,048 | 8.14% | 39.66% |
| soft `q20_p005` | 5% cap | 14.85% | -12.25% | 72.19 | 141 | 0.77% | 32.98% |

Decision: soft `q20_p002` is the best capacity-aware lottery variant so far.
It fixes the 5% same-bar capacity guardrail and preserves a solid 20.37%
full-base return with positive 2023/2024/2025 slices. However, it still more
than doubles turnover versus the original entry-exclusion branch and fails the
2% capacity stress. High-cost return falls to 13.18%, which is acceptable for a
watchlist branch but not enough for allocator promotion.

Keep `q20_p002` as the new capacity-aware watchlist reference. Do not promote it
to the allocator registry. The next improvement should move from score
penalties to true capacity-aware sizing: penalize or cap target weight by
lagged turnover so low-liquidity names can remain ranked but receive smaller
orders rather than causing wholesale rank reshuffling.

### Optimizer liquidity risk-penalty probe

Instead of modifying `score`, this probe leaves ranking and `entry_eligible`
unchanged and injects a liquidity-dependent `optimizer_risk_penalty_bps` field:

```text
optimizer_risk_penalty_bps =
  penalty_bps * max(0, 0.20 - liquidity_rank) / 0.20
```

The existing cost-aware optimizer subtracts this field from expected edge before
selection and utility weighting. This is closer to capacity-aware sizing because
it affects optimizer net edge without forcing a full score-rank reshuffle.

Generated score directories:

- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_risk_q20_bps002`
- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_risk_q20_bps005`
- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_risk_q20_bps010`

Full-base screen:

| branch | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| entry exclusion w05 | 21.31% | -12.22% | 24.86 | 36,955 |
| soft score `q20_p002` | 20.37% | -11.13% | 53.18 | 56,066 |
| risk penalty `bps002` | 21.68% | -11.29% | 41.98 | 46,586 |
| risk penalty `bps005` | 16.31% | -11.01% | 71.87 | 66,950 |
| risk penalty `bps010` | 13.37% | -10.81% | 79.09 | 69,794 |

The 2bps setting is the only useful setting. Larger penalties over-rotate the
optimizer and raise turnover.

`bps002` robustness:

| scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| full base | 21.68% | -11.29% | 41.98 | 46,586 |
| high cost | 12.14% | -11.45% | 62.11 | 90,030 |
| 2023 base | 0.39% | -3.46% | 30.87 | 28,478 |
| 2024 base | 4.04% | -9.08% | 22.82 | 23,323 |
| 2025 base | 8.01% | -4.57% | 22.04 | 23,802 |

Capacity diagnostics:

| branch | scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|---:|
| entry exclusion w05 | 5% cap | 21.29% | -12.21% | 24.23 | 411 | 6.01% | 49.65% |
| soft score `q20_p002` | 5% cap | 20.28% | -11.15% | 52.80 | 159 | 1.09% | 32.71% |
| risk penalty `bps002` | 5% cap | 21.51% | -11.32% | 41.68 | 139 | 1.11% | 30.81% |
| risk penalty `bps002` | 2% cap | 21.16% | -11.41% | 39.51 | 846 | 8.26% | 39.71% |

Decision: `risk_q20_bps002` is the best capacity-aware MAX overlay variant so
far. It improves full-base return versus entry-exclusion w05, fixes the 5%
same-bar capacity guardrail, and has lower turnover/cost than the score-penalty
and hard-filter variants. It still fails the 2% capacity stress and high-cost
return falls to 12.14%, so it remains a watchlist candidate rather than a
registry promotion.

Updated next step: make this risk-penalty mechanism framework-native. The
research result suggests the correct integration point is a score-overlay
postprocessor that can add `optimizer_risk_penalty_bps` from lagged liquidity.
That should be expressed in allocator config and validated through the
score-overlay registry path before any promotion decision.

### Framework-native liquidity risk postprocessor replay

The score-overlay framework now supports a postprocessor that joins an external
same-partition penalty stream onto generated overlay scores:

```json
{
  "type": "optimizer_risk_penalty_join",
  "penalty_dir": ".../scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_risk_q20_bps002",
  "penalty_column": "optimizer_risk_penalty_bps",
  "fill_value": 0.0
}
```

Implementation points:

- `examples/run_score_overlay_validation.py` accepts
  `--optimizer-risk-penalty-dir`, joins by `timestamp,instrument_id`, and writes
  `optimizer_risk_penalty_bps` into the generated score partitions.
- `examples/run_allocator_validation.py` maps `score.postprocessors` from the
  allocator registry into the score-overlay validation runner.
- `quant_research/portfolio/allocator_registry.py` validates the
  `optimizer_risk_penalty_join` schema and penalty score directory.

Temporary registry artifact:

- `runs/factor_research/lottery_max_2026_05_27/lottery_max_score_overlay_liq_risk_postprocessor_allocator_registry.json`

Registry validation status is `warn`, with no schema errors. The only warning
is the expected 2% same-bar capacity breach:

- `capacity_2pct.capacity_unfilled_vs_traded=8.26%` versus the `5.00%`
  warning threshold.

Framework-native standard replay:

- `runs/factor_research/lottery_max_2026_05_27/allocator_replay_liq_risk_postprocessor_standard/allocator_validation_summary.json`

| scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| full base | 21.68% | -11.29% | 41.98 | 46,586 |
| high cost | 12.14% | -11.45% | 62.11 | 90,030 |
| 2023 base | 0.39% | -3.46% | 30.87 | 28,478 |
| 2024 base | 4.04% | -9.08% | 22.82 | 23,323 |
| 2025 base | 8.01% | -4.57% | 22.04 | 23,802 |

The replay passes all standard validation checks and reproduces the direct
`risk_q20_bps002` result through the registry-controlled score-overlay path.
This confirms the framework extension is usable for governed allocator
experiments. Decision remains unchanged: watchlist only, not promotion, because
2% capacity stress and high-cost robustness are still not strong enough.

### Dynamic target-weight cap extension

The optimizer now accepts per-name target weight caps through score columns:

- `max_target_weight`
- `target_weight_cap`
- `optimizer_max_target_weight`

The score-overlay framework also supports a `target_weight_cap_join`
postprocessor that joins a same-partition cap stream by
`timestamp,instrument_id` and writes `max_target_weight`. The first
implementation is deliberately conservative: caps reduce final target weight
without redistributing the unused risk budget to other names.

Implementation points:

- `quant_research/strategies/policy.py` clips optimizer `aim_weight` by dynamic
  per-name caps after gross-exposure scaling.
- `examples/run_tree_score_backtest.py` forwards the cap columns from score
  parquet into the optimizer policy.
- `examples/run_score_overlay_validation.py` accepts `--target-weight-cap-dir`
  and writes `max_target_weight` into generated overlay scores.
- `examples/run_allocator_validation.py` and
  `quant_research/portfolio/allocator_registry.py` support
  `score.postprocessors[].type = "target_weight_cap_join"`.

Two quick full-base screens converted the previous 2bps liquidity penalty into
absolute caps for low-liquidity tail names:

- hard cap artifact:
  `runs/factor_research/lottery_max_2026_05_27/target_caps_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_q20_base008`
- mild floor artifact:
  `runs/factor_research/lottery_max_2026_05_27/target_caps_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_q20_floor004_base008`

Quick results:

| branch | full-base return | max drawdown | gross turnover | avg target gross | transaction cost |
|---|---:|---:|---:|---:|---:|
| risk penalty `bps002` | 21.68% | -11.29% | 41.98 | 34.02% | 46,586 |
| hard target cap 0-0.8% | -0.58% | -1.02% | 13.13 | 1.20% | 14,297 |
| floor target cap 0.4-0.8% | -0.39% | -3.72% | 33.57 | 7.99% | 76,898 |

Decision: reject the no-redistribution dynamic cap form. The mechanism is
valid, but the naive cap transforms low-liquidity selection into a large
uninvested cash position and destroys the MAX overlay edge. The next capacity
sizing attempt should either redistribute capped residual weight to uncapped
selected names or perform cap-aware selection with replacement names, then rerun
capacity diagnostics.

### Dynamic target-weight cap with redistribution

The dynamic cap framework now supports `optimizer_target_cap_mode`:

- `clip`: cap selected names and leave unused gross exposure in cash.
- `redistribute`: cap selected names, then iteratively redistribute residual
  target weight to selected names with remaining cap capacity.

Implementation points:

- `quant_research/strategies/policy.py` adds cap-mode validation and residual
  redistribution in the cost-aware optimizer.
- `examples/run_tree_score_backtest.py` exposes
  `--optimizer-target-cap-mode {clip,redistribute,replace}` and includes the mode in
  run summaries.
- `examples/run_score_overlay_validation.py`,
  `examples/run_allocator_validation.py`, and
  `quant_research/portfolio/allocator_registry.py` pass and validate the mode
  for score-overlay allocator runs.

Standard replay artifact:

- `runs/factor_research/lottery_max_2026_05_27/overlay_entry_excl_w05_liqturn48_targetcap_q20_floor004_base008_redistribute_standard/validation_summary.json`

The standard validation set passes all checks:

| scenario | return | max drawdown | gross turnover | avg target gross | transaction cost |
|---|---:|---:|---:|---:|---:|
| full base | 20.97% | -12.31% | 16.79 | 37.53% | 31,213 |
| high cost | 23.49% | -13.30% | 13.88 | n/a | 34,156 |
| 2023 base | 4.77% | -3.62% | 15.09 | 32.95% | 22,754 |
| 2024 base | 4.16% | -7.80% | 18.03 | n/a | 26,072 |
| 2025 base | 15.52% | -4.17% | 16.46 | n/a | 24,700 |

Capacity diagnostics artifact:

- `runs/factor_research/lottery_max_2026_05_27/capacity_diagnostics_targetcap_floor004_redist_summary.json`

| branch | scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|---:|
| target cap floor 0.4-0.8%, redistribute | no cap | 20.97% | -12.31% | 16.79 | 0 | 0.00% | 0.00% |
| target cap floor 0.4-0.8%, redistribute | 5% cap | 19.83% | -12.33% | 14.84 | 84 | 10.92% | 56.31% |
| target cap floor 0.4-0.8%, redistribute | 2% cap | 20.69% | -12.33% | 13.32 | 285 | 24.85% | 62.25% |

Decision: reject this target-cap replacement for promotion. Redistribution
solves the cash-drag failure from the first dynamic-cap implementation, but the
capacity profile is worse than `risk_q20_bps002`: even the 5% participation
case breaches the 5% unfilled/traded warning threshold, and the 2% stress is far
outside the current capacity guardrail. The high-cost return being higher than
base is treated as an execution-path artifact from lower realized turnover, not
as a robustness improvement.

Updated next step: keep `risk_q20_bps002` as the best watchlist variant for the
MAX overlay, and use the new target-cap machinery only as framework
infrastructure. The next research round should test cap-aware selection with
replacement names or a smoother liquidity risk penalty curve rather than hard
per-name target caps.

### Next-round capacity sizing comparison

Two follow-up ideas were tested:

1. Cap-aware selection with replacement names. The optimizer now supports
   `optimizer_target_cap_mode = "replace"`: after selecting the original target
   basket, it detects names whose uncapped aim weight breaches their dynamic
   `max_target_weight`, then continues down the candidate list to add replacement
   names before redistributing capped residual weight.
2. Smooth liquidity risk penalty curve. The linear `bps002` liquidity penalty
   was converted into a convex tail penalty:

```text
depth = linear_bps002 / 2
optimizer_risk_penalty_bps = 5 * depth^2
```

The comparison artifact is:

- `runs/factor_research/lottery_max_2026_05_27/next_round_capacity_curve_comparison_summary.json`

Full-window and robustness results:

| branch | scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|---:|
| current `risk_q20_bps002` | full base | 21.68% | -11.29% | 41.98 | 46,586 |
| current `risk_q20_bps002` | high cost | 12.14% | -11.45% | 62.11 | 90,030 |
| target cap `replace` | full base | 21.76% | -13.12% | 12.55 | 25,602 |
| target cap `replace` | high cost | 21.16% | -12.74% | 12.45 | 31,115 |
| smooth penalty `bps005_pow2` | full base | 17.17% | -11.76% | 49.29 | 51,232 |
| smooth penalty `bps005_pow2` | high cost | 14.25% | -11.49% | 50.12 | 74,067 |

Replacement-name yearly slices:

| scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| 2023 base | 5.12% | -3.89% | 10.80 | 17,352 |
| 2024 base | 3.35% | -7.80% | 14.62 | 22,184 |
| 2025 base | 14.07% | -4.37% | 13.97 | 21,902 |

Capacity diagnostics:

| branch | scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|---:|
| current `risk_q20_bps002` | 5% cap | 21.51% | -11.32% | 41.68 | 139 | 1.11% | 30.81% |
| current `risk_q20_bps002` | 2% cap | 21.16% | -11.41% | 39.51 | 846 | 8.26% | 39.71% |
| target cap `replace` | 5% cap | 20.98% | -13.09% | 11.10 | 63 | 11.27% | 59.81% |
| target cap `replace` | 2% cap | 20.79% | -13.08% | 10.15 | 223 | 24.24% | 64.13% |
| smooth penalty `bps005_pow2` | 5% cap | 17.00% | -11.77% | 48.94 | 130 | 1.07% | 34.18% |
| smooth penalty `bps005_pow2` | 2% cap | 16.67% | -11.78% | 46.32 | 887 | 7.85% | 38.03% |

Decision:

- Empirically, `replace` is the best return/cost branch: it keeps full-window
  return slightly above `risk_q20_bps002`, cuts turnover by roughly 70%, and
  remains positive in every calendar-year slice. However, it fails the capacity
  guardrail worse than `risk_q20_bps002` because target caps change intended
  weights but do not guarantee that the actual open auction/next-bar volume can
  absorb the residual orders. It is a useful sizing infrastructure improvement,
  but not a promotion candidate.
- Empirically, `bps005_pow2` is the better capacity-risk expression: it preserves
  the 5% capacity pass and slightly improves the 2% stress relative to
  `risk_q20_bps002`, but gives up too much full-base return.
- Theoretically, the smoother liquidity penalty is more coherent for this MAX
  overlay. The original factor thesis is behavioral mispricing from lottery
  demand; liquidity should enter as an execution/risk cost, not as a hard
  portfolio construction constraint that changes the number of names after the
  signal is formed. A continuous penalty also preserves the cross-sectional MAX
  ranking better and avoids arbitrary discontinuities around a cap threshold.

Updated next step: keep `risk_q20_bps002` as the active watchlist reference. The
most logical next refinement is a gentler smooth curve, for example a convex
penalty with lower max bps or exponent between 1 and 2, calibrated to retain the
capacity profile of `bps005_pow2` while recovering more of the `bps002` return.

### Gentler smooth liquidity penalty curve refinement

Three intermediate smooth curves were generated from the existing linear
`bps002` depth:

```text
depth = linear_bps002 / 2
optimizer_risk_penalty_bps = max_bps * depth^exponent
```

Generated score directories:

- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_risk_q20_bps003_pow15`
- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_risk_q20_bps004_pow15`
- `scores_frontier_lottery_nonfullrisk_entry_excl_w05_liqturn48_risk_q20_bps003_pow2`

The refinement summary artifact is:

- `runs/factor_research/lottery_max_2026_05_27/smooth_penalty_curve_refinement_summary.json`

Full-base screen:

| branch | mean penalty | max penalty | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|---:|---:|
| linear `bps002` | n/a | 2.0bps | 21.68% | -11.29% | 41.98 | 46,586 |
| `bps003_pow15` | 0.188bps | 3.0bps | 15.80% | -11.87% | 55.38 | 57,441 |
| `bps004_pow15` | 0.251bps | 4.0bps | 20.81% | -10.54% | 48.22 | 49,653 |
| `bps003_pow2` | 0.152bps | 3.0bps | 21.51% | -10.87% | 45.46 | 50,457 |
| `bps005_pow2` | 0.253bps | 5.0bps | 17.17% | -11.76% | 49.29 | 51,232 |

`bps003_pow2` was the only intermediate curve worth extending. It keeps almost
all of the linear `bps002` return while improving drawdown and preserving the
capacity profile of the stronger convex penalty:

| branch | scenario | return | max drawdown | gross turnover | limited events | unfilled / traded | unfilled / desired |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear `bps002` | high cost | 12.14% | -11.45% | 62.11 | n/a | n/a | n/a |
| `bps003_pow2` | high cost | 16.77% | -12.14% | 39.88 | n/a | n/a | n/a |
| linear `bps002` | 5% cap | 21.51% | -11.32% | 41.68 | 139 | 1.11% | 30.81% |
| `bps003_pow2` | 5% cap | 21.39% | -10.86% | 45.08 | 133 | 1.03% | 32.75% |
| linear `bps002` | 2% cap | 21.16% | -11.41% | 39.51 | 846 | 8.26% | 39.71% |
| `bps003_pow2` | 2% cap | 20.94% | -10.88% | 42.79 | 887 | 7.73% | 38.58% |

Decision: replace `risk_q20_bps002` with `risk_q20_bps003_pow2` as the active
watchlist reference. It is not a promotion candidate yet because the 2% capacity
stress still breaches the 5% unfilled/traded guardrail, but it is a strictly
better research branch on the current evidence: comparable full-base return,
better high-cost return, better drawdown, and slightly better 2% capacity
stress.

Updated next step: run this `bps003_pow2` branch through the governed allocator
postprocessor path, then test one final low-impact execution overlay aimed only
at the 2% capacity breach, such as a dynamic `max_gross_turnover_per_rebalance`
or min-trade threshold adjustment, without changing the MAX signal score.

### Governed replay and execution-layer capacity tuning

The `bps003_pow2` smooth liquidity penalty was replayed through the governed
score-overlay allocator path. Temporary registry artifact:

- `runs/factor_research/lottery_max_2026_05_27/lottery_max_score_overlay_liq_risk_pow2_postprocessor_allocator_registry.json`

Capacity-monitoring rows:

- `runs/factor_research/lottery_max_2026_05_27/capacity_diagnostics_entry_exclusion_w05_liqturn48_risk_q20_bps003_pow2_registry_rows.json`

Governed standard replay:

- `runs/factor_research/lottery_max_2026_05_27/allocator_replay_liq_risk_pow2_postprocessor_standard/allocator_validation_summary.json`

The registry validation status is `warn`, with the expected 2% capacity warning
only:

- `capacity_2pct.capacity_unfilled_vs_traded=7.73%` versus the `5.00%` warning
  threshold.

The governed replay passes all standard checks and reproduces the direct
`bps003_pow2` results:

| scenario | return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| full base | 21.51% | -10.87% | 45.46 | 50,457 |
| high cost | 16.77% | -12.14% | 39.88 | 62,773 |
| 2023 base | 1.94% | -3.68% | 32.53 | 30,128 |
| 2024 base | 2.18% | -9.07% | 28.54 | 27,826 |
| 2025 base | 8.49% | -5.14% | 22.43 | 23,816 |

Execution-layer 2% capacity tuning artifact:

- `runs/factor_research/lottery_max_2026_05_27/execution_tuning_bps003_pow2_capacity_2pct_summary.json`

| execution variant | return | max drawdown | gross turnover | avg target gross | unfilled / traded |
|---|---:|---:|---:|---:|---:|
| baseline `bps003_pow2` | 20.94% | -10.88% | 42.79 | 33.45% | 7.73% |
| turnover cap 0.003 | 16.03% | -10.48% | 56.73 | 31.09% | 6.48% |
| min trade 0.001 | 21.06% | -11.04% | 42.54 | 33.45% | 7.77% |
| gross exposure 0.35 | 16.50% | -11.23% | 26.53 | 31.20% | 6.84% |
| gross exposure 0.30 | 16.03% | -9.09% | 26.08 | 26.29% | 4.99% |
| max name weight 0.015 | -3.23% | -5.23% | 35.37 | 6.07% | 0.28% |
| max name weight 0.0125 | -3.80% | -4.79% | 28.25 | 5.06% | 0.16% |

Decision: reject these execution-layer tuning variants. Only gross exposure
0.30 and single-name caps clear the 2% capacity warning, but both do so by
cutting effective exposure or target gross enough to destroy the branch's
economic profile. Lower turnover cap and higher min-trade threshold do not solve
the capacity breach. The current best state remains `bps003_pow2` as watchlist,
not promotion.

Updated next step: stop trying to force this branch through the 2% capacity
guardrail with generic execution knobs. The remaining useful path is a true
capacity-aware optimizer term, using lagged turnover or open-bar executable
notional to penalize expected order size directly before target weights are
formed. That is a framework extension, not another parameter sweep.
