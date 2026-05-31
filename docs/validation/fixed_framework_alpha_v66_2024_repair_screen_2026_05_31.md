# Fixed-Framework Alpha v66 2024 Repair Screen - 2026-05-31

This note records the first controlled repair screen after the 2024 attribution
of the alpha-only v66 candidate baseline.

## Evidence

- Attribution report:
  `docs/validation/fixed_framework_alpha_v66_2024_attribution_2026_05_31.md`
- Baseline:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/validation_summary.json`
- Static weight-cap screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_weight_cap20_2024_screen/validation_summary.json`
- Row contribution-cap screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_contrib_cap25_2024_screen/validation_summary.json`
- Gap/tape removal screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_no_gap_tape_2024_screen/validation_summary.json`
- Targeted gap/tape gate screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gap_tape_gate_2024_screen/validation_summary.json`
- Targeted gate plus contribution-cap screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_contrib_cap25_2024_screen/validation_summary.json`
- Targeted gate with January protection screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_jan_protect_2024_screen/validation_summary.json`
- Targeted gate floor50 screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_floor50_2024_screen/validation_summary.json`
- Targeted gate blend50 screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_blend50_2024_screen/validation_summary.json`
- Targeted gate deep25 screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_2024_screen/validation_summary.json`
- Targeted gate deep25 standard validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_2026_05_31_standard/validation_summary.json`
- Targeted gate deep25 overnight-only 2024 slice:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_2024_slice/validation_summary.json`
- Targeted gate deep25 weak-tape-only 2024 slice:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_weak_tape_only_2024_slice/validation_summary.json`
- Targeted gate deep25 overnight-only plus contribution-cap 2024 slice:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2024_slice/validation_summary.json`
- Targeted gate deep25 overnight-only plus contribution-cap standard validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- Promoted benchmark attribution:
  `docs/validation/fixed_framework_alpha_rank_repaired_benchmark_attribution_2026_05_31.md`
- Promoted benchmark 2025 degradation attribution:
  `docs/validation/fixed_framework_alpha_rank_repaired_benchmark_2025_degradation_attribution_2026_05_31.md`
- Drawdown overlay screen:
  `docs/validation/fixed_framework_alpha_rank_drawdown_overlay_screen_2026_05_31.md`
- State overlay screen:
  `docs/validation/fixed_framework_alpha_rank_state_overlay_screen_2026_05_31.md`
- Conservative top-score-loss gate screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_top_score_loss_gate_conservative_2024_screen/validation_summary.json`
- Targeted gap/tape factor scale schedule:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gap_tape_gate_2024_screen/schedules/gap_tape_factor_weight_scale_schedule.csv`
- Top-score-loss factor scale schedule:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_top_score_loss_gate_conservative_2024_screen/schedules/factor_weight_scale_schedule.csv`
- January-protected factor scale schedule:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_jan_protect_2024_screen/schedules/gap_tape_factor_weight_scale_schedule.csv`
- Deep25 standard factor scale schedule:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_2026_05_31_standard/schedules/gap_tape_factor_weight_scale_schedule.csv`
- Comparison CSV:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_v66_2024_repair_screen_year2024_comparison.csv`

All completed repair screens use the fixed standard dataset, admission report,
correlation matrix, registry v66, `decorrelated`, `partial_rebalance_daily`,
and `factor_health_mode=off`. The 2024-only profile is a screen, not a
replacement validation; only the standard 2023-2025 promotion runs are
replacement evidence.

## Variants

| variant | change |
| --- | --- |
| baseline | alpha-only v66 baseline, 20 alpha-rank factors |
| weight cap 20% | `--factor-max-weight 0.20` |
| contribution cap 25% | `--factor-max-contribution-share 0.25` |
| no gap/tape | exclude `intraday_overnight_gap_5m` and `intraday_weak_tape_gap_up_risk_5m_w48` |
| targeted gate | apply lagged factor-health scale only to the two gap/tape factors |
| targeted gate + January protection | targeted gate, but force both target factor scales to `1.0` during 2024-01 |
| targeted gate floor50 | targeted gate, but raise minimum scale to `0.50` |
| targeted gate blend50 | targeted gate, but blend scale halfway back to `1.0` |
| targeted gate deep25 | targeted gate, but map original `0.25..1.00` scale to `0.00..1.00` |
| targeted gate deep25 overnight-only | apply deep25 only to `intraday_overnight_gap_5m` |
| targeted gate deep25 weak-tape-only | apply deep25 only to `intraday_weak_tape_gap_up_risk_5m_w48` |
| targeted gate deep25 overnight-only + contribution cap 25% | overnight-only deep25 plus `--factor-max-contribution-share 0.25` |
| targeted gate + contribution cap 25% | targeted gate plus `--factor-max-contribution-share 0.25` |
| conservative top-score-loss gate | scale the two gap/tape factors from lagged composite top-score basket loss |

The targeted gate uses the schedule generated by the incomplete
`factor_health_mode=shrink` screen, filtered to only:

- `intraday_overnight_gap_5m`
- `intraday_weak_tape_gap_up_risk_5m_w48`

The schedule is aggressive. Average 2024 weight scale is `0.5510` for
`intraday_overnight_gap_5m` and `0.6022` for
`intraday_weak_tape_gap_up_risk_5m_w48`; the minimum scale is `0.25`.

The conservative top-score-loss gate is built by
`examples/build_top_score_loss_factor_gate.py`. It uses the composite
`top_score_mean_label` from score diagnostics, shifted by `49` windows before
rolling, with `lookback_windows=96`, `min_periods=48`,
`reduced_scale=0.75`, and `blocked_scale=0.5`. Average 2024 weight scale is
`0.8239` for both target factors.

## 2024 Results

| variant | 2024 base | worst month | Jan | Jun | Sep | Nov | base cost | trades |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | -4.59% | 2024-01 | -11.50% | -10.10% | 18.27% | 4.32% | 54,097 | 8,177 |
| weight cap 20% | -5.07% | 2024-01 | -11.83% | -9.94% | 16.73% | 5.73% | 54,097 | 8,181 |
| contribution cap 25% | -3.24% | 2024-01 | -10.61% | -9.69% | 17.65% | 5.87% | 54,671 | 8,273 |
| no gap/tape | -1.64% | 2024-01 | -12.60% | -9.13% | 16.48% | 7.10% | 53,970 | 8,177 |
| targeted gate | -1.11% | 2024-01 | -12.01% | -9.81% | 17.84% | 5.02% | 54,360 | 8,217 |
| targeted gate + January protection | -5.06% | 2024-01 | -11.50% | -9.99% | 18.28% | 4.31% | 54,120 | 8,188 |
| targeted gate floor50 | -2.84% | 2024-01 | -11.63% | -9.63% | 18.28% | 4.35% | 54,464 | 8,237 |
| targeted gate blend50 | -4.38% | 2024-01 | -11.33% | -10.14% | 18.35% | 4.33% | 54,235 | 8,194 |
| targeted gate deep25 | 2.33% | 2024-01 | -12.42% | -8.89% | 18.12% | 5.11% | 55,401 | 8,375 |
| targeted gate deep25 overnight-only | -0.29% | 2024-01 | -12.39% | -9.57% | 18.33% | 4.48% | 54,860 | 8,286 |
| targeted gate deep25 weak-tape-only | -4.93% | 2024-01 | -12.86% | -9.66% | 16.68% | 5.66% | 53,605 | 8,143 |
| targeted gate deep25 overnight-only + contribution cap 25% | 0.90% | 2024-01 | -11.47% | -9.10% | 18.08% | 5.64% | 55,138 | 8,320 |
| targeted gate + contribution cap 25% | -3.19% | 2024-01 | -11.44% | -9.77% | 17.19% | 5.76% | 54,221 | 8,205 |
| conservative top-score-loss gate | -4.37% | 2024-01 | -11.87% | -10.08% | 18.09% | 4.51% | 54,311 | 8,224 |

## Readout

Static factor-weight capping is not a repair. It reduces concentration but also
damages the remaining score balance; 2024 worsens from `-4.59%` to `-5.07%`.

Row-level contribution capping is directionally useful. It lowers the 2024 loss
to `-3.24%` and reduces the worst drawdown in the 2024-only screen, but it does
not clear the stability problem. It also increases trade count and cost
slightly, so it should not be promoted by itself.

Removing the two gap/tape drivers gives the best 2024 screen result at
`-1.64%` among the static variants, but it still does not turn 2024 positive.
It also makes January worse
(`-12.60%` versus baseline `-11.50%`). The improvement mainly comes from later
months, especially June, November, and December. This says the two factors are
not simply bad factors; their usefulness is state dependent.

The targeted gate is the best screen so far, improving 2024 from `-4.59%` to
`-1.11%` while preserving more of the September rebound than direct removal.
However it still fails the yearly stability requirement and worsens January to
`-12.01%`.

Forcing January scales back to `1.0` does not fix the problem. It restores
January to the baseline `-11.50%`, but the full 2024 result falls to `-5.06%`,
worse than baseline. The target gate's benefit is path dependent; a simple
calendar-month protection rule changes later positioning enough to destroy the
improvement.

Combining the targeted gate with contribution capping is worse than the targeted
gate alone (`-3.19%` versus `-1.11%`). The two controls are not additive in this
configuration; capping appears to distort the already-shrunk score mix.

Less aggressive path-continuous refinements also fail. `floor50` improves on
baseline but is worse than the original targeted gate (`-2.84%` versus
`-1.11%`), while `blend50` nearly reverts to baseline (`-4.38%`). The repair is
not coming from mild de-concentration; it requires strong state-dependent
suppression of the two gap/tape legs.

The stronger `deep25` transform is the first 2024-only screen that turns the
year positive (`2.33%` base, `0.71%` high-cost). Its benefit comes mainly from
June and late-year path repair, but it worsens January to `-12.42%` and raises
trade count/cost modestly.

The full-window factor-leg split identifies the useful repair leg. Applying
deep25 only to `intraday_overnight_gap_5m` nearly fixes 2024 (`-0.29%`), while
applying it only to `intraday_weak_tape_gap_up_risk_5m_w48` is worse than
baseline (`-4.93%`). Adding the row contribution cap to the overnight-only gate
turns the 2024 slice positive (`0.90%`) and restores January close to baseline
(`-11.47%` versus `-11.50%`). The repair is therefore not a generic gap/tape
suppression rule; it is an overnight-gap health gate plus contribution
concentration control.

The conservative top-score-loss gate is also not enough. It leaves the 2024
result close to baseline (`-4.37%` versus `-4.59%`) and does not materially
repair January or June. Composite top-score loss appears too delayed or too
coarse to control the gap/tape failure by itself.

## Promotion Check

Because `deep25` passed the 2024-only screen, it was rebuilt as a full
2023-2025 lagged schedule and tested under the standard validation profile. The
promotion check does not pass:

| variant | full base | high cost | 2023 | 2024 | 2025 | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 27.00% | 21.19% | 1.45% | -4.59% | 18.09% | fail: 2024 negative |
| deep25 standard | 23.28% | 17.53% | 1.00% | -1.22% | 18.35% | warn: 2024 negative |
| overnight-only deep25 + contribution cap 25% | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | pass |

The full-window `deep25` schedule still improves 2024 materially
(`-4.59%` to `-1.22%`), but it gives back too much full-window return and does
not clear the yearly stability requirement. The difference versus the
2024-only `deep25` screen is path dependent: the full schedule enters January
with matured 2023 health history, especially lower scale on
`intraday_weak_tape_gap_up_risk_5m_w48`, while the 2024-only screen starts with
fresh warmup state. Therefore the full-window result is the controlling
evidence.

The overnight-only deep25 plus contribution-cap variant passes the standard
profile. It improves full-base return (`27.97%` versus `27.00%`), high-cost
return (`22.15%` versus `21.19%`), and all yearly slices. Max drawdown is worse
than the alpha-only v66 baseline (`-30.77%` versus `-28.53%` full-base), so this
is a return/stability improvement with a drawdown tradeoff, not a pure risk
reduction.

## Non-Result

A lagged `factor_health_mode=shrink` screen was started, but it did not complete
within the 15-minute command window. It produced a factor-health schedule under:

`runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_health_shrink_2024_screen/full_base/factor_health/factor_health_schedule.csv`

Do not use it as evidence until the associated score/backtest run completes.

## Decision

Promote the overnight-only deep25 plus contribution-cap variant as the current
research benchmark candidate:

`runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`

Do not promote the broader two-factor deep25 gate. It fixes the 2024-only screen
but fails full-window validation. Do not promote weak-tape-only gating, calendar
January protection, static weight capping, direct factor removal, or the
top-score-loss gate.

## Next Test

Use the promoted variant as the new comparison point for near-term alpha-rank
portfolio work. The next tests should focus on whether the full-base drawdown
increase is acceptable or can be reduced without giving back the repaired 2024
slice:

1. robustness-test the state-aware `budget_min90_l96` overlay before replacing
   the promoted benchmark;
2. require future factor additions to beat both the old alpha-only v66 baseline
   and this promoted research benchmark after costs.
