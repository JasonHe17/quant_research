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
- Comparison CSV:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_v66_2024_repair_screen_year2024_comparison.csv`

All completed repair screens use the fixed standard dataset, admission report,
correlation matrix, registry v66, `decorrelated`, `partial_rebalance_daily`,
`factor_health_mode=off`, and a 2024-only standard profile. The 2024-only
profile is a screen, not a replacement validation.

## Variants

| variant | change |
| --- | --- |
| baseline | alpha-only v66 baseline, 20 alpha-rank factors |
| weight cap 20% | `--factor-max-weight 0.20` |
| contribution cap 25% | `--factor-max-contribution-share 0.25` |
| no gap/tape | exclude `intraday_overnight_gap_5m` and `intraday_weak_tape_gap_up_risk_5m_w48` |

## 2024 Results

| variant | 2024 base | worst month | Jan | Jun | Sep | Nov | base cost | trades |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | -4.59% | 2024-01 | -11.50% | -10.10% | 18.27% | 4.32% | 54,097 | 8,177 |
| weight cap 20% | -5.07% | 2024-01 | -11.83% | -9.94% | 16.73% | 5.73% | 54,097 | 8,181 |
| contribution cap 25% | -3.24% | 2024-01 | -10.61% | -9.69% | 17.65% | 5.87% | 54,671 | 8,273 |
| no gap/tape | -1.64% | 2024-01 | -12.60% | -9.13% | 16.48% | 7.10% | 53,970 | 8,177 |

## Readout

Static factor-weight capping is not a repair. It reduces concentration but also
damages the remaining score balance; 2024 worsens from `-4.59%` to `-5.07%`.

Row-level contribution capping is directionally useful. It lowers the 2024 loss
to `-3.24%` and reduces the worst drawdown in the 2024-only screen, but it does
not clear the stability problem. It also increases trade count and cost
slightly, so it should not be promoted by itself.

Removing the two gap/tape drivers gives the best 2024 screen result at
`-1.64%`, but it still does not turn 2024 positive. It also makes January worse
(`-12.60%` versus baseline `-11.50%`). The improvement mainly comes from later
months, especially June, November, and December. This says the two factors are
not simply bad factors; their usefulness is state dependent.

## Non-Result

A lagged `factor_health_mode=shrink` screen was started, but it did not complete
within the 15-minute command window. It produced a factor-health schedule under:

`runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_health_shrink_2024_screen/full_base/factor_health/factor_health_schedule.csv`

Do not use it as evidence until the associated score/backtest run completes.

## Decision

No screened variant is eligible to replace the alpha-only v66 baseline:

- none restores 2024 to positive;
- static capping is worse than baseline;
- contribution capping helps but is insufficient;
- removing the two factors helps the year but worsens January and loses
  potentially useful state-dependent alpha.

The next repair should be a factor-specific dynamic gate, not a blunt static
cap. The gate should scale down `intraday_overnight_gap_5m` and
`intraday_weak_tape_gap_up_risk_5m_w48` only when lagged evidence indicates the
gap/tape regime is adverse. It must use only matured labels or observable
market-state proxies, then be tested first on 2024 and only afterward on the
full 2023-2025 standard profile.

## Next Test

Build a targeted weight-scale schedule for the two gap/tape factors:

1. derive a lagged health signal from each factor leg's top-minus-bottom label
   or top-score label using the fixed label lag of `49` windows;
2. apply the schedule through `--factor-weight-scale-schedule` with
   `--factor-weight-scale-combine-mode min`;
3. screen on 2024 against the same baseline;
4. promote to full standard validation only if 2024 turns positive without
   materially reducing the 2023-2025 full/high-cost edge.
