# Fixed-Framework Alpha-Rank Drawdown Overlay Screen - 2026-05-31

This note records the first drawdown-control overlay screen against the repaired
alpha-rank research benchmark.

The tested control is the built-in policy drawdown brake:

- if live equity drawdown is below a threshold, reduce target gross exposure;
- otherwise keep the repaired benchmark unchanged.

The repaired benchmark remains:

`intraday_overnight_gap_5m` deep25 gate plus row contribution cap 25%.

## Evidence

- Repaired benchmark:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- Alpha-only v66 control:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/validation_summary.json`
- Drawdown-brake standard candidate:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_dd_brake_t0p25_s0p90_2026_05_31_standard/validation_summary.json`
- Full-base screen outputs:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_dd_brake_t*_full_base_screen/summary.json`

The full-base screens reuse the repaired benchmark score files and only rerun
the policy backtest with different drawdown-brake settings.

## Full-Base Screen

| run | full return | max DD | return delta vs repaired | DD delta vs repaired | brake active | avg brake scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repaired benchmark | 27.97% | -30.77% | +0.00pp | +0.00pp | 0 | - |
| alpha-only v66 control | 27.00% | -28.53% | -0.97pp | +2.23pp | 0 | - |
| brake t25 s90 | 27.05% | -30.53% | -0.93pp | +0.23pp | 41 | 0.9943 |
| brake t22 s95 | 26.51% | -30.57% | -1.46pp | +0.19pp | 75 | 0.9948 |
| brake t20 s90 | 25.38% | -30.39% | -2.59pp | +0.37pp | 89 | 0.9877 |
| brake t18 s85 | 24.00% | -29.50% | -3.97pp | +1.26pp | 98 | 0.9797 |
| brake t10 s85 | 23.14% | -28.13% | -4.83pp | +2.64pp | 210 | 0.9566 |
| brake t15 s90 | 21.21% | -29.73% | -6.76pp | +1.03pp | 147 | 0.9797 |
| brake t15 s85 | 20.22% | -29.54% | -7.75pp | +1.23pp | 161 | 0.9667 |
| brake t10 s75 | 19.79% | -25.48% | -8.18pp | +5.29pp | 215 | 0.9259 |
| brake t08 s75 | 18.64% | -25.39% | -9.33pp | +5.37pp | 266 | 0.9083 |
| brake t12 s75 | 17.31% | -27.56% | -10.66pp | +3.20pp | 225 | 0.9224 |

The simple drawdown brake has a steep return cost. The only setting that keeps
full-base return above the alpha-only control is `threshold=-0.25`,
`reduced_scale=0.90`, and even that gives up `0.93pp` versus the repaired
benchmark for only `0.23pp` of full-base drawdown improvement.

## Standard Check For Best Screen

| run | full base | high cost | 2023 | 2024 | 2025 | full-base DD | high-cost DD | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alpha-only v66 | 27.00% | 21.19% | 1.45% | -4.59% | 18.09% | -28.53% | -29.52% | warn |
| repaired benchmark | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | -30.77% | -31.70% | pass |
| brake t25 s90 | 27.05% | 21.71% | 1.09% | 2.87% | 19.60% | -30.53% | -31.33% | pass |

The best drawdown-brake screen passes the standard validation gates. It keeps
2024 positive and improves 2024 from `0.90%` to `2.87%`. It also improves
high-cost drawdown from `-31.70%` to `-31.33%`.

However, the tradeoff is not attractive enough to replace the repaired
benchmark as the main research frontier:

- full-base return falls from `27.97%` to `27.05%`;
- high-cost return falls from `22.15%` to `21.71%`;
- full-base drawdown improves only from `-30.77%` to `-30.53%`;
- high-cost drawdown improves only from `-31.70%` to `-31.33%`.

The overlay is therefore a valid risk-version candidate, not a superior
benchmark.

## Readout

The drawdown brake is reactive. It cuts gross exposure after the path has
already entered a drawdown, so it can reduce the next part of the trough but
also suppresses recovery and profitable continuation. This explains why
moderate thresholds such as `-0.08` to `-0.15` reduce drawdown more visibly but
destroy return.

The lightest useful setting, `t25/s90`, only activates 41 decision timestamps in
full-base. It is late enough to preserve most return, but also too late to fix
the main drawdown problem materially.

## Decision

Do not replace the repaired alpha-rank research benchmark with the drawdown
brake variant.

Keep `t25/s90` as an optional risk-version reference because it passes standard
validation and keeps full/high-cost returns slightly above alpha-only v66, but
it is not the new frontier.

## Next Test

Move from reactive path drawdown control to a state-aware overlay. The next
candidate should be built from lagged realized basket quality or factor-state
evidence, and should decide before a deep drawdown has already occurred. The
minimum promotion bar is:

1. full-base return remains close to the repaired benchmark;
2. high-cost return remains above alpha-only v66;
3. 2024 remains positive after costs;
4. full/high-cost max drawdown improves materially, not just by a few basis
   points.
