# Fixed-Framework Repaired Alpha-Rank Benchmark Attribution - 2026-05-31

This note records the attribution follow-up for the promoted alpha-rank
research benchmark:

`intraday_overnight_gap_5m` deep25 lagged gate plus row contribution cap 25%.

The purpose is to explain what the repair changed, why the 2024 yearly slice
improved, and why the full-window max drawdown still worsened.

## Evidence

- Alpha-only v66 baseline attribution:
  `docs/validation/fixed_framework_alpha_v66_2024_attribution_2026_05_31.md`
- Repair screen and promotion decision:
  `docs/validation/fixed_framework_alpha_v66_2024_repair_screen_2026_05_31.md`
- Candidate baseline report:
  `docs/validation/fixed_framework_candidate_baseline_2026_05_31.md`
- Promoted validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- Promoted 2024 year-slice attribution:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/attribution_2024/attribution_report.md`
- Promoted 2024 full-path attribution:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/attribution_2024_full_path/attribution_report.md`
- Year-slice comparison to alpha-only v66:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/attribution_2024/comparison_to_alpha_only_v66/monthly_attribution_comparison.csv`
- Full-path monthly comparison to alpha-only v66:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/attribution_2024_full_path/comparison_to_alpha_only_v66/monthly_full_path_comparison.csv`

All comparisons use `decorrelated`, `partial_rebalance_daily`, the fixed
standard dataset, and `factor_health_mode=off`. The promoted benchmark uses an
external lagged scale schedule for `intraday_overnight_gap_5m` plus
`--factor-max-contribution-share 0.25`.

## Standard Validation Recap

| run | full base | high cost | 2023 | 2024 | 2025 | full-base max DD | high-cost max DD | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alpha-only v66 | 27.00% | 21.19% | 1.45% | -4.59% | 18.09% | -28.53% | -29.52% | yearly stability fail |
| overnight gate + cap25 | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | -30.77% | -31.70% | pass |

The promoted benchmark improves full-window return, high-cost return, and all
yearly slices. It is not a pure risk improvement: max drawdown worsens by about
`2.24pp` in full-base and `2.18pp` in high-cost.

## 2024 Year-Slice Attribution

| month | baseline return | promoted return | delta | baseline dominant | promoted dominant | largest-share delta | top-2-share delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 2024-01 | -11.50% | -11.47% | +0.03pp | `intraday_overnight_gap_5m` | `intraday_weak_tape_gap_up_risk_5m_w48` | -7.05pp | -7.89pp |
| 2024-03 | 2.22% | 3.35% | +1.13pp | `intraday_weak_tape_gap_up_risk_5m_w48` | `intraday_weak_tape_gap_up_risk_5m_w48` | -6.57pp | -6.30pp |
| 2024-06 | -10.10% | -9.10% | +1.00pp | `intraday_overnight_gap_5m` | `intraday_weak_tape_gap_up_risk_5m_w48` | -7.09pp | -8.19pp |
| 2024-09 | 18.27% | 18.08% | -0.19pp | `intraday_weak_tape_gap_up_risk_5m_w48` | `intraday_weak_tape_gap_up_risk_5m_w48` | -8.18pp | -10.04pp |
| 2024-11 | 4.32% | 5.64% | +1.32pp | `intraday_overnight_gap_5m` | `intraday_weak_tape_gap_up_risk_5m_w48` | -6.98pp | -8.61pp |
| 2024-12 | -3.81% | -3.16% | +0.66pp | `intraday_weak_tape_gap_up_risk_5m_w48` | `intraday_weak_tape_gap_up_risk_5m_w48` | -8.05pp | -8.19pp |

The 2024 repair is broad, not a single-month artifact. The promoted benchmark
improves 9 of 12 months in the 2024 year slice. The largest monthly
improvements are November (`+1.32pp`), March (`+1.13pp`), June (`+1.00pp`),
October (`+0.78pp`), July (`+0.74pp`), and December (`+0.66pp`). The main
degradations are smaller: February (`-0.40pp`), September (`-0.19pp`), and
April (`-0.08pp`).

Contribution concentration falls every month. The largest absolute contribution
share drops by roughly `6.6pp` to `14.0pp`; the top-two contribution share drops
by roughly `6.3pp` to `13.4pp`. This confirms that the cap is doing its
intended mechanical job.

## What Changed

The promoted benchmark no longer lets `intraday_overnight_gap_5m` dominate the
bad 2024 months. In the alpha-only baseline, January, February, June, and
November are dominated by the overnight-gap factor. In the promoted benchmark,
the dominant feature in those months shifts to
`intraday_weak_tape_gap_up_risk_5m_w48`.

This is expected. The repair is not a generic suppression of all gap/tape
signals. The full-window factor-leg split showed that directly gating
`intraday_weak_tape_gap_up_risk_5m_w48` was harmful. The useful repair is:

1. suppress the overnight-gap leg when its lagged health state is poor;
2. cap row-level score contribution so no single leg can dominate the selected
   basket;
3. keep the weak-tape leg active unless a separate, better state rule is found.

## Remaining Failure Mode

The remaining 2024 losses are weak-tape dominated:

| promoted worst month | return | max DD | top-score label | positive rate | dominant feature | dominant obs share |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2024-01 | -11.47% | -12.44% | -0.62% | 35.13% | `intraday_weak_tape_gap_up_risk_5m_w48` | 76.42% |
| 2024-06 | -9.10% | -10.07% | -0.41% | 34.76% | `intraday_weak_tape_gap_up_risk_5m_w48` | 80.48% |
| 2024-08 | -3.84% | -7.41% | -0.13% | 40.53% | `intraday_weak_tape_gap_up_risk_5m_w48` | 74.05% |
| 2024-12 | -3.16% | -8.34% | -0.06% | 50.66% | `intraday_weak_tape_gap_up_risk_5m_w48` | 86.65% |

This does not justify directly deep-gating the weak-tape factor. The controlled
weak-tape-only screen returned `-4.93%` in 2024, worse than the alpha-only
baseline. Any next repair should target a more specific observable state or a
portfolio-level drawdown overlay, not reuse the overnight health gate on
weak-tape.

## Full-Path Drawdown Tradeoff

The 2024 year slice starts from a fresh yearly path, while `full_base` carries
the live 2023-2025 equity path and schedule history. The drawdown tradeoff is
therefore visible in the full-path comparison:

| month | baseline return | promoted return | return delta | baseline monthly DD | promoted monthly DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-01 | -9.80% | -11.98% | -2.18pp | -12.17% | -13.69% | -1.51pp |
| 2024-04 | -0.93% | -1.59% | -0.67pp | -9.03% | -10.41% | -1.37pp |
| 2024-06 | -8.14% | -8.19% | -0.05pp | -9.18% | -9.65% | -0.47pp |
| 2024-09 | 15.45% | 15.74% | +0.29pp | -15.08% | -14.79% | +0.29pp |
| 2025-04 | -0.77% | 0.56% | +1.34pp | -12.33% | -12.99% | -0.66pp |
| 2025-10 | 4.17% | 1.67% | -2.50pp | -2.56% | -2.88% | -0.33pp |

The most important drawdown cost is January 2024. In the isolated 2024 slice,
January is almost flat versus baseline (`+0.03pp`), but in the full path it is
`-2.18pp` worse and has `1.51pp` deeper monthly drawdown. This is a path and
schedule-history effect, not a contradiction in the attribution.

The repair also gives back return in 2025-10 and 2025-08, while gaining in
2025-12, 2025-02, 2024-03, 2025-07, and 2023-05. The full-window return still
improves because the gains exceed these givebacks, but the equity path reaches
a worse trough.

## Decision

Keep the overnight-only deep25 plus contribution-cap variant as the current
alpha-rank research benchmark. It clears the standard validation gates and
removes the 2024 yearly stability blocker.

Do not treat it as a completed benchmark replacement for the production/default
framework. The max drawdown tradeoff remains an explicit open item.

## Next Tests

1. Run attribution on the full-path degradation months, especially 2025-10 and
   2025-08, to determine whether they share the same weak-tape residual mode or
   a different state.
2. Test a small drawdown-control overlay against the promoted benchmark, not
   against the superseded alpha-only baseline.
3. Explore a weak-tape state rule only if it uses an observable state distinct
   from the overnight health gate and can beat the promoted benchmark after
   costs.
