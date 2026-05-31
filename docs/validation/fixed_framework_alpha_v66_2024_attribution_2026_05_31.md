# Fixed-Framework Alpha v66 2024 Attribution - 2026-05-31

This note explains why post-fix returns look weaker in some comparisons and
records the 2024 attribution for the alpha-only v66 candidate baseline.

## Evidence

- Candidate baseline report:
  `docs/validation/fixed_framework_candidate_baseline_2026_05_31.md`
- Candidate validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/validation_summary.json`
- 2024 attribution summary:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/attribution_2024/monthly_score_contribution_attribution.csv`
- 2024 dominant-feature attribution:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/attribution_2024/dominant_feature_attribution_by_month.csv`
- Generated attribution report:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/attribution_2024/attribution_report.md`

## What Changed In Evaluation

The framework iteration was an evaluation-quality fix, not an alpha-improving
strategy change. The main changes that can lower previously reported returns
are:

- forward-return labels now use executable open-to-open pricing on the next-bar
  grid;
- dataset construction filters non-tradable entries, entry limit-up rows,
  non-tradable exits, and exit limit-down rows;
- label-derived, entry-metadata, and exit-metadata columns are excluded from
  default feature inference;
- factor admission uses registry role and expected direction instead of
  inferring production direction from the same evaluation sample;
- candidate combinations use equal admission-evidence weights by default;
- universe wildcard selection and real backtests use point-in-time `as_of`
  dates from the validation start.

Therefore a lower result versus pre-fix or legacy reports is expected when the
old result relied on optimistic pricing, unavailable rows, direction inference,
or broader leakage-prone feature inference. The fixed framework is stricter; it
does not automatically make the factor set better.

## Result Versus Current Fixed Benchmark

Against the current fixed standard benchmark, the alpha-only v66 baseline is
not a simple degradation:

| run | full base | high cost | worst year | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous fixed standard benchmark | 25.23% | 19.65% | 1.00% | -29.96% | 73.09 |
| alpha-only v66 candidate | 27.00% | 21.19% | -4.59% | -29.52% | 73.61 |

The full-window and high-cost results improved, but yearly stability worsened.
The candidate should therefore remain a research baseline, not replace the
accepted standard benchmark yet.

## 2024 Weakness

The 2024 slice returned `-4.59%`, with max drawdown `-25.81%`,
`8,177` trades, and total transaction cost `54,097`. Losses are concentrated:

| month | return | max drawdown | cost | top-score label | positive rate | dominant feature |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2024-01 | -11.50% | -12.29% | 5,372 | -0.64% | 35.51% | `intraday_overnight_gap_5m` |
| 2024-06 | -10.10% | -11.08% | 4,124 | -0.38% | 37.83% | `intraday_overnight_gap_5m` |
| 2024-08 | -4.02% | -7.39% | 4,552 | -0.10% | 41.48% | `intraday_weak_tape_gap_up_risk_5m_w48` |
| 2024-12 | -3.81% | -9.71% | 5,013 | -0.09% | 49.15% | `intraday_weak_tape_gap_up_risk_5m_w48` |
| 2024-05 | -1.33% | -5.13% | 4,364 | -0.13% | 45.73% | `intraday_weak_tape_gap_up_risk_5m_w48` |

January and June together account for most of the yearly damage. Their combined
transaction cost is about `9,496`, far smaller than the combined equity loss of
roughly `207,278`; this is not primarily a cost-overrun problem.

## Attribution

The alpha-only v66 decorrelated weights are concentrated in two gap/tape
signals:

| feature | weight |
| --- | ---: |
| `intraday_weak_tape_gap_up_risk_5m_w48` | 29.33% |
| `intraday_overnight_gap_5m` | 28.22% |

In the weak months, these same two signals dominate the top-score contribution:

| month | feature | obs share | top-score label | largest share | top-2 share |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024-01 | `intraday_overnight_gap_5m` | 53.50% | -0.98% | 29.75% | 45.09% |
| 2024-01 | `intraday_weak_tape_gap_up_risk_5m_w48` | 46.50% | -0.24% | 28.08% | 44.16% |
| 2024-06 | `intraday_overnight_gap_5m` | 53.95% | -0.40% | 28.75% | 44.20% |
| 2024-06 | `intraday_weak_tape_gap_up_risk_5m_w48` | 46.05% | -0.36% | 30.32% | 44.68% |
| 2024-08 | `intraday_weak_tape_gap_up_risk_5m_w48` | 57.29% | -0.29% | 29.30% | 44.15% |
| 2024-08 | `intraday_overnight_gap_5m` | 42.71% | 0.16% | 32.22% | 45.41% |
| 2024-12 | `intraday_weak_tape_gap_up_risk_5m_w48` | 63.83% | -0.37% | 31.36% | 44.63% |
| 2024-12 | `intraday_overnight_gap_5m` | 36.17% | 0.40% | 31.31% | 46.21% |

The failure mode is a regime inversion in the selected top-score basket. During
the worst months, the score still selects names with high exposure to the two
largest gap/tape factors, but those selected baskets have negative forward
labels. Contribution concentration is material, but not a standalone engine
bug: the top two contribution share is around `44%` to `45%` in the worst
months, while the core problem is that the dominant legs point the portfolio
into the wrong side of the 2024 regime.

## Conclusion

The framework iteration did have a positive effect on research quality: it
removed optimistic assumptions and made the weak 2024 regime visible. It did
not improve the alpha portfolio by itself. The current candidate has better
full-window and high-cost returns than the fixed standard benchmark, but it
fails the stability bar because 2024 turns negative.

Do not promote alpha-only v66 to the accepted benchmark until the 2024 gap/tape
failure is repaired.

## Next Tests

Run small, controlled variants against the same fixed baseline:

1. cap individual factor weights or contribution from
   `intraday_overnight_gap_5m` and
   `intraday_weak_tape_gap_up_risk_5m_w48`;
2. test an event/regime gate that scales these two factors down when their
   recent top-score labels or market-state proxies are adverse;
3. test `intraday_eod_reversal_5m_lb1_tail6` only as a gated overlay, not as a
   default alpha-rank member;
4. promote only a variant that keeps the full/high-cost improvement while
   lifting the 2024 yearly slice back above the fixed benchmark stability bar.
