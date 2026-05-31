# Fixed-Framework Repaired Alpha-Rank 2025 Degradation Attribution - 2026-05-31

This note follows up on the full-path degradation months identified after the
overnight-gap deep25 gate plus contribution-cap repair. The focus is 2025-08
and 2025-10, where the repaired benchmark underperforms alpha-only v66 despite
remaining profitable.

## Evidence

- Repaired benchmark attribution:
  `docs/validation/fixed_framework_alpha_rank_repaired_benchmark_attribution_2026_05_31.md`
- Alpha-only v66 2025 full-path attribution:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/attribution_2025_full_path/attribution_report.md`
- Repaired benchmark 2025 full-path attribution:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/attribution_2025_full_path/attribution_report.md`
- 2025 comparison CSV:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/attribution_2025_full_path/comparison_to_alpha_only_v66/monthly_attribution_comparison.csv`

All runs use `full_base`, `decorrelated`, and `partial_rebalance_daily`.

## Monthly Delta

| month | baseline return | repaired return | delta | baseline DD | repaired DD | DD delta | largest-share delta | top-2-share delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-10 | 4.17% | 1.67% | -2.50pp | -2.56% | -2.88% | -0.33pp | -6.95pp | -8.75pp |
| 2025-08 | 5.70% | 4.05% | -1.65pp | -4.18% | -4.56% | -0.39pp | -6.77pp | -8.00pp |
| 2025-05 | 4.49% | 3.30% | -1.20pp | -2.52% | -3.05% | -0.53pp | -4.32pp | -6.63pp |
| 2025-06 | 1.55% | 1.32% | -0.23pp | -3.88% | -4.36% | -0.48pp | -8.78pp | -8.34pp |
| 2025-12 | -0.34% | 2.30% | +2.63pp | -5.97% | -4.61% | +1.36pp | -6.39pp | -6.72pp |
| 2025-02 | 3.68% | 5.47% | +1.80pp | -3.61% | -3.66% | -0.05pp | -6.24pp | -6.26pp |

The same mechanical change is visible in both good and bad deltas: the repaired
benchmark consistently lowers contribution concentration. This is not a failed
cap. The tradeoff is that lower concentration and lower overnight exposure
sometimes remove profitable exposure.

## Focus Months

| month | return delta | baseline dominant obs | repaired dominant obs | baseline overnight obs | repaired overnight obs | baseline top-score label | repaired top-score label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-08 | -1.65pp | 80.95% | 86.31% | 19.05% | 13.69% | 0.20% | 0.21% |
| 2025-10 | -2.50pp | 76.47% | 92.40% | 23.53% | 4.78% | 0.22% | 0.20% |
| 2025-12 | +2.63pp | 71.28% | 89.76% | 28.72% | 9.19% | -0.02% | -0.01% |

The repaired benchmark does not lose money in 2025-08 or 2025-10. It makes
less money than alpha-only v66. The attribution points to opportunity cost:

- in 2025-08, the overnight-dominant rows have high positive labels in both
  runs (`0.52%` baseline and `0.94%` repaired), but repaired overnight
  observation share is lower;
- in 2025-10, overnight observation share falls from `23.53%` to `4.78%`, while
  repaired weak-tape dominance rises to `92.40%`;
- the repaired top-score label remains positive, but the selected score mix is
  less exposed to the profitable overnight leg.

December is the counterexample that explains why the repair is still useful.
In 2025-12, baseline overnight-dominant rows have negative mean labels
(`-0.20%`), so suppressing overnight helps the repaired benchmark turn the
month from `-0.34%` to `+2.30%`.

## Readout

The 2025 underperformance months do not show the same failure mode as January
and June 2024. In 2024, overnight dominance contributed to negative selected
top-score baskets. In 2025-08 and 2025-10, the problem is that overnight
exposure would have been useful. The repair is therefore paying an opportunity
cost for a stability improvement.

This makes a blunt undo of the overnight gate unattractive. Removing or broadly
weakening the gate would likely recover some 2025 upside, but it risks
reopening the 2024 yearly stability failure. A better next test is a drawdown
or regime overlay that is allowed to restore more overnight exposure only when
recent realized basket quality and path risk are both acceptable.

## Decision

Keep the repaired benchmark unchanged. The 2025 degradation is real, but it is
not a validation failure: full 2025 return still improves from `18.09%` to
`19.71%`, and the standard validation passes.

The next experiment should not directly relax the overnight gate. It should
test a small path-aware overlay against the repaired benchmark and require:

1. 2024 remains positive after costs;
2. full-base max drawdown improves versus `-30.77%`;
3. high-cost max drawdown improves versus `-31.70%`;
4. full-base return does not give back the repaired benchmark's advantage.
