# Fixed-Framework Alpha-Rank Incremental Daily-MA d10 Deviation Screen - 2026-06-01

This note records the first incremental alpha-rank factor test after promoting
`budget_min90_l120` as the state-aware research frontier.

The tested factor is:

`intraday_daily_ma_deviation_5m_d10`

It is a registry `watchlist` alpha-rank feature with fixed-framework admission
status `candidate`. It was not part of the 20-factor repaired alpha-rank
benchmark because the benchmark only used registry status `candidate`.

## Evidence

- Research benchmark stack:
  `docs/validation/fixed_framework_alpha_rank_research_benchmark_replacement_2026_06_01.md`
- Repaired no-overlay control:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard/validation_summary.json`
- State-aware frontier:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard/validation_summary.json`
- d10 no-overlay validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_single_control_2026_06_01_standard/validation_summary.json`
- d10 state-frontier validation:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_single_state_l120_2026_06_01_standard/validation_summary.json`
- Monthly comparison:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_single_2026_06_01_attribution/monthly_comparison.csv`
- 2025 generalization attribution:
  `docs/validation/fixed_framework_alpha_rank_daily_ma_d10dev_2025_generalization_attribution_2026_06_01.md`

## Candidate Setup

The screen adds only `intraday_daily_ma_deviation_5m_d10` to the existing
20-factor repaired alpha-rank pool. All other settings are held fixed:

- registry statuses: `candidate`, `watchlist`;
- admission statuses: `candidate`, `watchlist`;
- evaluation role: `alpha_rank`;
- method: `decorrelated`;
- score transform: `rank`;
- factor contribution cap: `25%`;
- overnight-gap factor-weight schedule: unchanged;
- state schedule, when enabled: `budget_min90_l120`.

The new factor receives a decorrelated weight of `4.13%`. Its admission evidence
is strong in isolation:

| metric | value |
| --- | ---: |
| admission status | candidate |
| registry status | watchlist |
| family | momentum |
| expected direction | mixed |
| Spearman rank IC mean | -0.0288 |
| rank IC t-stat | -31.25 |
| directional top-minus-bottom label | 0.6102% |
| cost-adjusted top-minus-bottom label | 0.6067% |
| stable years | 3 |

## Standard Results

| run | status | full base | high cost | 2023 | 2024 | 2025 | full DD | high-cost DD | mean turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-overlay control | pass | 27.97% | 22.15% | 1.72% | 0.90% | 19.71% | -30.77% | -31.70% | 73.23 |
| no-overlay + d10 | pass | 32.59% | 26.59% | 3.09% | 0.94% | 13.19% | -28.85% | -29.81% | 73.34 |
| state frontier | pass | 30.32% | 24.39% | 1.47% | 0.86% | 18.28% | -28.92% | -29.87% | 71.02 |
| state frontier + d10 | pass | 31.32% | 25.66% | 2.39% | 0.71% | 8.97% | -27.51% | -28.45% | 70.93 |

The aggregate screen looks attractive:

- no-overlay full-base return improves by `+4.62pp`;
- no-overlay high-cost return improves by `+4.44pp`;
- no-overlay full-base max drawdown improves by `+1.92pp`;
- state-frontier full-base return improves by `+1.00pp`;
- state-frontier high-cost return improves by `+1.27pp`;
- state-frontier full-base max drawdown improves by `+1.41pp`.

The yearly slices block promotion:

- no-overlay 2025 falls from `19.71%` to `13.19%`;
- state-frontier 2024 falls from `0.86%` to `0.71%`;
- state-frontier 2025 falls from `18.28%` to `8.97%`.

The standard validation gates still pass because all yearly slices remain
positive, but this is not stable enough to promote into the research frontier.

## Monthly Full-Path Readout

Largest full-path gains versus the no-overlay control:

| month | control return | d10 return | delta | control DD | d10 DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-05 | -2.83% | -2.24% | +0.59pp | -5.13% | -4.61% | +0.53pp |
| 2025-09 | 1.77% | 2.32% | +0.55pp | -3.69% | -3.57% | +0.12pp |
| 2025-10 | 1.67% | 2.21% | +0.54pp | -2.88% | -2.79% | +0.10pp |
| 2023-03 | -2.64% | -2.19% | +0.45pp | -5.05% | -4.66% | +0.38pp |

Largest full-path givebacks versus the no-overlay control:

| month | control return | d10 return | delta | control DD | d10 DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-09 | 15.74% | 15.03% | -0.71pp | -14.79% | -14.70% | +0.08pp |
| 2025-04 | 0.56% | 0.09% | -0.47pp | -12.99% | -13.07% | -0.08pp |
| 2024-07 | 1.69% | 1.26% | -0.43pp | -6.26% | -6.40% | -0.14pp |
| 2023-02 | 2.86% | 2.52% | -0.35pp | -2.26% | -2.27% | -0.01pp |

State-frontier full-path gains are also mixed:

| month | state return | d10 state return | delta | state DD | d10 state DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-09 | 1.76% | 2.41% | +0.66pp | -3.20% | -3.43% | -0.23pp |
| 2024-04 | -0.81% | -0.16% | +0.65pp | -9.42% | -9.19% | +0.23pp |
| 2023-11 | 1.08% | 1.66% | +0.58pp | -2.46% | -2.26% | +0.19pp |
| 2025-10 | 1.76% | 2.27% | +0.51pp | -2.73% | -2.71% | +0.03pp |

Largest state-frontier givebacks:

| month | state return | d10 state return | delta | state DD | d10 state DD | DD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-04 | 0.01% | -0.62% | -0.63pp | -12.85% | -13.24% | -0.39pp |
| 2025-05 | 2.93% | 2.31% | -0.62pp | -3.38% | -3.64% | -0.26pp |
| 2024-07 | 1.61% | 1.17% | -0.44pp | -6.15% | -6.09% | +0.07pp |
| 2024-10 | 5.74% | 5.32% | -0.43pp | -10.96% | -10.83% | +0.13pp |

The full-path monthly changes are mostly moderate, so they do not by themselves
explain the large isolated 2025 year-slice degradation. This is a path
dependence warning: the factor interacts with the existing score stack and
state schedule, and its standalone yearly behavior is not stable enough.

## Decision

Do not promote `intraday_daily_ma_deviation_5m_d10` into the alpha-rank
research frontier yet.

Classify it as a high-potential interaction candidate:

- it improves full-window and high-cost aggregate results;
- it improves drawdown in both no-overlay and state-frontier tests;
- it receives a modest `4.13%` decorrelated weight, so the result is not from
  overwhelming the existing stack;
- but it materially damages the 2025 isolated yearly slice, especially under
  the state frontier.

The 2025 attribution shows that the useful next test is not a broader daily-MA
grid. It should be a coarse state-conditioned sleeve test that asks when
daily-MA deviation improves marginal selection quality, without fitting a
calendar-year-specific switch.
