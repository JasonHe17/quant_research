# Fixed-Framework Daily-MA d10 2025 Generalization Attribution - 2026-06-01

This note follows the `intraday_daily_ma_deviation_5m_d10` incremental screen.
The purpose is not to tune around 2025. The purpose is to extract a reusable
rule candidate from the observed failure mode.

The tested factor has strong standalone admission evidence, but it degrades the
isolated 2025 alpha-rank portfolio slice after being added to the repaired
score stack and to the state-aware frontier.

## Evidence

- Incremental screen:
  `docs/validation/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_screen_2026_06_01.md`
- Selection-displacement diagnostics:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_single_2026_06_01_attribution/generalization_diagnostics/`
- Monthly selection-displacement table:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_single_2026_06_01_attribution/generalization_diagnostics/monthly_selection_displacement_2025.csv`
- State bucket diagnostics:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_incremental_daily_ma_d10dev_single_2026_06_01_attribution/generalization_diagnostics/selection_state_buckets_2025.csv`

The diagnostics use existing score and label artifacts only. No new parameter
grid or market-period fit was run.

## Question

Why can a factor with strong standalone 2025 rank IC still hurt the 2025
portfolio slice?

The right unit of analysis is not standalone factor IC. It is the marginal
effect of the factor on the portfolio's selected top-50 basket:

1. Which names does d10 add to the top-50?
2. Which names does it remove?
3. Is that replacement quality stable across observable market states?

## Selection-Displacement Results

In the isolated 2025 slice, adding d10 changes roughly 22% to 26% of the top-50
basket per rebalance. The replacement quality is usually negative.

| month | top-label delta | overlap | added minus removed label | no-overlay return delta | state-frontier return delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-01 | -0.0215pp | 73.61% | -0.0535pp | -0.95pp | -0.71pp |
| 2025-02 | +0.0123pp | 77.64% | +0.0317pp | +2.93pp | +2.96pp |
| 2025-03 | +0.0033pp | 77.72% | -0.0141pp | -1.39pp | -1.53pp |
| 2025-04 | -0.0179pp | 77.18% | -0.0294pp | -0.72pp | -1.16pp |
| 2025-05 | -0.0158pp | 76.36% | -0.0742pp | -1.85pp | -1.77pp |
| 2025-06 | -0.0059pp | 75.61% | -0.0234pp | -0.70pp | -0.97pp |
| 2025-07 | -0.0032pp | 75.94% | -0.0048pp | +0.33pp | -0.32pp |
| 2025-08 | -0.0094pp | 77.06% | -0.0452pp | +0.28pp | -0.05pp |
| 2025-09 | -0.0060pp | 79.34% | -0.0225pp | -0.18pp | -0.17pp |
| 2025-10 | -0.0330pp | 78.66% | -0.1550pp | +0.01pp | -0.18pp |
| 2025-11 | +0.0081pp | 77.04% | +0.0235pp | -0.80pp | -0.98pp |
| 2025-12 | +0.0089pp | 78.04% | +0.0468pp | -2.57pp | -3.35pp |

The important pattern is not a single bad month. The d10 overlay often replaces
names with lower realized forward returns than the names it removes. That means
the standalone d10 signal is not additive to the current alpha-rank stack in a
stable way.

## State Dependence

The replacement effect is state dependent. Using observable same-bar market
state diagnostics for attribution, the marginal top-label delta is negative in
weak or neutral market-return states and positive only in positive market-return
states.

| bucket type | bucket | top-label delta | added minus removed label | overlap | avg market return | avg breadth |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| market return | negative | -0.0073pp | -0.0298pp | 77.30% | -0.1429% | 17.84% |
| market return | near zero | -0.0090pp | -0.0406pp | 76.86% | -0.0003% | 35.81% |
| market return | positive | +0.0036pp | +0.0309pp | 77.23% | +0.1314% | 61.38% |
| breadth | weakest | -0.0088pp | -0.0362pp | 77.06% | -0.1218% | 16.56% |
| breadth | low | -0.0072pp | -0.0306pp | 77.02% | -0.0198% | 29.67% |
| breadth | middle | -0.0118pp | -0.0548pp | 76.93% | -0.0020% | 40.29% |
| breadth | strongest | +0.0030pp | +0.0245pp | 77.16% | +0.0999% | 58.18% |

This points to a general mechanism:

- d10 is a reversal/deviation-style signal;
- in broad, positive tape, deviation can identify recoverable names and
  improve replacement quality;
- in weak or neutral tape, deviation more often selects names whose apparent
  discount is not compensated by the existing alpha stack.

This is a regime-dependence hypothesis, not a fitted rule.

## Interpretation

The failure is not that d10 has no signal. The failure is that the factor is
not unconditionally additive to the existing score stack.

The current alpha-rank stack already has strong event, weak-tape, overnight-gap,
liquidity, and volatility components. Adding a daily-MA deviation signal changes
the selected basket, but in 2025 the replacements are mostly worse in weak or
neutral market states. This is why the aggregate full-window result can improve
while the isolated 2025 slice degrades.

The deeper lesson is methodological:

1. Single-factor IC is necessary but not sufficient.
2. Incremental factor admission should measure selection displacement, not only
   portfolio return.
3. Reversal/deviation-style factors should be treated as state-conditional
   candidates by default.
4. State rules should be based on broad observable tape quality, not on a
   calendar year or a hand-picked bad month.

## General Rule Candidate

Do not add `intraday_daily_ma_deviation_5m_d10` as an unconditional alpha-rank
factor.

A generalizable next hypothesis is:

Use daily-MA deviation only as a conditional alpha sleeve when broad tape
quality is positive enough that deviation is more likely to mean recovery than
uncompensated weakness.

The rule should be tested with coarse, predeclared state families rather than a
fine parameter grid:

- tape return sign or broad market return bucket;
- market breadth / weak-breadth state;
- possibly a monotone budget exposure or factor-weight sleeve, not a hard
  month-specific switch.

The acceptance condition should require:

- positive marginal selection-displacement quality in the enabled states;
- no material degradation in yearly slices;
- improvement versus both the repaired no-overlay control and the
  `budget_min90_l120` state-aware frontier;
- stable behavior under coarse state definitions, not one threshold.

## Decision

Keep d10 as a high-potential conditional candidate, but do not promote it to the
alpha-rank research frontier.

The next experiment should be a coarse state-conditioned sleeve test, not a
daily-MA parameter grid. The test should use a small number of generic
observable tape states and should report both:

1. portfolio metrics against the no-overlay control and state-aware frontier;
2. selection-displacement quality inside enabled and disabled states.
