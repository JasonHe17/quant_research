# Fixed-Framework Daily-MA d10 State Sleeve Screen - 2026-06-01

This note tests the general rule candidate from the d10 generalization
attribution:

Use `intraday_daily_ma_deviation_5m_d10` only when broad tape quality is
positive, instead of adding it unconditionally to the alpha-rank stack.

The test avoids calendar-month tuning and does not search a daily-MA parameter
grid.

## Evidence

- d10 generalization attribution:
  `docs/validation/fixed_framework_alpha_rank_daily_ma_d10dev_2025_generalization_attribution_2026_06_01.md`
- Lagged broad-tape factor sleeve schedules:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_d10_state_sleeve_2026_06_01_schedules/`
- Factor-weight sleeve full-base screen:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_d10_lagged_positive_tape_sleeve_2026_06_01_full_base_screen/summary.json`
- Score-switch sleeve scores:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_d10_lagged_positive_tape_score_switch_2026_06_01_full_base_screen/summary.json`
- Score-switch sleeve backtest:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_d10_lagged_positive_tape_score_switch_2026_06_01_full_base_screen/backtests/decorrelated/partial_rebalance_daily/summary.json`
- Score-switch sleeve analysis:
  `runs/candidate_factor_portfolios/fixed_framework_alpha_rank_d10_lagged_positive_tape_score_switch_2026_06_01_full_base_screen/analysis/`

## Rule Tested

The sleeve uses only lagged observable broad-tape state:

- target feature: `intraday_daily_ma_deviation_5m_d10`;
- active state: previous timestamp has positive `market_state_return_5m` and
  majority `market_state_breadth_5m`;
- blocked state: otherwise;
- active timestamps: `6,919 / 34,799`, or `19.9%`.

In this dataset, the predeclared variants `return_positive`,
`return_and_breadth`, and `monotone_breadth_return` collapse to the same
schedule because positive broad-market return coincides with majority breadth
at the lagged timestamp.

## Implementation Finding

The first implementation used a factor-weight schedule inside the 21-factor
d10 score stack. That result is not a clean conditional-sleeve test.

When d10 is scaled to zero inside a 21-factor decorrelated stack, the run does
not exactly revert to the original 20-factor control. The base decorrelated
weights were already recomputed with d10 in the pool. This creates a weight
structure change even during blocked states.

That implementation produced:

| run | full return | max DD |
| --- | ---: | ---: |
| no-overlay control | 27.97% | -30.77% |
| factor-weight d10 sleeve | 22.34% | -28.07% |

The drawdown reduction is not enough to offset the return loss, and the run is
not a valid test of "use control when disabled, d10 when enabled".

## Score-Switch Test

The cleaner test switches score streams directly:

- blocked timestamps use the original 20-factor repaired control score;
- active timestamps use the existing unconditional 21-factor d10 score;
- all backtest policy, cost, universe, and execution settings are held fixed.

| run | full return | max DD | turnover | trades | cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-overlay control | 27.97% | -30.77% | 121.27 | 24,030 | 163,936 |
| unconditional d10 | 32.59% | -28.85% | 121.56 | 24,062 | 164,943 |
| state frontier | 30.32% | -28.92% | 117.63 | 25,427 | 170,445 |
| score-switch d10 sleeve | 27.14% | -31.03% | 121.25 | 24,034 | 163,760 |

The score-switch sleeve fails the screen:

- full return is `-0.83pp` below the no-overlay control;
- max drawdown is `-0.27pp` worse than the no-overlay control;
- it is far below both unconditional d10 and the state-aware frontier.

## Selection Displacement

The score-switch test confirms that blocked timestamps are a true control:

| state | timestamps | overlap | top-label delta | added count | added minus removed label |
| --- | ---: | ---: | ---: | ---: | ---: |
| blocked | 27,880 | 100.00% | 0.0000pp | 0.00 | 0.0000pp |
| active | 6,919 | 76.06% | -0.0038pp | 11.97 | -0.0020pp |

The active sleeve still replaces about `24%` of the top-50 basket, but the
replacement quality is negative on average. This invalidates the simple
lagged-positive-tape rule.

## Interpretation

The previous attribution found that same-bar positive tape was the only bucket
where d10 replacement quality looked positive in 2025. The clean lagged
score-switch test shows that this observation does not convert into a useful
generic trading rule.

The likely reason is timing:

- same-bar attribution describes the environment in which the label was
  realized;
- the strategy can only use lagged observable state;
- once the state is lagged, the d10 reversal/deviation sleeve no longer has
  positive marginal selection quality.

This is a useful negative result. It prevents us from turning an attribution
pattern into an overfit calendar or threshold rule.

## Decision

Do not promote d10 as:

- an unconditional alpha-rank factor;
- a lagged-positive-tape conditional sleeve;
- a factor-weight schedule inside the 21-factor stack.

Do not proceed to standard validation for this sleeve.

## Next Step

Stop tuning the d10 state threshold. The next useful work should keep the new
score-level sleeve harness, but apply it only to future candidates whose
enabled-state marginal selection quality is positive under a no-leak lagged
state definition.

For d10 specifically, keep it as a documented non-promoted interaction
candidate. Further work should require a new general mechanism, not another
threshold around the same broad-tape state.
