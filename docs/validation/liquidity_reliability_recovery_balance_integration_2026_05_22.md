# Liquidity Reliability Recovery Balance Integration Review

Date: 2026-05-22

Status note, 2026-05-26: this is a historical integration review. Its
"current leading" equal annual-budget-52 baseline was the then-current
comparison target for this factor, not the latest research frontier. For
current baseline hierarchy and frontier selection, use
`docs/validation/factor_development_standard.md`.

## Decision

Do not integrate `intraday_liquidity_reliability_recovery_balance_5m` into the
then-leading equal-weight candidate portfolio.

The factor remains useful research evidence, but it is not a production-ready
complement to the then-leading combination. It passes standalone factor and
standalone policy validation, yet it is strongly dilutive when added to the
then-leading equal annual-budget-52 framework.

Registry action: move from `candidate` to `watchlist` with decision reason
`portfolio_negative`.

## Baseline

Reference portfolio:
`runs/candidate_factor_portfolios/equal_annual_budget_standard_budget52`

Configuration:

- Dataset: `runs/framework_v1_acceptance/standard/alpha_dataset`
- Method: `equal`
- Policy: `single`
- Trade policy: `cost_aware_optimizer`
- Annual gross turnover budget: 52
- Forecast calibration: `score_bucket`
- Factor health mode: `off`

Baseline metrics:

| Scenario | Return | Max drawdown | Gross turnover |
| --- | ---: | ---: | ---: |
| full base | 33.44% | -7.85% | 149.42 |
| full high cost | 22.05% | -8.72% | 149.47 |
| 2023 base | 8.51% | -6.44% | 50.69 |
| 2024 base | 16.69% | -7.23% | 48.33 |
| 2025 base | 6.83% | -4.89% | 49.13 |

## Integration Tests

All integration runs used the merged alpha dataset:
`runs/factor_research/liquidity_reliability_recovery_balance_integration_2026_05_21/alpha_dataset`

The run used `--no-enforce-registry` so the comparison stayed aligned with the
historical leading candidate set. The test only changes factor membership.

| Test | Added balance features | Output | Return | Max drawdown | Gross turnover |
| --- | --- | --- | ---: | ---: | ---: |
| leading set + both windows | `l48_c12_r24`, `l96_c24_r48` | `runs/candidate_factor_portfolios/liquidity_recovery_balance_equal_annual_budget52_quick` | 4.49% | -25.11% | 135.03 |
| leading set + l48 only | `l48_c12_r24` | `runs/candidate_factor_portfolios/liquidity_recovery_balance_l48_equal_annual_budget52_quick` | 2.80% | -22.85% | 136.99 |

Both runs passed mechanical validation, but both failed the economic acceptance
standard because they sharply reduced return and expanded drawdown versus the
baseline.

## Interpretation

The standalone signal is not enough. The l48 member has strong rank IC
(`0.05908`), strong t-stat (`65.11`), positive cost-adjusted spread
(`19.33 bps`), and low top-N turnover (`7.30%`). Standalone standard policy
validation also produced strong cost-aware full-base and high-cost results.

The failure appears at portfolio integration. The equal score blend changes the
selected frontier enough to damage the existing robust inverse-volatility and
liquidity/capacity balance. Removing the highly correlated l96 variant did not
help, so the failure is not mainly caused by duplicate windows. It is a
combination problem: this signal needs an allocator, a regime gate, or an
explicit capped factor budget before it can be considered again.

## Follow-Up Rule

Future factor development should require an incremental portfolio test against
the then-leading baseline before keeping a factor as a default `candidate`.
Strong standalone IC and standalone policy validation are necessary but not
sufficient.

For this branch, do not spend more time adding local variants of the same smooth
recovery-balance construction. The next useful work is either:

1. a portfolio allocator/gate that can use the recovery-balance signal without
   damaging the leading equal portfolio, or
2. a genuinely orthogonal factor with a clear economic mechanism and immediate
   incremental validation against the leading baseline.
