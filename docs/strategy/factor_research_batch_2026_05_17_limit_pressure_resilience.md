# Factor Research Batch 2026-05-17 Limit Pressure Resilience

This note records the next intraday factor round:
`intraday_limit_pressure_resilience_5m_w48`.

## Hypothesis

The factor measures how well a stock holds up when the market is under
explicit limit-pressure stress, defined by a positive excess of limit-down
opens over limit-up opens at the same timestamp. The idea is not raw momentum
or own-path volatility. It is a conditional resilience measure under a market
state that is economically distinct in A-share trading.

This should be more useful than a plain beta or gap transform because it ties
the security's return to a concrete execution-relevant stress regime:

- the stress state is observable from live open flags;
- the pressure term is market-wide and state dependent;
- the signal rewards names that are less damaged during disorderly tape.

## Implementation

- Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`
- Factor group: `limit_pressure_resilience`
- Feature column: `intraday_limit_pressure_resilience_5m_w48`
- Inputs: `instrument_id`, `bar_end_time`, `close_price`, `limit_up_open`, `limit_down_open`

## Registry

- Registry version: 7
- Status: `candidate`
- Expected direction: `invert`

## Next Step

Run the standard single-factor pipeline for the new group only, then decide
whether the factor belongs in `candidate`, `watchlist`, or `reject`.
