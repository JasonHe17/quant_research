# Factor Research Batch 2026-05-17 Sell Pressure Absorption

This note records the next intraday factor round:
`intraday_sell_pressure_absorption_5m_w48`.

## Hypothesis

The factor measures downside-only turnover absorption. It asks whether a name
can trade meaningful turnover while absorbing negative price pressure more
efficiently than peers.

This is intended to capture non-bull-market resilience that is not the same as
beta, reversal, or simple turnover expansion:

- high absorption should matter when broad tape quality deteriorates;
- the signal is downside-only, so it focuses on stress handling rather than
  general activity;
- the denominator is cumulative negative return, so the signal separates
  “busy but weak” from “busy and defended”.

## Implementation

- Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`
- Factor group: `sell_pressure_absorption`
- Feature column: `intraday_sell_pressure_absorption_5m_w48`
- Inputs: `instrument_id`, `bar_end_time`, `close_price`, `turnover`

## Registry

- Registry version: 9
- Status: `candidate`
- Expected direction: `long`

## Next Step

Run the standard single-factor pipeline for the new group only, then decide
whether the factor belongs in `candidate`, `watchlist`, or `reject`.
