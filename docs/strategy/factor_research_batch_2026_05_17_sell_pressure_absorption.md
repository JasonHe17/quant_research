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

## Standard Validation Result

The downside-volatility state gate was useful in the 2024 quick slice, but it
did not survive the standard full-window check. The 2023-2025 full-base run
finished at 39.69% return with -30.27% max drawdown, versus the standalone
decorrelated baseline at 60.79% return with -31.84% max drawdown. That is not
an acceptable tradeoff, so this gate path is recorded as a failed validation
for the current factor.
