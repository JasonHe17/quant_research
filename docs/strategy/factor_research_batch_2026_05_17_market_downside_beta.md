# Factor Research Batch 2026-05-17 Market Downside Beta

This note records the next intraday factor round: `intraday_market_downside_beta_5m_w48`.

## Hypothesis

The factor measures each stock's sensitivity to the cross-sectional market
return when the market is already down. It is a downside-risk penalty aimed at
long-only A-share selection: names that amplify market-wide weakness should be
less attractive even if their own realized volatility is moderate.

This is materially different from the rejected raw gap factor and from the
existing downside-volatility / negative-return-persistence family:

- it uses cross-sectional market state, not just one name's own path;
- it only activates on market-down bars;
- it measures co-movement with a weak tape, not simple return frequency or
  realized variance.

## Implementation

- Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`
- Factor group: `market_downside_beta`
- Feature column: `intraday_market_downside_beta_5m_w48`
- Inputs: `instrument_id`, `bar_end_time`, `close_price`

## Registry

- Registry version: 5
- Status: `candidate`
- Expected direction: `invert`

## Next Step

Run the standard single-factor pipeline for the new group only, then decide
whether the factor belongs in `candidate`, `watchlist`, or `reject`.
