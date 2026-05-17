# Factor Research Batch 2026-05-17 Downside Turnover Decay

This note records the next intraday factor round:
`intraday_downside_turnover_decay_5m_w48`.

## Hypothesis

The factor measures whether turnover on down bars is decaying or intensifying
inside a recent intraday window. It splits the 48-bar lookback into two 24-bar
halves, sums turnover only on negative-return bars in each half, and compares
the earlier sell-pressure turnover with the more recent sell-pressure turnover.

This targets non-bull-market handling more directly than a generic turnover or
volatility transform:

- high values mean downside turnover was heavier in the earlier half and has
  faded recently, which may indicate sell-pressure exhaustion;
- low values mean downside turnover is concentrated in the recent half, which
  may indicate intensifying distribution;
- the transform uses the time profile of downside turnover, not only the total
  amount of downside turnover or the frequency of negative bars.

## Implementation

- Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`
- Factor group: `downside_turnover_decay`
- Feature column: `intraday_downside_turnover_decay_5m_w48`
- Inputs: `instrument_id`, `bar_end_time`, `close_price`, `turnover`

## Registry

- Registry version: 14
- Status: `candidate`
- Expected direction: `long`

## Research Memory

The pre-development memory check matched turnover-family watchlist factors and
the rejected `intraday_vwap_deviation_5m_w48`. The VWAP rejection allows retry
only under a different microstructure hypothesis. This factor qualifies because
it does not measure price distance from VWAP or generic reversal; it measures
whether negative-return turnover is moving from the earlier half of the window
to the recent half.

## Admission Result

The standard single-factor run passed all admission checks in the long
direction:

- Spearman rank IC mean: `0.006878`
- Rank IC t-stat: `13.9275`
- Directional IC hit rate: `0.5339`
- Stable years: `3`
- Cost-adjusted top-minus-bottom spread: `0.001564`
- Top-N turnover: `0.2779`

## Next Step

Run the standard portfolio validation path next. The factor cleared admission
with positive annual IC in 2023, 2024, and 2025, so the remaining question is
whether it adds enough post-cost portfolio value to be useful beyond a raw
candidate signal.
