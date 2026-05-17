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
- Status: `watchlist`
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

Do not continue standalone portfolio validation for the current specification.
The factor should only be retried as a conditional overlay or interaction
feature, for example gated by broader market drawdown, downside volatility, or
liquidity state.

## Portfolio Validation Result

The standard standalone decorrelated validation was stopped after `full_base`
because the first scenario failed the risk-reward check. The 2023-2025
full-base equity curve ended at 0.73% total return with -34.29% max drawdown.
The result is not an outright negative final return, but it is economically
unusable as a standalone long-only signal despite the clean single-factor
admission profile.

## Failure Diagnosis

The main issue is long-only selection quality. The factor passed admission
because top-minus-bottom spread was positive, but score diagnostics show the
top-score basket itself had negative average forward labels in all three years:

- 2023 top-score mean label: `-0.00068`
- 2024 top-score mean label: `-0.00221`
- 2025 top-score mean label: `-0.00102`

This means the factor mostly identified names to avoid in the bottom bucket,
not names that were strong enough to buy. Transaction costs were not the
primary failure driver: total cost was about 15.27% of initial capital, while
pre-cost return was only about 16.0%, far below sell-pressure absorption's
roughly 78.1% pre-cost return over the comparable full-base run.
