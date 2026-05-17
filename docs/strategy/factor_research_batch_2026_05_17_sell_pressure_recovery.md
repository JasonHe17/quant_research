# Factor Research Batch 2026-05-17 Sell Pressure Recovery

This note records the next long-only oriented intraday factor round:
`intraday_sell_pressure_recovery_5m_w48`.

## Hypothesis

The previous downside-turnover-decay factor passed top-minus-bottom admission
but failed long-only diagnostics because its top-score basket still had negative
average forward labels in every year. This round therefore targets top-bucket
quality directly.

The factor measures recovery after sell pressure:

- rolling positive return over the window;
- divided by rolling downside return damage over the same window;
- multiplied by the share of turnover that occurred on positive-return bars.

High values require both price recovery and turnover confirmation. The intended
signal is not just "less bad than the bottom bucket"; it should identify names
where recent sell pressure has been followed by tradable recovery.

## Implementation

- Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`
- Factor group: `sell_pressure_recovery`
- Feature column: `intraday_sell_pressure_recovery_5m_w48`
- Inputs: `instrument_id`, `bar_end_time`, `close_price`, `turnover`

## Research Memory

The pre-development memory check matched several turnover and recovery-like
watchlist entries, plus the rejected `intraday_vwap_deviation_5m_w48`. The VWAP
retry condition allows a materially different microstructure hypothesis such as
liquidity-conditioned recovery. This factor qualifies because it does not use
price distance from VWAP; it uses recovery return after downside damage and
requires upside turnover participation.

## Registry

- Registry version: 17
- Status: `watchlist`
- Expected direction: `long`

## Admission Result

The standard single-factor run did not pass promotion gates:

- Admission status: `watchlist`
- Admission direction: `invert`
- Spearman rank IC mean: `-0.019643`
- Rank IC t-stat: `-26.1378`
- Cost-adjusted spread: `-0.001656`
- Failed check: `cost_adjusted_spread`

## Top-Basket Health

The additional long-only health check also failed. The highest quantile had
negative average forward labels in two of three years:

- 2023 q5 mean label: `-0.000448`
- 2024 q5 mean label: `-0.000096`
- 2025 q5 mean label: `0.000437`

This means the intended "recovery quality" top bucket was not a reliable
long-only buy basket. The likely interpretation is that strong immediate
recovery after sell pressure often marks short-horizon overextension or
exhaustion rather than a robust next-day entry.

## Next Step

Do not run standalone portfolio validation for this specification. Retry only
with a materially different recovery definition, such as separating early
rebound from late overextension or adding an explicit pullback/entry-price
condition.
