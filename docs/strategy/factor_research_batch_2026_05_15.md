# Factor Research Batch 2026-05-15

This note records the first controlled factor expansion after the Framework v1
research layer became usable for factor discovery.

## Candidate Set

The batch deliberately stays inside live 5-minute OHLCV data so that every
candidate can reuse the existing A-share point-in-time, T+1, ST, price-limit,
and liquidity-aware evaluation path.

| factor_id | family | hypothesis |
| --- | --- | --- |
| intraday_range_position_5m_w48 | momentum | A stock closing near the top of its recent intraday high-low range may have stronger short-horizon demand than a stock closing near the bottom of the range. |
| intraday_range_volatility_5m_w48 | volatility | High-low range volatility captures intrabar risk that close-to-close volatility can miss; high values should be penalized in long-only selection after costs. |
| intraday_efficiency_ratio_5m_w48 | momentum | A directional move that consumes little path length may be a cleaner trend than a noisy move of the same total return. |
| intraday_vwap_deviation_5m_w48 | reversal | A close price stretched above its rolling intraday VWAP may be vulnerable to short-horizon mean reversion after next-bar execution lag. |

## Implementation Boundary

- Input data: `instrument_id`, `bar_end_time`, `open_price`, `high_price`,
  `low_price`, `close_price`, `volume`, and `turnover`.
- Frequency: 5-minute CN equity bars.
- Label assumption for standard review: 48-bar forward return with 1-bar entry
  lag, matching the current Framework v1 benchmark.
- Initial status: `candidate`; promotion requires the standard single-factor
  admission report, candidate review artifact, and portfolio-level contribution
  evidence.

## Standard Admission Result

Standard Framework v1 validation was run under
`runs/framework_v1_acceptance/factor_batch_2026_05_15` on the 2023-01-03 to
2025-12-31 5-minute main-board window. The benchmark acceptance status was
`pass`; the single-factor admission report evaluated 23 features and classified
9 as `candidate`, 3 as `watchlist`, and 11 as `reject`.

| factor_id | status | direction | rank_ic | t_stat | hit_rate | stable_years | cost_adj_spread | turnover | review |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| intraday_range_volatility_5m_w48 | candidate | invert | -0.068756 | -85.62 | 0.7212 | 3 | 0.009800 | 0.0414 | `runs/factor_candidate_reviews/intraday_range_volatility_5m_w48/factor_candidate_review.md` |
| intraday_efficiency_ratio_5m_w48 | candidate | invert | -0.017126 | -29.90 | 0.6080 | 3 | 0.003577 | 0.2190 | `runs/factor_candidate_reviews/intraday_efficiency_ratio_5m_w48/factor_candidate_review.md` |
| intraday_range_position_5m_w48 | reject | long | 0.006922 | 9.26 | 0.5019 | 1 | 0.002600 | 0.6255 | `runs/factor_candidate_reviews/intraday_range_position_5m_w48/factor_candidate_review.md` |
| intraday_vwap_deviation_5m_w48 | reject | long | 0.006268 | 7.65 | 0.5021 | 1 | 0.003227 | 0.2676 | `runs/factor_candidate_reviews/intraday_vwap_deviation_5m_w48/factor_candidate_review.md` |

The two rejected features had positive full-window IC and cost-adjusted spread,
but failed the directional hit-rate and stable-year gates, so they should not
enter portfolio-level validation in the current framework.
