# Factor Research Batch: 2026-05-17 Breadth Resilience

## Hypothesis

`intraday_breadth_resilience_5m_w48` measures whether a stock holds up during
weak cross-sectional breadth states. For each 5-minute timestamp, the builder
computes the share of names with positive close-to-close returns. When that
advance ratio is below 50%, the factor weights each stock's return by the weak
breadth pressure and averages it over a 48-bar rolling window.

Expected direction is `long`: stocks with stronger returns during broad
advance-decline weakness should be more robust long-only holdings than stocks
that only perform when participation is already broad.

## Pre-Development Memory Check

Two earlier ideas were intentionally rejected before implementation:

- `intraday_upside_turnover_exhaustion_5m_w48` was blocked by the rejected VWAP
  deviation memory rule.
- `intraday_upside_volatility_share_5m_w48` was blocked by the rejected market
  downside beta memory rule.

The final breadth-resilience idea also matched
`intraday_market_downside_beta_5m_w48`, but that rejected factor's retry
condition explicitly allowed a materially different market-state construction
such as market breadth. This factor satisfies that condition because it uses an
advance-decline state instead of median market returns.

Memory artifact:
`runs/factor_research_memory/intraday_breadth_resilience_5m_w48/factor_research_memory_check.json`

## Implementation

- Added factor group: `breadth_resilience`
- Added config parameter: `breadth_resilience_windows`
- New feature column: `intraday_breadth_resilience_5m_w48`
- Implementation module: `quant_research.datasets.intraday_features`

## Admission Result

The standard single-factor admission rejected the factor:

| factor_id | status | direction | rank_ic | t_stat | hit_rate | stable_years | cost_adj_spread | top_n_turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| intraday_breadth_resilience_5m_w48 | reject | invert | -0.004190 | -4.20 | 49.81% | 1 | 16.64 bps | 13.00% |

The factor was statistically non-zero and cheap to trade, but the IC direction
was not reliable enough. Yearly rank IC means were mildly positive in 2023 and
2024, then strongly negative in 2025, so the result looks regime-dependent
rather than a robust cross-sectional alpha.

Artifacts:

- Alpha dataset:
  `runs/framework_v1_acceptance/factor_batch_2026_05_17_breadth_resilience/alpha_dataset`
- Factor evaluation:
  `runs/framework_v1_acceptance/factor_batch_2026_05_17_breadth_resilience/factor_evaluation/summary.json`
- Admission report:
  `runs/framework_v1_acceptance/factor_batch_2026_05_17_breadth_resilience/factor_admission/factor_admission_report.json`

## Registry

- Factor id: `intraday_breadth_resilience_5m_w48`
- Family: `market_regime`
- Status: `reject`
- Required inputs: `instrument_id`, `bar_end_time`, `close_price`

## Governance Decision

Reject as standalone alpha with `decision_reason=weak_hit_rate`. Retry only
with a materially different breadth construction, such as sector-relative
breadth, breadth shock persistence, or an explicit non-bull-market regime split
that improves 2023 and 2024 without relying on 2025 inversion.
