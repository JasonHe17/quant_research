# intraday_amihud_5m Validation Notes

Generated: 2026-05-18

## Summary

`intraday_amihud_5m` has a real standalone signal footprint, but it is not a stable
promotion candidate without regime control.

Full-window standard validation:

| Scenario | Return | Max drawdown | Gross turnover | Notes |
| --- | ---: | ---: | ---: | --- |
| full_base | 12.79% | -3.78% | 47.59 | Positive full-window result. |
| full_high_cost | 9.47% | -4.33% | 48.38 | Survives doubled transaction costs. |
| year_2023_base | 5.47% | -3.78% | 48.29 | Positive. |
| year_2024_base | -15.77% | -32.08% | 50.07 | Fails yearly stability. |
| year_2025_base | 12.62% | -8.38% | 45.97 | Positive. |

Validation artifact:
`runs/candidate_factor_portfolios/intraday_amihud_5m_standard_validation/validation_summary.json`

## 2024 Failure Diagnosis

The 2024 failure is concentrated early:

| Month | Portfolio return | Score IC | Score spread | Market label | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024-01 | -13.23% | -0.0305 | 0.06% | -0.82% | 1082 |
| 2024-02 | -6.83% | 0.0432 | 0.45% | 0.47% | 748 |
| 2024-03 | 3.90% | 0.0460 | 0.18% | 0.25% | 946 |

After March, the path turnover budget is effectively exhausted in the yearly
scenario, so most later monthly rows have `trade_count = 0`. The bad 2024 result
is therefore not a broad all-year ranking failure; it is a concentrated early
portfolio construction and risk-control failure.

Regime diagnostic artifact:
`runs/candidate_factor_portfolios/intraday_amihud_5m_standard_validation/regime_diagnostics_2024/summary.json`

## Mitigation Checks

| Check | 2024 return | Max drawdown | Gross turnover | Assessment |
| --- | ---: | ---: | ---: | --- |
| Baseline yearly path budget | -15.77% | -32.08% | 50.07 | Fails. |
| Path budget pacing = 1.0 | -12.11% | -32.08% | 50.41 | Insufficient. |
| Drawdown brake -7%, reduced scale 0 | 5.91% | -31.20% | 54.03 | Improves final return, does not control tail drawdown. |
| Score-health gate v1 | -4.28% | -27.88% | 41.92 | Helps, but still fails yearly stability. |
| Volatility gate v1 | -15.96% | -31.57% | 48.25 | Does not identify the bad regime. |
| Score-health gate v1 + drawdown brake | 8.18% | -27.19% | 46.49 | Best tested variant, still fails drawdown control. |
| Market downside gate w48, 240-lookback, 70/90 q | -13.77% | -30.71% | 47.01 | Directionally helpful, too slow. |
| Market downside gate w48, 120-lookback, 50/75 q, zero scale | -8.94% | -30.14% | 39.64 | Cuts trades and loss, still misses drawdown control. |

The drawdown brake can stop additional damage after the drawdown is observed,
but it does not prevent the initial drawdown event. That makes it a damage
control tool, not a sufficient promotion gate. The lagged score-health gate is
the best pre-trade proxy tested so far, but it reacts too late to prevent the
January/February 2024 loss.

## Decision

Do not promote `intraday_amihud_5m` as a standalone factor yet.

Next research step: add earlier market-state features to the alpha dataset and
build a pre-trade regime gate that blocks or scales down the factor before
January/February 2024-like conditions. The gate should be based on observable
state available before entry, not on realized portfolio drawdown. Promising
inputs include recent market breadth, market-level downside momentum, liquidity
stress, and limit-down pressure. A simple cross-sectional volatility proxy was
tested and rejected.

## Market-State Gate Prototype

Added a dedicated `market_state` feature group for risk-control inputs rather
than ordinary alpha admission. The generated columns use a `market_state_`
prefix so they are not picked up by the existing `intraday_` alpha feature
allowlist. Current outputs include market median return, downside pressure,
breadth, weak breadth, limit-down rate, limit pressure, and rolling mean risk
proxies.

Artifacts:

- Monthly 2024 dataset:
  `runs/factor_research_smoke/2026_05_18_amihud_market_state_2024_monthly_dataset`
- Conservative gate validation:
  `runs/candidate_factor_portfolios/intraday_amihud_5m_market_state_gate_2024_validation_v2/validation_summary.json`
- Fast gate validation:
  `runs/candidate_factor_portfolios/intraday_amihud_5m_market_state_gate_fast_2024_validation/validation_summary.json`

The market-state gate improves final 2024 PnL only when made aggressive, but it
does not materially reduce the worst drawdown. This suggests the remaining
problem is not just detecting stressed market state; the strategy also needs
entry timing and portfolio construction changes that avoid large early exposure
before the regime proxy has enough history.
