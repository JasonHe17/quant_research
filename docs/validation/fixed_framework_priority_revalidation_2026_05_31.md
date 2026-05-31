# Fixed-Framework Priority Revalidation - 2026-05-31

This report records the priority factor retest after the framework-v1 leakage
and governance fixes.

## Evidence

- Summary:
  `runs/legacy_factor_revalidation/fixed_framework_priority_decorr_2026_05_31/legacy_factor_revalidation_summary.json`
- Table:
  `runs/legacy_factor_revalidation/fixed_framework_priority_decorr_2026_05_31/legacy_factor_revalidation_summary.csv`
- Shared benchmark:
  `runs/framework_v1_acceptance/standard/benchmark_summary.json`
- Methods: `decorrelated`
- Policy: `partial_rebalance_daily`
- Profile: `standard`
- Label horizon: `48` bars
- Data access: `fast_parquet`

The run completed with 7 result rows and exit status `0`.

During this run, yearly and high-cost validation scenarios were changed to
reuse the already-built `full_base` score partitions. This does not change the
score values or backtest policy. It only avoids rebuilding identical monthly
score files for each scenario.

## Decision Summary

| factor | registry action | validation | full base | high cost | worst year | note |
|---|---|---:|---:|---:|---:|---|
| `intraday_amihud_5m` | keep `candidate` | warn | 54.39% | 47.60% | -1.79% | strong portfolio result; 2024 stability warning |
| `intraday_sell_pressure_recovery_5m_w48` | downgrade to `reject` | fail | -1.29% | -5.00% | -14.28% | current admission reject and portfolio fail agree |
| `intraday_daily_moving_average_state_5m` | downgrade to `watchlist` | warn | 9.08% | 4.55% | -7.45% | weak and negative in 2023/2024 |
| `intraday_eod_reversal_5m_lb1_tail6` | keep `candidate` | warn | 20.71% | 19.82% | -8.02% | strongest EOD tail overlay candidate |
| `intraday_eod_reversal_5m_lb6_tail6` | downgrade to `watchlist` | warn | 0.77% | 0.18% | -15.97% | near-zero edge despite positive admission |
| `intraday_eod_reversal_5m_lb12_tail6` | downgrade to `watchlist` | warn | 8.17% | 7.39% | -11.32% | positive full-window result but unstable years |
| `intraday_eod_reversal_5m_lb24_tail6` | downgrade to `watchlist` | warn | 7.55% | 6.71% | -14.27% | unstable years and prior compact-core overlay was negative |

## Registry Update

`configs/factors/factor_registry.json` was updated to version `66`.

Status counts after the update:

| status | count |
|---|---:|
| candidate | 24 |
| watchlist | 46 |
| reject | 25 |

Each retested entry now records:

- `evaluation.fixed_framework_revalidation`
- `evaluation.fixed_framework_revalidation_summary`
- `evaluation.fixed_framework_revalidation_review`
- `evaluation.fixed_framework_revalidation_validation_status`
- `evaluation.fixed_framework_revalidation_full_base_return`
- `evaluation.fixed_framework_revalidation_full_high_cost_return`
- `evaluation.fixed_framework_revalidation_negative_years`
- `evaluation.fixed_framework_registry_action`

## Readout

The fixed framework materially changes the interpretation of several factors.
Positive admission alone is not enough for promotion. The EOD reversal tail
features have low turnover and positive full-window returns, but most of the
edge is concentrated in 2025 while 2023 and 2024 are negative. Only
`intraday_eod_reversal_5m_lb1_tail6` remains candidate after role-specific
portfolio revalidation.

`intraday_sell_pressure_recovery_5m_w48` should leave the active research
queue. Both current admission and current portfolio validation are negative.

`intraday_daily_moving_average_state_5m` is no longer supported as a promoted
challenger by the fixed framework. It can be retried only as an incremental or
gated portfolio component that improves the benchmark after costs and repairs
the 2023/2024 losses.

## Required Follow-Ups

1. Rebuild the candidate portfolio baseline from the confirmed fixed-framework
   candidates, starting with `intraday_amihud_5m` and
   `intraday_eod_reversal_5m_lb1_tail6`.
2. Run an incremental overlay test for the EOD tail family against the current
   selected policy before admitting any `lb6`, `lb12`, or `lb24` variant back to
   candidate.
3. Keep `intraday_sell_pressure_recovery_5m_w48` in research memory only unless
   a new definition passes current admission and fixed-framework portfolio
   validation.
