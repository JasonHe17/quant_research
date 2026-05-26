# Factor Research Batch - 2026-05-25 Cross-Sectional Reversal

This batch implements the next reversal-first factor development round. The
starting point is the existing 5-minute reversal code path, but the registered
research surface is now feature-level and ready for dataset, admission, and
portfolio validation.

## Hypothesis

Short-horizon intraday loser reversal should be a structural A-share alpha
source because T+1 constraints make 5-minute reversal signals natural inputs for
next-bar entry and overnight holding. The intended premium is liquidity
provision after intraday extrapolation rather than continuation.

The batch deliberately separates this from the prior rejected
`intraday_vwap_deviation_5m_w48`: these features use recent close-return
reversal, market-demeaned reversal, state conditioning, or end-of-day timing;
they do not reuse distance from a rolling VWAP anchor.

## Implemented Feature Groups

| group | features | design |
|---|---|---|
| `reversal` | `intraday_reversal_5m_lb1`, `lb6`, `lb12`, `lb24` | Negative own close-to-close return over multiple 5-minute horizons. |
| `cross_sectional_reversal` | `intraday_cross_sectional_reversal_5m_lb1`, `lb6`, `lb12`, `lb24` | Negative own return residual after subtracting the timestamp cross-sectional median return. |
| `conditional_reversal` | `intraday_low_vol_volume_confirmed_reversal_5m_lb{1,6,12,24}_w12` | Reversal only for recent losers with low 12-bar realized volatility and volume at least equal to the 12-bar average. |
| `eod_reversal` | `intraday_eod_reversal_5m_lb{1,6,12,24}_tail6` | Reversal emitted only in the last six 5-minute bars, corresponding to 14:35-15:00 when timestamps are parseable. |

Default reversal lookbacks are now `(1, 6, 12, 24)` to match the requested
multi-scale research grid.

## Validation Plan

1. Build a new-factor-only dataset:
   `--factor-groups reversal cross_sectional_reversal conditional_reversal eod_reversal`.
2. Use `--lookback-bars 1 6 12 24` and the matching cross-sectional,
   conditional, and EOD lookback arguments.
3. Run single-factor evaluation and admission against `forward_return_48b`.
4. For admitted features, run candidate review and then incremental portfolio
   validation against the compact-core basket.
5. Inspect January and June 2024 separately, because the current factor stack
   failed most clearly in trend-down stress.

## Current Status

Implementation, unit tests, full dataset build, single-factor evaluation, and
standard admission are complete.

Dataset build:

- Output: `research_store/reversal_batch_2026_05_25_alpha_dataset`
- Window: 2023-01-01 09:35 to 2025-12-31 15:00 Asia/Shanghai
- Partitions: 36 monthly parquet files
- Rows: 103,495,412 joined feature/label rows
- Label: `forward_return`, horizon 48 bars, entry lag 1 bar
- Entry filters: ST excluded, price-limit aware, entry tradability and entry
  limit-up filters enabled

Evaluation and admission:

- Evaluation: `research_store/reversal_batch_2026_05_25_factor_evaluation/summary.json`
- Admission: `runs/factor_research/reversal_batch_2026_05_25/factor_admission/factor_admission_report.json`
- Result: 0 candidates, 4 watchlist, 12 rejects

| feature group | admission result | key finding |
|---|---:|---|
| `reversal` | 2 watchlist, 2 reject | 12/24-bar reversal passes statistical gates but fails the cost-adjusted spread gate; 1/6-bar reversal is too weak or unstable. |
| `cross_sectional_reversal` | 2 watchlist, 2 reject | Metrics are identical to raw reversal in timestamp-wise rank evaluation because subtracting a cross-sectional median does not change cross-sectional ordering. Keep only for absolute-score models or downstream residual features, not as a separate ranked alpha. |
| `conditional_reversal` | 4 reject | Sparse low-vol/volume-confirmed loser signals fail the 95% coverage gate; 6/12/24-bar variants work only in inverted IC direction and remain negative after costs. |
| `eod_reversal` | 4 reject | Tail-only coverage is 12.48%, so the standard full-coverage gate rejects all variants. The 24-bar tail has positive cost-adjusted spread, but it needs a tail-only overlay or portfolio policy test rather than standalone admission. |

The only statistically usable standalone reversal signals in this batch are
`intraday_reversal_5m_lb12`, `intraday_reversal_5m_lb24`, and their
cross-sectional residual aliases. They are not promoted because their
top-minus-bottom spread is negative after the 13 bps round-trip cost proxy.

The next research step should not be another plain rank-equivalent
cross-sectional demeaning pass. The useful branch is a lower-turnover reversal
policy or a sparse tail-only overlay test that evaluates EOD 24-bar reversal
under a portfolio policy designed for tail signals.

## Portfolio Overlay Test

The EOD 24-bar tail signal was tested as a portfolio-layer overlay instead of a
standalone admission candidate.

Artifacts:

- Satellite score stream:
  `runs/candidate_factor_portfolios/eod_reversal_lb24_tail6_satellite_2026_05_25/summary.json`
- Tail schedule:
  `runs/candidate_factor_portfolios/eod_reversal_lb24_tail6_satellite_2026_05_25/eod_tail_condition_schedule.csv`
  with 4,355 active tail timestamps out of 34,846 timestamps
- Close-only schedule:
  `runs/candidate_factor_portfolios/eod_reversal_lb24_tail6_satellite_2026_05_25/eod_close_condition_schedule.csv`
  with 725 active close timestamps out of 34,846 timestamps

Three policy variants were evaluated:

| test | artifact | result |
|---|---|---|
| 48-bar sampled overlay | `runs/candidate_factor_portfolios/eod_reversal_lb24_tail6_compact_core_overlay_2026_05_25_standard/validation_summary.json` | All overlay weights reproduced the compact-core baseline because 48-bar decision sampling missed the EOD condition timestamps. Treat this as a no-op control, not evidence for promotion. |
| Tail-6 sparse overlay | `runs/candidate_factor_portfolios/eod_reversal_lb24_tail6_compact_core_sparse_overlay_2026_05_25_standard/validation_summary.json` | Rebalancing on all six tail bars re-ranked the compact core intraday and exploded turnover. The weight-0 control lost 54.06% full-window with 602.48 turnover, so this policy design is rejected. |
| Close-only anchored overlay | `runs/candidate_factor_portfolios/eod_reversal_lb24_tail6_compact_core_close_overlay_2026_05_25_standard/validation_summary.json` | Uses daily first compact-core ranking as the primary leg and applies the EOD reversal only at 15:00. This is the cleanest tested overlay design, but it still underperforms the original compact-core policy. |

Close-only anchored overlay results:

| method | overlay weight | full base | high cost | 2024 | full turnover |
|---|---:|---:|---:|---:|---:|
| `eod_lb24_close_w00` | 0% | 19.05% | 9.56% | -13.29% | 214.64 |
| `eod_lb24_close_w01` | 1% | 20.32% | 10.69% | -11.62% | 214.71 |
| `eod_lb24_close_w02` | 2% | 25.23% | 15.42% | -13.03% | 215.40 |
| `eod_lb24_close_w05` | 5% | 23.41% | 13.92% | -12.15% | 215.18 |
| `eod_lb24_close_w10` | 10% | 22.94% | 13.13% | -11.16% | 214.76 |

Compact-core reference, using the original `decorrelated` partial-rebalance
daily policy:

| method | full base | high cost | 2024 | full turnover |
|---|---:|---:|---:|---:|
| `decorrelated` compact core | 42.92% | 36.39% | -5.96% | 111.60 |

Decision: keep `intraday_eod_reversal_5m_lb24_tail6` rejected. The standalone
tail feature has attractive tail-only IC and positive after-cost spread inside
the sparse EOD slice, but the tested portfolio overlays either fail to activate,
overtrade, or dilute the compact-core economics. Do not continue local EOD
overlay weight searches under the same policy family. A future retry should use
a materially different allocator, such as using EOD information as a next-day
single-rebalance tie-breaker, or move to the orthogonal time-series
decomposition branch.
