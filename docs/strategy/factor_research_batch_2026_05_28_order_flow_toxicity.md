# Factor Research Batch - 2026-05-28 Order Flow Toxicity

This batch adds a first-pass order-flow imbalance and VPIN-style toxicity
feature group for 5-minute A-share bars.

## Hypothesis

Informed trading should make recent order flow directional and toxic before the
full price correction is visible in ordinary realized volatility or Amihud
impact. The current feature set has several non-directional or downside-only
microstructure proxies, but it does not directly decompose buy-pressure versus
sell-pressure volume.

The new group is intentionally different from nearby existing features:

- `intraday_signed_turnover_imbalance` signs turnover and was historically
  cost-fragile.
- `intraday_amihud_5m` measures absolute impact per turnover, not the source or
  direction of the pressure.
- `intraday_sell_pressure_absorption_5m_w48` measures capacity to absorb
  downside turnover, not symmetric buy versus sell imbalance.

## Implementation

Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`

Factor group: `order_flow_toxicity`

Default windows: `6`, `12`, and `48` five-minute bars.

Emitted features:

| feature | interpretation |
|---|---|
| `intraday_order_flow_imbalance_5m_w{window}` | Rolling signed volume divided by rolling total volume. Positive means buy-pressure dominance; negative means sell-pressure dominance. |
| `intraday_vpin_5m_w{window}` | Absolute aggregate signed-volume imbalance divided by rolling total volume, used as a 5-minute-bar VPIN proxy. |
| `intraday_adverse_selection_5m_w{window}` | Rolling correlation between one-bar-lagged signed volume and current 5-minute return. |

The bar-level classification is:

```text
bar_return = close / previous_close - 1
signed_volume = sign(bar_return) * volume

OFI = rolling_sum(signed_volume, window) / rolling_sum(volume, window)
VPIN_proxy = abs(rolling_sum(signed_volume, window)) / rolling_sum(volume, window)
AdverseSelection = rolling_corr(lag(signed_volume, 1), bar_return, window)
```

The VPIN implementation uses the absolute aggregate imbalance over the rolling
window. With only 5-minute OHLCV bars and a whole-bar sign classifier, summing
per-bar absolute buy-minus-sell volume would collapse to total volume and lose
cross-sectional information.

## Registry Entries

The initial registry added the 6-bar variants for first-pass testing. After
admission, all three were moved to `watchlist` because their cost-adjusted
spreads were negative:

| factor | admitted direction | status | rationale |
|---|---:|---|---|
| `intraday_order_flow_imbalance_5m_w6` | `invert` | `watchlist` | Statistically strong but too turnover-heavy after 13 bps costs. |
| `intraday_vpin_5m_w6` | `invert` | `watchlist` | Correct toxicity direction, but short-window turnover erased spread. |
| `intraday_adverse_selection_5m_w6` | `invert` | `watchlist` | Stable IC but highest turnover in the batch. |

After the 2026-05-28 validation run, the registry also adds these 48-bar
variants as monitored candidates:

| factor | direction | status |
|---|---:|---|
| `intraday_vpin_5m_w48` | `invert` | `candidate` |
| `intraday_adverse_selection_5m_w48` | `invert` | `candidate` |

## Validation Run

Dataset build:

```bash
conda run -n quant python examples/build_baseline_a_alpha_dataset.py \
  --catalog-path ../quant_dataset/canonical_store/catalog/quant_research.duckdb \
  --start 2023-01-03T09:35:00+08:00 \
  --end 2025-12-31T15:00:00+08:00 \
  --factor-groups order_flow_toxicity \
  --order-flow-toxicity-windows 6 12 48 \
  --label-name forward_return \
  --horizon-bars 6 48 \
  --entry-lag-bars 1 \
  --output-dir runs/factor_research/order_flow_toxicity_2026_05_28/alpha_dataset \
  --workers 1 \
  --partition monthly \
  --padding-days 30 \
  --data-snapshot 2026-05-28
```

Single-factor evaluation used `forward_return_6b` as the primary label and
`forward_return_48b` as a horizon diagnostic:

```bash
conda run -n quant python examples/evaluate_alpha_dataset.py \
  --dataset-dir runs/factor_research/order_flow_toxicity_2026_05_28/alpha_dataset \
  --output-dir runs/factor_research/order_flow_toxicity_2026_05_28/factor_evaluation_6b \
  --label-column forward_return_6b \
  --horizon-label-columns forward_return_48b \
  --top-n 50 \
  --quantiles 5 \
  --workers 6 \
  --backend process
```

Admission used the standard framework thresholds with 13 bps costs:

```bash
conda run -n quant python examples/analyze_framework_v1_acceptance.py \
  --benchmark-summary runs/framework_v1_acceptance/standard/benchmark_summary.json \
  --factor-summary runs/factor_research/order_flow_toxicity_2026_05_28/factor_evaluation_6b/summary.json \
  --by-timestamp runs/factor_research/order_flow_toxicity_2026_05_28/factor_evaluation_6b/single_factor_by_timestamp.csv \
  --output-dir runs/factor_research/order_flow_toxicity_2026_05_28/factor_admission_6b \
  --cost-bps 13.0
```

## Single-Factor Results

Admission results: 9 factors tested, 2 candidates, 7 watchlist, 0 rejects.

| factor | status | direction | rank IC | t-stat | hit rate | cost-adjusted spread | turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| `intraday_vpin_5m_w48` | `candidate` | `invert` | -0.010559 | -31.13 | 0.6020 | 0.000255 | 0.1906 |
| `intraday_adverse_selection_5m_w48` | `candidate` | `invert` | -0.010384 | -32.15 | 0.5904 | 0.000017 | 0.1475 |
| `intraday_order_flow_imbalance_5m_w48` | `watchlist` | `invert` | -0.010996 | -19.66 | 0.5390 | -0.000167 | 0.1711 |
| `intraday_order_flow_imbalance_5m_w12` | `watchlist` | `invert` | -0.015646 | -33.18 | 0.5922 | -0.000244 | 0.4047 |
| `intraday_order_flow_imbalance_5m_w6` | `watchlist` | `invert` | -0.015694 | -35.57 | 0.6013 | -0.000492 | 0.5607 |
| `intraday_vpin_5m_w12` | `watchlist` | `invert` | -0.006971 | -22.56 | 0.5833 | -0.000801 | 0.4639 |
| `intraday_vpin_5m_w6` | `watchlist` | `invert` | -0.007235 | -24.25 | 0.5909 | -0.001009 | 0.6134 |
| `intraday_adverse_selection_5m_w12` | `watchlist` | `invert` | -0.007177 | -26.73 | 0.5785 | -0.000464 | 0.4686 |
| `intraday_adverse_selection_5m_w6` | `watchlist` | `invert` | -0.004525 | -19.20 | 0.5603 | -0.000818 | 0.7058 |

The 48-bar VPIN and adverse-selection features had low mutual correlation
(`0.166`) and low correlation with the 48-bar directional imbalance feature
(`-0.071` and `0.020`, respectively). The shorter same-family windows are more
correlated with each other and more turnover-heavy.

## Portfolio Validation

Candidate validation used the two admitted w48 features under a daily-like
rank-buffer policy:

```bash
conda run -n quant python examples/run_candidate_policy_validation.py \
  --dataset-dir runs/factor_research/order_flow_toxicity_2026_05_28/alpha_dataset \
  --label-column forward_return_6b \
  --admission-report runs/factor_research/order_flow_toxicity_2026_05_28/factor_admission_6b/factor_admission_report.json \
  --factor-correlation runs/factor_research/order_flow_toxicity_2026_05_28/factor_evaluation_6b/feature_correlation.csv \
  --output-dir runs/factor_research/order_flow_toxicity_2026_05_28/candidate_policy_validation_6b_single \
  --profile standard \
  --methods decorrelated equal ic_weighted \
  --primary-method decorrelated \
  --policy single \
  --include-features intraday_vpin_5m_w48 intraday_adverse_selection_5m_w48 \
  --no-enforce-registry \
  --backtest-policy-set single \
  --trade-policy rank_buffer_drop \
  --rebalance-every-n-bars 48 \
  --policy-entry-rank 50 \
  --policy-exit-rank 150 \
  --policy-max-entries-per-rebalance 10 \
  --policy-max-exits-per-rebalance 10 \
  --policy-partial-rebalance-rate 0.5 \
  --policy-estimated-cost-bps 13.0 \
  --policy-no-trade-weight-band 0.002 \
  --resume-existing
```

Validation status: `warn`, with zero hard failures and one warning for negative
2023 returns.

| method | full-base return | high-cost return | mean turnover | worst year return | worst drawdown |
|---|---:|---:|---:|---:|---:|
| `equal` | 13.39% | 8.26% | 72.32 | -3.99% | -35.15% |
| `decorrelated` | 11.88% | 7.10% | 72.25 | -4.08% | -35.29% |
| `ic_weighted` | 11.62% | 6.74% | 72.24 | -4.24% | -35.39% |

Primary decorrelated annual slices:

| scenario | return | max drawdown | gross turnover |
|---|---:|---:|---:|
| 2023 | -4.08% | -16.56% | 40.40 |
| 2024 | 2.56% | -27.11% | 42.84 |
| 2025 | 28.73% | -17.54% | 40.64 |

Decision: promote `intraday_vpin_5m_w48` and
`intraday_adverse_selection_5m_w48` to monitored `candidate` status. Do not
promote the shorter windows or the raw OFI variant until they show positive
cost-adjusted spread or a lower-turnover portfolio integration.

## Compact-Core Incremental Test

The next-step incremental test blended the toxicity score into the existing
compact-core decorrelated score stream. The toxicity satellite used the equal
two-factor score from `intraday_vpin_5m_w48` and
`intraday_adverse_selection_5m_w48`.

```bash
conda run -n quant python examples/run_score_overlay_validation.py \
  --primary-score-dir runs/candidate_factor_portfolios/compact_core_legacy_revalidation_2026_05_20_standard/full_base/scores/decorrelated \
  --satellite-score-dir runs/factor_research/order_flow_toxicity_2026_05_28/candidate_policy_validation_6b_single/full_base/scores/equal \
  --output-dir runs/factor_research/order_flow_toxicity_2026_05_28/compact_core_incremental_blend_standard \
  --method-prefix toxicity_blend \
  --overlay-mode blend \
  --overlay-weights 0 0.025 0.05 0.10 0.15 \
  --profile standard \
  --policy partial_rebalance_daily \
  --trade-policy rank_buffer_drop \
  --rebalance-every-n-bars 48 \
  --policy-entry-rank 50 \
  --policy-exit-rank 150 \
  --policy-max-entries-per-rebalance 10 \
  --policy-max-exits-per-rebalance 10 \
  --policy-partial-rebalance-rate 0.5 \
  --policy-estimated-cost-bps 13.0 \
  --policy-no-trade-weight-band 0.002 \
  --job-workers 2 \
  --resume-existing
```

Validation status: `warn`, with zero hard failures and five annual-stability
warnings because every tested weight still had a negative 2024 slice.

| method | toxicity weight | full-base | high-cost | 2023 | 2024 | 2025 | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `toxicity_blend_w00` | 0.0% | 42.92% | 36.39% | 14.97% | -5.96% | 31.85% | -32.48% |
| `toxicity_blend_w025` | 2.5% | 34.91% | 28.85% | 9.67% | -1.20% | 33.57% | -30.90% |
| `toxicity_blend_w05` | 5.0% | 42.00% | 36.39% | 9.24% | -4.34% | 29.08% | -32.63% |
| `toxicity_blend_w10` | 10.0% | 34.05% | 28.32% | 8.25% | -5.94% | 32.71% | -31.74% |
| `toxicity_blend_w15` | 15.0% | 39.17% | 33.44% | 10.25% | -7.83% | 34.25% | -30.56% |

Decision: do not add the toxicity basket to compact core as a fixed-weight
linear blend. The best full-base blend was `5%`, but it still trailed the
control by 0.93 percentage points and slightly trailed high-cost performance.
The `2.5%` blend materially improved 2024 but paid too much in full-window,
high-cost, and 2023 returns.

The useful signal from this test is conditional: toxicity helps some 2024 stress
selection in a small linear blend, but the full-window trade-off is not
acceptable for core inclusion.

## Compact-Core Avoidance Overlay Tests

Two follow-up overlays tested whether the toxicity basket works better as a
risk-control signal than as linear alpha:

- `entry_exclusion`: keep the compact-core score unchanged, but block entry for
  the worst toxicity satellite tail.
- `downside_penalty`: keep all names eligible, but subtract a penalty from the
  worst 20% toxicity satellite tail.

Both validations used the same compact-core primary score stream, the same
equal-weight VPIN/adverse-selection toxicity satellite, the standard
`partial_rebalance_daily` policy, and 13 bps estimated trading cost.

Entry-exclusion validation:
`runs/factor_research/order_flow_toxicity_2026_05_28/compact_core_toxicity_entry_exclusion_standard/validation_summary.json`

| method | excluded toxicity tail | full-base | high-cost | 2023 | 2024 | 2025 | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `toxicity_exclude_w00` | 0% | 42.92% | 36.39% | 14.97% | -5.96% | 31.85% | -32.34% |
| `toxicity_exclude_w05` | 5% | 41.70% | 35.15% | 14.97% | -5.96% | 29.69% | -32.54% |
| `toxicity_exclude_w10` | 10% | 41.70% | 35.15% | 14.97% | -6.05% | 32.40% | -32.54% |
| `toxicity_exclude_w20` | 20% | 42.23% | 36.02% | 13.58% | -6.93% | 30.55% | -31.97% |

Downside-penalty validation:
`runs/factor_research/order_flow_toxicity_2026_05_28/compact_core_toxicity_downside_penalty_standard/validation_summary.json`

| method | penalty strength | full-base | high-cost | 2023 | 2024 | 2025 | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `toxicity_penalty_w00` | 0.00 | 42.92% | 36.39% | 14.97% | -5.96% | 31.85% | -32.34% |
| `toxicity_penalty_w25` | 0.25 | 42.23% | 36.21% | 14.13% | -6.28% | 31.68% | -32.04% |
| `toxicity_penalty_w50` | 0.50 | 40.87% | 34.97% | 13.91% | -7.29% | 31.77% | -32.05% |
| `toxicity_penalty_w01` | 1.00 | 41.58% | 35.14% | 14.42% | -7.51% | 31.53% | -32.15% |

Decision: do not add the toxicity basket to compact core as a fixed linear
blend, entry-exclusion filter, or downside-penalty overlay. The control remains
the best full-base and high-cost configuration. The avoidance overlays reduce
or roughly preserve drawdown in a few cases, but they do not repair 2024 and
they dilute full-window return. Keep `intraday_vpin_5m_w48` and
`intraday_adverse_selection_5m_w48` as monitored standalone candidate and
diagnostic signals rather than compact-core components.
