# Factor Research Batch - 2026-05-29 Intraday Quality

This batch adds an `intraday_quality` feature group for 5-minute A-share bars.
The goal is to translate the quality/profitability idea into observable
intraday execution proxies using only point-in-time OHLCV fields.

## Hypothesis

Classic quality and profitability factors are persistent because they capture
the ability to earn returns efficiently. At 5-minute frequency, the closest
tradable proxy is execution efficiency: names whose prints stay near bar VWAP,
whose high-turnover bars do not pay large directional price concessions, and
whose low-cost execution persists should have better short-horizon
cost-adjusted behavior.

This group is intentionally different from the current price/volume families:

- `intraday_vwap_deviation_5m_w48` measures where close sits versus rolling
  VWAP; the new cost proxy normalizes the bar-level close-to-VWAP miss by the
  same bar's spread proxy.
- `intraday_amihud_5m` measures absolute return per turnover; the new large
  trade proxy measures directional close-to-VWAP cost on high-turnover bars.
- `order_flow_toxicity` decomposes signed volume pressure; the new features
  focus on whether that pressure is executed efficiently.

## Implementation

Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`

Factor group: `intraday_quality`

Default EWMA halflives: `6` and `12` five-minute bars.

Default large-trade window: `48` five-minute bars.

Emitted features:

| feature | interpretation |
|---|---|
| `intraday_price_efficiency_cost_5m` | `abs(close - bar_vwap) / (high - low)`. Lower is better. |
| `intraday_signed_trade_efficiency_5m` | `sign(bar_return) * (1 - clipped_price_efficiency_cost)`. Positive means efficient upside pressure; negative means efficient downside pressure. |
| `intraday_execution_quality_5m_hl{halflife}` | EWMA of clipped execution advantage. Higher means persistent low execution cost. |
| `intraday_large_trade_cost_5m_w{window}` | Turnover-weighted directional close-to-VWAP cost on high-turnover bars. Lower is better. |

Formula sketch:

```text
bar_return = close / previous_close - 1
bar_vwap = turnover / volume
spread_proxy = high - low

price_efficiency_cost = abs(close - bar_vwap) / spread_proxy
price_advantage = 1 - clip(price_efficiency_cost, 0, 1)
signed_trade_efficiency = sign(bar_return) * price_advantage
execution_quality = ewma(price_advantage, halflife)

large_trade_bar = turnover >= rolling_quantile(turnover, window, 0.8)
directional_cost = sign(bar_return) * (close - bar_vwap) / bar_vwap
large_trade_cost = rolling_sum(directional_cost * turnover * large_trade_bar)
                   / rolling_sum(turnover * large_trade_bar)
```

## Planned Validation

Dataset build:

```bash
conda run -n quant python examples/build_baseline_a_alpha_dataset.py \
  --catalog-path ../quant_dataset/canonical_store/catalog/quant_research.duckdb \
  --start 2023-01-03T09:35:00+08:00 \
  --end 2025-12-31T15:00:00+08:00 \
  --factor-groups intraday_quality \
  --intraday-quality-halflives 6 12 \
  --intraday-quality-large-trade-windows 12 48 \
  --intraday-quality-large-trade-quantile 0.8 \
  --label-name forward_return \
  --horizon-bars 6 48 \
  --entry-lag-bars 1 \
  --output-dir runs/factor_research/intraday_quality_2026_05_29/alpha_dataset \
  --workers 1 \
  --partition monthly \
  --padding-days 30 \
  --data-snapshot 2026-05-29
```

Single-factor evaluation:

```bash
conda run -n quant python examples/evaluate_alpha_dataset.py \
  --dataset-dir runs/factor_research/intraday_quality_2026_05_29/alpha_dataset \
  --output-dir runs/factor_research/intraday_quality_2026_05_29/factor_evaluation_6b \
  --label-column forward_return_6b \
  --horizon-label-columns forward_return_48b \
  --top-n 50 \
  --quantiles 5 \
  --workers 6 \
  --backend process
```

Admission:

```bash
conda run -n quant python examples/analyze_framework_v1_acceptance.py \
  --benchmark-summary runs/framework_v1_acceptance/standard/benchmark_summary.json \
  --factor-summary runs/factor_research/intraday_quality_2026_05_29/factor_evaluation_6b/summary.json \
  --by-timestamp runs/factor_research/intraday_quality_2026_05_29/factor_evaluation_6b/single_factor_by_timestamp.csv \
  --output-dir runs/factor_research/intraday_quality_2026_05_29/factor_admission_6b \
  --cost-bps 13.0
```

## Results

Dataset:
`runs/factor_research/intraday_quality_2026_05_29/alpha_dataset`

Rows after entry filters: `103,497,494`.

Admission:
`runs/factor_research/intraday_quality_2026_05_29/factor_admission_6b/factor_admission_report.json`

| factor | status | direction | rank IC | t-stat | hit rate | cost-adjusted spread | turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| `intraday_price_efficiency_cost_5m` | `candidate` | `long` | 0.032482 | 65.26 | 0.6662 | 0.000292 | 0.8389 |
| `intraday_large_trade_cost_5m_w48` | `watchlist` | `long` | 0.011802 | 23.34 | 0.5499 | -0.000207 | 0.1184 |
| `intraday_large_trade_cost_5m_w12` | `reject` | `long` | 0.014803 | 37.59 | 0.5992 | -0.000486 | 0.3764 |
| `intraday_execution_quality_5m_hl12` | `reject` | `long` | - | - | - | -0.000002 | 0.0012 |
| `intraday_execution_quality_5m_hl6` | `reject` | `long` | - | - | - | -0.000002 | 0.0012 |
| `intraday_signed_trade_efficiency_5m` | `reject` | `long` | - | - | - | -0.000008 | 0.0062 |

The admitted direction for `intraday_price_efficiency_cost_5m` is `long`, not
the originally expected inverted "lower cost is better" direction. Empirically,
larger close-to-bar-VWAP displacement normalized by the intrabar range appears
to behave more like profitable/informed intraday pressure than a pure execution
drag proxy.

Portfolio validation:
`runs/factor_research/intraday_quality_2026_05_29/candidate_policy_validation_6b_single/validation_summary.json`

Validation status: `pass`, with zero failed checks and zero warnings.

Primary `decorrelated` results:

| scenario | return | max drawdown | gross turnover |
|---|---:|---:|---:|
| full base | 17.60% | -22.31% | 122.79 |
| high cost | 12.06% | -23.19% | 122.64 |
| 2023 | 0.90% | -13.76% | 41.89 |
| 2024 | 3.15% | -20.53% | 42.55 |
| 2025 | 8.08% | -8.22% | 40.32 |

Because this is a single-feature candidate, `decorrelated`, `equal`, and
`ic_weighted` are equivalent in this validation run.

Compact-core incremental validation:
`runs/factor_research/intraday_quality_2026_05_29/compact_core_incremental_blend_standard/validation_summary.json`

Validation status: `warn`, with zero failed checks and five warnings. All
warnings are the same 2024 negative-return check that also applies to the
`w00` compact-core baseline.

Blend setup: existing compact-core `decorrelated` score plus the quality
candidate score, using weights `0`, `0.025`, `0.05`, `0.10`, and `0.15`.

| blend | full return | high-cost return | max drawdown | gross turnover | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `w00` | 42.92% | 36.39% | -32.34% | 111.60 | 14.97% | -5.96% | 31.85% |
| `w025` | 33.16% | 27.33% | -31.01% | 111.81 | 9.05% | -5.09% | 29.09% |
| `w05` | 28.41% | 23.03% | -30.54% | 113.34 | 8.91% | -4.22% | 26.57% |
| `w10` | 31.56% | 25.52% | -30.45% | 113.52 | 8.74% | -3.69% | 25.72% |
| `w15` | 31.91% | 26.07% | -31.29% | 113.38 | 9.09% | -2.15% | 23.62% |

Incremental read: the quality candidate reduces drawdown modestly and improves
the weak 2024 sleeve as weight rises, but it lowers full-sample and high-cost
returns versus the compact-core baseline. The trade-off is therefore not strong
enough for immediate core promotion.

Downside-penalty validation:
`runs/factor_research/intraday_quality_2026_05_29/compact_core_downside_penalty_standard/validation_summary.json`

Validation status: `warn`, with zero failed checks and five warnings. The
warnings again come from the 2024 negative-return check.

This mode keeps the compact-core score unchanged except for a penalty applied
to the bottom 20% of the quality candidate rank. It is less destructive than a
linear blend, but still does not clear the incremental hurdle.

| penalty | full return | high-cost return | max drawdown | gross turnover | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `w00` | 42.92% | 36.39% | -32.34% | 111.60 | 14.97% | -5.96% | 31.85% |
| `w10` | 40.19% | 34.09% | -31.91% | 111.67 | 14.44% | -6.94% | 33.79% |
| `w20` | 39.14% | 32.99% | -31.94% | 111.61 | 14.65% | -6.97% | 33.85% |
| `w30` | 39.60% | 33.28% | -31.94% | 111.64 | 14.89% | -7.08% | 33.96% |
| `w50` | 40.57% | 34.35% | -31.91% | 111.69 | 14.79% | -6.35% | 34.10% |

Entry-exclusion validation:
`runs/factor_research/intraday_quality_2026_05_29/compact_core_entry_exclusion_standard/validation_summary.json`

Validation status: `warn`, with zero failed checks and four warnings. The
warnings again come from the 2024 negative-return check.

This mode keeps the compact-core score and marks low-quality-tail names as
ineligible for new entries. The 5% and 10% exclusions had no realized effect
under the existing rank-buffer policy, which indicates those names were not
binding entry candidates. The 20% exclusion affected trades, but reduced
full-sample and high-cost returns.

| excluded tail | full return | high-cost return | max drawdown | gross turnover | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `w00` | 42.92% | 36.39% | -32.34% | 111.60 | 14.97% | -5.96% | 31.85% |
| `w05` | 42.92% | 36.39% | -32.34% | 111.60 | 14.97% | -5.96% | 31.85% |
| `w10` | 42.92% | 36.39% | -32.34% | 111.60 | 14.97% | -5.96% | 31.85% |
| `w20` | 40.42% | 34.29% | -31.90% | 111.72 | 14.57% | -6.63% | 32.89% |

Overlay-suite decision: simple blend, downside penalty, and entry exclusion all
fail to improve compact-core full-sample return. The factor remains useful as a
standalone monitored candidate and a research signal, but it should not be used
as a compact-core component or generic risk gate in the current policy.

## Registry Decision

Initial planned entries and final decisions:

| factor | expected direction | reason |
|---|---:|---|
| `intraday_price_efficiency_cost_5m` | `long` | promoted to monitored `candidate` after admission and portfolio validation passed |
| `intraday_execution_quality_5m_hl12` | `long` | `reject`; EWMA clipped advantage was nearly cross-sectionally constant and failed IC/stability/cost gates |
| `intraday_large_trade_cost_5m_w48` | `long` | `watchlist`; strong IC but failed the 13 bps cost-adjusted spread gate |

Decision: keep `intraday_price_efficiency_cost_5m` as a monitored standalone
candidate. Do not promote it to core after the compact-core overlay suite: the
standalone candidate passed admission and portfolio validation, but blend,
downside-penalty, and entry-exclusion overlays did not improve the existing
compact-core stack.
