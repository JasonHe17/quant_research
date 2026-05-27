# Factor Research Batch - 2026-05-27 Microstructure Exhaustion Alert

This batch adds an imminent-exhaustion warning signal instead of another
realized sell-pressure confirmation factor.

## Hypothesis

Liquidity providers have finite risk capacity. After they have absorbed a
sequence of sell orders, the next small sell program can move price sharply if
the same turnover starts causing larger losses, intraday rebounds stop repairing
the damage, and market-wide limit pressure rises. This should behave as a
risk/avoidance signal for long-only selection.

The mechanism is intentionally different from the existing sell-pressure
family:

- `sell_pressure_absorption` measures downside turnover per unit loss over a
  completed window.
- `sell_pressure_exhaustion` requires recovery and downside-turnover decay that
  have already happened.
- `microstructure_exhaustion_alert` looks for capacity strain before confirmed
  relief: recent impact is worsening versus the previous half-window, rebound
  quality is decaying, and limit-pressure withdrawal is rising.

The data does not expose order-book cancellation events directly, so this first
implementation uses a point-in-time proxy from cross-sectional
`limit_down_open - limit_up_open` pressure and its shock versus the lagged
rolling state.

## Implementation

Builder: `quant_research.datasets.intraday_features.build_intraday_feature_matrix`

Factor group: `microstructure_exhaustion_alert`

Default window: `48` five-minute bars.

Emitted features:

| feature | interpretation |
|---|---|
| `intraday_microstructure_impact_strain_5m_w48` | Absorption load times recent/previous downside-turnover speed and positive downside-impact acceleration. |
| `intraday_microstructure_exhaustion_alert_5m_w48` | Impact strain amplified by rebound decay and rising limit-pressure withdrawal proxy. |

For each window, the calculation splits the window into previous and recent
halves:

```text
absorption_load = log1p(rolling downside turnover / rolling downside return)
absorption_speed = recent downside turnover / previous downside turnover
impact_acceleration = positive_part((recent loss/turnover - previous loss/turnover) / sum)
recovery_decay = positive_part((previous recovery - recent recovery) / sum_abs)
limit_withdrawal_proxy = current limit pressure + positive shock over lagged state

impact_strain = absorption_load * log1p(absorption_speed) * impact_acceleration
exhaustion_alert = impact_strain * (1 + recovery_decay) * (1 + limit_withdrawal_proxy)
```

Expected direction for admission review: `invert`.

## Validation Plan

Build a new-factor-only dataset:

```bash
conda run -n quant python examples/build_baseline_a_alpha_dataset.py \
  --start 2023-01-01T00:00:00+08:00 \
  --end 2025-12-31T23:59:59+08:00 \
  --factor-groups microstructure_exhaustion_alert \
  --output-dir runs/factor_research/microstructure_exhaustion_2026_05_27/alpha_dataset \
  --workers 1
```

Then run single-factor evaluation and admission against `forward_return`.
Promotion should require incremental portfolio evidence against the current
portfolio-native frontier, not just standalone IC.

## Research Memory Readout

The registry memory check marked both features as `blocked` because they share
the liquidity family, 48-bar horizon, and inputs with prior microstructure
recovery/acceleration, sell-pressure quality, and limit-pressure work.

This batch proceeded only as a materially different mechanism test: it targets
pre-confirmation impact strain and rebound decay, while the prior rejected
microstructure recovery variants measured realized recovery speed or recovery
acceleration after the move. The test result below now supersedes that planned
retry path for this exact transform.

Evidence:

- `runs/factor_research_memory/intraday_microstructure_impact_strain_5m_w48/factor_research_memory_check.json`
- `runs/factor_research_memory/intraday_microstructure_exhaustion_alert_5m_w48/factor_research_memory_check.json`

## Standard Results

Dataset build completed for 2023-01-03 through 2025-12-31:

- Dataset: `runs/factor_research/microstructure_exhaustion_2026_05_27/alpha_dataset`
- Rows: `102771651`
- Evaluation: `runs/factor_research/microstructure_exhaustion_2026_05_27/factor_evaluation/summary.json`
- Admission: `runs/factor_research/microstructure_exhaustion_2026_05_27/factor_admission/factor_admission_report.json`

Admission rejected both features:

| feature | status | direction | rank IC | t-stat | hit rate | cost-adjusted spread | failed checks |
|---|---:|---:|---:|---:|---:|---:|---|
| `intraday_microstructure_impact_strain_5m_w48` | `reject` | `long` | 0.000993 | 2.94 | 50.90% | -0.001836 | `abs_rank_ic_mean`, `directional_ic_hit_rate`, `cost_adjusted_spread` |
| `intraday_microstructure_exhaustion_alert_5m_w48` | `reject` | `long` | 0.000656 | 1.88 | 50.64% | -0.001963 | `abs_rank_ic_mean`, `abs_rank_ic_t_stat`, `directional_ic_hit_rate`, `cost_adjusted_spread` |

The admission-selected direction was `long`, not the intended `invert` risk
direction, and the top-minus-bottom label was negative for both features. This
means the standalone rank transform is not a usable imminent-exhaustion warning
signal under the current 5m OHLCV plus limit-pressure proxy.

No portfolio-level validation was run. The standard workflow allows portfolio
validation only after candidate admission and registry readiness; this batch had
zero admitted candidates.

## Decision

Mark both new features as `reject` in the registry. Do not continue the current
standalone formula or force a portfolio backtest. A legitimate retry needs
materially stronger microstructure inputs, preferably direct order-book depth or
cancellation data, or a state-conditioned gate that is evaluated as a risk
control rather than as a standalone rank factor.
