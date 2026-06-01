# Factor Research Batch - 2026-06-01 Regime-Conditioned Factor Weights

This batch implements a framework-level challenger for regime-conditioned
factor weights. It does not add a new raw factor. The goal is to test whether
the fixed full-sample decorrelation matrix should be replaced by lagged rolling
correlation memories that change across observable market regimes.

## Hypothesis

The 2024 weak-market failures are partly a factor-weighting problem. A fixed
2023-2025 decorrelation matrix assumes factor relationships are stable, but the
portfolio attribution indicates that stress months were dominated by volatility,
Amihud, and sell-pressure absorption legs. In high-volatility or weak-breadth
states, the useful diversification structure can differ from calm states.

The implemented challenger therefore:

1. estimates per-timestamp cross-sectional factor correlations;
2. builds rolling lagged decorrelation weights over a configurable window,
   defaulting to `48` score bars;
3. classifies lagged observable state using cross-sectional volatility and
   breadth rolling quantiles;
4. when enough same-regime history exists, uses only that regime's correlation
   memory to compute the current `decorrelated` weights.

## Implementation

Core API:

- `RegimeConditionedFactorWeightConfig`
- `build_regime_conditioned_factor_weight_schedule`

Runner integration:

- `examples/run_candidate_factor_portfolios.py`
- new mode: `--factor-weight-regime-mode state_conditioned_decorrelated`
- output: `factor_weights/decorrelated_factor_weight_schedule.csv`

The score builder now accepts a timestamp/feature `weight` schedule. Existing
static weights remain the fallback, and existing factor-health
`weight_scale` schedules still multiply the selected base weight.

## Default State Definition

The current defaults are intentionally observable and lagged:

- volatility state column: `intraday_bar_return_5m`
- volatility aggregation: cross-sectional `std`
- breadth column: `market_state_breadth_5m`
- high-vol threshold: rolling `0.75` quantile
- weak-breadth threshold: rolling `0.25` quantile
- selector: `volatility_or_weak_breadth`
- state lag: `1` bar
- correlation lag: `1` bar

## Validation Command

Use this as the first full-sample challenger against the repaired no-overlay
alpha-rank control:

```bash
conda run -n quant python examples/run_candidate_factor_portfolios.py \
  --dataset-dir runs/framework_v1_acceptance/standard/alpha_dataset \
  --admission-report runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json \
  --registry configs/factors/factor_registry.json \
  --registry-statuses candidate promoted \
  --evaluation-roles alpha_rank \
  --methods decorrelated \
  --weight-evidence-mode equal \
  --score-transform rank \
  --factor-weight-regime-mode state_conditioned_decorrelated \
  --factor-weight-regime-lookback-windows 48 \
  --factor-weight-regime-min-periods 24 \
  --factor-weight-regime-state-lookback-windows 240 \
  --factor-weight-regime-state-min-periods 48 \
  --factor-weight-regime-selector volatility_or_weak_breadth \
  --factor-max-contribution-share 0.25 \
  --score-diagnostics-top-n 50 \
  --run-backtests \
  --backtest-policy-set comparison \
  --backtest-policies partial_rebalance_daily cost_aware_optimizer_daily \
  --start 2023-01-01T09:35:00+08:00 \
  --end 2025-12-31T15:00:00+08:00 \
  --top-n 50 \
  --output-dir runs/candidate_factor_portfolios/regime_conditioned_factor_weights_2026_06_01_standard
```

## Decision Criteria

Compare against both current alpha-rank benchmark layers:

- no-overlay control:
  `fixed_framework_alpha_rank_v66_target_gate_deep25_overnight_only_contrib_cap25_2026_05_31_standard`
- state-aware frontier:
  `fixed_framework_alpha_rank_repaired_state_overlay_budget_min90_l120_2026_06_01_standard`

Required readout:

- full-base and high-cost total return;
- yearly 2023/2024/2025 slices;
- January 2024 and June 2024 month slices;
- max drawdown and mean turnover;
- dynamic factor-weight attribution by regime.

Promotion requires improved 2024 stress-month losses without giving back the
full-window and high-cost economics that the state-aware frontier already
protects.
