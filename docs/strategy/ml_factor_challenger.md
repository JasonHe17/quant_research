# ML Factor Challenger

Status: experimental challenger pipeline. It does not replace the promoted
`decorrelated + partial_rebalance_daily` baseline.

The entrypoint is:

```bash
conda run -n quant python examples/run_ml_factor_challenger.py
```

## Standard No-Leak Workflow

New LightGBM factor-pool or primary-pool rerank experiments must use the
standard workflow wrapper unless the task is explicitly a low-level debugging
run:

```bash
conda run -n quant python examples/run_ml_challenger_standard_workflow.py \
  --dataset-dir runs/ml_factor_challenger/alpha_dataset_2023_2026_candidate27_48b_live_like_2026_05_30 \
  --admission-report runs/legacy_factor_revalidation/role_aware_alpha_rank_top5_standard_2026_05_29/shared_benchmark/factor_admission/factor_admission_report.json \
  --primary-score-dir runs/ml_factor_challenger/baselines/legacy_top2_2023_2026_live_like_scores_2026_05_29/scores/decorrelated \
  --baseline-backtest-dir runs/ml_factor_challenger/backtest_adaptive_inputs_2025_2026_2026_05_30/baseline \
  --output-dir runs/ml_factor_challenger/<run_id> \
  --include-features <feature_1> <feature_2> ... \
  --train-start 2023-01-01T00:00:00+08:00 \
  --history-train-end 2024-12-31T23:59:59+08:00 \
  --history-test-start 2025-01-01T00:00:00+08:00 \
  --history-test-end 2025-12-31T23:59:59+08:00 \
  --live-train-end 2025-12-31T23:59:59+08:00 \
  --live-start 2026-01-01T00:00:00+08:00 \
  --live-end 2026-05-15T15:00:00+08:00
```

Without `--execute`, the wrapper writes
`standard_workflow_plan.json` and exits. Review the plan first; then rerun with
`--execute` to run every stage in order. Logs are written under
`<output-dir>/logs/`.

The wrapper is the standard path because it enforces all required stages:

- train two explicit walk-forward folds, normally `2023-2024 -> 2025` and
  `2023-2025 -> current 2026`;
- use `run_ml_factor_challenger.py`, which blocks label/entry/exit-derived
  features by default and validates required columns in every dataset
  partition before training;
- build primary/ML blend scores from one ML-only primary-pool rerank score set;
- run fixed-candidate backtests on both the full walk-forward span and the live
  window with production-like CN A-share costs and constraints;
- build adaptive score switches from prior backtest performance only;
- backtest adaptive score streams with source-transition exits enabled, so a
  source switch can actually enter the portfolio instead of silently retaining
  the old score source through the exit buffer.

Do not add `--allow-label-derived-features` to standard runs. That switch is
reserved only for controlled leakage diagnostics and is intentionally not
exposed by the standard workflow wrapper. Do not promote a fixed blend from a
single live-like window unless the walk-forward span and adaptive selector are
also reported.

It trains purged walk-forward LightGBM regressors on selected alpha-rank
features, writes OOS prediction scores under `scores/lightgbm/score_*.parquet`,
and emits diagnostics for feature importance, SVD redundancy, high-correlation
pairs, and drop suggestions. Trading validation remains in
`examples/run_tree_score_backtest.py`.

Two score modes are supported:

- `standalone`: writes the model score for the whole prediction universe.
- `primary_pool_rerank`: keeps only names already inside a primary score pool,
  then uses LightGBM to re-rank that pool. This is the preferred first use when
  the manual baseline is stronger than the standalone model.

The recommended first production-grade challenger run for the current compact
core is:

```bash
conda run -n quant python examples/run_ml_factor_challenger.py \
  --dataset-dir runs/legacy_factor_revalidation/role_aware_alpha_rank_top5_standard_2026_05_29/shared_benchmark/alpha_dataset \
  --admission-report runs/legacy_factor_revalidation/role_aware_alpha_rank_top5_standard_2026_05_29/shared_benchmark/factor_admission/factor_admission_report.json \
  --output-dir runs/ml_factor_challenger/compact_core_lightgbm_2026_05_29 \
  --statuses candidate watchlist \
  --evaluation-roles alpha_rank \
  --include-features \
    intraday_sell_pressure_absorption_5m_w48 \
    intraday_amihud_5m \
    intraday_volatility_5m_w6 \
    intraday_efficiency_ratio_5m_w48 \
  --label-column forward_return_48b \
  --score-transform rank \
  --max-train-rows 2000000 \
  --max-valid-rows 500000 \
  --redundancy-sample-rows 1000000 \
  --num-boost-round 200 \
  --early-stopping-rounds 25 \
  --num-threads 8
```

The recommended secondary-layer run keeps the current absorption core as the
candidate generator and lets LightGBM re-rank the top 150 names only. Set
`--primary-blend-weight` above zero when the primary score should remain part of
the final ordering; for example, `0.7` means 70% primary rank and 30% ML rank
inside the retained pool.

```bash
conda run -n quant python examples/run_ml_factor_challenger.py \
  --dataset-dir runs/legacy_factor_revalidation/role_aware_alpha_rank_top5_standard_2026_05_29/shared_benchmark/alpha_dataset \
  --admission-report runs/legacy_factor_revalidation/role_aware_alpha_rank_top5_standard_2026_05_29/shared_benchmark/factor_admission/factor_admission_report.json \
  --output-dir runs/ml_factor_challenger/compact_core_lightgbm_primary_pool_rerank_2026_05_29 \
  --statuses candidate watchlist \
  --evaluation-roles alpha_rank \
  --include-features \
    intraday_sell_pressure_absorption_5m_w48 \
    intraday_amihud_5m \
    intraday_volatility_5m_w6 \
    intraday_efficiency_ratio_5m_w48 \
  --label-column forward_return_48b \
  --score-transform rank \
  --score-mode primary_pool_rerank \
  --primary-score-dir runs/candidate_factor_portfolios/legacy_top2_alpha_rank_rank_standard_2026_05_29/scores/decorrelated \
  --primary-pool-rank 150 \
  --primary-blend-weight 0.7 \
  --sample-weight-mode top_bottom \
  --sample-weight-top-quantile 0.2 \
  --sample-weight-multiplier 3.0 \
  --max-train-rows 2000000 \
  --max-valid-rows 500000 \
  --redundancy-sample-rows 1000000 \
  --num-boost-round 200 \
  --early-stopping-rounds 25 \
  --num-threads 8
```

Then backtest the OOS score:

```bash
conda run -n quant python examples/run_tree_score_backtest.py \
  --predictions-path 'runs/ml_factor_challenger/compact_core_lightgbm_2026_05_29/scores/lightgbm/*.parquet' \
  --start 2024-01-01T00:00:00+08:00 \
  --end 2025-12-31T23:59:59+08:00 \
  --top-n 50 \
  --trade-policy rank_buffer_drop \
  --rebalance-every-n-bars 48 \
  --policy-entry-rank 50 \
  --policy-exit-rank 150 \
  --policy-max-entries-per-rebalance 10 \
  --policy-max-exits-per-rebalance 10 \
  --policy-no-trade-weight-band 0.002 \
  --policy-partial-rebalance-rate 0.5 \
  --data-access-mode fast_parquet \
  --streaming-chunk month \
  --output-dir runs/ml_factor_challenger/compact_core_lightgbm_2026_05_29/backtests/lightgbm/partial_rebalance_daily
```

Promotion rule: compare only OOS backtests against the current
`intraday_sell_pressure_absorption_5m_w48` core. Full-window return, high-cost
stress, annual slices, drawdown, and turnover all need to clear the same
standard validation bar before this challenger can replace or augment the
manual-weight baseline.

For fast blend-weight sweeps after an ML-only primary-pool score already
exists, reuse the model scores instead of retraining:

```bash
conda run -n quant python examples/build_primary_pool_score_blends.py \
  --primary-score-dir runs/candidate_factor_portfolios/legacy_top2_alpha_rank_rank_standard_2026_05_29/scores/decorrelated \
  --ml-pool-score-dir runs/ml_factor_challenger/compact_core_lightgbm_primary_pool_rerank_2026_05_29/scores/lightgbm \
  --output-dir runs/ml_factor_challenger/primary_pool_blend_grid_2026_05_29 \
  --primary-blend-weights 0.5 0.6 0.7 0.8 0.9
```

To derive a stricter pool from an existing wider ML pool, add
`--primary-pool-rank`; for example, rank100 can be derived from rank150 scores
without retraining:

```bash
conda run -n quant python examples/build_primary_pool_score_blends.py \
  --primary-score-dir runs/candidate_factor_portfolios/legacy_top2_alpha_rank_rank_standard_2026_05_29/scores/decorrelated \
  --ml-pool-score-dir runs/ml_factor_challenger/compact_core_lightgbm_primary_pool_rerank_2026_05_29/scores/lightgbm \
  --output-dir runs/ml_factor_challenger/primary_pool_rank100_blend050_2026_05_29 \
  --primary-blend-weights 0.5 \
  --primary-pool-rank 100
```

Current 2024-2025 standard-constraint sweep:

| Method | Primary weight | Total return | Max drawdown |
| --- | ---: | ---: | ---: |
| Baseline absorption | - | 22.42% | -29.77% |
| Standalone LightGBM | - | 16.51% | -30.69% |
| Pure ML pool rerank | 0.0 | 19.45% | -27.66% |
| Primary-pool blend | 0.5 | 26.94% | -27.30% |
| Primary-pool blend | 0.6 | 22.82% | -28.29% |
| Primary-pool blend | 0.7 | 24.31% | -27.66% |
| Primary-pool blend | 0.8 | 24.54% | -28.12% |
| Primary-pool blend | 0.9 | 25.21% | -28.79% |

The current best challenger is `primary_w050`, stored under
`runs/ml_factor_challenger/primary_pool_blend_grid_2026_05_29`.  Treat it as a
candidate for stress validation, not as promoted production logic, until annual
slices, high-cost stress, and pool-rank sensitivity are reviewed.

Stability validation for `primary_w050`:

| Scenario | Method | Pool rank | Total return | Max drawdown |
| --- | --- | ---: | ---: | ---: |
| Full standard | Baseline absorption | - | 22.42% | -29.77% |
| Full standard | Primary-pool blend | 150 | 26.94% | -27.30% |
| 2024 standard | Baseline absorption | - | 3.68% | -29.77% |
| 2024 standard | Primary-pool blend | 150 | 4.99% | -27.30% |
| 2025 standard | Baseline absorption | - | 61.20% | -16.72% |
| 2025 standard | Primary-pool blend | 150 | 49.03% | -15.57% |
| Full doubled-cost | Baseline absorption | - | 18.91% | -30.32% |
| Full doubled-cost | Primary-pool blend | 150 | 23.18% | -27.90% |
| Pool sensitivity | Primary-pool blend | 100 | 21.49% | -28.60% |
| Pool sensitivity | Primary-pool blend | 150 | 26.94% | -27.30% |
| Pool sensitivity | Primary-pool blend | 200 | 23.84% | -27.74% |

Read: rank150 is the current sweet spot. Rank100 cuts off too much of the ML
repair space and falls below baseline; rank200 remains acceptable but dilutes
the edge. The challenger improves the weak 2024 slice and high-cost robustness,
but gives up part of the strong 2025 baseline upside, so promotion should still
include regime and capacity checks.

Capacity stress for `primary_w050`:

| Scenario | Method | Total return | Max drawdown | Unfilled / traded | Unfilled / desired |
| --- | --- | ---: | ---: | ---: | ---: |
| 5% bar participation | Baseline absorption | 23.80% | -29.12% | 1.34% | 62.99% |
| 5% bar participation | Primary-pool blend | 27.30% | -27.00% | 0.95% | 54.10% |
| 2% bar participation | Baseline absorption | 24.00% | -28.84% | 3.43% | 58.23% |
| 2% bar participation | Primary-pool blend | 27.71% | -26.77% | 2.73% | 55.67% |

The capacity check does not show hidden fragility. Under both 5% and 2%
same-bar participation assumptions, `primary_w050` keeps higher return, lower
drawdown, and lower unfilled/traded notional than the absorption baseline. The
2% unfilled/traded ratio remains below the 5% monitoring warning threshold.
Detailed rows are stored in
`runs/ml_factor_challenger/primary_w050_capacity_2026_05_29.csv`.

Monthly/state attribution for `primary_w050`:

```bash
conda run -n quant python examples/analyze_ml_challenger_attribution.py \
  --baseline-backtest-dir runs/ml_factor_challenger/baselines/absorption_core_2024_2025_partial_rebalance_daily_standard_constraints \
  --challenger-backtest-dir runs/ml_factor_challenger/primary_pool_blend_grid_2026_05_29/backtests/primary_w050/partial_rebalance_daily_standard_constraints \
  --baseline-score-dir runs/candidate_factor_portfolios/legacy_top2_alpha_rank_rank_standard_2026_05_29/scores/decorrelated \
  --challenger-score-dir runs/ml_factor_challenger/primary_pool_blend_grid_2026_05_29/scores/primary_w050 \
  --dataset-dir runs/legacy_factor_revalidation/role_aware_alpha_rank_top5_standard_2026_05_29/shared_benchmark/alpha_dataset \
  --output-dir runs/ml_factor_challenger/primary_w050_attribution_2026_05_29 \
  --start 2024-01-01T00:00:00+08:00 \
  --end 2025-12-31T23:59:59+08:00
```

Key attribution read:

| View | Result |
| --- | --- |
| Monthly hit rate | `primary_w050` beats baseline in 14 of 24 months |
| Best delta months | 2024-07, 2025-10, 2024-11, 2025-07, 2024-02 |
| Worst delta months | 2024-10, 2025-12, 2024-08, 2025-02, 2025-08 |
| Score replacement signal | Monthly return delta is more correlated with replacement-label delta (`0.30`) than broad market-state aggregates |
| State signal | Prior-month downside state is the most promising observable switch proxy in this small sample |

The top sample-in rule from the exploratory switch table is: use ML blend after
months whose prior `market_state_downside_mean_5m_w48_mean` is above the lower
third of observations. In-sample compound return rises to `31.20%`, versus
`26.94%` for always-on `primary_w050` and `22.42%` for baseline. This is not
yet production evidence: it has only 23 lagged monthly decisions and must be
converted to a daily observable gate before validation. Attribution outputs are
stored in `runs/ml_factor_challenger/primary_w050_attribution_2026_05_29`.

Daily state switch validation:

```bash
conda run -n quant python examples/build_state_conditioned_score_switch.py \
  --baseline-score-dir runs/candidate_factor_portfolios/legacy_top2_alpha_rank_rank_standard_2026_05_29/scores/decorrelated \
  --challenger-score-dir runs/ml_factor_challenger/primary_pool_blend_grid_2026_05_29/scores/primary_w050 \
  --dataset-dir runs/legacy_factor_revalidation/role_aware_alpha_rank_top5_standard_2026_05_29/shared_benchmark/alpha_dataset \
  --output-dir runs/ml_factor_challenger/primary_w050_downside_state_switch_2026_05_29 \
  --method-name downside_q33_lag1_switch \
  --state-column market_state_downside_mean_5m_w48 \
  --activation-quantile 0.33 \
  --min-history-days 20 \
  --active-when gte \
  --start 2024-01-01T00:00:00+08:00 \
  --end 2025-12-31T23:59:59+08:00
```

The switch uses only lagged daily state: each trading day is gated by the prior
day's state value against an expanding quantile threshold. The generated score
files also include `signal_source`, so `run_tree_score_backtest.py` can test a
hard migration mode with `--policy-reset-on-source-change`.

Current daily switch read:

| Scenario | Total return | Max drawdown | Gross turnover | Notes |
| --- | ---: | ---: | ---: | --- |
| No source reset | 22.42% | -29.77% | 75.95 | Path-dependent; legacy holdings remain inside the broad exit buffer, so it reproduces baseline |
| Source reset, standard cost | 53.72% | -30.86% | 217.35 | 126 source switches; strong but high-turnover diagnostic |
| Source reset, doubled cost | 34.63% | -32.72% | 217.23 | Survives higher cost but drawdown worsens |
| Source reset, 5% capacity | 70.28% | -25.30% | 193.80 | Capacity cap changes execution path; treat as stress evidence, not promotion proof |
| Source reset, 2024 | -0.38% | -30.86% | 109.18 | Weak year remains weak |
| Source reset, 2025 | 55.69% | -13.08% | 108.05 | Most of the improvement is concentrated in 2025 |

Soft source-transition validation:

`run_tree_score_backtest.py` supports
`--policy-force-source-transition-exits` for state-switch score streams. This
keeps policy state intact, but records the source that introduced each held
name. When the current signal source changes, held names from another source
are removed from the current forecast so the rank-buffer policy treats them as
exit candidates. The transition still respects `--policy-max-exits-per-rebalance`.
Use `--policy-source-transition-exit-rate 1.0` so selected transition exits do
not remain as tiny residual policy positions and block new entries.
`--policy-source-transition-turnover-cap` can additionally cap gross turnover
only on source-transition decisions; this is separate from the ordinary
rebalance turnover cap.

```bash
conda run -n quant python examples/run_tree_score_backtest.py \
  --predictions-path 'runs/ml_factor_challenger/primary_w050_downside_state_switch_2026_05_29/scores/downside_q33_lag1_switch/score_*.parquet' \
  --start 2024-01-01T00:00:00+08:00 \
  --end 2025-12-31T23:59:59+08:00 \
  --top-n 50 \
  --trade-policy rank_buffer_drop \
  --rebalance-every-n-bars 48 \
  --policy-entry-rank 50 \
  --policy-exit-rank 150 \
  --policy-max-entries-per-rebalance 10 \
  --policy-max-exits-per-rebalance 10 \
  --policy-no-trade-weight-band 0.002 \
  --policy-partial-rebalance-rate 0.5 \
  --policy-force-source-transition-exits \
  --policy-source-transition-exit-rate 1.0 \
  --commission-bps 3.0 \
  --slippage-bps 1.0 \
  --sell-stamp-tax-bps 5.0 \
  --min-commission 5.0 \
  --min-trade-weight 0.0005 \
  --exclude-st \
  --limit-up-bps 980.0 \
  --limit-down-bps 980.0 \
  --data-access-mode fast_parquet \
  --streaming-chunk month \
  --streaming-chunk-padding-days 10 \
  --output-dir runs/ml_factor_challenger/primary_w050_downside_state_switch_2026_05_29/backtests/downside_q33_lag1_switch/full_standard_constraints_source_soft_exit_v2
```

Updated read:

| Scenario | Total return | Max drawdown | Gross turnover | Notes |
| --- | ---: | ---: | ---: | --- |
| No source reset | 22.42% | -29.77% | 75.95 | Path-dependent; legacy holdings remain inside the broad exit buffer |
| Hard source reset | 53.72% | -30.86% | 217.35 | Too aggressive; bypasses normal migration |
| Soft source transition | 67.44% | -28.72% | 132.12 | Preferred validation path so far |
| Soft transition, doubled cost | 54.86% | -30.03% | 132.20 | Cost stress remains strong |
| Soft transition, 5% capacity | 76.41% | -26.66% | 128.61 | Capacity stress remains strong; same-bar volume assumption applies |
| Soft transition, 2024 | 7.25% | -28.72% | 68.52 | Fixes hard-reset 2024 weakness |
| Soft transition, 2025 | 56.67% | -18.08% | 65.80 | Keeps most of the 2025 upside |

Parameter sweep read:

| Variant | Active days | Total return | Max drawdown | Gross turnover | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| q33, exit rate 1.0 | 228 | 67.44% | -28.72% | 132.12 | Strong, but not the best gate |
| q50, exit rate 1.0 | 142 | 68.09% | -27.02% | 120.66 | Current best standard setting |
| q67, exit rate 1.0 | 87 | 61.01% | -30.08% | 108.03 | Too conservative |
| q33, exit rate 0.75 | 228 | 21.59% | -31.22% | 106.47 | Too slow; old-source residuals block entries |
| q33, exit rate 0.5 | 228 | 21.86% | -31.69% | 82.11 | Too slow |
| q33, source turnover cap 0.1 | 228 | 23.57% | -30.01% | 58.65 | Too restrictive |
| q33, source turnover cap 0.2 | 228 | 18.61% | -31.35% | 81.53 | Too restrictive |
| q33, source turnover cap 0.3 | 228 | 21.70% | -31.26% | 104.51 | Too restrictive |
| q33, source turnover cap 0.5 | 228 | 67.44% | -28.72% | 132.12 | Does not bind versus uncapped |

q50 stress validation:

| Scenario | Total return | Max drawdown | Gross turnover |
| --- | ---: | ---: | ---: |
| Full standard | 68.09% | -27.02% | 120.66 |
| Full doubled-cost | 56.30% | -28.40% | 120.80 |
| 5% bar participation | 75.94% | -26.66% | 117.65 |
| 2024 standard | 9.08% | -27.02% | 68.99 |
| 2025 standard | 57.17% | -16.88% | 55.71 |

Frozen-configuration validation report:

```bash
conda run -n quant python examples/build_state_switch_validation_report.py \
  --baseline-backtest-dir runs/ml_factor_challenger/baselines/absorption_core_2024_2025_partial_rebalance_daily_standard_constraints \
  --challenger-backtest-dir runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/backtests/downside_q50_lag1_switch/full_standard_constraints_source_soft_exit_v2 \
  --schedule-path runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/state_switch_schedule.csv \
  --output-dir runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/validation_report \
  --scenario-summary standard=runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/backtests/downside_q50_lag1_switch/full_standard_constraints_source_soft_exit_v2/summary.json \
  --scenario-summary high_cost=runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/backtests/downside_q50_lag1_switch/full_high_cost_source_soft_exit_v2/summary.json \
  --scenario-summary capacity_5pct=runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/backtests/downside_q50_lag1_switch/capacity_5pct_source_soft_exit_v2/summary.json \
  --scenario-summary year_2024=runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/backtests/downside_q50_lag1_switch/year_2024_source_soft_exit_v2/summary.json \
  --scenario-summary year_2025=runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/backtests/downside_q50_lag1_switch/year_2025_source_soft_exit_v2/summary.json
```

Report outputs are under
`runs/ml_factor_challenger/primary_w050_downside_state_switch_q50_2026_05_29/validation_report`.
The frozen q50 challenger beats the absorption baseline in 19 of 24 months and
6 of 8 quarters. The weak relative periods are mainly 2024-05, 2024-10,
2024-12, and 2025-02. The trade profile shows the expected cost of the stronger
result: gross notional rises from `75.46M` to `138.87M`, average daily gross
notional rises from `169k` to `311k`, and the traded instrument count expands
from `60` to `549`. This remains acceptable under the 5% capacity stress, but
it should be monitored as a deployment constraint.

Earlier-year independent validation is not yet complete because the current
primary-pool ML score stream only covers 2024-2025, while the baseline and
dataset extend back to 2023. A true 2023 out-of-window test requires rerunning
the ML challenger score generation for 2023 with a valid purged training setup,
then rebuilding the same q50 state-switch stream without changing the frozen
parameters.

Conclusion: the daily observable gate plus soft source transition is now a
serious challenger to both always-on `primary_w050` and the absorption baseline.
The current preferred configuration is q50 downside-state activation,
`--policy-force-source-transition-exits`, and
`--policy-source-transition-exit-rate 1.0`, with no additional source-transition
turnover cap unless future capacity work requires it. The q50 gate improves
return, drawdown, and turnover versus q33, while retaining strong high-cost,
capacity, and annual-slice behavior.
