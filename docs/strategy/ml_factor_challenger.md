# ML Factor Challenger

Status: experimental challenger pipeline. It does not replace the promoted
`decorrelated + partial_rebalance_daily` baseline.

The entrypoint is:

```bash
conda run -n quant python examples/run_ml_factor_challenger.py
```

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
