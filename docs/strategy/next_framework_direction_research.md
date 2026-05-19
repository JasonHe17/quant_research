# Next Framework Direction Research

Status note, 2026-05-19: this is a strategy-framework research log, not the
canonical factor-development workflow. Some sections describe historical
"promoted" candidates inside optimizer and turnover-budget experiments. The
current compatibility and factor-promotion workflow is documented in
`docs/validation/factor_development_standard.md`.

This note decides the next framework task after the first policy-level
gross-exposure gate experiments.

## Current Evidence

The current production-oriented candidate is
`decorrelated/partial_rebalance_daily`. It is operationally tractable and
cost-resilient, but it is not production-promotable because the 2024 validation
slice loses money across all score-combination methods.

The 2024 failure is not one homogeneous problem:

- 2024-01 and 2024-06: the composite score still has positive relative ranking
  power, but the selected top basket has negative absolute forward returns.
  A market or basket-health exposure gate helps these months.
- 2024-02: the composite score/factor legs invert. The H1 2024 budget gate does
  not fix this, because reducing gross exposure after composite-score health
  deterioration is too late and too blunt.
- 2024-04: the budget gate creates a clear false positive and opportunity cost.
  More gate tuning risks overfitting this small sample.

## External Research And Engineering Read

Mature engineering systems separate prediction, portfolio construction, and
execution. Qlib's portfolio strategy layer consumes forecast scores and supports
custom strategies; its Topk-Drop implementation is a simple rank-driven
portfolio strategy, not a complete risk model. Its own documentation also notes
that rank-only strategies may ignore score scale, while optimization-based
strategies need meaningful score calibration.

Optimization-based portfolio construction is the right long-term architecture.
The Boyd/Busseti/Diamond/Kahn/Koh/Nystrup/Speth framework trades off expected
return, risk, transaction cost, and holding cost, and its multi-period version
executes the first slice of a re-planned trajectory. Cvxportfolio implements
that style with single-period and multi-period optimization policies,
constraints, and transaction-cost models using volume, volatility, and spread
terms. However, those methods assume the alpha forecasts and risk inputs are
usable; they exploit predictions, but do not solve bad or inverted predictions.

The factor-timing literature is mixed but useful here. Broad contrarian factor
timing is difficult and should not be treated as an easy alpha source. At the
same time, factor momentum research shows that recent factor performance can
contain persistence, including evidence in the Chinese stock market. For our
case, the relevant use is conservative factor-leg health and shrinkage, not
aggressive all-in/all-out factor timing.

Market-state exposure control is also legitimate, especially for long-only
books. Long-only A-share trading faces T+1 sellability, board-lot, price-limit,
and liquidity constraints, so gross exposure changes must be slow and
auditable. Trend-following evidence supports defensive state awareness, but the
current H1 2024 replay shows that a basket-health gate alone can add false
positives.

## Decision

The next implementation should prioritize factor-leg health and shrinkage before
more regime-gate tuning or a full optimizer.

Reasons:

- It directly addresses the 2024-02 inversion, which policy-level gross exposure
  cannot explain or repair.
- It reduces concentration risk already observed in the score construction:
  short-horizon volatility and Amihud legs dominate top-score contribution in
  the main failure months.
- It is compatible with the existing score-partition pipeline and can be tested
  with the current RankBufferDropPolicy without first building a full optimizer.
- It is a prerequisite for optimizer work. A cost-aware optimizer will allocate
  more cleanly if the input score is already leg-capped, health-shrunk, and
  auditable.
- It is less likely than more gate tuning to overfit a handful of 2024 months,
  because the acceptance target is cross-sectional signal quality and exposure
  concentration before trading.

Do not promote this as "factor timing alpha" by default. The first version
should be a risk-control transform:

- cap any single factor's contribution to composite score;
- shrink factors after lagged rolling IC or top-minus-bottom deterioration;
- keep a minimum residual weight unless a hard health rule is breached;
- record per-factor raw weight, health score, effective weight, and shrink
  reason for every partition or decision date.

## Comparison Experiment

Run the comparison at score-construction level first, then policy replay.

Candidate variants:

1. Baseline: current `decorrelated` composite.
2. Contribution cap only: cap per-factor absolute contribution to top-score
   names before final score ranking.
3. Lagged leg-health shrinkage: shrink each factor by lagged rolling IC and
   top-minus-bottom health.
4. Cap + leg-health shrinkage.
5. Optional control: market/basket-health gate from the existing budget
   deadband schedule.
6. Optional combined: cap + leg-health shrinkage + existing gate.

Primary diagnostics before backtest:

- monthly composite rank IC;
- monthly top-minus-bottom label;
- top-score forward return;
- factor contribution concentration: largest factor share and top-two share;
- factor-leg inversion count;
- score rank autocorrelation and top-N name turnover.

Backtest acceptance diagnostics:

- total return, max drawdown, monthly worst return;
- gross turnover, trade count, transaction cost;
- average target gross exposure if a gate is used;
- month-level attribution for 2024-01, 2024-02, 2024-04, 2024-06, and 2024-12.

Minimum acceptance bar for choosing the next implementation:

- improve 2024-02 versus the baseline without materially worsening 2024-01 and
  2024-06;
- reduce top-score contribution concentration;
- no higher full-window turnover than `partial_rebalance_daily` baseline by
  more than 10%;
- preserve 2023 and 2025 positive full-year behavior under base costs;
- remain better than or equal to baseline under high-cost validation after
  transaction costs.

## Implementation Order

- [x] Add a score-construction transform that emits factor contribution columns
  or sidecar diagnostics before aggregation.
- [x] Add contribution-cap mode to `run_candidate_factor_portfolios.py`.
- [x] Add lagged rolling factor-leg health schedules using only matured labels.
- [x] Add shrinkage mode that combines static decorrelated weights with
  lagged health scores.
- [x] Add factor-leg health summaries to validation outputs.
- [x] Run the score-only diagnostics over selected 2024 failure windows.
- [x] Run policy replays for baseline, static cap + health shrinkage, and row
  contribution cap + health shrinkage.
- [ ] Run broader split-variant replays for cap-only, health-only,
  contribution-cap-only, and gate combinations if we continue this branch.
- [x] Promote only if the acceptance bar above is met; otherwise keep the
  current policy baseline and move to optimizer research with explicit input
  limitations documented.

## Initial Implementation Notes

The first implementation adds explicit, opt-in score-construction controls:

- `--factor-max-weight`: caps static normalized factor weights before composite
  score construction.
- `--factor-health-mode shrink`: builds a lagged rolling factor-health schedule
  from matured labels and scales per-factor weights by timestamp.
- `--factor-max-contribution-share`: caps row-level factor contribution share
  after static weights and health shrinkage.
- `--score-diagnostics-top-n`: writes top-score factor-contribution diagnostics
  per partition.

Validation now also writes factor-health and factor-contribution summary CSVs
when those artifacts exist.

Initial smoke checks:

| Window | Variant | Return | Max drawdown | Gross turnover | Trade count | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Q1 2023 | Baseline decorrelated | 4.76% | -5.82% | 11.89 | 2,274 | Existing partial-rebalance baseline |
| Q1 2023 | Static cap + health shrink | 5.28% | -5.46% | 11.90 | 2,273 | Improves Q1 without turnover increase |
| H1 2024 | Baseline decorrelated | -24.06% | -32.13% | 22.08 | 4,621 | Failure window |
| H1 2024 | Static cap + health shrink | -23.72% | -32.46% | 22.19 | 4,608 | Improves 2024-02 but worsens drawdown |
| H1 2024 | Static cap + row contribution cap + health shrink | -23.44% | -30.67% | 22.22 | 4,639 | Better drawdown and failure-month behavior, but higher cost and 2024-05 drag |

The row-level contribution cap reduced H1 2024 average largest top-score
factor-contribution share to roughly 47% in every month, with a hard observed
maximum of 50%. It improved 2024-01, 2024-02, 2024-04, and 2024-06 versus the
baseline, but materially worsened 2024-05. This is promising enough to enter
selected-year validation, but not enough to promote as a default.

Selected-year validation rejected promotion:

| Scenario | Baseline return | Cap+health+contribution-cap return | Baseline max drawdown | New max drawdown | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| full_base | 34.97% | 25.39% | -32.79% | -33.50% | Worse full-window return and drawdown |
| year_2023_base | 8.39% | 5.54% | -13.88% | -14.46% | Loses too much good-regime return |
| year_2024_base | -11.55% | -12.80% | -32.13% | -31.97% | H1 improvement is offset by H2 drag |
| year_2025_base | 26.39% | 24.21% | -13.49% | -15.06% | Still positive, but weaker |
| full_high_cost | 28.04% | 18.66% | -33.77% | -34.57% | High-cost resilience worsens |

The implementation is useful as a research and diagnostic tool, but it should
not become the default score construction. It confirms that factor-leg controls
can address parts of the failure window, especially 2024-02 inversion and H1
drawdown, but static contribution clipping is too blunt and creates opportunity
cost in later 2024 months.

Next decision point:

- Continue within the signal layer only if we are willing to research smoother
  factor-leg controls, such as shrink-only schedules, factor-family caps, or
  regime-conditioned caps instead of hard row-level clipping.
- Otherwise move to the cost-aware optimizer layer with the current baseline
  score, while keeping factor-health diagnostics as risk inputs rather than
  score modifiers.

## Cost-Aware Optimizer MVP

Implemented a `CostAwareOptimizerPolicy` behind the same strategy contracts as
`RankBufferDropPolicy`. The MVP is deliberately conservative: it is not a
full convex multi-period optimizer yet, but it moves the framework away from
rank-only trade triggers and toward explicit forecast, cost, risk, and state
inputs.

Implemented behavior:

- Converts score or optional expected-edge columns into expected edge bps.
- Subtracts estimated trading cost and optional risk penalties before a name is
  eligible.
- Preserves existing holdings before admitting new entries when the existing
  name remains in the candidate set and has positive net edge.
- Supports max entries/exits, partial rebalance, no-trade band, T+1 sellability,
  and gross exposure scale through the standard policy contract.
- Supports an optimizer-specific turnover-budget allocator through
  `max_gross_turnover_per_rebalance`: empty or under-invested portfolios may use
  the exposure gap to build toward the target gross, while subsequent
  replacement trades compete for a fixed turnover budget.
- Applies a final incremental gross-budget clamp so T+1-blocked sells cannot be
  offset by hidden extra buys.
- Emits the same portfolio intent, trade decision, order intent, policy state,
  and diagnostics artifacts as the rank-buffer policy.

Initial comparable smoke checks use the same Q1 2023 and H1 2024 baseline score
files and base-cost execution assumptions:

| Window | Policy | Return | Max drawdown | Gross turnover | Trade count | Avg target gross | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q1 2023 | Rank buffer partial rebalance | 4.76% | -5.82% | 11.89 | 2,274 | 0.96 | Existing baseline |
| Q1 2023 | Cost-aware optimizer, equal full rebalance | 8.17% | -5.65% | 59.68 | 1,069 | 1.00 | Strong return, but spends too much turnover |
| Q1 2023 | Cost-aware optimizer, turnover budget 0.25 | 6.57% | -5.85% | 14.33 | 1,014 | 1.00 | Better return with near-baseline turnover |
| H1 2024 | Rank buffer partial rebalance | -24.06% | -32.13% | 22.08 | 4,621 | 0.96 | Failure window |
| H1 2024 | Cost-aware optimizer, equal full rebalance | -23.88% | -26.83% | 114.25 | 2,281 | 1.00 | Better drawdown but too much turnover |
| H1 2024 | Cost-aware optimizer, turnover budget 0.25 | -19.74% | -31.66% | 28.45 | 2,373 | 1.00 | Return and trade count improve; drawdown still needs regime gate |
| Q1 2023 | Optimizer 0.25 + budget-deadband gate | 5.67% | -3.32% | 26.89 | 1,835 | 0.55 | Keeps Q1 alpha while materially reducing drawdown |
| H1 2024 | Optimizer 0.25 + budget-deadband gate | -9.57% | -11.98% | 43.41 | 4,804 | 0.40 | Solves failure-window loss and drawdown, but trades too much |
| Q1 2023 | Optimizer 0.15 + budget-deadband gate | 4.17% | -3.24% | 21.22 | 1,397 | 0.55 | Conservative turnover budget; gives up Q1 upside |
| H1 2024 | Optimizer 0.15 + budget-deadband gate | -7.94% | -9.21% | 33.93 | 3,616 | 0.39 | Best H1 loss/drawdown tradeoff so far |

Promotion read:

- Do not promote as default yet. The optimizer plus budget-deadband gate is the
  first variant that materially improves both Q1 2023 and H1 2024 drawdown. The
  0.25 turnover-budget variant preserves more upside, while the 0.15 variant is
  the best H1 failure-window risk control so far.
- Slower re-risk schedules and wider gate deadbands did not reduce trading
  enough and weakened Q1/H1 tradeoffs in smoke tests. The next validation step
  should compare the 0.15 and 0.25 optimizer budgets across the standard
  multi-year suite before adding more knobs.
- Next research step is to pass calibrated expected-edge and risk-penalty
  columns into the optimizer instead of deriving edge from raw score scale.

Standard multi-year validation, using a full 2023-2025 budget-deadband gate
schedule and parallel yearly backtests where memory allowed:

| Scenario | Policy | Return | Max drawdown | Gross turnover | Trade count | Avg target gross | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_base | Baseline rank buffer | 34.97% | -32.79% | 128.79 | 29,052 | n/a | High absolute return, unacceptable drawdown |
| full_base | Optimizer 0.15 + gate | 2.94% | -18.08% | 222.93 | 20,926 | 0.43 | Strong drawdown reduction, but return is mostly consumed |
| full_base | Optimizer 0.25 + gate | -2.53% | -19.43% | 283.88 | 27,890 | 0.44 | Worse than 0.15 |
| full_high_cost | Baseline rank buffer | 28.04% | -33.77% | 128.31 | 28,936 | n/a | Still high return, drawdown remains too high |
| full_high_cost | Optimizer 0.15 + gate | -9.51% | -23.12% | 222.22 | 20,873 | 0.43 | Fails cost stress |
| full_high_cost | Optimizer 0.25 + gate | -17.17% | -27.06% | 282.82 | 27,730 | 0.44 | Fails cost stress |
| year_2023_base | Baseline rank buffer | 8.39% | -13.88% | 43.86 | 9,568 | n/a | Baseline good-regime reference |
| year_2023_base | Optimizer 0.15 + gate | -0.38% | -9.10% | 79.88 | 6,524 | 0.46 | Low drawdown, gives up too much return |
| year_2024_base | Baseline rank buffer | -11.55% | -32.13% | 43.92 | 9,561 | n/a | Failure year |
| year_2024_base | Optimizer 0.15 + gate | -3.07% | -11.74% | 68.34 | 7,515 | 0.38 | Best failure-year improvement |
| year_2025_base | Baseline rank buffer | 26.39% | -13.49% | 43.40 | 9,411 | n/a | Good-regime reference |
| year_2025_base | Optimizer 0.15 + gate | 9.24% | -7.35% | 74.98 | 6,904 | 0.44 | Lower drawdown, too much upside loss |

Validation read: the joint optimizer-gate framework is directionally useful as
a risk-control layer, especially in 2024, but it should not replace the current
baseline as the main policy. The main blocker is not trade count, which improves
under the 0.15 budget, but gross turnover and low average exposure: the gate
cuts risk, then optimizer re-risking spends too much turnover to rebuild a
lower-gross book. Next implementation should add calibrated expected-edge/risk
inputs and/or a re-risk-specific budget before another full-suite run.

Implementation follow-up:

- The optimizer turnover allocator now carries `net_edge_bps` into trade rows,
  so budget-limited buys are prioritized by net edge rather than silently
  falling back to rank/priority ordering.
- Added optional
  `max_gross_exposure_increase_per_rebalance` / CLI
  `--optimizer-max-gross-exposure-increase-per-rebalance`. When unset, the
  previous budget behavior is preserved for existing experiments. When set,
  sells still use the configured replacement-turnover budget, while re-risking
  from a lower gross book is capped by a separate net gross-exposure increase
  budget.
- This is an experiment control, not a promotion. The next validation should
  compare calibrated-edge variants with and without this re-risk cap before any
  full-suite run is repeated.

Small-window re-risk cap smoke:

| Window | Variant | Return | Max drawdown | Gross turnover | Trade count | Avg target gross | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q1 2023 | Optimizer 0.15 + gate, no re-risk cap | 4.17% | -3.24% | 21.22 | 1,397 | 0.55 | Prior reference |
| Q1 2023 | Re-risk cap 0.05 | 2.25% | -2.37% | 16.14 | 1,610 | 0.36 | Cuts turnover and drawdown, but gives up too much upside |
| Q1 2023 | Re-risk cap 0.10 | 4.25% | -2.84% | 16.67 | 1,409 | 0.50 | Best Q1 tradeoff in this smoke |
| H1 2024 | Optimizer 0.15 + gate, no re-risk cap | -7.94% | -9.21% | 33.93 | 3,616 | 0.39 | Prior reference |
| H1 2024 | Re-risk cap 0.05 | -7.28% | -7.64% | 31.44 | 5,033 | 0.27 | Lower drawdown, but still high trade count |
| H1 2024 | Re-risk cap 0.10 | -9.12% | -9.80% | 32.04 | 4,617 | 0.32 | Worse than no-cap on H1 loss and drawdown |

Read: the re-risk cap is useful as a control surface and confirms that
exposure rebuilding was part of the turnover problem, but it is not sufficient
to promote the optimizer-gate stack. The 0.10 cap looks reasonable in Q1 2023
but worsens H1 2024; the 0.05 cap helps H1 drawdown but suppresses good-regime
exposure too much. Further optimizer work should prioritize calibrated
expected-edge/risk inputs over additional gate-speed tuning.

## Forecast Calibration Layer

The next framework step is now implemented as an opt-in score-bucket
calibration layer. It converts raw composite scores into optimizer-ready
forecast columns:

- `expected_edge_bps`
- `risk_penalty_bps`
- `forecast_calibration_bucket`
- `forecast_calibration_reason`

The first version uses timestamp-level score buckets and lagged rolling
`forward_return` observations. Current labels are never used for the current
timestamp: bucket statistics are shifted by `label_lag_windows` before rolling
means and risk estimates are assigned back to score rows.

CLI entry point:

```bash
conda run -n quant python examples/run_candidate_factor_portfolios.py \
  --forecast-calibration-mode score_bucket \
  --forecast-calibration-lookback-windows 20 \
  --forecast-calibration-min-periods 5 \
  --forecast-calibration-label-lag-windows 48 \
  --forecast-calibration-bucket-count 5
```

Q1 2023 smoke over the decorrelated score partitions wrote 8,284,467 calibrated
score rows under
`runs/candidate_factor_portfolios/calibration_smoke_q1_2023`. The first January
windows are `warmup`; later January plus February/March are `calibrated`. This
validates the I/O and no-lookahead mechanics, but it is not yet a promotion
result. The next experiment should run optimizer validation with
`optimizer-score-to-edge-bps=0` so the optimizer consumes only calibrated
`expected_edge_bps` instead of raw score scaling.

Initial optimizer smoke:

| Window | Variant | Return | Max drawdown | Gross turnover | Trade count | Avg target gross | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q1 2023 | Raw score edge, optimizer 0.15 + gate | 4.17% | -3.24% | 21.22 | 1,397 | 0.55 | Prior reference |
| Q1 2023 | Calibrated edge + calibrated risk | 1.84% | -2.34% | 5.35 | 296 | 0.16 | Too conservative |
| Q1 2023 | Calibrated edge only | 6.04% | -2.49% | 16.81 | 1,071 | 0.51 | Better return, drawdown, and turnover |
| H1 2024 | Raw score edge, optimizer 0.15 + gate | -7.94% | -9.21% | 33.93 | 3,616 | 0.39 | Prior reference |
| H1 2024 | Calibrated edge + calibrated risk | -6.37% | -10.92% | 1.91 | 127 | 0.03 | Risk penalty suppresses exposure too much |
| H1 2024 | Calibrated edge only | -2.43% | -7.27% | 22.63 | 2,430 | 0.28 | Strong failure-window improvement |

Read: calibrated expected edge is the first framework-side change that improves
both the good-regime Q1 2023 and the 2024 failure window while lowering
turnover. The current `risk_penalty_bps` estimate is too blunt because it uses
raw bucket volatility as a direct bps deduction; keep it as an experimental
field and use `optimizer-risk-penalty-multiplier=0` for the next validation
until downside-specific risk calibration is implemented.

Standard multi-year validation:

```bash
conda run -n quant python examples/run_candidate_policy_validation.py \
  --profile standard \
  --methods decorrelated \
  --primary-method decorrelated \
  --backtest-policy-set single \
  --policy single \
  --trade-policy cost_aware_optimizer \
  --rebalance-every-n-bars 48 \
  --policy-estimated-cost-bps 9 \
  --policy-no-trade-weight-band 0.002 \
  --policy-max-gross-turnover-per-rebalance 0.10 \
  --policy-gross-exposure-scale-path \
    runs/candidate_factor_portfolios/optimizer_regime_gate_validation/gate_budget_deadband_full/decorrelated/gross_exposure_schedule.csv \
  --optimizer-candidate-rank 150 \
  --optimizer-score-to-edge-bps 0 \
  --optimizer-min-net-edge-bps 1 \
  --optimizer-risk-penalty-multiplier 0 \
  --optimizer-weighting equal \
  --forecast-calibration-mode score_bucket \
  --forecast-calibration-lookback-windows 3 \
  --forecast-calibration-min-periods 1 \
  --forecast-calibration-label-lag-windows 48 \
  --forecast-calibration-bucket-count 5
```

Official promoted run output:

```text
runs/candidate_factor_portfolios/calibrated_edge_optimizer_validation_promoted_budget010
```

Wrapper status: `pass`, with zero failed checks and zero warnings.

Results:

| Scenario | Variant | Return | Max drawdown | Gross turnover | Trade count | Avg target gross | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023-2025 full | Calibrated edge, turnover budget 0.15 | 27.15% | -9.46% | 182.96 | 16,043 | 0.38 | Positive, but above 160 turnover gate |
| 2023-2025 full | Calibrated edge, turnover budget 0.10 | 32.21% | -8.49% | 156.37 | 13,663 | 0.40 | Promotable default for next experiments |
| 2023-2025 high cost | Calibrated edge, turnover budget 0.15 | 14.11% | -10.37% | 182.49 | 16,011 | 0.38 | Positive, but turnover warning remains |
| 2023-2025 high cost | Calibrated edge, turnover budget 0.10 | 20.31% | -9.28% | 156.19 | 13,595 | 0.40 | Positive under doubled costs |
| 2023 | Calibrated edge, turnover budget 0.10 | 7.97% | -7.47% | 57.95 | 4,067 | 0.47 | Positive yearly slice |
| 2024 | Calibrated edge, turnover budget 0.10 | 12.18% | -7.23% | 44.05 | 4,650 | 0.32 | Positive yearly slice |
| 2025 | Calibrated edge, turnover budget 0.10 | 5.10% | -7.10% | 54.25 | 4,934 | 0.40 | Positive yearly slice |

Parameter read: reducing the per-rebalance gross turnover budget from `0.15` to
`0.10` is the cleanest turnover control found so far. It lowered full-window
turnover below the current 160 gate and improved return/drawdown. By contrast,
raising the no-trade band to `0.003` kept full-window turnover at 182.62, and
raising `optimizer-min-net-edge-bps` to `3` kept turnover at 180.21 while
reducing return. Use the calibrated edge-only optimizer with budget `0.10` as
the next framework default; keep calibrated risk penalties disabled until their
risk model is redesigned.

## Candidate Combination Recheck Under Promoted Framework

The first combination-layer expansion under the promoted framework compared
`equal`, `ic_weighted`, and `decorrelated` while holding the trading policy,
calibration, regime gross-exposure gate, optimizer settings, and cost model
constant.

Quick full-window output:

```text
runs/candidate_factor_portfolios/promoted_combination_methods_full_base_quick
```

| Method | Return | Max drawdown | Gross turnover | Trades | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| equal | 36.40% | -7.89% | 166.00 | 14,357 | Best return, fails turnover gate |
| decorrelated | 32.21% | -8.49% | 156.37 | 13,663 | Current default, passes gate |
| ic_weighted | 16.15% | -9.69% | 157.91 | 14,260 | Not competitive |

Expanded `equal` standard validation:

| Run | Status | Full return | Full drawdown | Full turnover | High-cost return | Yearly returns | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `promoted_combination_methods_equal_standard_budget010` | `warn` | 36.40% | -7.89% | 166.00 | 23.47% | 8.93%, 16.69%, 7.07% | Strong, but breaches turnover |
| `promoted_combination_methods_equal_standard_budget0095` | `warn` | 37.18% | -9.08% | 162.63 | 24.42% | 6.56%, 15.31%, 6.91% | Still breaches turnover |

Decision: do not replace the promoted `decorrelated` default yet. `equal` is
now the leading score-combination research candidate because it improves the
2024 failure window and remains positive under cost stress, but it still cannot
pass the full-window turnover gate through a small per-rebalance budget tweak.
This points to a framework-side issue in the trade optimizer or turnover budget
allocation, not to a need for more factor-side sweeps.

Near-term framework tasks:

- [x] Recheck candidate combination methods under the promoted calibrated
  optimizer framework.
- [x] Expand `equal` to standard validation and run a narrow turnover-budget
  sensitivity check.
- [x] Redesign turnover handling so the optimizer allocates a path-level
  turnover budget instead of only clipping each rebalance independently.
- [x] Revalidate `equal` and `decorrelated` after the turnover allocator change
  using the same standard gate.
- [ ] Revisit calibrated risk penalties only after turnover allocation is
  stable.

Implemented path-level turnover budget:

- `run_tree_score_backtest.py` now carries `PolicyBudgetState` across streaming
  chunks, tracks remaining path turnover, and applies a hard post-decision cap
  through `--policy-total-gross-turnover-budget`.
- `--policy-turnover-budget-pacing` is optional and defaults to `0`, meaning no
  time-slicing. A positive value adds an additional remaining-budget pacing cap.
  Early quick checks showed strict even pacing under-invested the book.
- Candidate portfolio and policy-validation wrappers pass through the new
  budget arguments and surface `average_dynamic_turnover_cap` plus
  `turnover_path_budget_remaining` in summaries.

Quick validation after the allocator change:

```text
runs/candidate_factor_portfolios/equal_path_turnover_budget_quick_budget155_v5
```

| Method | Path budget | Pacing | Return | Max drawdown | Gross turnover | Planned turnover | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| equal | 155 | 0 | 33.86% | -7.89% | 148.11 | 155.00 | Passes 160 turnover gate; budget consumed by 2025-09 |

Decision: path-level budgeting fixes the immediate gate breach without the
overly conservative behavior caused by even pacing. It should now be validated
on the standard suite for both `equal` and the promoted `decorrelated` default.
The implementation still warrants a performance pass if full validation shows
the post-decision cap is a bottleneck.

Standard validation after the allocator change:

```text
runs/candidate_factor_portfolios/equal_path_turnover_budget_standard_budget155
runs/candidate_factor_portfolios/decorrelated_promoted_standard_after_path_budget
```

Wrapper status: both runs `pass`, with zero warnings and zero failed checks.

| Method | Turnover control | Full return | Full drawdown | Full turnover | High-cost return | Yearly returns | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| equal | Path budget 155, no pacing | 33.86% | -7.89% | 148.11 | 22.21% | 8.93%, 16.69%, 7.07% | Strongest standard result; path budget exhausted by 2025-09 on the full path |
| decorrelated | Per-rebalance budget 0.10 | 32.21% | -8.49% | 156.37 | 20.31% | 7.97%, 12.18%, 5.10% | Current default remains valid, but now dominated by equal path-budget candidate |

Decision: promote `equal` with `--policy-total-gross-turnover-budget 155` to
the leading implementation candidate. It passes the same standard gate as the
current `decorrelated` default while improving full-window return, drawdown,
high-cost robustness, and every yearly slice. Do not make a silent default
switch yet: the full-path budget is exhausted around 2025-09, leaving no
turnover for late-year adaptation. The next framework task should turn the
static validation-path budget into a production-usable budget horizon, for
example rolling annual/monthly envelopes or a replenishing budget ledger, before
we treat the candidate as the live-trading default.

Rolling budget ledger implementation:

- `--policy-turnover-budget-period` now supports `path`, `year`, and `month`.
  The default is `path`, so existing full-path budget experiments are unchanged.
- For `year` and `month`, the configured
  `--policy-total-gross-turnover-budget` is the per-period gross-turnover
  envelope. The remaining budget is carried across streaming chunks within the
  same calendar period and replenished when the period key changes.
- Policy diagnostics now include `turnover_budget_period`,
  `turnover_budget_period_key`, and `turnover_budget_period_count`, so run
  summaries can detect whether a rolling ledger was actually active.

Quick smoke for monthly replenishment:

```text
runs/candidate_factor_portfolios/equal_monthly_budget_quick_smoke
```

| Method | Period | Per-period budget | Return | Max drawdown | Gross turnover | Period count | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| equal | month | 6 | 37.27% | -7.24% | 163.53 | 36 | CLI and streaming ledger work; budget is too loose for 160 turnover gate |

Decision: the production-oriented ledger is now available, but the monthly
budget needs calibration. A monthly envelope of `6` keeps the strategy active
through the whole path and improves return, but it exceeds the current full
turnover gate. The next experiment should sweep monthly budgets around
`4.0-4.5` or test an annual replenishing envelope around `52`, then standard
validate the best candidate against `equal` path-budget `155`.

## References

- Qlib: An AI-oriented Quantitative Investment Platform:
  https://arxiv.org/abs/2009.11189
- Qlib portfolio strategy documentation:
  https://qlib.readthedocs.io/en/latest/component/strategy.html
- Multi-Period Trading via Convex Optimization:
  https://web.stanford.edu/~boyd/papers/cvx_portfolio.html
- Cvxportfolio optimization policies:
  https://www.cvxportfolio.com/en/stable/optimization_policies.html
- Cvxportfolio cost models:
  https://www.cvxportfolio.com/en/1.3.1/costs.html
- Factor Timing is Hard:
  https://www.aqr.com/Insights/Perspectives/Factor-Timing-is-Hard
- Factor Momentum Everywhere:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3300728
- Factor Momentum and the Momentum Factor:
  https://ideas.repec.org/a/bla/jfinan/v77y2022i3p1877-1919.html
- Factor Momentum in the Chinese Stock Market:
  https://cirforum.org/cirf2022/forum_files/papers/CIRF-160.pdf
- A Century of Evidence on Trend-Following Investing:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026
- Shanghai Stock Exchange trading mechanism:
  https://english.sse.com.cn/start/trading/mechanism/
- HKEX Stock Connect product leaflet:
  https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Getting-Started/Information-Booklet-and-FAQ/HKEX_Stock-Connect_EN_Oct-2024.pdf
