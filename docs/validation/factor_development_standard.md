# Factor Development Standard

This document defines the governance layer that must be in place before the next
round of factor discovery. The goal is to let factor research scale without
turning into an uncontrolled factor zoo.

## External Design Inputs

The standard follows five practical lessons from prior research and production
practice:

- Harvey, Liu, and Zhu's factor-zoo work warns that many published factors are
  the result of repeated testing and weak multiple-testing control. New factors
  therefore need explicit hypotheses, family tags, and promotion gates instead
  of single-run return claims.
- Bailey, Borwein, Lopez de Prado, and Zhu's backtest-overfitting work motivates
  a strict separation between discovery, candidate review, and standard
  validation. The registry records every promoted or rejected decision so that
  selection pressure is visible.
- Novy-Marx and Velikov's trading-cost work supports treating turnover and
  implementation cost as first-class factor diagnostics, not as a final
  afterthought.
- Alphalens-style engineering practice motivates the single-factor diagnostics
  already in the framework: rank IC, quantile returns, top-minus-bottom spread,
  turnover, and tear-sheet-like review artifacts.
- A-share execution constraints require every live candidate to be compatible
  with long-only operation, T+1 holding rules, ST filtering, price-limit-aware
  entry/exit logic, and liquidity-aware execution.

Latest legacy-factor revalidation:

- `docs/validation/legacy_factor_revalidation_2026_05_20.md` records the first
  full revalidation run under factor-health monitor mode. Use it as the current
  source for confirmed legacy factors, upgrade-review candidates, and
  horizon/policy review queues before starting the next discovery batch.

Reference links:

- Harvey, Liu, Zhu, "... and the Cross-Section of Expected Returns":
  <https://academic.oup.com/rfs/article/29/1/5/1843824>
- Bailey et al., "The Probability of Backtest Overfitting":
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
- Novy-Marx and Velikov, "A Taxonomy of Anomalies and Their Trading Costs":
  <https://www.nber.org/papers/w20721>
- Alphalens engineering reference:
  <https://github.com/quantopian/alphalens>
- Shanghai Stock Exchange trading mechanism:
  <https://english.sse.com.cn/start/trading/mechanism/>

## Governance Objects

Factor research now has three required governance artifacts:

- Registry: `configs/factors/factor_registry.json`
- Registry validator: `examples/validate_factor_registry.py`
- Candidate review renderer: `examples/run_factor_candidate_review.py`

The registry is the source of truth for factor identity and lifecycle state. A
factor that is not registered is not eligible for portfolio-level testing.

## Baseline Hierarchy For Portfolio Review

Portfolio review must use a layered baseline stack. A factor or risk-control
overlay is not promotion-ready just because it beats a naive or historical
control. Each review must state which layer is being compared and must preserve
the evidence paths for all material comparisons.

- Naive/control baseline: a simple or historical benchmark used for sanity
  checks, plumbing regression, and long-horizon comparability. This layer
  answers whether the candidate has any absolute value, but it is not sufficient
  for promotion.
- Active/default baseline: the current default production-candidate policy. As
  of the 2026-05-19 daily moving-average review, this remains
  `score_budget_gate_v1`.
- Research frontier baseline: the strongest reviewed challenger that has not
  necessarily become the active default. The frontier is family-specific:
  daily-moving-average work should still compare against the fixed
  `high_dispersion_current` ribbon-dispersion gross-exposure gate from
  `runs/candidate_factor_portfolios/daily_ma_promoted_candidate_review_v1/`;
  optimizer-native portfolio work should compare against the latest
  volume-concentration cost-pressure frontier `vc_opt_risk_cp0010_w50` from
  `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_cost_pressure_cap0010_standard/`.

New factor batches must report marginal contribution against the research
frontier whenever the frontier is in the same strategy family or can be
composed without violating the candidate hypothesis. A result that beats only
the naive/control baseline is recorded as useful evidence, but it does not enter
the default combination unless it also improves or diversifies the active
baseline and the research frontier after costs.

The research frontier is a comparison target, not an automatic default. A
frontier candidate becomes the active/default baseline only through a separate
default-change review that links the candidate review, admission report,
standard validation, walk-forward evidence, concentration checks, cost stress,
and attribution evidence from the registry entry.

## Required Registry Fields

Every active factor must record:

- `factor_id`: stable snake-case identifier.
- `display_name`: human-readable name.
- `family`: one of the allowed factor families enforced by the validator.
- `status`: `planned`, `candidate`, `watchlist`, `reject`, `promoted`, or
  `deprecated`.
- `expected_direction`: `long`, `invert`, `neutral`, or `mixed`.
- `evaluation_role`: optional role for admission and downstream validation;
  defaults to `alpha_rank`. Use `risk_penalty`, `entry_filter`,
  `state_allocator`, or `event_overlay` when the signal is portfolio-native
  rather than a standalone rank alpha.
- `feature_columns`: dataset columns generated by the factor.
- `required_inputs`: raw fields needed to compute the factor.
- `frequency`, `lookback_bars`, and `label_lag_bars`.
- `point_in_time_safe`: must be true for active factors.
- `live_available`: must be true for active factors.
- `a_share_constraints`: must confirm `long_only`, `t_plus_one_safe`,
  `price_limit_aware`, and `st_aware`.
- `implementation`: module and callable used to build the feature.
- `evaluation`: latest admission status and evidence path.
- `research_memory`: required for `watchlist`, `reject`, and `deprecated`
  factors. It records why the idea did not advance and when, if ever, it may be
  retried.
- `description`, `hypothesis`, `tags`, and `references`.

The validator treats missing active-factor safety metadata as a hard error.

## Development Workflow

1. Register the hypothesis before implementation.
   New factors start as `planned` or `candidate` entries. The registry must
   state the expected direction, raw inputs, lookback horizon, and A-share
   execution assumptions.

2. Check research memory before implementation.
   Search the registry for matching `family`, `required_inputs`, lookback
   horizon, transform type, `similar_to`, tags, and notes. If a rejected or
   watchlist factor is similar, the new entry must explain what is materially
   different before code is written.

   ```bash
   conda run -n quant python examples/check_factor_research_memory.py \
     --factor-id intraday_new_reversal_5m_w48 \
     --family reversal \
     --required-inputs instrument_id bar_end_time close_price volume turnover \
     --lookback-bars 48 \
     --keywords vwap deviation reversal \
     --enforce-no-blocking
   ```

3. Implement within the declared boundary.
   The implementation may add columns only under the registered
   `feature_columns`. New raw data dependencies require a registry update before
   code changes.

4. Run single-factor evaluation.
   For a new factor batch, build and evaluate only the newly introduced factor
   groups first. Do not run `--factor-groups all` during iterative discovery
   unless the task is explicitly a full-regression benchmark or release gate.
   This keeps admission evidence focused on the new hypotheses and prevents
   full-feature memory growth from slowing every iteration.

   A new-factor-only dataset command should name the new groups explicitly:

   ```bash
   conda run -n quant python examples/build_baseline_a_alpha_dataset.py \
     --catalog-path ../quant_dataset/canonical_store/catalog/quant_research.duckdb \
     --start 2023-01-03T09:35:00+08:00 \
     --end 2025-12-31T15:00:00+08:00 \
     --output-dir runs/framework_v1_acceptance/<batch>/alpha_dataset \
     --factor-groups intraday_new_group another_new_group \
     --workers 1 \
     --memory-budget-gb 0 \
     --worker-memory-estimate-gb 10
   ```

   Full-suite benchmark runs remain available for framework regression, but
   dataset worker concurrency must be bounded by the dataset memory budget:

   ```bash
   conda run -n quant python examples/run_framework_v1_benchmark.py \
     --output-dir runs/framework_v1_acceptance/standard \
     --auto-factor-admission \
     --resume-existing \
     --enforce-gates
   ```

   If the benchmark was run without `--auto-factor-admission`, run
   `examples/analyze_framework_v1_acceptance.py` manually against the completed
   `benchmark_summary.json`.

5. Render a candidate review.

   ```bash
   conda run -n quant python examples/run_factor_candidate_review.py \
     --factor-id intraday_volatility_5m_w24 \
     --output-dir runs/factor_candidate_reviews/intraday_volatility_5m_w24 \
     --enforce-ready
   ```

6. Run portfolio-level validation only after the review is ready.
   The current controlled default for combination-layer review is
   `examples/run_candidate_policy_validation.py` with the standard comparison
   set, methods `decorrelated equal ic_weighted`, and primary gate
   `decorrelated + partial_rebalance_daily`. Keep scenario construction serial
   unless a task explicitly opts into scenario parallelism; the expensive
   score-backtest subprocess layer defaults to six workers. Cost-aware
   optimizer branches are not the default factor-promotion path, but they are
   no longer all historical rejects: as of the 2026-05-25
   time-series-decomposition review, `vc_opt_risk_cp0010_w50` is the latest
   optimizer-native research frontier. The older `equal` annual gross-turnover
   budget `52` branch remains historical unless a task explicitly reopens that
   path.

   Standard policy validation now builds lagged factor-leg health diagnostics in
   `monitor` mode by default. This writes per-factor rolling IC,
   top-minus-bottom, top-bucket label, health-state, contribution-concentration,
   and recommended-weight-scale artifacts, but it keeps applied factor
   `weight_scale=1.0` and does not change composite alpha scores. Use
   `--factor-health-mode shrink` only for an explicit score-construction
   experiment, and report it as an alpha-transform variant rather than as the
   default validation baseline. The validation wrapper also writes
   `validation_factor_health_attribution.csv`, which joins month-level
   performance with factor health and contribution dominance so failure months
   can be reviewed without manually joining score diagnostics, health schedules,
   and backtest equity curves.

   Before running portfolio validation, identify the baseline stack for the
   research family:

   - naive/control anchor;
   - active/default baseline;
   - research frontier baseline, when one exists.

   The validation report must include a comparison against the active/default
   baseline and the research frontier. If compute budget prevents a full
   frontier replay, the report must explain why and must not claim promotion.
   For the current daily moving-average research family, compare new candidates
   against both `score_budget_gate_v1` and the fixed
   `high_dispersion_current` frontier. Do not use the dynamic train-window
   selector from `daily_ma_ribbon_dispersion_walk_forward_v1` as a default
   comparator; it was explicitly rejected in the promoted-candidate review. For
   optimizer-native risk-penalty or cost-pressure research, compare against
   `vc_opt_risk_cp0010_w50` and state explicitly that it is a research frontier,
   not the active/default allocator.

   ```bash
   conda run -n quant python examples/run_candidate_policy_validation.py \
     --dataset-dir runs/framework_v1_acceptance/standard/alpha_dataset \
     --label-column forward_return \
     --admission-report runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json \
     --factor-correlation runs/framework_v1_acceptance/standard/factor_evaluation/feature_correlation.csv \
     --output-dir runs/framework_v1_acceptance/standard/candidate_policy_validation \
     --profile standard \
     --methods decorrelated equal ic_weighted \
     --primary-method decorrelated \
     --policy partial_rebalance_daily \
     --resume-existing
   ```

   Portfolio validation must use the current framework unless a framework issue
   is explicitly being tested. For factor-derived risk controls, prefer the
   integrated
   `examples/run_candidate_policy_validation.py --factor-risk-gate-feature ...`
   path so the gate schedule is rebuilt from the validation dataset and passed
   consistently to every full, yearly, and cost-stress scenario. If the
   validation alpha dataset does not yet include the risk-control feature, use
   `--factor-risk-gate-dataset-dir` to point the gate builder at the matching
   factor dataset while keeping the portfolio alpha dataset fixed.

   LightGBM or primary-pool rerank experiments have an additional standard
   wrapper and must not be run as loose one-off commands when they are used for
   candidate promotion evidence:

   ```bash
   conda run -n quant python examples/run_ml_challenger_standard_workflow.py \
     --dataset-dir runs/ml_factor_challenger/<alpha_dataset> \
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
     --live-end <latest_available_timestamp>
   ```

   The wrapper first writes a dry-run plan. Use `--execute` only after checking
   the plan. Standard ML challenger evidence must include the fixed blend
   backtests, the full walk-forward span, and adaptive source-switch backtests.
   The standard wrapper intentionally does not expose
   `--allow-label-derived-features`; label-derived columns, entry metadata, exit
   metadata, and missing partition columns are hard blockers for promotion
   evidence. If an adaptive score source changes, the live adaptive backtest
   must use source-transition exits so the selected score source can actually
   affect holdings.

7. Promote, watchlist, or reject with evidence.
   Promotion requires the registry entry, candidate review, admission report,
   and portfolio validation summary to be linked from the entry before default
   configuration changes. If a candidate is accepted as a research frontier but
   not as the active/default baseline, record that distinction in
   `evaluation.portfolio_validation_status` and keep the default configuration
   unchanged. Watchlist, reject, and deprecated decisions must also update
   `research_memory`.

## Research Memory

Research memory prevents repeated discovery of the same failed idea under a new
name. Any factor with status `watchlist`, `reject`, or `deprecated` must include:

- `decision_reason`: one of `weak_ic`, `unstable_years`, `weak_hit_rate`,
  `cost_fragile`, `portfolio_negative`, `duplicate_like`,
  `implementation_issue`, `data_quality`, `risk_concentration`, or `other`.
- `negative_findings`: concise explanation of what failed or why the result is
  not promotion-ready.
- `similar_to`: factor ids with close hypotheses, raw inputs, transforms, or
  portfolio behavior.
- `retry_conditions`: concrete condition required before the idea can be opened
  again. "Try another window" is not enough unless the evidence identifies
  window sensitivity as the specific failure.
- `evidence_artifacts`: admission reports, candidate reviews, portfolio
  validations, or research notes supporting the decision.

Default rule: a `reject` factor cannot enter portfolio-level validation again
unless the new registry entry documents a materially different transform or
satisfies the rejected factor's `retry_conditions`. A `watchlist` factor may be
used only for targeted combination, risk overlay, or conditional experiments
until it clears the missing gate.

The executable pre-development check is:

```bash
conda run -n quant python examples/check_factor_research_memory.py \
  --factor-id intraday_new_reversal_5m_w48 \
  --family reversal \
  --required-inputs close_price volume turnover \
  --lookback-bars 48 \
  --keywords vwap deviation reversal \
  --output-dir runs/factor_research_memory/intraday_new_reversal_5m_w48 \
  --enforce-no-blocking
```

`reject` and `deprecated` matches are blocking under `--enforce-no-blocking`.
`watchlist` matches are warnings and require an explicit material-difference
note in the registry before implementation.

## Evaluation Gates

Single-factor admission remains defined in
`docs/validation/factor_admission.md`. The factor development layer adds
pre-admission and post-admission gates:

| Gate | Hard requirement |
| --- | --- |
| Registry completeness | Validator status must be `pass` before batch discovery starts |
| Point-in-time safety | Active factors must not use future bars, future labels, or revised fields |
| Live availability | Inputs must be available at the decision time in a future real-time system |
| A-share tradability | Must explicitly support long-only, T+1, ST filtering, and price-limit-aware execution |
| Single-factor quality | Must pass or intentionally enter `watchlist` under the standard admission report |
| Portfolio contribution | Must improve or diversify the active/default baseline and the research frontier after costs |
| Robustness | Must be checked across full-window, annual slices, high-cost stress, and, when a frontier is involved, walk-forward or anchored forward windows |
| Baseline hierarchy | Must report naive/control, active/default, and research-frontier comparisons when those layers exist |

## Unified Candidate Review Format

The candidate review report has fixed sections:

- Factor identity and hypothesis.
- Registry validation result and factor-specific issues.
- Point-in-time and live-availability checklist.
- Single-factor admission rows for all registered `feature_columns`.
- Optional portfolio validation summary.
- Machine-readable `status`:
  - `ready_for_portfolio_review`
  - `watchlist`
  - `pending_single_factor_review`
  - `blocked`

The renderer writes both JSON and Markdown:

```bash
conda run -n quant python examples/run_factor_candidate_review.py \
  --factor-id intraday_turnover_ratio_5m_w48 \
  --output-dir runs/factor_candidate_reviews/intraday_turnover_ratio_5m_w48
```

## Parallelism Policy

Development decisions remain sequential. Implementation and evaluation may be
parallelized only within explicit family boundaries:

- At most one active implementation task per factor family unless the write
  paths are disjoint.
- Python compute parallelism must use process workers. Do not add
  `ThreadPoolExecutor` or `thread` backends for factor builds, evaluations,
  backtests, or validation orchestration.
- Evaluation can run in parallel under the existing memory-budget controls.
- Full-feature dataset builds must set an explicit worker memory estimate or
  memory budget. The builder may reduce requested worker count to stay within
  budget.
- Promotion decisions are not automated. Reports are inputs to review, not
  authority to change defaults.

## Registry Validation

Run this before starting any new factor batch:

```bash
conda run -n quant python examples/validate_factor_registry.py \
  --registry configs/factors/factor_registry.json \
  --output-dir runs/factor_registry_validation/current \
  --enforce-clean
```

`--enforce-clean` fails on errors or warnings. Use `--enforce-no-errors` when
planned factors are intentionally incomplete but active factors still need hard
validation.

## Current Seed State

The initial registry is seeded from the current standard admission report. It
tracks the current volatility, liquidity, turnover, volume, and momentum
features that are already part of the candidate-factor portfolio workflow. This
does not promote new factors by itself; it only establishes the controlled
starting inventory for the next discovery phase.
