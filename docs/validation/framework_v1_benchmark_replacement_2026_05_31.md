# Framework v1 Benchmark Replacement 2026-05-31

This note records the standard benchmark replacement after the
evaluation-framework fixes completed on 2026-05-31.

## Artifacts

- Current benchmark:
  `runs/framework_v1_acceptance/standard/benchmark_summary.json`
- Current dataset:
  `runs/framework_v1_acceptance/standard/alpha_dataset`
- Current factor evaluation:
  `runs/framework_v1_acceptance/standard/factor_evaluation/summary.json`
- Current factor admission:
  `runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json`
- Current candidate policy validation:
  `runs/framework_v1_acceptance/standard/candidate_policy_validation/validation_summary.json`
- Pre-fix report backup:
  `runs/framework_v1_acceptance/standard/legacy_before_framework_fix_20260530`

## Framework Changes

- Forward-return labels now explicitly use `open_price` for both entry and exit
  legs on the next-bar executable grid.
- Dataset construction filters non-tradable entries, entry limit-up bars,
  non-tradable exits, and exit limit-down bars. Exit limit-up rows remain
  diagnostics for long-only forward returns.
- Label-derived, entry-metadata, and exit-metadata columns are excluded from
  default feature inference.
- ML challenger splits purge training rows by label maturity, not only by
  feature timestamp.
- Rolling factor health and forecast calibration default label lag is the
  label horizon plus one execution bar, `49` windows for a 48-bar label.
- Factor admission uses registry role and expected-direction metadata instead
  of inferring production direction from the same evaluation run.
- Candidate factor combinations use equal admission-evidence weights by
  default. Admission-IC sizing is an explicit experiment.
- Universe wildcard selection and real backtests use point-in-time `as_of`
  dates from the validation start rather than the end of the window.
- Feature-correlation estimation is sampled by default in the top-level
  benchmark so the full all-factor run is reproducible on available memory.

## Replacement Results

The current standard benchmark status is `completed`, with acceptance
`pass`, `failed_count=0`, and `warning_count=0`.

Dataset and factor counts:

| item | pre-fix | current |
| --- | ---: | ---: |
| dataset rows | 103,498,445 | 102,974,156 |
| label rows | 103,498,445 | 103,118,656 |
| evaluated features | 17 | 148 |
| admission candidates | 5 | 43 |
| admission watchlist | 3 | 55 |
| admission rejects | 9 | 50 |

The dataset row reduction comes from the new exit-side tradability filters. The
large feature-count change is expected: the fixed builder expands `all` factor
groups and batches feature construction instead of silently evaluating only the
smaller pre-fix subset.

Top current single-factor IC entries:

| feature | rank IC | top-minus-bottom label |
| --- | ---: | ---: |
| `intraday_range_volatility_5m_w48` | -0.0667 | -0.00717 |
| `intraday_lottery_max_5m_w96` | 0.0656 | 0.00527 |
| `intraday_lottery_max_5m_w48` | 0.0604 | 0.00493 |
| `intraday_liquidity_reliability_recovery_balance_5m_l48_c12_r24` | 0.0572 | 0.00147 |
| `intraday_lottery_max_5m_w24` | 0.0529 | 0.00467 |

The best current candidate policy validation entry is
`decorrelated / partial_rebalance_daily`:

| metric | value |
| --- | ---: |
| full-base return | 25.23% |
| full-high-cost return | 19.65% |
| mean return | 13.34% |
| worst return | 1.00% |
| worst drawdown | -29.96% |
| mean gross turnover | 73.09 |

## Interpretation

Pre-fix admission results are no longer current evidence. They can be used for
historical comparison only. Any factor previously promoted, admitted, or placed
on the watchlist under the old framework should be revalidated against the
current standard artifacts before it is used for sizing, model inclusion, or
default strategy configuration.
