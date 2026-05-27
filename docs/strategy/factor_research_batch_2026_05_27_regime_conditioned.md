# Factor Research Batch - 2026-05-27 Regime-Conditioned Factors

This batch implements the policy-level market/regime gate direction before
another broad factor-combination sweep. The goal is not to find an all-period
factor. The goal is to expose factors whose sign or weight depends on an
observable market regime.

## Hypothesis

The 2024 failures are treated as a regime problem, not primarily a transaction
cost problem. In retail-heavy A-share conditions, breadth deterioration,
limit-down pressure, and downside market momentum can flip the effective IC of
short-horizon reversal, momentum, volatility, and crowding-sensitive signals.

The new feature group therefore separates the market state from the base
cross-sectional signal:

`alpha = base_signal * regime_probability`

and also emits an explicit sign-switch version:

`alpha = residual_reversal * (1 - 2 * regime_probability)`

When stress probability is near 0, the switch is reversal-like. When stress
probability is near 1, the switch becomes momentum-like.

## Implemented Feature Group

New group: `regime_conditioned`

Default parameters:

- `--regime-conditioned-lookback-bars 12 24`
- `--regime-conditioned-state-windows 48`

For each state window, the builder emits:

| feature | role |
|---|---|
| `market_regime_stress_probability_5m_w{window}` | Continuous observable stress state from weak breadth, downside momentum, and limit-down pressure. |
| `intraday_regime_stress_reversal_5m_lb{lookback}_w{window}` | Market-residual reversal active in stress states. |
| `intraday_regime_stress_momentum_5m_lb{lookback}_w{window}` | Market-residual momentum active in stress states. |
| `intraday_regime_calm_reversal_5m_lb{lookback}_w{window}` | Market-residual reversal active outside stress states. |
| `intraday_regime_switch_reversal_5m_lb{lookback}_w{window}` | One-piece conditional factor that flips reversal into momentum as stress probability rises. |

The stress probability uses only timestamp-observable inputs:

- weak breadth: `max(0, 0.5 - up_rate)`;
- downside momentum: `max(0, -market_median_return)`;
- limit pressure: `max(0, limit_down_rate - limit_up_rate)`.

## Validation Plan

Build a new-factor-only dataset:

```bash
conda run -n quant python examples/build_baseline_a_alpha_dataset.py \
  --start 2023-01-01T09:35:00+08:00 \
  --end 2025-12-31T15:00:00+08:00 \
  --factor-groups regime_conditioned \
  --regime-conditioned-lookback-bars 12 24 \
  --regime-conditioned-state-windows 48 \
  --output-dir runs/factor_research/regime_conditioned_2026_05_27/alpha_dataset
```

Then run standard factor evaluation and admission against `forward_return_48b`.
If any conditional leg passes admission, portfolio validation should compare:

- raw compact-core baseline;
- compact core plus admitted conditional leg;
- compact core with the stress probability used only as a policy-level gross
  exposure or allocator gate.

The required diagnostic table is January 2024, June 2024, full-window
high-cost, and yearly 2023/2024/2025 slices. A feature that improves full IC but
does not improve the January/June 2024 failure months should remain watchlist.

## Implementation Status

Implemented in `quant_research.datasets.intraday_features` and exposed through
`examples/build_baseline_a_alpha_dataset.py`.

Focused verification:

- `conda run -n quant pytest tests/unit/test_intraday_features.py`
- `conda run -n quant pytest tests/contracts/test_examples_contract.py`

## Full Dataset and Admission Results

Full 2023-2025 dataset build completed:

- Dataset: `research_store/regime_conditioned_2026_05_27_alpha_dataset`
- Partitions: `36`
- Rows: `103357495`
- Feature group: `regime_conditioned`
- Lookbacks: `12`, `24`
- State window: `48`

Single-factor evaluation:

- Output: `research_store/regime_conditioned_2026_05_27_factor_evaluation`
- Summary: `research_store/regime_conditioned_2026_05_27_factor_evaluation/summary.json`

Admission:

- Output: `runs/factor_research/regime_conditioned_2026_05_27/factor_admission`
- Report: `runs/factor_research/regime_conditioned_2026_05_27/factor_admission/factor_admission_report.json`
- Result: `0` candidates, `8` watchlist, `1` reject

| feature | status | direction | rank IC | t-stat | cost-adjusted spread | failed check |
|---|---:|---:|---:|---:|---:|---|
| `intraday_regime_switch_reversal_5m_lb24_w48` | watchlist | long | 0.006386 | 8.13 | -0.001651 | cost-adjusted spread |
| `intraday_regime_calm_reversal_5m_lb24_w48` | watchlist | long | 0.005178 | 6.59 | -0.001880 | cost-adjusted spread |
| `intraday_regime_stress_reversal_5m_lb24_w48` | watchlist | long | 0.005178 | 6.59 | -0.001880 | cost-adjusted spread |
| `intraday_regime_stress_momentum_5m_lb24_w48` | watchlist | invert | -0.005178 | -6.59 | -0.001843 | cost-adjusted spread |
| `intraday_regime_switch_reversal_5m_lb12_w48` | watchlist | long | 0.002979 | 4.09 | -0.001282 | cost-adjusted spread |
| `market_regime_stress_probability_5m_w48` | reject | long | NaN | NaN | -0.000002 | no cross-sectional IC |

The raw conditional features are not promoted as standalone alphas. Their
statistical IC is real, but the top-minus-bottom economics remain negative
after the 13 bps round-trip proxy.

## Regime-State Diagnostics

The continuous stress probability is useful as a state variable but poorly
calibrated as an absolute `0.5` switch:

| sample | count | mean | median | max | share > 0.5 |
|---|---:|---:|---:|---:|---:|
| full 2023-2025 | 34799 | 0.136953 | 0.128581 | 0.681051 | 0.38% |
| 2024-01 | 1056 | 0.171261 | 0.164650 | 0.305333 | 0.00% |
| 2024-06 | 912 | 0.157449 | 0.150095 | 0.252329 | 0.00% |

This means `intraday_regime_switch_reversal_*` did not actually flip sign in
the two known stress months. A fixed absolute switch threshold is therefore not
usable. Any policy-level gate must use rolling relative thresholds or a
recalibrated probability mapping.

Stress-month standalone diagnostics for the most relevant legs:

| feature | month | rank IC | top-minus-bottom | cost-adjusted | turnover |
|---|---:|---:|---:|---:|---:|
| `stress_momentum_lb24` | 2024-01 | 0.064572 | 0.003502 | 0.003503 | 0.222 |
| `stress_momentum_lb24` | 2024-06 | 0.000027 | -0.001753 | -0.001786 | 0.212 |
| `switch_reversal_lb24` | 2024-01 | -0.064572 | -0.003502 | -0.003503 | 0.227 |
| `switch_reversal_lb24` | 2024-06 | -0.000027 | 0.001753 | 0.001786 | 0.236 |

The 2024-01 behavior is exactly the factor-direction inversion case: the
momentum-like leg works while the reversal-like switch loses. June is mixed and
does not provide a clean standalone fix.

## Policy-Level Gate Test

A rolling relative threshold gate was built from
`market_regime_stress_probability_5m_w48`:

- Gate output:
  `runs/factor_research/regime_conditioned_2026_05_27/stress_probability_gate_default`
- Gate rule: 240-bar lagged rolling thresholds, high `0.80`, extreme `0.95`
- Schedule counts: `26456` full, `4541` reduced, `3754` blocked, `48` warmup

Stress-month gate activation:

| month | blocked | reduced | full | average exposure |
|---|---:|---:|---:|---:|
| 2024-01 | 171 | 248 | 637 | 0.720644 |
| 2024-06 | 151 | 94 | 667 | 0.782895 |

The gate was then tested as a gross-exposure schedule on the existing
2026-05-22 three-factor candidate basket:

- Output:
  `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gate_quick`
- Profile: `quick`, full-base only
- Candidate basket:
  `intraday_false_absorption_risk_5m_w48`,
  `intraday_overnight_gap_5m`,
  `intraday_overnight_gap_down_recovery_5m`

Full-window comparison against the ungated quick run:

| method | gate | full return | max DD | turnover | transaction cost |
|---|---:|---:|---:|---:|---:|
| decorrelated | none | 0.170905 | -0.366323 | 122.46 | 162006 |
| decorrelated | stress-probability gate | 0.079836 | -0.325308 | 168.47 | 221050 |
| equal | none | 0.175797 | -0.361537 | 121.97 | 162562 |
| equal | stress-probability gate | 0.135408 | -0.327406 | 168.53 | 223152 |

Stress-month comparison:

| method | month | ungated return | gated return | ungated DD | gated DD |
|---|---:|---:|---:|---:|---:|
| decorrelated | 2024-01 | -0.143815 | -0.127709 | -0.148978 | -0.138339 |
| decorrelated | 2024-06 | -0.091364 | -0.076526 | -0.106269 | -0.085643 |
| equal | 2024-01 | -0.147629 | -0.128007 | -0.153534 | -0.135132 |
| equal | 2024-06 | -0.092737 | -0.075998 | -0.109457 | -0.092629 |

Decision: do not promote this gate. It improves the known stress months and
reduces max drawdown, but it gives back too much full-window return and raises
turnover above the standard 160 gate. The same state variable should be retried
only with a smoother exposure budget, such as `blocked_scale=0.5`, slower
hysteresis, or turnover-budget-aware scaling. Do not run more simple
full/reduced/block parameter searches until the gate is redesigned to reduce
turnover rather than creating extra rebalance churn.

## Smoothed Gate Follow-Up

The factor-risk gate now supports confirmation and step-limited exposure
changes:

- `--state-confirmation-windows`
- `--max-scale-change-per-window`
- `--max-scale-increase-per-window`
- `--max-scale-decrease-per-window`
- `--scale-change-deadband`

The schedule writes raw state, confirmed target state, and final executable
state separately. This keeps the observable regime signal auditable while
allowing the executable gross exposure path to move more slowly.

Tested schedule:

- Output:
  `runs/factor_research/regime_conditioned_2026_05_27/stress_probability_gate_smooth_confirm3_step010`
- Gate rule: same 240-bar lagged thresholds, high `0.80`, extreme `0.95`
- Smoothing: 3-window confirmation, max scale step `0.10`, deadband `0.05`
- Final state counts: `25418` full, `2796` reduced, `2807` blocked,
  `1884` step-limited down, `1844` step-limited up, `50` warmup

Quick validation output:

`runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gate_smooth_quick`

Full-window comparison:

| method | gate | full return | max DD | turnover | transaction cost | avg exposure |
|---|---:|---:|---:|---:|---:|---:|
| decorrelated | none | 0.170905 | -0.366323 | 122.46 | 162006 | 1.000 |
| decorrelated | default gate | 0.079836 | -0.325308 | 168.47 | 221050 | 0.764 |
| decorrelated | smoothed gate | 0.085293 | -0.320704 | 163.38 | 216131 | 0.769 |
| equal | none | 0.175797 | -0.361537 | 121.97 | 162562 | 1.000 |
| equal | default gate | 0.135408 | -0.327406 | 168.53 | 223152 | 0.764 |
| equal | smoothed gate | 0.118966 | -0.325698 | 163.50 | 218129 | 0.769 |

Stress-month comparison:

| method | month | ungated return | default gated return | smoothed gated return | ungated DD | default gated DD | smoothed gated DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| decorrelated | 2024-01 | -0.143815 | -0.127709 | -0.118375 | -0.148978 | -0.138339 | -0.129101 |
| decorrelated | 2024-06 | -0.091364 | -0.076526 | -0.082316 | -0.106269 | -0.085643 | -0.091225 |
| equal | 2024-01 | -0.147629 | -0.128007 | -0.124434 | -0.153534 | -0.135132 | -0.131216 |
| equal | 2024-06 | -0.092737 | -0.075998 | -0.078882 | -0.109457 | -0.092629 | -0.095330 |

Decision: smoothing improves the default gate's implementation quality, but
does not change the admission decision. It reduces turnover from roughly `168`
to `163`, improves decorrelated max drawdown, and gives better January 2024
protection, but it still fails the standard turnover gate and remains materially
worse than the ungated basket on full-window return. The next useful experiment
is not another threshold grid. The gate should be integrated into a
turnover-budget-aware policy or optimizer risk budget, where exposure changes
are paced together with name-level rebalance decisions.

## Path-Budget Gate Follow-Up

The smoothed gate was next combined with the existing path-level gross-turnover
budget machinery:

- Output:
  `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gate_smooth_path_budget155_quick`
- Policy turnover budget: `--policy-total-gross-turnover-budget 155`
- Budget period: `path`
- Budget pacing: `0`
- Validation status: `pass`

Full-window comparison:

| method | gate / budget | full return | max DD | gross turnover | planned turnover | transaction cost | avg exposure |
|---|---:|---:|---:|---:|---:|---:|---:|
| decorrelated | none | 0.170905 | -0.366323 | 122.46 | n/a | 162006 | 1.000 |
| decorrelated | smoothed gate | 0.085293 | -0.320704 | 163.38 | 175.64 | 216131 | 0.769 |
| decorrelated | smoothed gate + path budget 155 | 0.119753 | -0.320704 | 141.86 | 155.00 | 192772 | 0.781 |
| equal | none | 0.175797 | -0.361537 | 121.97 | n/a | 162562 | 1.000 |
| equal | smoothed gate | 0.118966 | -0.325698 | 163.50 | 175.63 | 218129 | 0.769 |
| equal | smoothed gate + path budget 155 | 0.154587 | -0.325698 | 141.81 | 155.00 | 194803 | 0.781 |

Stress-month comparison:

| method | month | ungated return | smoothed gated return | smoothed gated + budget return | ungated DD | smoothed gated DD | smoothed gated + budget DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| decorrelated | 2024-01 | -0.143815 | -0.118375 | -0.118375 | -0.148978 | -0.129101 | -0.129101 |
| decorrelated | 2024-06 | -0.091364 | -0.082316 | -0.082316 | -0.106269 | -0.091225 | -0.091225 |
| equal | 2024-01 | -0.147629 | -0.124434 | -0.124434 | -0.153534 | -0.131216 | -0.131216 |
| equal | 2024-06 | -0.092737 | -0.078882 | -0.078882 | -0.109457 | -0.095330 | -0.095330 |

Decision: the path budget fixes the turnover warning but does not make the
regime gate promotable. The pressure-month protection is unchanged because the
path budget binds late in the sample, not during January or June 2024. The
budgeted gate still trails the ungated basket on full-window return, while the
ungated basket already sits below the `160` turnover gate. This confirms the
state variable is useful for diagnostics and stress drawdown control, but this
gross-exposure gate is not a better production policy for the 2026-05-22
candidate basket.

Next research step: use `market_regime_stress_probability_5m_w48` as an
optimizer-side risk penalty or factor-leg allocator, not as a portfolio-level
gross exposure scaler. The desired behavior is to reduce entry aggressiveness
or reverse-sensitive leg weight in stressed states while keeping the existing
low-turnover basket structure intact.

## Factor-Leg Allocator Follow-Up

The portfolio runner now accepts an external per-factor weight scale schedule:

- `--factor-weight-scale-schedule`
- `--factor-weight-scale-combine-mode {min,multiply,override}`

The schedule is merged with the existing factor-health schedule before score
partition writing, so it changes factor-leg weights directly instead of
changing portfolio gross exposure after the score has already been formed. The
validation wrapper passes the same arguments through to each scenario.

Tested allocator:

- Schedule:
  `runs/factor_research/regime_conditioned_2026_05_27/factor_leg_allocator/stress_gap_leg_weight_scale.csv`
- Source state:
  `stress_probability_gate_smooth_confirm3_step010/gross_exposure_schedule.csv`
- Target leg: `intraday_overnight_gap_5m`
- Combine mode: `min`
- Scale distribution: mean `0.831`, median `1.000`, min approximately `0`,
  max `1.000`

This is intentionally narrower than the gross-exposure gate. It shrinks the
previously weak `intraday_overnight_gap_5m` leg during stressed states while
leaving the rest of the basket and its low-turnover rebalance structure mostly
intact.

Quick validation output:

`runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gap_leg_allocator_quick`

Full-window quick comparison:

| method | allocator | full return | max DD | gross turnover | transaction cost | status |
|---|---:|---:|---:|---:|---:|---:|
| decorrelated | none | 0.170905 | -0.366323 | 122.46 | 162006 | pass |
| decorrelated | stress gap-leg allocator | 0.212910 | -0.340217 | 121.69 | 162749 | pass |
| equal | none | 0.175797 | -0.361537 | 121.97 | 162562 | pass |
| equal | stress gap-leg allocator | 0.189215 | -0.361099 | 123.41 | 163809 | pass |

Stress-month quick comparison:

| method | month | ungated return | allocator return | ungated DD | allocator DD |
|---|---:|---:|---:|---:|---:|
| decorrelated | 2024-01 | -0.143815 | -0.138139 | -0.148978 | -0.148660 |
| decorrelated | 2024-06 | -0.091364 | -0.093790 | -0.106269 | -0.101859 |
| equal | 2024-01 | -0.147629 | -0.148835 | -0.153534 | -0.153419 |
| equal | 2024-06 | -0.092737 | -0.101434 | -0.109457 | -0.114401 |

The allocator is clearly better than the gross-exposure gate for full-window
return and turnover. It is also better than the ungated decorrelated basket on
full return and max drawdown. However, it does not provide the same 2024 stress
drawdown protection as the gross-exposure gate, and the equal-weight version is
not robust in the known stress months.

Standard decorrelated validation output:

`runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gap_leg_allocator_standard_decorrelated`

Standard scenario results:

| scenario | total return | max DD | gross turnover | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 0.212910 | -0.340217 | 121.69 | 162749 | 24132 |
| year_2023_base | -0.015141 | -0.148734 | 41.65 | 54055 | 7871 |
| year_2024_base | 0.022102 | -0.229134 | 42.55 | 56070 | 8422 |
| year_2025_base | 0.148290 | -0.137799 | 39.21 | 53140 | 7674 |
| full_high_cost | 0.157236 | -0.350233 | 121.63 | 203868 | 24088 |

Standard validation status: `warn`, with `0` failures and `1` warning. The
warning is `primary_yearly_base_positive_returns` because `year_2023_base` is
negative.

Standard stress-month diagnostics:

| scenario | month | return | max DD | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 2024-01 | -0.138139 | -0.148660 | 4872 | 735 |
| full_base | 2024-06 | -0.093790 | -0.101859 | 4081 | 636 |
| year_2024_base | 2024-01 | -0.114451 | -0.118869 | 5576 | 735 |
| year_2024_base | 2024-06 | -0.069181 | -0.078303 | 4220 | 646 |
| full_high_cost | 2024-01 | -0.139807 | -0.150288 | 6061 | 740 |
| full_high_cost | 2024-06 | -0.094640 | -0.102507 | 4925 | 632 |

Decision: keep the factor-leg allocator as the next active research branch for
the decorrelated portfolio only. It is a substantial improvement over the
portfolio-level gross-exposure gate and passes turnover and high-cost checks,
but the 2023 negative yearly slice blocks promotion. The next step should be a
narrow allocator sweep across known weak or inversion-prone legs, with the
acceptance bar set to: keep full-base return above the ungated basket, keep
gross turnover below `160`, keep full-high-cost return positive, and remove the
negative yearly-slice warning without worsening January/June 2024 drawdowns.

## Narrow Allocator Sweep

The first allocator sweep stayed intentionally small and used the same
observable smoothed stress state. It tested only candidate-basket legs:

- `intraday_overnight_gap_5m`
- `intraday_false_absorption_risk_5m_w48`
- `intraday_overnight_gap_down_recovery_5m`

Generated schedules:

| schedule | target legs | note |
|---|---|---|
| `stress_gap_leg_weight_scale_strong.csv` | gap | squared stress scale |
| `stress_false_absorption_leg_weight_scale.csv` | false absorption | base stress scale |
| `stress_gap_false_absorption_leg_weight_scale.csv` | gap, false absorption | base stress scale |
| `stress_gap_recovery_leg_weight_scale.csv` | gap, gap-down recovery | base stress scale |
| `stress_gap_false_absorption_false_strong_leg_weight_scale.csv` | gap, false absorption | false absorption squared |
| `stress_gap_false_absorption_both_strong_leg_weight_scale.csv` | gap, false absorption | both squared |

The first screen ran only the 2023 decorrelated slice because the previous best
allocator was blocked by a negative 2023 yearly result.

2023 screening results:

| variant | 2023 return | max DD | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| gap + false absorption, both strong | 0.001232 | -0.132316 | 41.64 | 54126 |
| gap + false absorption | -0.001365 | -0.135270 | 41.66 | 54105 |
| gap + false absorption, false strong | -0.010216 | -0.149603 | 41.58 | 53678 |
| false absorption | -0.012097 | -0.150866 | 41.61 | 53690 |
| gap + recovery | -0.017739 | -0.150439 | 41.62 | 53986 |
| gap strong | -0.021779 | -0.157410 | 41.64 | 53663 |

The best screen was therefore:

`runs/factor_research/regime_conditioned_2026_05_27/factor_leg_allocator/stress_gap_false_absorption_both_strong_leg_weight_scale.csv`

Standard decorrelated validation output:

`runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gap_false_absorption_both_strong_standard_decorrelated`

Standard scenario results:

| scenario | total return | max DD | gross turnover | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 0.234021 | -0.333693 | 121.86 | 163203 | 24163 |
| year_2023_base | 0.001232 | -0.132316 | 41.64 | 54126 | 7877 |
| year_2024_base | -0.001255 | -0.243719 | 41.75 | 54116 | 8146 |
| year_2025_base | 0.117157 | -0.139741 | 38.63 | 51560 | 7469 |
| full_high_cost | 0.176923 | -0.343132 | 121.66 | 204672 | 24125 |

Standard validation status remains `warn`, with `0` failures and `1` warning.
The negative yearly slice moved from `year_2023_base` to `year_2024_base`.

Stress-month diagnostics:

| scenario | month | return | max DD | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 2024-01 | -0.140414 | -0.149922 | 4876 | 733 |
| full_base | 2024-06 | -0.101100 | -0.107941 | 4073 | 636 |
| year_2024_base | 2024-01 | -0.126333 | -0.128841 | 5468 | 716 |
| year_2024_base | 2024-06 | -0.082328 | -0.086189 | 4074 | 628 |
| full_high_cost | 2024-01 | -0.141399 | -0.150878 | 6097 | 739 |
| full_high_cost | 2024-06 | -0.102106 | -0.108941 | 4915 | 633 |

Decision: do not promote this stronger two-leg allocator. It produces the best
full-window return so far, keeps turnover controlled, and passes high-cost, but
it worsens the full-window 2024 stress months versus the single gap-leg
allocator and merely moves the yearly-stability warning from 2023 to 2024. The
useful lesson is that stress-state allocation should not be a single global
stress mapping across all years. The next allocator should condition on the
type of stress: January 2024 crash/limit-down diffusion needs different leg
behavior from the 2023 weak tape and the June 2024 decay episode.

## Multi-Regime Allocator Attempt

The next attempt merged the smoothed stress-probability gate with the lagged
event-state diagnostics from the event-shock proxy research. The first
multi-regime hypothesis was:

- always shrink `intraday_overnight_gap_5m` during stress;
- shrink `intraday_false_absorption_risk_5m_w48` only during weak-tape stress;
- do not shrink false absorption during lagged `limit_diffusion`,
  `limit_diffusion_extreme`, `shock_elevated`, or `shock_extreme`.

Generated schedules:

| schedule | false absorption rule |
|---|---|
| `multi_regime_gap_all_false_weak_tape.csv` | stress active and not toxic event state |
| `multi_regime_gap_all_false_weak_tape_strong.csv` | same rule, squared false scale |
| `multi_regime_gap_all_false_weak_tape_reduced_only.csv` | weak tape and reduced state only |

Focused screen results:

| variant | slice | return | max DD | gross turnover | transaction cost |
|---|---:|---:|---:|---:|---:|
| weak tape | 2023 | -0.018886 | -0.150520 | 41.58 | 53927 |
| weak tape | 2024-01 | -0.114451 | -0.118869 | 5.53 | 5576 |
| weak tape | 2024-06 | -0.080056 | -0.089604 | 5.05 | 4822 |
| weak tape strong | 2023 | -0.019991 | -0.153074 | 41.62 | 53952 |

Decision: stop this branch. The weak-tape rule improves January and June 2024
versus the single gap-leg allocator, but it worsens the already-blocking 2023
year. Excluding toxic event states is the wrong split: 2023 repair also appears
to need false-absorption shrinkage in some limit-diffusion-like windows. Event
labels are useful diagnostics, but the allocator needs a severity or payoff
condition, not a simple toxic/non-toxic label.

## Non-Crash Stress Severity Attempt

A follow-up tested whether false-absorption shrinkage should be applied during
non-crash stress while preserving the 2024 crash-like pressure months. The
rules used only observable state from the smoothed stress schedule and lagged
event diagnostics:

- gap leg: shrink during all stress states using the smoothed stress scale;
- false-absorption leg: shrink with squared scale only when stress is active
  and the crash filter does not trigger;
- crash filters tested: `risk_value >= 0.17`, lagged
  `limit_pressure_rate >= 0.003`, and a combined version with
  `limit_pressure_rate >= 0.0035`; lagged `shock_extreme` was always excluded.

Generated schedules:

| schedule | false absorption active rows | note |
|---|---:|---|
| `stress_gap_false_non_crash_risk017_false_strong.csv` | 6086 | excludes high stress probability and shock extreme |
| `stress_gap_false_non_crash_limit003_false_strong.csv` | 6384 | excludes high lagged limit pressure and shock extreme |
| `stress_gap_false_non_crash_combined_false_strong.csv` | 5303 | excludes either high stress probability, high lagged limit pressure, or shock extreme |

Focused screen results:

| variant | 2023 return | 2024-01 return | 2024-06 return | 2023 max DD | 2024-01 max DD | 2024-06 max DD |
|---|---:|---:|---:|---:|---:|---:|
| risk017 | -0.015392 | -0.116707 | -0.090318 | -0.147660 | -0.120115 | -0.101738 |
| limit003 | -0.015300 | -0.138212 | -0.091086 | -0.147479 | -0.140717 | -0.100537 |
| combined | -0.015392 | -0.116707 | -0.084046 | -0.147660 | -0.120115 | -0.093575 |

Decision: do not run standard validation. None of the non-crash variants beats
the single gap-leg allocator on the blocking 2023 slice (`-0.015141`), and the
limit-pressure variant also gives back the January 2024 improvement. The best
combination improves June 2024 versus single gap but still fails the 2023
acceptance bar. This means the current obstacle is not just identifying crash
months. The same false-absorption leg has different marginal value across
weak-tape, limit-diffusion, and crash-decay windows, and the split is not
captured by one severity threshold.

Current conclusion: continuing is still possible, but the next round should
move from rule thresholds to payoff-conditioned leg attribution. Before another
allocator is proposed, compute timestamp-level or month-regime-level
contribution deltas for the three candidate legs under the existing accepted
score construction. The next gate should be learned or selected from observed
leg payoff by regime bucket, with an embargoed validation split, instead of
hand-authoring more stress labels.

## Regime-Leg Payoff Attribution

Added a reusable diagnostic:

`examples/analyze_regime_factor_leg_payoff.py`

The diagnostic reconstructs each candidate leg's timestamp-level directional
rank payoff, contribution rank IC, effective weight, and top-minus-bottom label
spread, then aggregates by observable regime buckets. It uses only lagged
event-state fields and the already-built smoothed stress schedule.

Focus-month output:

`runs/factor_research/regime_conditioned_2026_05_27/regime_leg_payoff_focus_months`

Focus months were `2023-03`, `2023-04`, `2023-05`, `2023-08`, `2023-10`,
`2023-12`, `2024-01`, and `2024-06`.

Key attribution findings:

| leg | bucket | top-bottom payoff | interpretation |
|---|---:|---:|---|
| `intraday_false_absorption_risk_5m_w48` | `stress_high_limit_pressure` + `shock_elevated` | -0.001800 | false absorption should be shrunk in this shock-elevated high-pressure bucket |
| `intraday_false_absorption_risk_5m_w48` | `stress_high_risk` + `shock_elevated` | -0.000512 | same direction, but smaller effect |
| `intraday_false_absorption_risk_5m_w48` | `stress_high_limit_pressure` + `limit_diffusion_extreme` | 0.005585 | do not globally shrink false absorption in all limit-diffusion stress |
| `intraday_false_absorption_risk_5m_w48` | 2024-01 `stress_high_limit_pressure` | 0.013333 | explains why all-stress false shrink worsened January 2024 |
| `intraday_overnight_gap_down_recovery_5m` | `stress_high_limit_pressure` + `shock_elevated` | -0.008404 | recovery leg is weak in high-pressure shock-elevated states |
| `intraday_overnight_gap_5m` | `stress_shock_extreme` | -0.015258 | gap leg still justifies stress shrink |

This attribution explains the earlier failures. Event labels alone were too
coarse, and crash severity alone was too blunt: the false-absorption leg is
positive in some severe limit-diffusion buckets and negative in some
shock-elevated buckets.

## Payoff-Informed Static Allocator Screen

Two payoff-informed static schedules were generated:

| schedule | rule |
|---|---|
| `payoff_gap_selective_false_strong.csv` | shrink gap in all stress; shrink false absorption only in shock-extreme or shock-elevated high-risk/high-limit buckets |
| `payoff_gap_selective_false_recovery_strong.csv` | same, plus shrink recovery in high-limit-pressure buckets except `limit_diffusion_extreme` |

Schedule coverage:

| schedule | false rows | recovery rows |
|---|---:|---:|
| `payoff_gap_selective_false_strong.csv` | 497 | 0 |
| `payoff_gap_selective_false_recovery_strong.csv` | 497 | 975 |

Focused screen results:

| variant | 2023 return | 2024-01 return | 2024-06 return | 2023 max DD |
|---|---:|---:|---:|---:|
| selective false | -0.005766 | -0.135866 | -0.090113 | -0.141666 |
| selective false + recovery | -0.005766 | -0.135866 | -0.090113 | -0.141666 |

Decision: do not run standard validation. The payoff-informed static gate is a
clear improvement over the prior non-crash rules and beats the single gap-leg
allocator in 2024-01 and 2024-06, but it still leaves 2023 negative. The
recovery-leg static shrink did not change the tested portfolio path.

## Gap Allocator Plus Factor-Health Shrink

The next screen kept the stress gap-leg allocator but changed factor health
from monitor-only to lagged shrink mode:

- Factor schedule:
  `runs/factor_research/regime_conditioned_2026_05_27/factor_leg_allocator/stress_gap_leg_weight_scale.csv`
- Factor health mode: `shrink`
- Lookback: `20` windows
- Label lag: `48` windows
- Min scale: `0.25`
- Combine mode: `min`

Focused screen results:

| slice | return | max DD | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| 2023 | 0.001264 | -0.128464 | 41.56 | 53826 |
| 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| 2024-06 | -0.077267 | -0.090585 | 5.03 | 4843 |

This was the first variant to fix the blocking 2023 slice while preserving the
January and June 2024 stress-month protection.

Standard decorrelated validation output:

`runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gap_health_shrink_standard_decorrelated`

Standard scenario results:

| scenario | total return | max DD | gross turnover | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 0.231632 | -0.327544 | 121.68 | 162805 | 24101 |
| year_2023_base | 0.001264 | -0.128464 | 41.56 | 53826 | 7836 |
| year_2024_base | 0.012112 | -0.230205 | 41.75 | 54844 | 8240 |
| year_2025_base | 0.137951 | -0.133289 | 39.36 | 53258 | 7704 |
| full_high_cost | 0.175710 | -0.336608 | 121.53 | 204322 | 24070 |

Standard validation status: `pass`, with `0` failures and `0` warnings.

Stress-month diagnostics:

| scenario | month | return | max DD | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 2024-01 | -0.139601 | -0.149115 | 4876 | 733 |
| full_base | 2024-06 | -0.099654 | -0.106277 | 4073 | 636 |
| year_2024_base | 2024-01 | -0.115054 | -0.117856 | 5542 | 729 |
| year_2024_base | 2024-06 | -0.073961 | -0.081347 | 4120 | 631 |
| full_high_cost | 2024-01 | -0.140636 | -0.150080 | 6037 | 730 |
| full_high_cost | 2024-06 | -0.100667 | -0.107308 | 4917 | 632 |

Full-base factor-health scale distribution:

| feature | min | mean | median | max | impaired count |
|---|---:|---:|---:|---:|---:|
| `intraday_false_absorption_risk_5m_w48` | 0.25 | 0.649 | 0.711 | 1.00 | 12583 |
| `intraday_overnight_gap_5m` | ~0.00 | 0.486 | 0.372 | 1.00 | 16277 |
| `intraday_overnight_gap_down_recovery_5m` | 0.25 | 0.617 | 0.639 | 1.00 | 14127 |

Decision: this is the first promotable regime-conditioned allocator in this
batch. It keeps the full-window return close to the best two-leg allocator,
keeps turnover far below `160`, passes high-cost, and removes the yearly
stability warning. The remaining caveat is that full-base January/June 2024
stress-month diagnostics are still weaker than the year-specific 2024 run,
which indicates path dependence in the lagged health state. The next production
step should be robustness validation, not another static schedule sweep.

## Robust and Parameter-Stability Follow-Up

Robust validation was run on the same accepted configuration with
`--profile robust --resume-existing`, adding only the zero-cost diagnostic to
the already completed standard scenarios.

Robust scenario results:

| scenario | total return | max DD | gross turnover | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_zero_cost | 0.450597 | -0.302286 | 122.28 | 0 | 24295 |
| full_base | 0.231632 | -0.327544 | 121.68 | 162805 | 24101 |
| full_high_cost | 0.175710 | -0.336608 | 121.53 | 204322 | 24070 |
| year_2023_base | 0.001264 | -0.128464 | 41.56 | 53826 | 7836 |
| year_2024_base | 0.012112 | -0.230205 | 41.75 | 54844 | 8240 |
| year_2025_base | 0.137951 | -0.133289 | 39.36 | 53258 | 7704 |

Robust validation status: `pass`, with `0` failures and `0` warnings. The
zero-cost to high-cost spread confirms that costs are meaningful, but the
high-cost scenario remains strongly positive. This reinforces the previous
conclusion that the 2024 failures were not primarily a transaction-cost
problem.

A first health-shrink parameter screen then tested whether the accepted
`lookback=20`, `min_scale=0.25` configuration sits in a broad stable region.
The screen varied lookback and min scale while keeping the stress gap-leg
schedule fixed.

Focused screen results:

| variant | 2023 return | 2024-01 return | 2024-06 return | 2023 max DD |
|---|---:|---:|---:|---:|
| lookback 12, min scale 0.4 | -0.010016 | -0.116785 | -0.076589 | -0.142878 |
| lookback 12, min scale 0.5 | -0.010044 | -0.116785 | -0.076589 | -0.142424 |
| lookback 36, min scale 0.4 | -0.013039 | -0.114302 | -0.075827 | -0.143038 |
| accepted: lookback 20, min scale 0.25 | 0.001264 | -0.115054 | -0.073961 | -0.128464 |

The remaining `lookback=36`, `min_scale=0.5` run was stopped after the
`lookback=36`, `min_scale=0.4` 2023 slice failed; the partial output was
removed.

Decision: robust validation passes, but the health-shrink parameter screen is a
caution. The accepted configuration does not yet look like a broad plateau:
short lookback (`12`) and long lookback (`36`) both fail the 2023 blocking
slice even though they preserve 2024 pressure-month protection. Before
production promotion, run a narrower stability map around the accepted
configuration, especially `lookback=16/20/24/28` and `min_scale=0.20/0.25/0.30`,
with the same 2023, 2024-01, and 2024-06 admission screen. If only
`lookback=20,min_scale=0.25` passes, treat the allocator as promising but
parameter-fragile rather than production-ready.

## Narrow Health-Shrink Stability Map

The next screen narrowed the parameter search around the accepted
`lookback=20`, `min_scale=0.25` setting. The stress gap-leg weight schedule,
lagged health label, and portfolio policy were unchanged. The first pass used
the 2023 blocking slice because prior failed variants all failed there first.

2023 focused results:

| variant | 2023 return | 2023 max DD | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| lookback 16, min scale 0.20 | -0.001048 | -0.128769 | 41.57 | 53809 |
| lookback 16, min scale 0.25 | -0.000387 | -0.135092 | 41.56 | 53832 |
| lookback 16, min scale 0.30 | 0.005265 | -0.131250 | 41.60 | 53971 |
| lookback 20, min scale 0.20 | 0.000099 | -0.125900 | 41.58 | 53912 |
| accepted: lookback 20, min scale 0.25 | 0.001264 | -0.128464 | 41.56 | 53826 |
| lookback 20, min scale 0.30 | 0.000378 | -0.128596 | 41.59 | 53902 |

The automatic grid was stopped after `lookback=20,min_scale=0.30` completed;
24/28 window variants were not rerun because the wider `lookback=36` screen had
already shown that longer health memory degraded the 2023 slice.

The two most informative variants were then run on the pressure months:

| variant | slice | return | max DD | gross turnover | transaction cost |
|---|---|---:|---:|---:|---:|
| accepted: lookback 20, min scale 0.25 | 2023 | 0.001264 | -0.128464 | 41.56 | 53826 |
| accepted: lookback 20, min scale 0.25 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| accepted: lookback 20, min scale 0.25 | 2024-06 | -0.077267 | -0.090585 | 5.03 | 4843 |
| lookback 16, min scale 0.30 | 2023 | 0.005265 | -0.131250 | 41.60 | 53971 |
| lookback 16, min scale 0.30 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| lookback 16, min scale 0.30 | 2024-06 | -0.082442 | -0.092048 | 5.02 | 4821 |
| lookback 20, min scale 0.20 | 2023 | 0.000099 | -0.125900 | 41.58 | 53912 |
| lookback 20, min scale 0.20 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| lookback 20, min scale 0.20 | 2024-06 | -0.075454 | -0.089789 | 5.02 | 4850 |

Decision: the accepted configuration is no longer an isolated single point,
but the plateau is still narrow. `lookback=16,min_scale=0.30` improves the 2023
return but gives back protection in June 2024, so it should not replace the
accepted configuration. `lookback=20,min_scale=0.20` has the best drawdown on
2023 and June 2024, but its 2023 return is only barely positive. Treat it as a
conservative sensitivity variant rather than a superior production candidate.

Research implication: the market-state gate plus lagged factor-health shrink is
still the best direction found in this batch, but its remaining weakness is
state-memory calibration. The next useful experiment is not another static
factor sweep; it is an ensemble or adaptive health-memory rule that blends
nearby lookbacks instead of selecting a single brittle window.

## Ensemble Health-Memory Prototype

Implemented an ensemble health-memory option in the candidate portfolio runner:

- New CLI option: `--factor-health-ensemble-lookbacks`, for example `16,20`
- New CLI option: `--factor-health-ensemble-combine-mode`, with `mean`, `min`,
  or `max`
- The existing single-window path remains unchanged when the ensemble option is
  omitted.
- The first implementation builds one lagged health schedule per lookback and
  blends the final health scores. This is acceptable for research screens, but
  production use should refactor it into a single-read, multi-rolling
  implementation.

Prototype outputs:

- `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_health_ensemble_lb16_20_mean_ms025_year2023`
- `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_health_ensemble_lb16_20_mean_ms025_2024_01`
- `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_health_ensemble_lb16_20_mean_ms025_2024_06`
- `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_health_ensemble_lb16_20_mean_ms020_year2023`
- `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_health_ensemble_lb16_20_mean_ms020_2024_01`
- `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_health_ensemble_lb16_20_mean_ms020_2024_06`

Focused screen results:

| variant | slice | return | max DD | gross turnover | transaction cost |
|---|---|---:|---:|---:|---:|
| accepted: lookback 20, min scale 0.25 | 2023 | 0.001264 | -0.128464 | 41.56 | 53826 |
| accepted: lookback 20, min scale 0.25 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| accepted: lookback 20, min scale 0.25 | 2024-06 | -0.077267 | -0.090585 | 5.03 | 4843 |
| ensemble mean 16/20, min scale 0.25 | 2023 | 0.000391 | -0.128493 | 41.56 | 53787 |
| ensemble mean 16/20, min scale 0.25 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| ensemble mean 16/20, min scale 0.25 | 2024-06 | -0.077237 | -0.089773 | 5.03 | 4829 |
| ensemble mean 16/20, min scale 0.20 | 2023 | 0.000434 | -0.125900 | 41.57 | 53898 |
| ensemble mean 16/20, min scale 0.20 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| ensemble mean 16/20, min scale 0.20 | 2024-06 | -0.077267 | -0.090585 | 5.03 | 4843 |

Decision: the ensemble prototype reduces single-window brittleness but does not
create a clearly superior candidate yet. The `16/20 mean, min_scale=0.20`
variant gives the best 2023 drawdown while preserving the pressure-month
profile, but its 2023 return is only barely positive. The `16/20 mean,
min_scale=0.25` variant is almost identical to the accepted configuration, with
a small June 2024 drawdown improvement.

Research implication: blending nearby health-memory windows is viable as a
stability control, but the current equal-weight ensemble is too conservative to
raise expected return. The next research step should test asymmetric or
state-conditioned ensemble combinations, for example using the faster 16-window
only when stress diffusion is elevated and otherwise defaulting to the more
stable 20-window configuration.

## State-Conditioned Health Memory

The next implementation added a standalone schedule builder:

`examples/build_state_conditioned_factor_health_schedule.py`

It takes a normal health schedule, a stress health schedule, and an observable
regime schedule. The initial regime proxy was the existing
`intraday_overnight_gap_5m` row in
`stress_gap_leg_weight_scale.csv`. Two modes are supported:

- `select`: hard-switch to the stress health schedule when the regime selector
  scale is below a threshold.
- `blend`: linearly blend normal and stress schedules using
  `1 - regime_selector_scale` as the regime probability.

The first test used `accepted` as the normal schedule and `ensemble 16/20,
min_scale=0.20` as the stress schedule. It produced the same 2023 backtest path
as `accepted`, despite score-level differences. A stronger stress schedule
(`lookback=16,min_scale=0.30`) also produced the same 2023 backtest path when
used only during stress states. This indicates that the 2023 improvement from
the faster health memory does not come from the stress-state timestamps.

The direction was therefore reversed: use `lookback=16,min_scale=0.30` in
normal/calm states, but fall back to the accepted `lookback=20,min_scale=0.25`
configuration during stress states.

Focused screen results:

| variant | slice | return | max DD | gross turnover | transaction cost |
|---|---|---:|---:|---:|---:|
| accepted: lookback 20, min scale 0.25 | 2023 | 0.001264 | -0.128464 | 41.56 | 53826 |
| accepted: lookback 20, min scale 0.25 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| accepted: lookback 20, min scale 0.25 | 2024-06 | -0.077267 | -0.090585 | 5.03 | 4843 |
| lookback 16, min scale 0.30 | 2023 | 0.005265 | -0.131250 | 41.60 | 53971 |
| lookback 16, min scale 0.30 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| lookback 16, min scale 0.30 | 2024-06 | -0.082442 | -0.092048 | 5.02 | 4821 |
| state select: normal 16/0.30, stress 20/0.25 | 2023 | 0.005265 | -0.131250 | 41.60 | 53971 |
| state select: normal 16/0.30, stress 20/0.25 | 2024-01 | -0.115054 | -0.117856 | 5.51 | 5542 |
| state select: normal 16/0.30, stress 20/0.25 | 2024-06 | -0.076264 | -0.089611 | 5.03 | 4834 |
| state blend: normal 16/0.30, stress 20/0.25 | 2023 | 0.005265 | -0.131250 | 41.60 | 53971 |
| state blend: normal 16/0.30, stress 20/0.25 | 2024-06 | -0.078157 | -0.089894 | 5.03 | 4813 |

Decision: this is the first state-conditioned health-memory variant that
improves the accepted pressure-month profile while keeping the stronger 2023
return from the faster memory. The hard `select` mode is better than soft
`blend` on June 2024 in this screen. This supports the original thesis: the
useful rule is not a universal health-memory parameter, but a market-state
conditional rule. The sign of the rule is also informative: faster health memory
helps in normal/calm states, while stress states should fall back to the more
conservative accepted memory.

Next validation step: build the same state-conditioned schedule over the full
2023-2025 validation span and run standard/robust validation. The current
implementation is a research screen because it combines prebuilt schedules by
slice; production promotion requires generating both component health memories
inside one full-window pipeline.

## Full-Window State-Conditioned Validation

The slice-level state-conditioned rule was then rebuilt over the full
2023-2025 span:

- Normal/calm schedule:
  `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_health_narrow_lb16_ms030_full_scores/factor_health/factor_health_schedule.csv`
- Stress fallback schedule:
  `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_next_round_stress_gap_health_shrink_standard_decorrelated/full_base/factor_health/factor_health_schedule.csv`
- Full state-conditioned schedule:
  `runs/factor_research/regime_conditioned_2026_05_27/state_conditioned_health_lb16_ms030_to_lb20_ms025_select_full/factor_health_schedule.csv`
- Validation output:
  `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_state_conditioned_health_select_standard_decorrelated`

Full schedule summary:

| metric | value |
|---|---:|
| row count | 104397 |
| feature count | 3 |
| stress-regime weight mean | 0.268140 |
| final weight scale mean | 0.602075 |
| final weight scale median | 0.574913 |

Robust validation scenario results:

| scenario | total return | max DD | gross turnover | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_zero_cost | 0.471292 | -0.299759 | 122.37 | 0 | 24326 |
| full_base | 0.249559 | -0.327245 | 121.90 | 163414 | 24140 |
| full_high_cost | 0.193803 | -0.336601 | 121.67 | 205317 | 24096 |
| year_2023_base | 0.005265 | -0.131250 | 41.60 | 53971 | 7853 |
| year_2024_base | 0.017663 | -0.231099 | 41.60 | 54028 | 8095 |
| year_2025_base | 0.146337 | -0.130685 | 39.32 | 53327 | 7717 |

Robust validation status: `pass`, with `0` failures and `0` warnings. Relative
to the accepted `lookback=20,min_scale=0.25` allocator, this state-conditioned
variant improves full-base return from `0.231632` to `0.249559`, high-cost
return from `0.175710` to `0.193803`, 2023 from `0.001264` to `0.005265`, 2024
from `0.012112` to `0.017663`, and 2025 from `0.137951` to `0.146337`.

Stress-month diagnostics:

| scenario | month | return | max DD | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 2024-01 | -0.139863 | -0.150024 | 4907 | 738 |
| full_base | 2024-06 | -0.097963 | -0.104576 | 4089 | 637 |
| year_2024_base | 2024-01 | -0.119102 | -0.121807 | 5487 | 719 |
| year_2024_base | 2024-06 | -0.080047 | -0.089452 | 4000 | 609 |
| full_high_cost | 2024-01 | -0.141073 | -0.151175 | 6063 | 731 |
| full_high_cost | 2024-06 | -0.098706 | -0.105378 | 4935 | 633 |

Decision: this is now the strongest full-window allocator in the batch on
overall return, yearly stability, and high-cost robustness. The result supports
the original regime-conditioned thesis: use a faster health memory in
normal/calm states and fall back to the accepted slower memory when the
observable stress proxy is active.

Caveat: stress-month diagnostics are mixed. Full-base June 2024 improves versus
the accepted allocator, but the isolated `year_2024_base` January and June
diagnostics are weaker than the accepted year-specific run. This is another
path-dependence signal in the lagged health state. Treat the allocator as a
promotable research candidate, but require one more productionization step:
generate both health memories inside the portfolio runner from the same
scenario window instead of stitching prebuilt schedules, then rerun robust
validation.

## Productionized State-Conditioned Health Memory

The state-conditioned health-memory rule was then moved into the candidate
portfolio runner so each validation scenario generates both component health
memories from the same scenario window:

- Normal/calm health memory: `lookback=16`, `min_scale=0.30`
- Stress fallback health memory: `lookback=20`, `min_scale=0.25`
- State selector: `intraday_overnight_gap_5m` row from
  `stress_gap_leg_weight_scale.csv`
- Selector mode: hard `select` with threshold `0.999`
- Final combination: `min` with the stress gap-leg weight schedule

Implementation changes:

- `build_state_conditioned_factor_health_schedule(...)` now lives in the
  portfolio layer and combines normal/stress health schedules with an
  observable regime schedule.
- `examples/run_candidate_factor_portfolios.py` can now build scenario-local
  state-conditioned health schedules via `--factor-health-state-regime-mode`
  and `--factor-health-stress-*` parameters.
- `examples/run_candidate_policy_validation.py` forwards the new parameters,
  so robust validation no longer needs a prebuilt stitched health CSV.

Productionized robust validation output:

`runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_state_conditioned_health_internal_robust_decorrelated`

Full-base generated schedule summary:

| metric | value |
|---|---:|
| row count | 104397 |
| feature count | 3 |
| stress-regime weight mean | 0.268140 |
| final weight scale mean | 0.602075 |
| final weight scale median | 0.574913 |

Full-base feature scale distribution:

| feature | min | mean | median | max |
|---|---:|---:|---:|---:|
| `intraday_false_absorption_risk_5m_w48` | 0.250000 | 0.665280 | 0.720756 | 1.000000 |
| `intraday_overnight_gap_5m` | ~0.000000 | 0.506731 | 0.397222 | 1.000000 |
| `intraday_overnight_gap_down_recovery_5m` | 0.250000 | 0.634215 | 0.650890 | 1.000000 |

Robust validation scenario results:

| scenario | total return | max DD | gross turnover | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 0.249559 | -0.327245 | 121.90 | 163414 | 24140 |
| year_2023_base | 0.005265 | -0.131250 | 41.60 | 53971 | 7853 |
| year_2024_base | 0.016717 | -0.230205 | 41.78 | 54916 | 8250 |
| year_2025_base | 0.146337 | -0.130685 | 39.32 | 53327 | 7717 |
| full_high_cost | 0.193803 | -0.336601 | 121.67 | 205317 | 24096 |
| full_zero_cost | 0.471292 | -0.299759 | 122.37 | 0 | 24326 |

Robust validation status: `pass`, with `0` failures and `0` warnings.

Stress-month diagnostics:

| scenario | month | return | max DD | transaction cost | trades |
|---|---:|---:|---:|---:|---:|
| full_base | 2024-01 | -0.139863 | -0.150024 | 4907 | 738 |
| full_base | 2024-06 | -0.097963 | -0.104576 | 4089 | 637 |
| year_2024_base | 2024-01 | -0.115054 | -0.117856 | 5542 | 729 |
| year_2024_base | 2024-06 | -0.073966 | -0.081351 | 4119 | 631 |
| full_high_cost | 2024-01 | -0.141073 | -0.151175 | 6063 | 731 |
| full_high_cost | 2024-06 | -0.098706 | -0.105378 | 4935 | 633 |

Decision: the productionized implementation confirms the previous external
schedule result. Full-window base, high-cost, and zero-cost results are
effectively unchanged, while the scenario-local 2024 slice changes modestly
because it now builds health memory only from the 2024 scenario window. The
variant remains a promotable research candidate: positive in every yearly
slice, high-cost robust, and turnover remains below the `160` guardrail.

Remaining engineering note: the current implementation is correct but not yet
fast. State-conditioned health builds the normal and stress schedules in two
separate passes over the dataset. Production promotion should refactor this
into one read pass with multiple rolling windows, then register the allocator
configuration for ongoing monitoring.

## Health-Memory Performance Optimization

The remaining performance issue was addressed by separating factor-health
processing into two stages:

1. Read each dataset partition once and build timestamp-level factor-health
   observations.
2. Reuse those observations to generate one or more rolling health schedules.

This changes both multi-lookback paths:

- `build_factor_health_ensemble_schedule(...)` now reads each partition once
  and reuses observations for all ensemble lookbacks.
- `build_state_conditioned_factor_health_schedule_from_partitions(...)` builds
  normal/calm and stress fallback schedules from the same observations, then
  applies the regime selector. The state-conditioned allocator no longer reads
  the same parquet partitions separately for normal and stress memory.

Implementation entry points:

- `quant_research.portfolio.factor_portfolios._factor_health_observation_frame`
- `quant_research.portfolio.factor_portfolios._factor_health_schedule_from_observations`
- `quant_research.portfolio.factor_portfolios.build_state_conditioned_factor_health_schedule_from_partitions`

Validation:

- Unit tests assert that ensemble health and state-conditioned health read each
  input partition exactly once.
- A real-data smoke run generated a one-partition state-conditioned schedule
  and score file successfully:
  `runs/candidate_factor_portfolios/regime_conditioned_2026_05_27_state_conditioned_health_single_read_smoke`

Performance decision: the allocator is now productionizable from a data-access
standpoint. Further optimization can still vectorize parts of the rolling
calculation, but the largest avoidable IO cost has been removed.
