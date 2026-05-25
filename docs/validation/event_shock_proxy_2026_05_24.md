# Event Shock Proxy Acceptance - 2026-05-24

## Dataset

- Dataset: `runs/factor_research/event_shock_proxy_2026_05_24/alpha_dataset`
- Factor group: `event_shock_proxy`
- Event shock windows: `48`
- Sample window: `2023-01-03 09:35:00+08:00` to `2025-12-31 15:00:00+08:00`
- Partitions: `36`
- Dataset rows: `103,351,780`
- Duplicate `(timestamp, instrument_id)` keys: `0`
- `forward_return` null count: `0`
- QA artifact: `runs/factor_research/event_shock_proxy_2026_05_24/alpha_dataset/event_shock_dataset_qa.csv`

Feature coverage was clean enough for standard admission. The minimum partition
coverage was `98.86%` for synchronized-downside resilience, `100.00%` for limit
diffusion resilience, `93.48%` for turnover-dislocation recovery in the first
month because of the 96-bar effective warmup, and `99.996%` for open-jump
recovery quality. Full-window coverage for all four features was above the
`95%` admission gate.

## Admission

- Factor evaluation: `runs/factor_research/event_shock_proxy_2026_05_24/factor_evaluation/summary.json`
- Admission report: `runs/factor_research/event_shock_proxy_2026_05_24/factor_admission/factor_admission_report.json`
- Admission table: `runs/factor_research/event_shock_proxy_2026_05_24/factor_admission/factor_admission_table.csv`
- Candidate review, limit diffusion: `runs/factor_candidate_reviews/intraday_event_limit_diffusion_resilience_5m_w48/factor_candidate_review.json`
- Candidate review, turnover dislocation: `runs/factor_candidate_reviews/intraday_event_turnover_dislocation_recovery_5m_w48/factor_candidate_review.json`
- Candidate review, sync down: `runs/factor_candidate_reviews/intraday_event_sync_down_resilience_5m_w48/factor_candidate_review.json`
- Candidate review, open jump: `runs/factor_candidate_reviews/intraday_event_open_jump_recovery_quality_5m_w48/factor_candidate_review.json`

| feature | decision | direction | rank IC | t-stat | hit rate | cost-adjusted spread | stable years | failed checks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `intraday_event_limit_diffusion_resilience_5m_w48` | `candidate` | `invert` | -0.011386 | -15.1538 | 0.546970 | 0.000681 | 2 | - |
| `intraday_event_turnover_dislocation_recovery_5m_w48` | `watchlist` | `invert` | -0.016964 | -23.0591 | 0.572945 | -0.001370 | 3 | `cost_adjusted_spread` |
| `intraday_event_sync_down_resilience_5m_w48` | `reject` | `long` | 0.000978 | 1.0412 | 0.497470 | 0.000491 | 1 | `abs_rank_ic_mean`, `abs_rank_ic_t_stat`, `directional_ic_hit_rate`, `stable_year_count` |
| `intraday_event_open_jump_recovery_quality_5m_w48` | `reject` | `invert` | -0.000822 | -1.5418 | 0.524440 | -0.002230 | 2 | `abs_rank_ic_mean`, `abs_rank_ic_t_stat`, `cost_adjusted_spread` |

## Portfolio Validation

Portfolio-level validation was run only for the inverted limit-diffusion member
because it was the sole admission candidate.

- Validation summary: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_standard_primary/validation_summary.json`
- Scenario table: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_standard_primary/validation_summary.csv`
- Factor-health diagnostics: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_standard_primary/validation_factor_health_summary.csv`

Overall result: `fail`.

| scenario | total return | max drawdown | gross turnover | total transaction cost | final equity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_base` | -0.008138 | -0.363312 | 115.711448 | 151616.45 | 991862.46 |
| `year_2023_base` | -0.105487 | -0.164583 | 39.773125 | 51694.13 | 894513.04 |
| `year_2024_base` | -0.089842 | -0.371290 | 39.882823 | 50061.42 | 910157.69 |
| `year_2025_base` | 0.122431 | -0.170140 | 40.599533 | 54507.32 | 1122431.33 |
| `full_high_cost` | -0.052705 | -0.374866 | 115.307357 | 185062.12 | 947295.12 |

The hard failures were `primary_full_base_positive_return` and
`primary_full_high_cost_positive_return`. Annual stability also warned because
2023 and 2024 were negative. Turnover control passed, so the failure is not a
turnover-budget issue; it is an economic and drawdown issue. Factor-health
monitoring also showed many impaired observations in the full window
(`18,438` impaired observations out of `34,799`), with frequent rolling IC,
spread, and top-label deterioration.

### Control Follow-Ups

Two controlled follow-ups were run to test whether the admitted inverted signal
could be rescued without changing the factor definition.

- Health-shrink validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_health_shrink_standard/validation_summary.json`
- Downside-volatility gate validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_downside_vol_gate_standard/validation_summary.json`
- Downside-volatility gate schedule: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_downside_vol_gate_standard/factor_risk_gate/downside_volatility_w48/summary.json`

Both controls failed.

| control | overall | full return | high-cost return | 2023 | 2024 | 2025 | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| factor-health shrink | `fail` | -0.008138 | -0.052705 | -0.105487 | -0.089842 | 0.122431 | Shrink lowered the single-leg score scale but did not change the single-factor ranking or trading path. |
| downside-volatility exposure gate | `fail` | -0.091824 | -0.142823 | -0.136620 | -0.005661 | 0.004899 | The gate reduced average exposure to about `0.76`, but raised turnover to `159.52` and worsened full/high-cost economics. |

The downside-volatility gate used the previously validated
`intraday_downside_volatility_5m_w48` market-wide risk proxy with lagged rolling
thresholds. Its schedule had `26,472` full, `3,978` reduced, `4,301` blocked,
and `48` warmup observations. This generic exposure control is therefore not a
valid rescue path for this factor.

### Event-State Regime Diagnostic

A timestamp-level event-state diagnostic was added after the generic controls
failed.

- Diagnostic summary: `runs/factor_research/event_shock_proxy_2026_05_24/event_state_regime_diagnostics/summary.json`
- Event-state table: `runs/factor_research/event_shock_proxy_2026_05_24/event_state_regime_diagnostics/event_state_performance.csv`
- Monthly state summary: `runs/factor_research/event_shock_proxy_2026_05_24/event_state_regime_diagnostics/monthly_event_state_summary.csv`
- Diagnostic report: `runs/factor_research/event_shock_proxy_2026_05_24/event_state_regime_diagnostics/event_state_regime_report.md`

The diagnostic classifies each 5-minute timestamp using only lagged market
observables: price-limit pressure, limit-down imbalance, and cross-sectional
dispersion of the four event-shock proxy features. Rolling z-scores are
clipped at `6.0` for report stability. This does not change the raw factor
values.

| state | timestamps | share | top-N label | top-universe | score IC | intensity | limit pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `limit_diffusion_extreme` | 4,928 | 14.16% | -0.71% | -0.61% | 0.0218 | 1.5346 | 1.39% |
| `limit_diffusion` | 3,054 | 8.78% | -0.64% | -0.41% | 0.0130 | 0.7270 | 0.29% |
| `calm` | 18,887 | 54.27% | -0.30% | -0.36% | 0.0077 | 0.1636 | 0.13% |
| `post_shock_stabilization` | 5,375 | 15.45% | -0.23% | -0.31% | 0.0135 | 0.3826 | 0.11% |
| `shock_elevated` | 2,065 | 5.93% | -0.22% | -0.32% | 0.0122 | 0.9949 | 0.19% |
| `shock_extreme` | 442 | 1.27% | 0.22% | -0.31% | -0.0055 | 1.8674 | 1.08% |

The failure is therefore more specific than a generic downside-volatility
problem. The most toxic states for the selected top-N names were
`limit_diffusion_extreme` and `limit_diffusion`. `shock_extreme` had only `442`
timestamps and did not show the same aggregate top-N loss, so a simple
"block all shock" rule is not supported by this diagnostic.

The worst month, `2024-01`, had a `-16.89%` portfolio return, `-1.40%` top-N
label, `-0.58%` top-minus-universe label, and `-0.0738` score rank IC. Its state
mix included `16.10%` `limit_diffusion_extreme`, `16.76%` `limit_diffusion`,
and `17.42%` `post_shock_stabilization`. Other losing months such as
`2024-06`, `2024-12`, `2024-08`, `2023-08`, and `2024-04` also had negative
top-N labels, but the state mix was not always dominated by a single regime.

### Event-State Gate Follow-Ups

Two fixed, point-in-time event-state exposure schedules were tested. Both use
the prior 5-minute timestamp's event state and do not use month, year, or
portfolio-return information in the rule.

- Gate builder: `examples/build_event_state_exposure_schedule.py`
- Half-block gate validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_event_state_gate_standard/validation_summary.json`
- Half-block gate schedule: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_event_state_gate_standard/event_state_exposure_gate/summary.json`
- Full-block gate validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_event_state_block_limit_standard/validation_summary.json`
- Full-block gate schedule: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_event_state_block_limit_standard/event_state_exposure_gate/summary.json`
- Full-block plus drawdown-brake validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_event_state_block_limit_drawdown_brake_standard/validation_summary.json`

| control | overall | full return | high-cost return | 2023 | 2024 | 2025 | full turnover | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lagged `limit_diffusion_extreme` block, `limit_diffusion` 0.5x | `fail` | 0.052913 | -0.001173 | -0.103918 | -0.024481 | 0.107290 | 130.60 | Full-window economics improved, but high-cost stayed slightly negative and 2023/2024 remained negative. |
| lagged `limit_diffusion` and `limit_diffusion_extreme` block | `warn` | 0.086797 | 0.025446 | -0.097806 | 0.002989 | 0.104472 | 136.92 | Full/high-cost gates cleared and 2024 turned slightly positive, but 2023 stayed materially negative. |
| lagged `limit_diffusion` and `limit_diffusion_extreme` block plus -7%/0.5x drawdown brake | `fail` | -0.079866 | -0.078060 | -0.115409 | -0.079991 | -0.076902 | 56.87 | Drawdown fell to `-21.45%` and turnover fell, but the path was over-delevered after losses and all annual slices were negative. |

The full-block schedule blocked `7,981` of `34,799` timestamp states
(`22.94%`) and left the rest at full exposure. It is a stronger and cleaner
validation of the cross-sectional limit-diffusion state than the generic
downside-volatility gate, but it is not a full promotion because yearly
stability remains incomplete and drawdown is still deep (`-34.15%` full-base
max drawdown, `-35.47%` high-cost max drawdown).

A fixed portfolio-level drawdown brake was then stacked on top of the same
lagged full-block event-state schedule. This was a general path-level control,
not a month-specific parameter patch. It failed: full-base return fell to
`-7.99%`, high-cost return fell to `-7.81%`, and 2023, 2024, and 2025 were all
negative. The lower drawdown is not a valid tradeoff here because the brake
mostly de-risked after losses and removed the remaining positive contribution.
Do not continue threshold searches on this drawdown-brake path for this factor.

### Fixed Complementary Selection Follow-Up

A general selection/replacement experiment was then run instead of another
gross-exposure threshold search. The event dataset was left as the base sample
and joined with four previously governed candidate features:
`intraday_sell_pressure_absorption_5m_w48`, `intraday_volatility_5m_w6`,
`intraday_amihud_5m`, and `intraday_efficiency_ratio_5m_w48`.
The selected features were fixed before validation because they are already
registry candidates from different mechanisms: sell-pressure/liquidity,
short-horizon risk, impact/liquidity, and price-path efficiency. The rule does
not depend on any month or year result.

- Joined dataset: `runs/factor_research/event_shock_proxy_2026_05_24/joined_selection_alpha_dataset`
- Joined dataset builder: `examples/build_joined_alpha_dataset.py`
- Joined admission report: `runs/factor_research/event_shock_proxy_2026_05_24/joined_selection_admission/factor_admission_report.json`
- Joined factor evaluation: `runs/factor_research/event_shock_proxy_2026_05_24/joined_selection_evaluation/summary.json`
- Joined correlation matrix: `runs/factor_research/event_shock_proxy_2026_05_24/joined_selection_evaluation/feature_correlation.csv`
- Standard validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_block_standard/validation_summary.json`
- Residual-risk diagnostic: `runs/factor_research/event_shock_proxy_2026_05_24/joined_selection_residual_risk/summary.json`
- Lagged factor-health shrink validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_standard/validation_summary.json`
- Lagged factor-health shrink residual-risk diagnostic: `runs/factor_research/event_shock_proxy_2026_05_24/joined_selection_health_shrink_residual_risk/summary.json`

The joined dataset retained the event-study base sample exactly:
`36` partitions and `103,351,780` rows. Minimum joined-feature coverage was
`99.89%` for sell-pressure absorption, `100.00%` for 6-bar volatility,
`99.88%` for Amihud, and `99.9996%` for 48-bar efficiency ratio.

The decorrelated weights were mechanically derived from the full joined
admission ICs and same-sample feature correlations:

| feature | weight |
| --- | ---: |
| `intraday_sell_pressure_absorption_5m_w48` | 0.4482 |
| `intraday_volatility_5m_w6` | 0.3461 |
| `intraday_event_limit_diffusion_resilience_5m_w48` | 0.0809 |
| `intraday_efficiency_ratio_5m_w48` | 0.0763 |
| `intraday_amihud_5m` | 0.0485 |

The portfolio kept the validated one-bar-lagged full block for
`limit_diffusion` and `limit_diffusion_extreme` states. It did not use a
drawdown brake.

| validation | overall | full return | high-cost return | 2023 | 2024 | 2025 | full turnover | max drawdown | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed complementary blend plus lagged event-state full block, health monitor only | `warn` | 0.430555 | 0.349927 | 0.088907 | -0.028863 | 0.309637 | 140.44 | -0.319455 | Full/high-cost and turnover gates passed. The combination fixed the 2023 loss and preserved 2025, but 2024 stayed slightly negative and drawdown remained high. |
| same blend and event-state block, with lagged factor-health shrink | `pass` | 0.338077 | 0.262703 | 0.056356 | 0.024503 | 0.361835 | 141.01 | -0.297943 | Full/high-cost, turnover, and all yearly gates passed. Full-window return is lower than monitor-only, but 2024 turns positive and drawdown improves. |

The monitor-only blend was the strongest gross-exposure follow-up but still had
residual 2024 instability. A residual-risk diagnostic was added to avoid
hand-tuning weights against 2024. The diagnostic joined monthly portfolio
returns, event-state shares, the actual lagged gate schedule, factor-health
states, and top-score factor contribution diagnostics. It found that losing
months were not caused by a one-off contribution-concentration spike:
`intraday_sell_pressure_absorption_5m_w48` was the largest contribution feature
in all `36` monitor-only months. The strongest generic warning variable was
entry-quality deterioration in the selected basket: monitor-only loss months
had average top-score label `-0.11%` versus `+0.15%` in non-loss months, and
`52.23%` negative top-label timestamps versus `40.40%` in non-loss months. The
lagged factor-health schedule is point-in-time safe for this use because it
uses matured labels shifted by `48` bars before computing rolling health.

The fixed factor-health shrink follow-up therefore changed only one general
control: the same lagged factor-health diagnostics that were previously
monitor-only were allowed to scale each factor leg between `0.25` and `1.0`.
It was not retuned by month or year. This passed standard validation with no
warnings. The tradeoff is explicit: full-base return falls from `+43.06%` to
`+33.81%`, and high-cost return falls from `+34.99%` to `+26.27%`, but 2024
improves from `-2.89%` to `+2.45%`, 2025 improves from `+30.96%` to
`+36.18%`, full-window drawdown improves from `-31.95%` to `-29.79%`, and all
yearly return gates are positive. This is a valid general robustness rule for
the complementary blend, not a standalone rescue of the raw event factor.

### Independent Robustness Follow-Up

The final fixed construction was then tested without changing factor weights,
event-state block rules, factor-health shrink thresholds, or calendar-specific
parameters.

- Robust validation: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_robust/validation_summary.json`
- Capacity 5% bar-participation stress: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_capacity_5pct_full_base/summary.json`
- Capacity 2% bar-participation stress: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_capacity_2pct_full_base/summary.json`
- Capacity diagnostic rerun summary: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_capacity_diagnostics_summary.json`
- Capacity 5% diagnostic rerun: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_capacity_5pct_diag_exst_full_base/summary.json`
- Capacity 2% diagnostic rerun: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_capacity_2pct_diag_exst_full_base/summary.json`
- Rebalance 24-bar sensitivity: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_rebalance_24_full_base_day_chunk/summary.json`
- Rebalance 96-bar sensitivity: `runs/candidate_factor_portfolios/event_limit_diffusion_2026_05_24_joined_selection_health_shrink_block_rebalance_96_full_base/summary.json`

The robust profile passed with no failures or warnings. It adds a zero-cost
upper-bound scenario to the standard profile, while preserving the same base,
high-cost, and yearly checks.

| cost profile | total return | max drawdown | gross turnover | total transaction cost |
| --- | ---: | ---: | ---: | ---: |
| zero cost, 48-bar rebalance | 0.691943 | -0.286787 | 141.87 | 0.00 |
| base cost, 48-bar rebalance | 0.338077 | -0.297943 | 141.01 | 247,863.26 |
| high cost, 48-bar rebalance | 0.262703 | -0.305607 | 140.65 | 298,459.95 |

The cost profile confirms that costs are a large drag, but the construction
keeps positive economics under the high-cost stress. This is materially better
than the raw event factor, whose standalone high-cost result was negative.

Two bar-participation capacity stresses were then run against the same 48-bar
base-cost construction. These checks used the explicit
`allow_same_bar_capacity` flag because the current open-price execution model
otherwise refuses to constrain trades with same-bar turnover/volume. Treat
these as bar-volume sensitivity diagnostics, not as a full production fill
guarantee.

| capacity stress | total return | max drawdown | gross turnover | total transaction cost |
| --- | ---: | ---: | ---: | ---: |
| no capacity cap | 0.338077 | -0.297943 | 141.01 | 247,863.26 |
| 5% max bar participation | 0.337597 | -0.298154 | 139.84 | 246,610.32 |
| 2% max bar participation | 0.343685 | -0.298035 | 135.59 | 242,995.22 |

The execution diagnostics were then extended to count capacity-bound events
and unfilled capacity notional in the same-bar stress runs. A rerun using the
same scores, 48-bar policy, event-state block, factor-health shrink, costs, and
`exclude_st` setting reproduced the 5% and 2% stress metrics exactly and added
the following binding statistics:

| capacity stress | limited events | capped events | zero-capacity events | unfilled notional | unfilled / traded notional | unfilled / desired notional in limited events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no capacity cap | 0 | 0 | 0 | 0.00 | 0.00% | 0.00% |
| 5% max bar participation | 271 | 249 | 22 | 1,358,464.86 | 0.91% | 45.37% |
| 2% max bar participation | 1,427 | 1,340 | 87 | 5,714,810.36 | 3.96% | 47.69% |

Capacity sensitivity is acceptable at this research size: both 5% and 2%
bar-participation caps keep returns positive, drawdown nearly unchanged, and
turnover slightly lower. The result does not justify increasing gross exposure;
it simply says the current top-N path is not fragile to these first-pass
participation caps. The new diagnostics show that the strict 2% cap does bind
more often, but the unfilled amount is still small relative to total traded
notional at the current `1,000,000` initial capital research size. This should
be monitored before scaling capital or widening the universe, and it remains a
same-bar bar-volume sensitivity rather than a production fill guarantee.
The allocator registry now records this as structured `monitor_only` governance
with warning thresholds: positive return, absolute drawdown no worse than
`35%`, unfilled capacity notional no more than `5%` of traded notional, and
unfilled notional no more than `55%` of desired notional inside capacity-bound
events. These thresholds are not used to change weights or exposure; they only
surface validation warnings if future capacity diagnostics deteriorate.

Fixed rebalance-cadence checks were run at 24, 48, and 96 bars. These are
symmetric robustness checks around the passing 48-bar cadence, not a search over
calendar months. The 24-bar rerun used day-level streaming chunks after a
month-chunk parquet read anomaly; the strategy, signal, cost, and gate rules
were unchanged.

| rebalance cadence | total return | max drawdown | gross turnover | total transaction cost | read |
| --- | ---: | ---: | ---: | ---: | --- |
| 24 bars | -0.012962 | -0.346616 | 305.43 | 503,963.05 | Fails economics; faster trading doubles turnover and costs. |
| 48 bars | 0.338077 | -0.297943 | 141.01 | 247,863.26 | Current passing baseline. |
| 96 bars | 0.425062 | -0.316918 | 73.73 | 135,104.05 | Higher return and lower turnover, but deeper drawdown than 48 bars. |

This cadence test supports the 48-bar baseline as the current governed default.
The 24-bar path is rejected. The 96-bar path is useful as a low-turnover
sensitivity result, but it should not replace 48 bars solely because its
full-window return is higher; the drawdown deterioration means it needs
separate future out-of-sample confirmation.

## Decisions

`intraday_event_limit_diffusion_resilience_5m_w48` moves from admission
candidate to governance watchlist after portfolio validation. The single-factor
evidence remains statistically useful in the inverted direction, but the raw
inverted score failed standalone standard portfolio validation with negative
full-window and high-cost returns, negative 2023 and 2024 annual slices, and a
deep full-window drawdown. Health shrink did not change the trading path, and
the downside-volatility exposure gate made the economics worse. Event-state
diagnostics show that `limit_diffusion` and `limit_diffusion_extreme` regimes
are the most toxic states for the selected top-N names. A fixed lagged
full-block gate for those two states cleared full-window and high-cost return
checks, but still left 2023 negative and drawdown high. Adding a fixed
drawdown brake reduced drawdown mechanically but failed full/high-cost and
yearly return checks, so the gross-exposure control path is exhausted for this
factor. A fixed complementary selection blend using already governed liquidity,
risk, and efficiency candidates plus the lagged event-state block produced the
best monitor-only result: full-base `+43.06%`, high-cost `+34.99%`, and 2023
`+8.89%`, but it remained warning-level because 2024 was `-2.89%`. The final
fixed follow-up kept the same blend and event-state block and enabled the
existing lagged factor-health shrink. That standard validation passed:
full-base `+33.81%`, high-cost `+26.27%`, 2023 `+5.64%`, 2024 `+2.45%`, and
2025 `+36.18%`, with full-window max drawdown improved to `-29.79%`. The raw
event factor is not eligible as a standalone portfolio leg, but the event-state
block plus complementary blend plus lagged factor-health shrink is a validated
portfolio construction path. Independent robustness checks confirmed the fixed
construction under the robust profile with zero failures or warnings, positive
high-cost return, and positive 5%/2% bar-participation capacity stresses. The
24-bar rebalance sensitivity failed after costs, while the 96-bar sensitivity
raised return but worsened drawdown. Keep 48 bars as the governed default unless
future out-of-sample evidence supports a slower cadence.

`intraday_event_turnover_dislocation_recovery_5m_w48` moves to watchlist. Its
inverted rank IC is strong and stable across all three years, but the top/bottom
economic spread conflicts with the inverted direction and is negative after the
standard 13 bps cost adjustment. Use it only for targeted risk-overlay or
conditional experiments until a design clears the cost-adjusted spread gate.

`intraday_event_sync_down_resilience_5m_w48` is rejected as a standalone alpha.
The full-window rank IC was below the hard threshold, the t-stat was weak, the
directional hit rate was below 50%, and only one annual slice matched the
selected long direction.

`intraday_event_open_jump_recovery_quality_5m_w48` is rejected. The direction
selected by admission was inverted, but IC strength and t-stat were below hard
thresholds and the cost-adjusted spread was negative.

## Completion Status and Handoff

Do not rerun the rejected synchronized-downside or open-jump variants without a
materially different event-state definition. The turnover-dislocation watchlist
entry can be retried only as a risk gate or with an execution-aware transform
that fixes the cost-adjusted spread sign. Do not continue the raw standalone
limit-diffusion portfolio path, the current downside-volatility gross-exposure
gate, the fixed drawdown-brake path, or parameter searches against specific
months. The validated continuation path is the fixed complementary blend with
the lagged `limit_diffusion`/`limit_diffusion_extreme` event-state block and
lagged factor-health shrink, using the 48-bar rebalance baseline. This path has
now been packaged as
`event_limit_diffusion_complementary_health_shrink_48b` in
`configs/allocators/candidate_allocator_registry.json`.

The round is complete as of `2026-05-25`. The allocator registry validates
cleanly, the capacity diagnostics are registered as warning-only governance,
the monitoring report is generated by
`examples/generate_allocator_monitoring_report.py`, and daily paper monitoring
is available through `examples/run_allocator_daily_monitoring.py`. Daily
history rows include `run_id` and `mode`, and same-allocator/same-run reruns
replace the existing row so retry attempts do not create artificial sustained
warnings. The current monitoring state is expected to be `warn`, not `fail`,
because the latest sample still has an active event-state block and active
factor-health shrink; registry validation, portfolio validation, and capacity
thresholds pass.

The handoff state is therefore: no further in-sample parameter search for this
round, no standalone promotion of the raw event factor, no manual weight or
cadence retuning, and no calendar-month tuning. Future work belongs to
out-of-sample paper/live-simulation monitoring on newly arriving data, plus a
separate promotion review only after the daily monitoring ledger contains enough
new observations.
