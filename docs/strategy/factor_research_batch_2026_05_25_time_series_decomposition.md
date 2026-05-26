# Factor Research Batch - 2026-05-25 Time-Series Decomposition

This batch follows the reversal overlay review. The conclusion from the EOD
24-bar reversal work was that another plain cross-sectional reversal variant is
unlikely to clear portfolio gates. The new branch therefore tests time-series
state features that are more orthogonal to the current cross-sectional compact
core.

## Hypothesis

The current compact core is dominated by cross-sectional ranking signals:
low volatility, low path inefficiency, liquidity cost, and sell-pressure
absorption. Those signals can cluster in trend-down stress. Time-series
decomposition should add a different axis by describing how each instrument's
own intraday state is changing:

- Volatility state change: acceleration or deceleration of realized volatility.
- Volume distribution shape: whether volume is concentrated, bursty, or
  back-loaded inside a rolling window.
- Microstructure recovery speed: whether positive returns are arriving after
  downside turnover pressure.

All implemented features are point-in-time rolling features. The volume shape
features deliberately use rolling windows only; no full-session volume share is
used.

## Implemented Feature Groups

| group | features | design |
|---|---|---|
| `volatility_state_change` | `intraday_volatility_state_change_5m_s12_l48`, `s24_l96`; `intraday_volatility_state_trend_5m_s12_l48`, `s24_l96` | Compare short-window realized volatility with longer-window realized volatility, plus the short-window volatility change normalized by long volatility. |
| `volume_distribution_shape` | `intraday_volume_burstiness_5m_w{24,48,96}`; `intraday_volume_back_loaded_5m_w{24,48,96}`; `intraday_volume_concentration_5m_w{24,48,96}`; `intraday_volume_u_shape_5m_w{24,48,96}` | Rolling volume z-score magnitude, recent-half versus prior-half volume balance, Herfindahl-like concentration, and a burstiness plus concentration composite. |
| `microstructure_recovery_speed` | `intraday_microstructure_recovery_speed_5m_w{24,48}`; `intraday_microstructure_recovery_acceleration_5m_s12_l48`, `s24_l96` | Positive-return recovery relative to downside-return pressure, enhanced when recent downside turnover pressure has faded; acceleration is short recovery speed minus long recovery speed. |

## Dataset And Evaluation

Dataset build:

- Output: `research_store/time_series_decomposition_2026_05_25_alpha_dataset`
- Window: 2023-01-01 09:30 to 2025-12-31 15:00 Asia/Shanghai
- Partitions: 36 monthly parquet files
- Rows: 103,428,197 joined feature/label rows
- Feature rows before label join: 109,320,078
- Label: `forward_return`, horizon 48 bars, entry lag 1 bar
- Entry filters: ST excluded, price-limit aware, entry tradability and entry
  limit-up filters enabled

Evaluation and admission:

- Evaluation: `research_store/time_series_decomposition_2026_05_25_factor_evaluation/summary.json`
- Admission: `runs/factor_research/time_series_decomposition_2026_05_25/factor_admission/factor_admission_report.json`
- Result: 7 candidates, 8 watchlist, 5 rejects under standalone admission

| feature | admission | direction | rank IC | t-stat | cost-adj spread | turnover | note |
|---|---|---|---:|---:|---:|---:|---|
| `intraday_volume_concentration_5m_w96` | candidate | invert | -0.0415 | -73.02 | 0.00791 | 0.044 | Strongest and lowest-turnover signal. |
| `intraday_volume_concentration_5m_w48` | candidate | invert | -0.0400 | -72.41 | 0.00773 | 0.065 | Very close to w96. |
| `intraday_volume_concentration_5m_w24` | candidate | invert | -0.0197 | -37.59 | 0.00470 | 0.139 | Still stable across all three years. |
| `intraday_volume_back_loaded_5m_w96` | candidate | invert | -0.0126 | -28.85 | 0.00266 | 0.070 | Useful but weaker than concentration. |
| `intraday_volume_back_loaded_5m_w48` | candidate | invert | -0.0077 | -15.99 | 0.00043 | 0.124 | Marginal after costs. |
| `intraday_volatility_state_change_5m_s12_l48` | candidate | long | 0.0074 | 14.85 | 0.00033 | 0.210 | Passes but with small spread. |
| `intraday_volatility_state_trend_5m_s24_l96` | candidate | invert | -0.0069 | -13.74 | 0.00100 | 0.132 | Stable and positive after costs. |

Watchlist results:

- `volume_burstiness` and `volume_u_shape` variants have positive IC but fail
  the cost-adjusted spread gate because turnover is too high.
- `microstructure_recovery_speed` variants have intuitive direction but fail
  cost-adjusted spread; their annual stability is weaker than the volume shape
  factors.

Reject results:

- `intraday_volatility_state_trend_5m_s12_l48` fails the directional hit-rate
  gate.
- `intraday_microstructure_recovery_acceleration_5m_s12_l48`,
  `intraday_microstructure_recovery_acceleration_5m_s24_l96`,
  `intraday_volume_back_loaded_5m_w24`, and
  `intraday_volatility_state_change_5m_s24_l96` are too weak or negative after
  costs.

## Portfolio Validation

The 7 standalone admission candidates were scored as an isolated candidate
portfolio with `equal`, `ic_weighted`, and `decorrelated` weights.

Artifact:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_standard/summary.json`

| method | policy | total return | max drawdown | gross turnover |
|---|---|---:|---:|---:|
| `decorrelated` | `partial_rebalance_daily` | -13.98% | -35.55% | 111.78 |
| `decorrelated` | `cost_aware_optimizer_daily` | -27.45% | -33.00% | 523.72 |
| `equal` | `partial_rebalance_daily` | -5.02% | -32.77% | 109.41 |
| `equal` | `cost_aware_optimizer_daily` | -26.66% | -32.09% | 618.30 |
| `ic_weighted` | `partial_rebalance_daily` | -13.98% | -35.55% | 111.78 |
| `ic_weighted` | `cost_aware_optimizer_daily` | -27.45% | -33.00% | 523.72 |

The isolated portfolio result is negative despite strong single-factor IC. This
means the current formulation should not enter the automatic `candidate` pool.
The evidence is more consistent with a sorting or risk-control overlay than a
standalone long-only alpha basket.

## Compact-Core Overlay Check

The time-series decomposition equal-weight score was also overlaid on the
existing compact-core `decorrelated` score using rank-normalized blending.

Artifact:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_compact_core_overlay_quick/validation_summary.json`

| method | overlay weight | full base return | max drawdown | gross turnover |
|---|---:|---:|---:|---:|
| `tsd_equal_w00` | 0% | 42.92% | -32.34% | 111.60 |
| `tsd_equal_w02` | 2% | 32.84% | -31.89% | 112.05 |
| `tsd_equal_w05` | 5% | 35.86% | -31.35% | 112.65 |
| `tsd_equal_w10` | 10% | 23.09% | -31.40% | 112.13 |
| `tsd_equal_w15` | 15% | 26.84% | -32.72% | 112.32 |

The 0% control exactly reproduces the compact-core reference. Every positive
overlay weight reduces full-window return. This is a negative incremental
portfolio result, not a promotion result.

## Volume Concentration Overlay Follow-Up

Because rolling volume concentration was the strongest single-factor family in
the batch, it was retested separately as a compact-core satellite rather than as
part of the full 7-feature time-series-decomposition basket.

Artifacts:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_satellite/summary.json`
- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_equal_overlay_quick/validation_summary.json`
- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_penalty_overlay_quick/validation_summary.json`
- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_entry_exclusion_quick/validation_summary.json`

The satellite used only the three admitted concentration windows:

- `intraday_volume_concentration_5m_w24`
- `intraday_volume_concentration_5m_w48`
- `intraday_volume_concentration_5m_w96`

First, the concentration satellite was used as a small rank-normalized blend
against the compact-core `decorrelated` score.

| method | overlay weight | full base return | max drawdown | gross turnover |
|---|---:|---:|---:|---:|
| `vc_equal_w00` | 0.0% | 42.92% | -32.34% | 111.60 |
| `vc_equal_w005` | 0.5% | 37.19% | -32.36% | 111.22 |
| `vc_equal_w01` | 1.0% | 34.45% | -32.64% | 111.24 |
| `vc_equal_w02` | 2.0% | 29.72% | -32.23% | 111.96 |
| `vc_equal_w03` | 3.0% | 32.52% | -32.58% | 111.91 |
| `vc_equal_w05` | 5.0% | 32.38% | -31.80% | 111.65 |

Second, the satellite was tested as a downside-only risk penalty. This keeps the
compact-core primary score unchanged except for names in the weak lower tail of
the concentration satellite ranking.

| method | penalty weight | full base return | max drawdown | gross turnover |
|---|---:|---:|---:|---:|
| `vc_penalty_w00` | 0% | 42.92% | -32.34% | 111.60 |
| `vc_penalty_w01` | 1% | 35.35% | -32.05% | 111.52 |
| `vc_penalty_w02` | 2% | 37.72% | -32.18% | 111.55 |
| `vc_penalty_w05` | 5% | 38.01% | -31.83% | 111.53 |
| `vc_penalty_w10` | 10% | 34.71% | -31.97% | 111.47 |

Both tests are negative on incremental return. The best positive equal-blend
weight returned 37.19%, and the best positive downside-penalty weight returned
38.01%, versus the 42.92% compact-core control. Turnover did not improve enough
to justify the return loss.

Third, the satellite was tested as a hard pre-trade entry exclusion. This keeps
the compact-core score unchanged and uses the concentration satellite only to
mark the lower tail as ineligible for new entries.

| method | excluded lower-tail quantile | full base return | max drawdown | gross turnover |
|---|---:|---:|---:|---:|
| `vc_entry_filter_w00` | 0% | 42.92% | -32.34% | 111.60 |
| `vc_entry_filter_w05` | 5% | 39.07% | -32.28% | 111.25 |
| `vc_entry_filter_w10` | 10% | 36.29% | -31.94% | 111.51 |
| `vc_entry_filter_w20` | 20% | 37.61% | -32.19% | 111.27 |
| `vc_entry_filter_w30` | 30% | 32.20% | -33.62% | 111.63 |

The hard-exclusion branch is also negative. It gives up return faster than it
reduces drawdown or cost, so the volume concentration signal should not be used
as a direct entry filter under the current compact-core partial-rebalance
policy.

## Volume Concentration Optimizer Overlay

The next test moved the concentration satellite into the optimizer rather than
altering the primary rank. The compact-core score remained unchanged, while
names in the weak lower tail of the concentration satellite received an
additional `risk_penalty_bps` inside the cost-aware optimizer.

Artifacts:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_quick/validation_summary.json`
- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_budget155_quick/validation_summary.json`
- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_budget155_standard/validation_summary.json`

The unconstrained optimizer result was directionally promising but turnover
blocked. It increased return versus the optimizer control, but gross turnover
remained near 475 and therefore could not pass the portfolio gate.

| method | risk penalty scale | full base return | max drawdown | gross turnover |
|---|---:|---:|---:|---:|
| `vc_opt_risk_w00` | 0 bps | 27.23% | -24.32% | 472.75 |
| `vc_opt_risk_w25` | 25 bps | 33.06% | -22.82% | 475.09 |
| `vc_opt_risk_w50` | 50 bps | 31.55% | -22.59% | 476.42 |

The same overlay was then rerun with a path-level gross-turnover budget of 155,
which matches the existing portfolio admission constraint. The quick profile
passed all checks: `w25` improved full-base return by 1.21 percentage points,
reduced drawdown, lowered turnover, and reduced transaction cost versus the
budgeted optimizer control.

| method | full base return | max drawdown | gross turnover | transaction cost |
|---|---:|---:|---:|---:|
| `vc_opt_risk_w00` | 5.92% | -8.66% | 152.48 | 109,661 |
| `vc_opt_risk_w25` | 7.13% | -8.42% | 151.28 | 109,459 |
| `vc_opt_risk_w50` | 6.96% | -8.32% | 151.28 | 109,383 |

The standard profile kept the same ranking under the full-base scenario and all
annual base slices stayed positive. However, the doubled-cost full-window stress
remained negative for every method, so the standard validation status is `warn`
rather than `pass`.

| method | full base | full high cost | 2023 base | 2024 base | 2025 base | mean turnover |
|---|---:|---:|---:|---:|---:|---:|
| `vc_opt_risk_w00` | 5.92% | -4.32% | 5.55% | 0.31% | 24.73% | 152.50 |
| `vc_opt_risk_w25` | 7.13% | -3.43% | 6.82% | 2.94% | 24.49% | 151.99 |
| `vc_opt_risk_w50` | 6.96% | -3.08% | 6.96% | 3.17% | 24.21% | 151.89 |

This changes the concentration conclusion. Rank blending, downside score
penalties, and hard entry exclusion are rejected, but constrained optimizer
risk-penalty integration is a viable next-round branch. `w25` is the best
full-base choice; `w50` is slightly more defensive in drawdown and high-cost
stress. Neither should be promoted directly while high-cost return remains
negative.

## Volume Concentration Cost Robustness Follow-Up

The cost-robustness pass tested whether a tighter path-level turnover budget
could turn the doubled-cost stress positive without losing the useful
optimizer-risk-penalty behavior.

Artifacts:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_cost_robust_budget_grid/budget_grid_summary.json`
- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_budget100_standard/validation_summary.json`

The high-cost-only budget grid shows that uniform budget compression helps only
at the tightest tested level. At budgets of 120 and 140, all tested methods
remain negative under doubled costs.

| path turnover budget | `w00` high-cost return | `w25` high-cost return | `w50` high-cost return |
|---:|---:|---:|---:|
| 100 | -0.71% | 0.34% | 0.93% |
| 120 | -2.97% | -2.31% | -2.05% |
| 140 | -4.52% | -4.43% | -4.56% |

The budget-100 standard validation confirms that `w25` and `w50` can clear the
full-window high-cost stress. However, the same configuration fails annual
stability because the 2024 base slice turns materially negative.

| method | full base | full high cost | 2023 base | 2024 base | 2025 base | mean turnover |
|---|---:|---:|---:|---:|---:|---:|
| `vc_opt_risk_w00` | 5.99% | -0.71% | 6.00% | -15.01% | 18.38% | 98.96 |
| `vc_opt_risk_w25` | 7.16% | 0.34% | 7.13% | -12.71% | 18.42% | 98.46 |
| `vc_opt_risk_w50` | 7.22% | 0.93% | 7.18% | -12.35% | 17.90% | 98.40 |

This rejects fixed uniform budget compression as a promotion configuration.
Budget 155 preserves annual stability but remains high-cost negative; budget
100 fixes high-cost for `w25`/`w50` but breaks 2024. The next retry should not
be another static budget grid. It should use regime- or state-aware budget
pacing, net-edge calibration, or an exposure constraint that can cut costly
trading without starving the 2024 annual slice.

### Net-Edge And Pacing Probe

A small follow-up probe tested the two simplest optimizer-native controls around
the budget-155 configuration: a positive `optimizer_min_net_edge_bps` threshold
and uniform path-budget pacing.

Artifact:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_net_edge_pacing_probe/probe_summary.json`

The probe only reran the two binding scenarios, `full_high_cost` and
`year_2024_base`, for `w25`/`w50`.

| method | control | full high-cost | 2024 base | note |
|---|---|---:|---:|---|
| `w50` | min net edge 5 bps | -3.08% | 3.17% | Same as baseline; no practical filter. |
| `w50` | min net edge 10 bps | -3.08% | 3.17% | Same as baseline; no practical filter. |
| `w50` | pacing 1.0 | -3.07% | -0.31% | High cost barely changes; 2024 turns negative. |
| `w25` | min net edge 5 bps | -3.43% | 2.94% | Same as baseline; no practical filter. |
| `w25` | pacing 1.0 | -3.27% | -0.33% | High cost improves only slightly; 2024 turns negative. |

This rules out the naive versions of the two proposed controls. The current
score-to-edge scale leaves all selected names above 5-10 bps of net edge, so
small positive thresholds do not change the chosen book. Uniform budget pacing
changes the execution path but hurts the 2024 slice, which is the slice that
budget compression already damaged. A useful next experiment needs a true state
variable, such as cost pressure, liquidity regime, drawdown state, or realized
turnover spend rate, rather than a constant threshold or constant pacing rate.

### State-Aware Policy Probe

The first state-aware probe used only existing policy hooks. It did not change
the optimizer. Three schedules scaled gross exposure when the `w50` score file
showed high concentration-tail risk among top-ranked names, and one branch used
the existing drawdown brake.

Artifact:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_state_policy_probe/probe_summary.json`

| policy | full high-cost | 2024 base | gross turnover, high-cost | note |
|---|---:|---:|---:|---|
| baseline `w50` budget155 | -3.08% | 3.17% | 156.53 | Reference from standard validation. |
| top150 risk q80 scale 0.70 | -3.18% | 2.84% | 157.00 | Worse than baseline. |
| top50 risk q80 scale 0.70 | -3.18% | 2.40% | 157.00 | Worse than baseline. |
| top150 risk q90 scale 0.50 | -3.08% | 5.07% | 156.53 | Improves 2024 but leaves high-cost unchanged. |
| drawdown brake -5%, scale 0.70 | -11.03% | -5.04% | 162.87 | Fails both binding scenarios. |

This is not a promotion result. The high-risk top150 q90 schedule is worth
remembering because it improves the weak 2024 slice without hurting high-cost,
but it does not solve the doubled-cost loss. The drawdown brake is rejected for
this overlay because it cuts exposure after losses have already occurred,
increases realized turnover, and materially worsens both binding scenarios.

### Cost-Pressure Turnover Cap

The final retry used realized transaction-cost pressure directly. When
realized transaction costs reached 1000 bps of initial capital, the optimizer
kept gross exposure unchanged but capped per-rebalance gross turnover at 0.01.
This was combined with the existing path-level gross-turnover budget of 155.

Artifacts:

- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_cost_pressure_turnover_probe/probe_summary.json`
- `runs/candidate_factor_portfolios/time_series_decomposition_2026_05_25_volume_concentration_optimizer_risk_penalty_cost_pressure_cap0010_standard/validation_summary.json`

The focused probe showed that exposure scaling was the wrong tool: reducing
gross exposure after cost pressure worsened high-cost results. A turnover cap
was the useful control. The standard validation then reran `w00`, `w25`, and
`w50` across full, high-cost, and annual base slices.

| method | full base | full high cost | 2023 base | 2024 base | 2025 base | full turnover |
|---|---:|---:|---:|---:|---:|---:|
| `vc_opt_risk_cp0010_w00` | 14.13% | 8.84% | 4.11% | 0.33% | 24.66% | 146.72 |
| `vc_opt_risk_cp0010_w25` | 14.68% | 8.19% | 5.21% | 3.00% | 24.26% | 146.45 |
| `vc_opt_risk_cp0010_w50` | 16.07% | 8.10% | 5.40% | 3.22% | 24.13% | 146.23 |

Validation status is `pass`: zero failures and zero warnings. `w50` is the best
full-base and best annual-stability choice; `w00` is slightly better in the
doubled-cost stress, but `w50` remains strongly positive there and improves
full-base return by 1.94 percentage points versus the no-penalty cost-pressure
control.

## Decision

Do not promote the full 7-feature time-series-decomposition basket into the
compact core, and do not promote any of these features as standalone long-only
alphas under the ordinary partial-rebalance policy. The isolated basket and the
direct compact-core rank overlays remain negative.

Keep the high-turnover burstiness and U-shape variants as `watchlist` with
`cost_fragile`. Register the weak acceleration and weak state-change variants
as `reject`.

The volume concentration follow-up closes the same-policy rank-blend,
downside-penalty, and direct hard-entry-exclusion branches. Do not continue by
adding more concentration thresholds under the current daily partial-rebalance
policy.

Promote the volume-concentration family only as a portfolio-native candidate:
`intraday_volume_concentration_5m_w24`, `w48`, and `w96` are candidate inputs
for the validated optimizer-risk-penalty satellite with path turnover budget
155 and cost-pressure turnover cap 0.01 after 1000 bps realized cost. The
recommended current branch is `vc_opt_risk_cp0010_w50`.
