# Factor Registry Admission Audit 2026-05-31

This audit compares `configs/factors/factor_registry.json` with the current
framework-fixed standard admission report:

`runs/framework_v1_acceptance/standard/factor_admission/factor_admission_report.json`

The registry itself validates cleanly:

```bash
conda run -n quant python examples/validate_factor_registry.py \
  --registry configs/factors/factor_registry.json
```

Current validation summary:

- Registry entries: 95
- Admission rows: 148
- Registry validation status: `pass`
- Direction conflicts: 0
- Role conflicts: 0

## Registry Sync Applied

The registry was updated conservatively after this audit:

- Registry version: `65`
- `updated_at`: `2026-05-31`
- Current status counts after sync: `28 candidate`, `43 watchlist`, `24 reject`
- Every status-difference entry now records
  `evaluation.current_standard_admission_report`,
  `evaluation.current_standard_benchmark_summary`,
  `evaluation.current_standard_registry_audit`,
  `evaluation.current_standard_admission_status`, and
  `evaluation.current_standard_admission_by_feature`.

Top-level status was changed only when single-factor admission is the relevant
governance state and no stronger portfolio conclusion should override it:

| factor_id | applied status | action |
| --- | --- | --- |
| `intraday_gap_5m` | watchlist | upgraded from reject to current-standard watchlist |
| `intraday_reversal_5m_lb1` | watchlist | upgraded from reject to current-standard watchlist |
| `intraday_cross_sectional_reversal_5m_lb1` | watchlist | upgraded from reject to current-standard watchlist |
| `intraday_eod_reversal_5m_lb1_tail6` | candidate | event-overlay candidate; requires role-specific portfolio validation |
| `intraday_reversal_5m_lb6` | watchlist | upgraded from reject to current-standard watchlist |
| `intraday_cross_sectional_reversal_5m_lb6` | watchlist | upgraded from reject to current-standard watchlist |
| `intraday_eod_reversal_5m_lb6_tail6` | candidate | event-overlay candidate; requires role-specific portfolio validation |
| `intraday_eod_reversal_5m_lb12_tail6` | candidate | event-overlay candidate; requires role-specific portfolio validation |
| `intraday_eod_reversal_5m_lb24_tail6` | candidate | event-overlay candidate; requires portfolio retest because older portfolio evidence was negative |
| `intraday_volume_u_shape_5m_w24` | reject | downgraded from watchlist to current-standard reject |
| `intraday_microstructure_recovery_acceleration_5m_s24_l96` | watchlist | upgraded from reject to current-standard watchlist |

Top-level status was intentionally not changed for:

- `intraday_amihud_5m`: current single-factor admission is watchlist, but
  prior portfolio validation was strongly positive, so it remains candidate
  pending portfolio revalidation under the fixed framework. This follow-up is
  now recorded in
  `docs/validation/fixed_framework_priority_revalidation_2026_05_31.md`; it
  remains candidate with a 2024 stability warning.
- `intraday_sell_pressure_recovery_5m_w48`: current single-factor admission is
  reject, but prior legacy revalidation was positive enough to keep it as
  watchlist pending horizon or policy review. The fixed-framework retest failed
  portfolio validation, so the registry now downgrades it to reject.
- `intraday_daily_moving_average_state_5m`: current per-feature admission is
  mixed and the entry has separate promoted-challenger portfolio evidence. The
  fixed-framework retest was weak and unstable, so the registry now keeps it on
  watchlist pending incremental portfolio review.
- `intraday_liquidity_reliability_recovery_balance_5m`: current l48 feature is
  candidate, but l96 is missing from current admission and prior integration
  validation was dilutive, so the combined registry entry remains watchlist.
- `intraday_volatility_state_change_5m_s12_l48`: already watchlist, now aligned
  with current-standard watchlist admission.

## Status Differences

These registered factors have a current standard admission status that differs
from the status currently stored in the registry:

| factor_id | registry status | current admission status |
| --- | --- | --- |
| `intraday_amihud_5m` | candidate | watchlist |
| `intraday_gap_5m` | reject | watchlist |
| `intraday_sell_pressure_recovery_5m_w48` | watchlist | reject |
| `intraday_daily_moving_average_state_5m` | missing | reject |
| `intraday_liquidity_reliability_recovery_balance_5m` | watchlist | candidate |
| `intraday_reversal_5m_lb1` | reject | watchlist |
| `intraday_cross_sectional_reversal_5m_lb1` | reject | watchlist |
| `intraday_eod_reversal_5m_lb1_tail6` | reject | candidate |
| `intraday_reversal_5m_lb6` | reject | watchlist |
| `intraday_cross_sectional_reversal_5m_lb6` | reject | watchlist |
| `intraday_eod_reversal_5m_lb6_tail6` | reject | candidate |
| `intraday_eod_reversal_5m_lb12_tail6` | reject | candidate |
| `intraday_eod_reversal_5m_lb24_tail6` | reject | candidate |
| `intraday_volatility_state_change_5m_s12_l48` | candidate | watchlist |
| `intraday_volume_u_shape_5m_w24` | watchlist | reject |
| `intraday_microstructure_recovery_acceleration_5m_s24_l96` | reject | watchlist |

These should be reviewed before editing the registry because several entries
have existing portfolio validation, compact-core review, or research-memory
metadata that should not be overwritten by single-factor admission alone.

## Missing From Current Admission

The registry entry below did not appear in the current standard admission rows:

| factor_id | feature_columns | registry status |
| --- | --- | --- |
| `intraday_liquidity_reliability_5m_w24` | `intraday_liquidity_reliability_5m_w24` | candidate |

Before changing this entry, confirm whether the current `all` factor-group
expansion should include this feature or whether the registry entry refers to a
retired/manual feature variant.

## Admission Rows Not Registered As Feature Columns

The current admission report contains 43 feature rows that do not map directly
to any registered `feature_columns` value. The candidate subset is:

| feature | current status | role |
| --- | --- | --- |
| `intraday_liquidity_reliability_5m_w48` | candidate | alpha_rank |
| `intraday_range_volatility_5m_w12` | candidate | alpha_rank |
| `intraday_downside_volatility_5m_w12` | candidate | alpha_rank |
| `intraday_weak_tape_gap_down_recovery_risk_5m_w48` | candidate | alpha_rank |
| `intraday_return_skewness_5m_w12` | candidate | alpha_rank |
| `intraday_efficiency_ratio_5m_w12` | candidate | alpha_rank |

The remaining unmatched rows are watchlist or reject diagnostics, including
market-state features and shorter-window variants. Do not add all unmatched rows
automatically. Register only features with a clear hypothesis, required inputs,
expected direction, and research-memory decision.

## Registry Update Guidance

Use the current standard admission report as the source of truth for new
single-factor evidence, but apply registry changes selectively:

- Update `admission_report` to the current standard report only when the entry's
  current evidence is intended to supersede prior single-factor evidence.
- Do not downgrade a factor with positive portfolio validation solely from
  single-factor watchlist status; add an explicit research-memory note and
  schedule portfolio revalidation.
- Do not promote newly admitted event-overlay features until role-specific
  portfolio validation confirms they improve the selected policy.
- For multi-feature entries such as daily moving-average state, review the
  per-feature rows before assigning one combined registry status.
