"""Portfolio risk and market-regime gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RiskConstraint:
    """One portfolio risk constraint."""

    name: str
    limit: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint name is required")
        if self.limit < 0:
            raise ValueError("constraint limit must be non-negative")


RegimeGateReason = Literal[
    "warmup",
    "full_exposure",
    "reduced_exposure",
    "blocked_exposure",
    "budget_exposure",
]
RegimeGateMode = Literal["threshold", "budget"]


@dataclass(frozen=True, slots=True)
class RollingRegimeGateConfig:
    """Configuration for a lagged rolling policy gross-exposure gate."""

    lookback_windows: int = 20
    min_periods: int = 5
    label_lag_windows: int = 48
    state_confirmation_windows: int = 1
    max_scale_change_per_window: float | None = None
    max_scale_increase_per_window: float | None = None
    max_scale_decrease_per_window: float | None = None
    scale_change_deadband: float = 0.0
    gate_mode: RegimeGateMode = "threshold"
    full_scale: float = 1.0
    reduced_scale: float = 0.5
    blocked_scale: float = 0.0
    warmup_scale: float = 1.0
    budget_min_scale: float = 0.25
    budget_max_scale: float = 1.0
    budget_top_return_floor: float = -0.001
    budget_top_return_ceiling: float = 0.001
    budget_spread_floor: float = -0.001
    budget_spread_ceiling: float = 0.001
    budget_rank_ic_floor: float = -0.05
    budget_rank_ic_ceiling: float = 0.05
    min_top_return: float = 0.0
    min_spread: float = 0.0
    min_rank_ic: float = 0.0
    block_top_return: float = -0.001
    block_spread: float = -0.001
    block_rank_ic: float = -0.05

    def __post_init__(self) -> None:
        if self.lookback_windows <= 0:
            raise ValueError("lookback_windows must be positive")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if self.min_periods > self.lookback_windows:
            raise ValueError("min_periods must be <= lookback_windows")
        if self.label_lag_windows <= 0:
            raise ValueError("label_lag_windows must be positive")
        if self.state_confirmation_windows <= 0:
            raise ValueError("state_confirmation_windows must be positive")
        if (
            self.max_scale_change_per_window is not None
            and not 0 < self.max_scale_change_per_window <= 1
        ):
            raise ValueError("max_scale_change_per_window must be in (0, 1]")
        for name in ("max_scale_increase_per_window", "max_scale_decrease_per_window"):
            value = getattr(self, name)
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.scale_change_deadband <= 1:
            raise ValueError("scale_change_deadband must be in [0, 1]")
        if self.gate_mode not in {"threshold", "budget"}:
            raise ValueError("gate_mode must be threshold or budget")
        for name in (
            "full_scale",
            "reduced_scale",
            "blocked_scale",
            "warmup_scale",
            "budget_min_scale",
            "budget_max_scale",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.budget_min_scale > self.budget_max_scale:
            raise ValueError("budget_min_scale must be <= budget_max_scale")
        for floor_name, ceiling_name in (
            ("budget_top_return_floor", "budget_top_return_ceiling"),
            ("budget_spread_floor", "budget_spread_ceiling"),
            ("budget_rank_ic_floor", "budget_rank_ic_ceiling"),
        ):
            if float(getattr(self, floor_name)) >= float(getattr(self, ceiling_name)):
                raise ValueError(f"{floor_name} must be below {ceiling_name}")


def build_rolling_regime_gate(
    diagnostics: pd.DataFrame,
    config: RollingRegimeGateConfig,
) -> pd.DataFrame:
    """Build timestamp-level gross exposure scales from lagged rolling diagnostics.

    The current row's forward-return diagnostics are deliberately shifted out of
    the rolling window by ``label_lag_windows``. A row can only use diagnostics
    whose forward-return horizon should have matured before the current
    timestamp.
    """

    required = (
        "timestamp",
        "score_top_n_mean_label",
        "score_top_minus_bottom_label",
        "score_rank_ic",
    )
    _require_columns(diagnostics, required)
    if diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "gross_exposure_scale",
                "gate_reason",
                "raw_gross_exposure_scale",
                "raw_gate_reason",
                "target_gross_exposure_scale",
                "target_gate_reason",
                "scale_step_limited",
                "scale_deadband_held",
                "budget_health_score",
                "rolling_observation_count",
                "rolling_score_top_n_mean_label",
                "rolling_score_top_minus_bottom_label",
                "rolling_score_rank_ic",
            ]
        )
    frame = diagnostics.loc[:, required].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    metric_columns = [
        "score_top_n_mean_label",
        "score_top_minus_bottom_label",
        "score_rank_ic",
    ]
    metrics = frame.loc[:, metric_columns].apply(pd.to_numeric, errors="coerce")
    lagged = metrics.shift(config.label_lag_windows)
    rolling = lagged.rolling(
        window=config.lookback_windows,
        min_periods=config.min_periods,
    )
    rolling_means = rolling.mean()
    observation_count = (
        lagged["score_top_n_mean_label"]
        .rolling(window=config.lookback_windows, min_periods=1)
        .count()
        .astype(int)
    )
    output = frame.loc[:, ["timestamp"]].copy()
    output["raw_gross_exposure_scale"] = config.warmup_scale
    output["raw_gate_reason"] = "warmup"
    output["budget_health_score"] = pd.NA
    output["rolling_observation_count"] = observation_count
    output["rolling_score_top_n_mean_label"] = rolling_means["score_top_n_mean_label"]
    output["rolling_score_top_minus_bottom_label"] = rolling_means[
        "score_top_minus_bottom_label"
    ]
    output["rolling_score_rank_ic"] = rolling_means["score_rank_ic"]
    _set_raw_gate_columns(output, config)
    return _apply_gate_hysteresis(output, config)


def _set_raw_gate_columns(
    output: pd.DataFrame,
    config: RollingRegimeGateConfig,
) -> None:
    ready = output["rolling_observation_count"].astype(int) >= config.min_periods
    if not ready.any():
        return
    top_return = output["rolling_score_top_n_mean_label"].astype(float)
    spread = output["rolling_score_top_minus_bottom_label"].astype(float)
    rank_ic = output["rolling_score_rank_ic"].astype(float)
    if config.gate_mode == "budget":
        top_score = _linear_score_series(
            top_return,
            floor=config.budget_top_return_floor,
            ceiling=config.budget_top_return_ceiling,
        )
        spread_score = _linear_score_series(
            spread,
            floor=config.budget_spread_floor,
            ceiling=config.budget_spread_ceiling,
        )
        rank_ic_score = _linear_score_series(
            rank_ic,
            floor=config.budget_rank_ic_floor,
            ceiling=config.budget_rank_ic_ceiling,
        )
        health_score = pd.concat([top_score, spread_score, rank_ic_score], axis=1).min(
            axis=1
        )
        scale = config.budget_min_scale + (
            config.budget_max_scale - config.budget_min_scale
        ) * health_score
        output.loc[ready, "raw_gross_exposure_scale"] = scale.loc[ready].astype(float)
        output.loc[ready, "raw_gate_reason"] = "budget_exposure"
        output.loc[ready, "budget_health_score"] = health_score.loc[ready].astype(float)
        return

    blocked = (
        top_return.le(config.block_top_return)
        | spread.le(config.block_spread)
        | rank_ic.le(config.block_rank_ic)
    )
    reduced = (
        top_return.lt(config.min_top_return)
        | spread.lt(config.min_spread)
        | rank_ic.lt(config.min_rank_ic)
    )
    output.loc[ready, "raw_gross_exposure_scale"] = config.full_scale
    output.loc[ready, "raw_gate_reason"] = "full_exposure"
    reduced_ready = ready & reduced & ~blocked
    output.loc[reduced_ready, "raw_gross_exposure_scale"] = config.reduced_scale
    output.loc[reduced_ready, "raw_gate_reason"] = "reduced_exposure"
    blocked_ready = ready & blocked
    output.loc[blocked_ready, "raw_gross_exposure_scale"] = config.blocked_scale
    output.loc[blocked_ready, "raw_gate_reason"] = "blocked_exposure"


def _linear_score_series(values: pd.Series, *, floor: float, ceiling: float) -> pd.Series:
    scaled = (values - floor) / (ceiling - floor)
    return pd.Series(
        np.clip(scaled.to_numpy(dtype=float), 0.0, 1.0),
        index=values.index,
    )


def _apply_gate_hysteresis(
    output: pd.DataFrame,
    config: RollingRegimeGateConfig,
) -> pd.DataFrame:
    target_reasons: list[str] = []
    target_scales: list[float] = []
    final_scales: list[float] = []
    step_limited: list[bool] = []
    deadband_held: list[bool] = []
    active_reason = "warmup"
    active_scale = config.warmup_scale
    pending_reason: str | None = None
    pending_scale: float | None = None
    pending_count = 0
    previous_scale = config.warmup_scale
    for row in output.itertuples(index=False):
        raw_reason = str(row.raw_gate_reason)
        raw_scale = float(row.raw_gross_exposure_scale)
        if config.gate_mode == "budget":
            active_reason = raw_reason
            active_scale = raw_scale
        elif raw_reason == "warmup":
            active_reason = "warmup"
            active_scale = config.warmup_scale
            pending_reason = None
            pending_scale = None
            pending_count = 0
        elif raw_reason == active_reason and abs(raw_scale - active_scale) <= 1e-12:
            pending_reason = None
            pending_scale = None
            pending_count = 0
        elif raw_reason == pending_reason and pending_scale is not None and (
            abs(raw_scale - pending_scale) <= 1e-12
        ):
            pending_count += 1
        else:
            pending_reason = raw_reason
            pending_scale = raw_scale
            pending_count = 1
        if pending_count >= config.state_confirmation_windows:
            active_reason = str(pending_reason)
            active_scale = float(pending_scale)
            pending_reason = None
            pending_scale = None
            pending_count = 0
        target_reasons.append(active_reason)
        target_scales.append(active_scale)
        scale, limited, held = _limit_scale_step(
            previous_scale=previous_scale,
            target_scale=active_scale,
            max_step=config.max_scale_change_per_window,
            max_increase=config.max_scale_increase_per_window,
            max_decrease=config.max_scale_decrease_per_window,
            deadband=config.scale_change_deadband,
        )
        final_scales.append(scale)
        step_limited.append(limited)
        deadband_held.append(held)
        previous_scale = scale
    output = output.copy()
    output["target_gross_exposure_scale"] = target_scales
    output["target_gate_reason"] = target_reasons
    output["gross_exposure_scale"] = final_scales
    output["gate_reason"] = target_reasons
    output["scale_step_limited"] = step_limited
    output["scale_deadband_held"] = deadband_held
    return output


def _limit_scale_step(
    *,
    previous_scale: float,
    target_scale: float,
    max_step: float | None,
    max_increase: float | None,
    max_decrease: float | None,
    deadband: float,
) -> tuple[float, bool, bool]:
    delta = target_scale - previous_scale
    if abs(delta) <= 1e-12:
        return target_scale, False, False
    if abs(delta) <= deadband:
        return previous_scale, False, True
    step = max_step
    if delta > 0 and max_increase is not None:
        step = max_increase
    elif delta < 0 and max_decrease is not None:
        step = max_decrease
    if step is None:
        return target_scale, False, False
    if abs(delta) <= step:
        return target_scale, False, False
    direction = 1.0 if delta > 0 else -1.0
    return previous_scale + direction * step, True, False


def _scale_from_rolling_metrics(
    *,
    top_return: float,
    spread: float,
    rank_ic: float,
    config: RollingRegimeGateConfig,
) -> tuple[float, RegimeGateReason, float | None]:
    if config.gate_mode == "budget":
        scale, health_score = _budget_scale_from_rolling_metrics(
            top_return=top_return,
            spread=spread,
            rank_ic=rank_ic,
            config=config,
        )
        return scale, "budget_exposure", health_score
    if (
        top_return <= config.block_top_return
        or spread <= config.block_spread
        or rank_ic <= config.block_rank_ic
    ):
        return config.blocked_scale, "blocked_exposure", None
    if (
        top_return < config.min_top_return
        or spread < config.min_spread
        or rank_ic < config.min_rank_ic
    ):
        return config.reduced_scale, "reduced_exposure", None
    return config.full_scale, "full_exposure", None


def _budget_scale_from_rolling_metrics(
    *,
    top_return: float,
    spread: float,
    rank_ic: float,
    config: RollingRegimeGateConfig,
) -> tuple[float, float]:
    components = [
        _linear_score(
            top_return,
            floor=config.budget_top_return_floor,
            ceiling=config.budget_top_return_ceiling,
        ),
        _linear_score(
            spread,
            floor=config.budget_spread_floor,
            ceiling=config.budget_spread_ceiling,
        ),
        _linear_score(
            rank_ic,
            floor=config.budget_rank_ic_floor,
            ceiling=config.budget_rank_ic_ceiling,
        ),
    ]
    health_score = min(components)
    scale = config.budget_min_scale + (
        config.budget_max_scale - config.budget_min_scale
    ) * health_score
    return scale, health_score


def _linear_score(value: float, *, floor: float, ceiling: float) -> float:
    if value <= floor:
        return 0.0
    if value >= ceiling:
        return 1.0
    return (value - floor) / (ceiling - floor)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
