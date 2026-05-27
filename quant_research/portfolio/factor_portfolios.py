"""Candidate-factor portfolio scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_research.portfolio.forecast_calibration import (
    ScoreForecastCalibrationConfig,
    apply_score_forecast_calibration,
    build_score_forecast_calibration_from_observations,
    score_forecast_calibration_observations,
)


@dataclass(frozen=True, slots=True)
class CandidateFactor:
    """One admitted factor and its preferred direction."""

    feature: str
    direction: int
    rank_ic_mean: float

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("feature must be non-empty")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")


@dataclass(frozen=True, slots=True)
class FactorHealthConfig:
    """Lagged rolling factor-leg health monitoring/shrinkage configuration."""

    lookback_windows: int = 20
    min_periods: int = 5
    label_lag_windows: int = 48
    min_scale: float = 0.25
    max_scale: float = 1.0
    rank_ic_floor: float = -0.05
    rank_ic_ceiling: float = 0.05
    spread_floor: float = -0.001
    spread_ceiling: float = 0.001

    def __post_init__(self) -> None:
        if self.lookback_windows <= 0:
            raise ValueError("lookback_windows must be positive")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if self.min_periods > self.lookback_windows:
            raise ValueError("min_periods must be <= lookback_windows")
        if self.label_lag_windows <= 0:
            raise ValueError("label_lag_windows must be positive")
        if not 0 <= self.min_scale <= self.max_scale <= 1:
            raise ValueError("health scales must satisfy 0 <= min_scale <= max_scale <= 1")
        if self.rank_ic_floor >= self.rank_ic_ceiling:
            raise ValueError("rank_ic_floor must be below rank_ic_ceiling")
        if self.spread_floor >= self.spread_ceiling:
            raise ValueError("spread_floor must be below spread_ceiling")


def load_candidate_factors(
    admission_report_path: Path,
    *,
    statuses: tuple[str, ...] = ("candidate",),
    include_features: tuple[str, ...] = (),
) -> tuple[CandidateFactor, ...]:
    """Load candidate factors from a factor admission report."""

    report = json.loads(admission_report_path.read_text(encoding="utf-8"))
    include_set = set(include_features)
    factors = []
    for row in report.get("factors", []):
        feature = str(row["feature"])
        if include_set and feature not in include_set:
            continue
        if row.get("admission_status") not in statuses:
            continue
        factors.append(
            CandidateFactor(
                feature=feature,
                direction=-1 if row.get("direction") == "invert" else 1,
                rank_ic_mean=float(row.get("spearman_rank_ic_mean") or 0.0),
            )
        )
    if not factors:
        message = f"no factors found for statuses: {statuses}"
        if include_set:
            message += f", include_features: {sorted(include_set)}"
        raise ValueError(message)
    return tuple(factors)


def factor_combination_weights(
    candidates: tuple[CandidateFactor, ...],
    *,
    method: str,
    correlation: pd.DataFrame | None = None,
    ridge: float = 0.05,
) -> dict[str, float]:
    """Compute non-negative combination weights for candidate factors."""

    if method == "equal":
        return _normalize({factor.feature: 1.0 for factor in candidates})
    base = {
        factor.feature: max(abs(float(factor.rank_ic_mean)), 1e-12)
        for factor in candidates
    }
    if method == "ic_weighted":
        return _normalize(base)
    if method != "decorrelated":
        raise ValueError("method must be equal, ic_weighted, or decorrelated")
    if correlation is None or correlation.empty:
        return _normalize(base)
    features = [factor.feature for factor in candidates]
    directions = np.array([factor.direction for factor in candidates], dtype=float)
    matrix = correlation.reindex(index=features, columns=features).astype(float)
    matrix = matrix.fillna(0.0)
    oriented = matrix.to_numpy(dtype=float) * np.outer(directions, directions)
    oriented = np.nan_to_num(oriented, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(oriented, 1.0)
    system = oriented + np.eye(len(features)) * ridge
    target = np.array([base[feature] for feature in features], dtype=float)
    try:
        raw = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        raw = target
    raw = np.clip(raw, 0.0, None)
    if float(raw.sum()) <= 0:
        raw = target
    return _normalize(dict(zip(features, raw.tolist(), strict=True)))


def cap_factor_weights(
    weights: dict[str, float],
    *,
    max_weight: float | None,
) -> dict[str, float]:
    """Cap static factor weights and redistribute excess conservatively."""

    if max_weight is None:
        return _normalize(weights)
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    capped = {key: max(float(value), 0.0) for key, value in weights.items()}
    if not capped:
        raise ValueError("weights must be non-empty")
    if max_weight * len(capped) < 1.0:
        raise ValueError("max_weight is too low for the number of factors")
    for _ in range(len(capped) + 1):
        over = {key: value for key, value in capped.items() if value > max_weight}
        if not over:
            return _normalize(capped)
        excess = sum(value - max_weight for value in over.values())
        for key in over:
            capped[key] = max_weight
        under_keys = [key for key, value in capped.items() if value < max_weight]
        if not under_keys:
            return _normalize(capped)
        under_total = sum(capped[key] for key in under_keys)
        if under_total <= 0:
            increment = excess / len(under_keys)
            for key in under_keys:
                capped[key] += increment
        else:
            for key in under_keys:
                capped[key] += excess * capped[key] / under_total
    return _normalize({key: min(value, max_weight) for key, value in capped.items()})


def build_factor_health_schedule(
    dataset_paths: list[Path],
    *,
    candidates: tuple[CandidateFactor, ...],
    config: FactorHealthConfig,
    top_n: int,
    label_column: str = "forward_return",
    apply_shrink: bool = True,
) -> pd.DataFrame:
    """Build lagged rolling per-factor health diagnostics from matured labels."""

    observations = _factor_health_observation_frame(
        dataset_paths,
        candidates=candidates,
        top_n=top_n,
        label_column=label_column,
    )
    return _factor_health_schedule_from_observations(
        observations,
        config=config,
        apply_shrink=apply_shrink,
    )


def _factor_health_observation_frame(
    dataset_paths: list[Path],
    *,
    candidates: tuple[CandidateFactor, ...],
    top_n: int,
    label_column: str,
) -> pd.DataFrame:
    if not label_column:
        raise ValueError("label_column must be non-empty")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    rows: list[dict[str, Any]] = []
    features = [factor.feature for factor in candidates]
    for dataset_path in dataset_paths:
        frame = pd.read_parquet(
            dataset_path,
            columns=["timestamp", "instrument_id", label_column, *features],
        )
        rows.extend(
            _factor_health_observations(
                frame,
                candidates=candidates,
                top_n=top_n,
                label_column=label_column,
            )
        )
        del frame
    if not rows:
        return _empty_factor_health_observations()
    return pd.DataFrame(rows).sort_values(["feature", "timestamp"])


def _empty_factor_health_observations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "feature",
            "label_column",
            "factor_valid_count",
            "factor_top_count",
            "factor_rank_ic",
            "factor_top_label",
            "factor_bottom_label",
            "factor_top_minus_bottom_label",
        ]
    )


def _empty_factor_health_schedule() -> pd.DataFrame:
    output = _empty_factor_health_observations()
    for column in [
        "rolling_rank_ic",
        "rolling_top_label",
        "rolling_bottom_label",
        "rolling_top_minus_bottom_label",
        "health_score",
        "health_state",
        "recommended_weight_scale",
        "weight_scale",
        "shrink_reason",
    ]:
        output[column] = pd.Series(dtype=object)
    return output


def _factor_health_schedule_from_observations(
    observations: pd.DataFrame,
    *,
    config: FactorHealthConfig,
    apply_shrink: bool,
) -> pd.DataFrame:
    if observations.empty:
        return _empty_factor_health_schedule()
    schedules: list[pd.DataFrame] = []
    for feature, group in observations.groupby("feature", sort=True):
        current = group.sort_values("timestamp").reset_index(drop=True).copy()
        current["rolling_rank_ic"] = (
            current["factor_rank_ic"]
            .shift(config.label_lag_windows)
            .rolling(config.lookback_windows, min_periods=config.min_periods)
            .mean()
        )
        current["rolling_top_label"] = (
            current["factor_top_label"]
            .shift(config.label_lag_windows)
            .rolling(config.lookback_windows, min_periods=config.min_periods)
            .mean()
        )
        current["rolling_bottom_label"] = (
            current["factor_bottom_label"]
            .shift(config.label_lag_windows)
            .rolling(config.lookback_windows, min_periods=config.min_periods)
            .mean()
        )
        current["rolling_top_minus_bottom_label"] = (
            current["factor_top_minus_bottom_label"]
            .shift(config.label_lag_windows)
            .rolling(config.lookback_windows, min_periods=config.min_periods)
            .mean()
        )
        rank_score = _linear_score(
            current["rolling_rank_ic"],
            floor=config.rank_ic_floor,
            ceiling=config.rank_ic_ceiling,
        )
        spread_score = _linear_score(
            current["rolling_top_minus_bottom_label"],
            floor=config.spread_floor,
            ceiling=config.spread_ceiling,
        )
        health_score = pd.concat([rank_score, spread_score], axis=1).min(axis=1)
        current["health_score"] = health_score
        current["recommended_weight_scale"] = (
            config.min_scale
            + (config.max_scale - config.min_scale) * health_score.fillna(1.0)
        )
        current["health_state"] = "warmup"
        current.loc[health_score.notna() & (health_score >= 0.75), "health_state"] = (
            "healthy"
        )
        current.loc[
            health_score.notna() & (health_score >= 0.25) & (health_score < 0.75),
            "health_state",
        ] = "watch"
        current.loc[health_score.notna() & (health_score < 0.25), "health_state"] = (
            "impaired"
        )
        if apply_shrink:
            current["weight_scale"] = current["recommended_weight_scale"]
            current.loc[health_score.isna(), "shrink_reason"] = "warmup"
            current.loc[
                health_score.notna() & (health_score >= 0.999), "shrink_reason"
            ] = "healthy"
            current.loc[
                health_score.notna() & (health_score < 0.999), "shrink_reason"
            ] = "lagged_health_shrink"
        else:
            current["weight_scale"] = 1.0
            current["shrink_reason"] = "monitor_only"
            current.loc[health_score.isna(), "shrink_reason"] = "warmup"
        current["feature"] = feature
        schedules.append(current)
    return (
        pd.concat(schedules, ignore_index=True)
        .sort_values(["timestamp", "feature"])
        .reset_index(drop=True)
    )


def build_factor_health_ensemble_schedule(
    dataset_paths: list[Path],
    *,
    candidates: tuple[CandidateFactor, ...],
    configs: tuple[FactorHealthConfig, ...],
    top_n: int,
    label_column: str = "forward_return",
    apply_shrink: bool = True,
    combine_mode: str = "mean",
) -> pd.DataFrame:
    """Blend multiple lagged health memories into one per-factor scale schedule."""

    if not configs:
        raise ValueError("configs must be non-empty")
    if combine_mode not in {"mean", "min", "max"}:
        raise ValueError("combine_mode must be mean, min, or max")
    reference = configs[0]
    for config in configs[1:]:
        if config.label_lag_windows != reference.label_lag_windows:
            raise ValueError("ensemble configs must use the same label_lag_windows")
        if config.min_scale != reference.min_scale or config.max_scale != reference.max_scale:
            raise ValueError("ensemble configs must use the same min/max scale")
        if (
            config.rank_ic_floor != reference.rank_ic_floor
            or config.rank_ic_ceiling != reference.rank_ic_ceiling
            or config.spread_floor != reference.spread_floor
            or config.spread_ceiling != reference.spread_ceiling
        ):
            raise ValueError("ensemble configs must use the same health score thresholds")
    observations = _factor_health_observation_frame(
        dataset_paths,
        candidates=candidates,
        top_n=top_n,
        label_column=label_column,
    )
    schedules: list[pd.DataFrame] = []
    for config in configs:
        schedule = _factor_health_schedule_from_observations(
            observations,
            config=config,
            apply_shrink=apply_shrink,
        )
        if schedule.empty:
            continue
        current = schedule.copy()
        current["ensemble_member_lookback_windows"] = config.lookback_windows
        current["ensemble_member_min_periods"] = config.min_periods
        schedules.append(current)
    if not schedules:
        return _factor_health_schedule_from_observations(
            observations,
            config=reference,
            apply_shrink=apply_shrink,
        )
    stacked = pd.concat(schedules, ignore_index=True)
    output_rows: list[dict[str, Any]] = []
    numeric_columns = [
        "factor_valid_count",
        "factor_top_count",
        "factor_rank_ic",
        "factor_top_label",
        "factor_bottom_label",
        "factor_top_minus_bottom_label",
        "rolling_rank_ic",
        "rolling_top_label",
        "rolling_bottom_label",
        "rolling_top_minus_bottom_label",
    ]
    for (timestamp, feature), group in stacked.groupby(["timestamp", "feature"], sort=True):
        row: dict[str, Any] = {
            "timestamp": timestamp,
            "feature": feature,
            "label_column": group["label_column"].iloc[0],
        }
        for column in numeric_columns:
            row[column] = _mean(group[column]) if column in group.columns else None
        health_values = pd.to_numeric(group["health_score"], errors="coerce")
        filled_scores = health_values.fillna(1.0)
        if combine_mode == "mean":
            health_score = float(filled_scores.mean())
        elif combine_mode == "min":
            health_score = float(filled_scores.min())
        else:
            health_score = float(filled_scores.max())
        row["health_score"] = None if bool(health_values.isna().all()) else health_score
        row["recommended_weight_scale"] = (
            reference.min_scale
            + (reference.max_scale - reference.min_scale) * health_score
        )
        if row["health_score"] is None:
            row["health_state"] = "warmup"
        elif health_score >= 0.75:
            row["health_state"] = "healthy"
        elif health_score >= 0.25:
            row["health_state"] = "watch"
        else:
            row["health_state"] = "impaired"
        row["weight_scale"] = row["recommended_weight_scale"] if apply_shrink else 1.0
        if row["health_score"] is None:
            row["shrink_reason"] = "warmup"
        elif not apply_shrink:
            row["shrink_reason"] = "monitor_only"
        elif row["weight_scale"] >= 0.999:
            row["shrink_reason"] = "healthy"
        else:
            row["shrink_reason"] = "ensemble_lagged_health_shrink"
        row["ensemble_combine_mode"] = combine_mode
        row["ensemble_lookback_windows"] = ",".join(
            str(config.lookback_windows) for config in configs
        )
        row["ensemble_min_periods"] = ",".join(str(config.min_periods) for config in configs)
        output_rows.append(row)
    return pd.DataFrame(output_rows).sort_values(["timestamp", "feature"]).reset_index(drop=True)


def build_state_conditioned_factor_health_schedule(
    normal: pd.DataFrame,
    stress: pd.DataFrame,
    regime: pd.DataFrame,
    *,
    regime_feature: str,
    mode: str = "select",
    threshold: float = 0.999,
) -> pd.DataFrame:
    """Select or blend factor health memories using an observable regime schedule."""

    if mode not in {"select", "blend"}:
        raise ValueError("mode must be select or blend")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not regime_feature:
        raise ValueError("regime_feature must be non-empty")
    normal_scales = _factor_health_component_scales(normal, label="normal")
    stress_scales = _factor_health_component_scales(stress, label="stress")
    regime_weight = _state_conditioned_regime_weight(
        regime,
        feature=regime_feature,
        mode=mode,
        threshold=threshold,
    )
    keys = ["timestamp", "feature"]
    joined = normal_scales.merge(stress_scales, on=keys, how="outer", sort=False)
    joined = joined.merge(regime_weight, on="timestamp", how="left", sort=False)
    joined["normal_weight_scale"] = joined["normal_weight_scale"].fillna(1.0)
    joined["stress_weight_scale"] = joined["stress_weight_scale"].fillna(1.0)
    joined["regime_weight"] = joined["regime_weight"].fillna(0.0).clip(0.0, 1.0)
    joined["weight_scale"] = (
        joined["normal_weight_scale"] * (1.0 - joined["regime_weight"])
        + joined["stress_weight_scale"] * joined["regime_weight"]
    ).clip(0.0, 1.0)
    joined["shrink_reason"] = "state_conditioned_health_memory"
    active = joined["regime_weight"] > 0.0
    joined.loc[active, "shrink_reason"] = (
        "state_conditioned_health_memory,stress_health_memory"
    )
    joined["state_conditioned_mode"] = mode
    return (
        joined.loc[
            :,
            [
                "timestamp",
                "feature",
                "weight_scale",
                "shrink_reason",
                "normal_weight_scale",
                "stress_weight_scale",
                "regime_weight",
                "regime_selector_scale",
                "state_conditioned_mode",
            ],
        ]
        .sort_values(keys)
        .reset_index(drop=True)
    )


def build_state_conditioned_factor_health_schedule_from_partitions(
    dataset_paths: list[Path],
    *,
    candidates: tuple[CandidateFactor, ...],
    normal_config: FactorHealthConfig,
    stress_config: FactorHealthConfig,
    regime: pd.DataFrame,
    regime_feature: str,
    top_n: int,
    label_column: str = "forward_return",
    apply_shrink: bool = True,
    mode: str = "select",
    threshold: float = 0.999,
) -> pd.DataFrame:
    """Build state-conditioned factor health with one dataset read pass."""

    observations = _factor_health_observation_frame(
        dataset_paths,
        candidates=candidates,
        top_n=top_n,
        label_column=label_column,
    )
    normal = _factor_health_schedule_from_observations(
        observations,
        config=normal_config,
        apply_shrink=apply_shrink,
    )
    stress = _factor_health_schedule_from_observations(
        observations,
        config=stress_config,
        apply_shrink=apply_shrink,
    )
    return build_state_conditioned_factor_health_schedule(
        normal,
        stress,
        regime,
        regime_feature=regime_feature,
        mode=mode,
        threshold=threshold,
    )


def build_composite_scores(
    frame: pd.DataFrame,
    *,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
    factor_health: pd.DataFrame | None = None,
    max_factor_contribution_share: float | None = None,
) -> pd.DataFrame:
    """Build timestamp-level composite scores from candidate factor columns."""

    features = tuple(factor.feature for factor in candidates)
    _require_columns(frame, ("timestamp", "instrument_id", *features))
    output = frame.loc[:, ["timestamp", "instrument_id"]].copy()
    contributions: dict[str, pd.Series] = {}
    available_weight = pd.Series(0.0, index=frame.index)
    health_scales = _factor_health_scale_lookup(factor_health)
    for factor in candidates:
        weight = float(weights.get(factor.feature, 0.0))
        if weight <= 0:
            continue
        effective_weight = _effective_factor_weight(
            frame["timestamp"],
            feature=factor.feature,
            base_weight=weight,
            health_scales=health_scales,
        )
        ranks = frame.groupby("timestamp", sort=False)[factor.feature].rank(
            method="average",
            pct=True,
        )
        oriented = (ranks - 0.5) * factor.direction
        valid = oriented.notna()
        contribution = pd.Series(0.0, index=frame.index)
        contribution.loc[valid] = oriented.loc[valid] * effective_weight.loc[valid]
        contributions[factor.feature] = contribution
        available_weight.loc[valid] += effective_weight.loc[valid]
    if contributions:
        contribution_frame = pd.DataFrame(contributions, index=frame.index)
        contribution_frame = _cap_factor_contributions(
            contribution_frame,
            max_share=max_factor_contribution_share,
        )
        weighted = contribution_frame.sum(axis=1)
    else:
        weighted = pd.Series(0.0, index=frame.index)
    output["score"] = weighted.where(available_weight <= 0, weighted / available_weight)
    output = output.loc[available_weight > 0].copy()
    return output.sort_values(["timestamp", "score", "instrument_id"], ascending=[True, False, True]).reset_index(drop=True)


def write_score_partitions(
    dataset_paths: list[Path],
    *,
    output_dir: Path,
    candidates: tuple[CandidateFactor, ...],
    weights_by_method: dict[str, dict[str, float]],
    max_factor_weight: float | None = None,
    max_factor_contribution_share: float | None = None,
    factor_health_schedule: pd.DataFrame | None = None,
    diagnostics_top_n: int | None = None,
    diagnostics_label_column: str = "forward_return",
    forecast_calibration_config: ScoreForecastCalibrationConfig | None = None,
) -> dict[str, Any]:
    """Write composite score parquet partitions for each method."""

    if not diagnostics_label_column:
        raise ValueError("diagnostics_label_column must be non-empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    features = [factor.feature for factor in candidates]
    methods: dict[str, dict[str, Any]] = {}
    for method, weights in weights_by_method.items():
        effective_weights = cap_factor_weights(weights, max_weight=max_factor_weight)
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for old_path in method_dir.glob("score_*.parquet"):
            old_path.unlink()
        diagnostics_dir = method_dir / "diagnostics"
        if diagnostics_top_n is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            for old_path in diagnostics_dir.glob("factor_contribution_*.csv"):
                old_path.unlink()
        row_count = 0
        partition_count = 0
        diagnostics_paths: list[str] = []
        calibration_observations: list[pd.DataFrame] = []
        calibration_paths: list[str] = []
        for dataset_path in dataset_paths:
            columns = ["timestamp", "instrument_id", *features]
            if diagnostics_top_n is not None:
                columns.append(diagnostics_label_column)
            if forecast_calibration_config is not None:
                columns.append(forecast_calibration_config.label_column)
            columns = list(dict.fromkeys(columns))
            frame = pd.read_parquet(
                dataset_path,
                columns=columns,
            )
            partition = dataset_path.stem.removeprefix("dataset_")
            health = _factor_health_for_partition(
                factor_health_schedule,
                timestamps=frame["timestamp"],
            )
            scores = build_composite_scores(
                frame,
                candidates=candidates,
                weights=effective_weights,
                factor_health=health,
                max_factor_contribution_share=max_factor_contribution_share,
            )
            if forecast_calibration_config is not None:
                joined = scores.loc[:, ["timestamp", "instrument_id", "score"]].merge(
                    frame.loc[
                        :,
                        [
                            "timestamp",
                            "instrument_id",
                            forecast_calibration_config.label_column,
                        ],
                    ],
                    on=["timestamp", "instrument_id"],
                    how="inner",
                )
                current_observations = score_forecast_calibration_observations(
                    joined,
                    forecast_calibration_config,
                )
                calibration_inputs = (
                    [*calibration_observations, current_observations]
                    if not current_observations.empty
                    else calibration_observations
                )
                calibration = (
                    build_score_forecast_calibration_from_observations(
                        pd.concat(calibration_inputs, ignore_index=True),
                        forecast_calibration_config,
                    )
                    if calibration_inputs
                    else pd.DataFrame()
                )
                scores = apply_score_forecast_calibration(
                    scores,
                    calibration,
                    forecast_calibration_config,
                )
                if not current_observations.empty:
                    calibration_observations.append(current_observations)
            score_path = method_dir / f"score_{partition}.parquet"
            scores.to_parquet(score_path, index=False)
            if diagnostics_top_n is not None:
                diagnostics = factor_contribution_diagnostics(
                    frame,
                    scores=scores,
                    candidates=candidates,
                    weights=effective_weights,
                    factor_health=health,
                    max_factor_contribution_share=max_factor_contribution_share,
                    top_n=diagnostics_top_n,
                    label_column=diagnostics_label_column,
                )
                diagnostics_path = diagnostics_dir / f"factor_contribution_{partition}.csv"
                diagnostics.to_csv(diagnostics_path, index=False)
                diagnostics_paths.append(str(diagnostics_path))
            row_count += len(scores)
            partition_count += 1
            del frame, scores
        if forecast_calibration_config is not None and calibration_observations:
            calibration_dir = method_dir / "calibration"
            calibration_dir.mkdir(parents=True, exist_ok=True)
            for old_path in calibration_dir.glob("score_forecast_calibration*.csv"):
                old_path.unlink()
            calibration = build_score_forecast_calibration_from_observations(
                pd.concat(calibration_observations, ignore_index=True),
                forecast_calibration_config,
            )
            calibration_path = calibration_dir / "score_forecast_calibration.csv"
            calibration.to_csv(calibration_path, index=False)
            calibration_paths.append(str(calibration_path))
        methods[method] = {
            "path": str(method_dir / "*.parquet"),
            "weights": effective_weights,
            "raw_weights": weights,
            "row_count": row_count,
            "partition_count": partition_count,
            "factor_contribution_diagnostics": diagnostics_paths,
            "score_forecast_calibration": calibration_paths,
        }
    return {
        "candidate_features": [factor.feature for factor in candidates],
        "methods": methods,
    }


def factor_contribution_diagnostics(
    frame: pd.DataFrame,
    *,
    scores: pd.DataFrame,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
    factor_health: pd.DataFrame | None,
    max_factor_contribution_share: float | None = None,
    top_n: int,
    label_column: str = "forward_return",
) -> pd.DataFrame:
    """Summarize factor contribution concentration in top-score baskets."""

    if not label_column:
        raise ValueError("label_column must be non-empty")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    _require_columns(frame, ("timestamp", "instrument_id", label_column))
    contribution_frame = _factor_contribution_frame(
        frame,
        candidates=candidates,
        weights=weights,
        factor_health=factor_health,
        max_factor_contribution_share=max_factor_contribution_share,
    )
    joined = scores.loc[:, ["timestamp", "instrument_id", "score"]].merge(
        contribution_frame,
        on=["timestamp", "instrument_id"],
        how="left",
    )
    joined = joined.merge(
        frame.loc[:, ["timestamp", "instrument_id", label_column]],
        on=["timestamp", "instrument_id"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    contribution_columns = [
        f"contribution_{factor.feature}" for factor in candidates if f"contribution_{factor.feature}" in joined.columns
    ]
    for timestamp, group in joined.groupby("timestamp", sort=True):
        top = group.nlargest(min(top_n, len(group)), "score")
        abs_by_factor = {
            column.removeprefix("contribution_"): float(top[column].abs().sum())
            for column in contribution_columns
        }
        total_abs = sum(abs_by_factor.values())
        ordered = sorted(abs_by_factor.items(), key=lambda item: item[1], reverse=True)
        largest_feature = ordered[0][0] if ordered else None
        largest_share = ordered[0][1] / total_abs if total_abs > 0 and ordered else 0.0
        top_two_share = (
            sum(value for _, value in ordered[:2]) / total_abs if total_abs > 0 else 0.0
        )
        rows.append(
            {
                "timestamp": timestamp,
                "top_n": int(len(top)),
                "label_column": label_column,
                "top_score_mean_label": _mean(top[label_column]),
                "largest_contribution_feature": largest_feature,
                "largest_abs_contribution_share": largest_share,
                "top_two_abs_contribution_share": top_two_share,
                "total_abs_contribution": total_abs,
            }
        )
    return pd.DataFrame(rows)


def _factor_health_observations(
    frame: pd.DataFrame,
    *,
    candidates: tuple[CandidateFactor, ...],
    top_n: int,
    label_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        for factor in candidates:
            valid = group.dropna(subset=[factor.feature, label_column]).copy()
            if valid.empty:
                rank_ic = None
                top_label = None
                bottom_label = None
                spread = None
                top_count = 0
            else:
                ranks = valid[factor.feature].rank(method="average", pct=True)
                oriented = (ranks - 0.5) * factor.direction
                rank_ic = _correlation(oriented, valid[label_column])
                n = min(top_n, len(valid))
                top = valid.loc[oriented.nlargest(n).index] if n else valid
                bottom = valid.loc[oriented.nsmallest(n).index] if n else valid
                top_label = _mean(top[label_column])
                bottom_label = _mean(bottom[label_column])
                spread = (
                    top_label - bottom_label
                    if top_label is not None and bottom_label is not None
                    else None
                )
                top_count = int(n)
            rows.append(
                {
                    "timestamp": timestamp,
                    "feature": factor.feature,
                    "label_column": label_column,
                    "factor_valid_count": int(len(valid)),
                    "factor_top_count": top_count,
                    "factor_rank_ic": rank_ic,
                    "factor_top_label": top_label,
                    "factor_bottom_label": bottom_label,
                    "factor_top_minus_bottom_label": spread,
                }
            )
    return rows


def _factor_health_scale_lookup(
    factor_health: pd.DataFrame | None,
) -> dict[str, pd.Series]:
    if factor_health is None or factor_health.empty:
        return {}
    _require_columns(factor_health, ("timestamp", "feature", "weight_scale"))
    lookup: dict[str, pd.Series] = {}
    for feature, group in factor_health.groupby("feature", sort=False):
        series = group.drop_duplicates("timestamp", keep="last").set_index("timestamp")[
            "weight_scale"
        ]
        lookup[str(feature)] = series.astype(float)
    return lookup


def _factor_health_component_scales(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    _require_columns(frame, ("timestamp", "feature", "weight_scale"))
    output = frame.loc[:, ["timestamp", "feature", "weight_scale"]].copy()
    output["timestamp"] = output["timestamp"].astype(str)
    output["feature"] = output["feature"].astype(str)
    output["weight_scale"] = pd.to_numeric(output["weight_scale"], errors="coerce")
    if output["weight_scale"].isna().any():
        raise ValueError(f"{label} factor health schedule contains invalid weight_scale")
    if not output["weight_scale"].between(0.0, 1.0).all():
        raise ValueError(f"{label} factor health schedule scales must be in [0, 1]")
    duplicates = output.duplicated(["timestamp", "feature"], keep=False)
    if bool(duplicates.any()):
        raise ValueError(f"{label} factor health schedule has duplicate rows")
    return output.rename(columns={"weight_scale": f"{label}_weight_scale"})


def _state_conditioned_regime_weight(
    frame: pd.DataFrame,
    *,
    feature: str,
    mode: str,
    threshold: float,
) -> pd.DataFrame:
    _require_columns(frame, ("timestamp", "feature", "weight_scale"))
    selected = frame.loc[frame["feature"].astype(str) == feature].copy()
    if selected.empty:
        raise ValueError(f"regime schedule has no rows for feature: {feature}")
    selected["timestamp"] = selected["timestamp"].astype(str)
    selected["regime_selector_scale"] = pd.to_numeric(
        selected["weight_scale"],
        errors="coerce",
    )
    if selected["regime_selector_scale"].isna().any():
        raise ValueError("regime schedule contains invalid weight_scale")
    if mode == "select":
        selected["regime_weight"] = (
            selected["regime_selector_scale"] < threshold
        ).astype(float)
    else:
        selected["regime_weight"] = (1.0 - selected["regime_selector_scale"]).clip(
            0.0,
            1.0,
        )
    return selected.loc[:, ["timestamp", "regime_selector_scale", "regime_weight"]]


def _effective_factor_weight(
    timestamps: pd.Series,
    *,
    feature: str,
    base_weight: float,
    health_scales: dict[str, pd.Series],
) -> pd.Series:
    if feature not in health_scales:
        return pd.Series(float(base_weight), index=timestamps.index)
    scale = timestamps.map(health_scales[feature]).fillna(1.0).astype(float)
    return scale * float(base_weight)


def _factor_health_for_partition(
    factor_health_schedule: pd.DataFrame | None,
    *,
    timestamps: pd.Series,
) -> pd.DataFrame | None:
    if factor_health_schedule is None or factor_health_schedule.empty:
        return None
    timestamp_values = pd.Index(timestamps.drop_duplicates())
    return factor_health_schedule[
        factor_health_schedule["timestamp"].isin(timestamp_values)
    ].copy()


def _factor_contribution_frame(
    frame: pd.DataFrame,
    *,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
    factor_health: pd.DataFrame | None,
    max_factor_contribution_share: float | None = None,
) -> pd.DataFrame:
    output = frame.loc[:, ["timestamp", "instrument_id"]].copy()
    health_scales = _factor_health_scale_lookup(factor_health)
    contributions: dict[str, pd.Series] = {}
    for factor in candidates:
        weight = float(weights.get(factor.feature, 0.0))
        effective_weight = _effective_factor_weight(
            frame["timestamp"],
            feature=factor.feature,
            base_weight=weight,
            health_scales=health_scales,
        )
        ranks = frame.groupby("timestamp", sort=False)[factor.feature].rank(
            method="average",
            pct=True,
        )
        contributions[factor.feature] = (ranks - 0.5) * factor.direction * effective_weight
    if contributions:
        contribution_frame = _cap_factor_contributions(
            pd.DataFrame(contributions, index=frame.index),
            max_share=max_factor_contribution_share,
        )
        for feature in contribution_frame.columns:
            output[f"contribution_{feature}"] = contribution_frame[feature]
    return output


def _cap_factor_contributions(
    contributions: pd.DataFrame,
    *,
    max_share: float | None,
) -> pd.DataFrame:
    if max_share is None or contributions.shape[1] <= 1:
        return contributions
    if not 0 < max_share <= 1:
        raise ValueError("max_share must be in (0, 1]")
    if max_share * contributions.shape[1] < 1.0:
        raise ValueError("max_share is too low for the number of contribution columns")
    capped = contributions.fillna(0.0).copy()
    for _ in range(capped.shape[1]):
        changed = False
        abs_values = capped.abs()
        total_abs = abs_values.sum(axis=1)
        for column in capped.columns:
            column_abs = abs_values[column]
            rest_abs = total_abs - column_abs
            limit = max_share * rest_abs / (1.0 - max_share)
            limit = limit.where(rest_abs > 0.0, 0.0)
            over = column_abs > limit
            if bool(over.any()):
                capped.loc[over, column] = (
                    capped.loc[over, column].clip(lower=-limit.loc[over], upper=limit.loc[over])
                )
                changed = True
        if not changed:
            break
    return capped


def _linear_score(series: pd.Series, *, floor: float, ceiling: float) -> pd.Series:
    return ((series - floor) / (ceiling - floor)).clip(lower=0.0, upper=1.0)


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 2:
        return None
    value = valid["left"].corr(valid["right"], method="spearman")
    if pd.isna(value):
        return None
    return float(value)


def _mean(series: pd.Series) -> float | None:
    value = series.dropna().mean()
    if pd.isna(value):
        return None
    return float(value)


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0:
        raise ValueError("cannot normalize zero weights")
    return {key: max(float(value), 0.0) / total for key, value in values.items()}


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
