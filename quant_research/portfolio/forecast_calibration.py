"""Forecast calibration utilities for score-based portfolio policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ScoreForecastCalibrationConfig:
    """Lagged rolling score-bucket calibration configuration."""

    lookback_windows: int = 20
    min_periods: int = 5
    label_lag_windows: int = 48
    bucket_count: int = 5
    label_column: str = "forward_return"
    default_edge_bps: float = 0.0
    default_risk_bps: float = 0.0
    risk_multiplier: float = 1.0
    max_abs_edge_bps: float | None = None

    def __post_init__(self) -> None:
        if self.lookback_windows <= 0:
            raise ValueError("lookback_windows must be positive")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if self.min_periods > self.lookback_windows:
            raise ValueError("min_periods must be <= lookback_windows")
        if self.label_lag_windows <= 0:
            raise ValueError("label_lag_windows must be positive")
        if self.bucket_count <= 1:
            raise ValueError("bucket_count must be greater than 1")
        if not self.label_column:
            raise ValueError("label_column must be non-empty")
        if self.default_risk_bps < 0:
            raise ValueError("default_risk_bps must be non-negative")
        if self.risk_multiplier < 0:
            raise ValueError("risk_multiplier must be non-negative")
        if self.max_abs_edge_bps is not None and self.max_abs_edge_bps <= 0:
            raise ValueError("max_abs_edge_bps must be positive")


def score_forecast_calibration_observations(
    scores_with_labels: pd.DataFrame,
    config: ScoreForecastCalibrationConfig,
) -> pd.DataFrame:
    """Aggregate timestamp/bucket score-label observations for calibration."""

    _require_columns(
        scores_with_labels,
        ("timestamp", "instrument_id", "score", config.label_column),
    )
    if scores_with_labels.empty:
        return _empty_observations()
    frame = scores_with_labels.loc[
        :,
        ["timestamp", "instrument_id", "score", config.label_column],
    ].dropna(subset=["score", config.label_column])
    if frame.empty:
        return _empty_observations()
    frame = frame.copy()
    frame["score_bucket"] = _score_buckets(frame, bucket_count=config.bucket_count)
    grouped = frame.groupby(["timestamp", "score_bucket"], sort=True)[config.label_column]
    observations = grouped.agg(
        bucket_mean_label="mean",
        bucket_std_label=lambda series: float(series.std(ddof=0) or 0.0),
        bucket_observation_count="count",
    ).reset_index()
    observations["score_bucket"] = observations["score_bucket"].astype(int)
    return observations.sort_values(["score_bucket", "timestamp"]).reset_index(drop=True)


def build_score_forecast_calibration(
    scores_with_labels: pd.DataFrame,
    config: ScoreForecastCalibrationConfig,
) -> pd.DataFrame:
    """Build a lagged rolling calibration schedule from raw score-label rows."""

    observations = score_forecast_calibration_observations(scores_with_labels, config)
    return build_score_forecast_calibration_from_observations(observations, config)


def build_score_forecast_calibration_from_partitions(
    score_dataset_paths: Iterable[tuple[Path, Path]],
    config: ScoreForecastCalibrationConfig,
) -> pd.DataFrame:
    """Build calibration schedule by streaming score/dataset partition pairs."""

    observations: list[pd.DataFrame] = []
    for score_path, dataset_path in score_dataset_paths:
        scores = pd.read_parquet(
            score_path,
            columns=["timestamp", "instrument_id", "score"],
        )
        labels = pd.read_parquet(
            dataset_path,
            columns=["timestamp", "instrument_id", config.label_column],
        )
        joined = scores.merge(labels, on=["timestamp", "instrument_id"], how="inner")
        observations.append(score_forecast_calibration_observations(joined, config))
        del scores, labels, joined
    if not observations:
        return _empty_calibration()
    observation_frame = pd.concat(observations, ignore_index=True)
    return build_score_forecast_calibration_from_observations(observation_frame, config)


def build_score_forecast_calibration_from_observations(
    observations: pd.DataFrame,
    config: ScoreForecastCalibrationConfig,
) -> pd.DataFrame:
    """Build lagged rolling expected-edge and risk schedule from bucket stats."""

    if observations.empty:
        return _empty_calibration()
    _require_columns(
        observations,
        (
            "timestamp",
            "score_bucket",
            "bucket_mean_label",
            "bucket_std_label",
            "bucket_observation_count",
        ),
    )
    schedules: list[pd.DataFrame] = []
    ordered = observations.sort_values(["score_bucket", "timestamp"]).reset_index(drop=True)
    for bucket, group in ordered.groupby("score_bucket", sort=True):
        current = group.sort_values("timestamp").reset_index(drop=True).copy()
        shifted_mean = current["bucket_mean_label"].shift(config.label_lag_windows)
        shifted_std = current["bucket_std_label"].shift(config.label_lag_windows)
        shifted_count = current["bucket_observation_count"].shift(config.label_lag_windows)
        window_count = shifted_mean.rolling(
            config.lookback_windows,
            min_periods=config.min_periods,
        ).count()
        edge_bps = (
            shifted_mean.rolling(config.lookback_windows, min_periods=config.min_periods).mean()
            * 10_000.0
        )
        if config.max_abs_edge_bps is not None:
            edge_bps = edge_bps.clip(
                lower=-config.max_abs_edge_bps,
                upper=config.max_abs_edge_bps,
            )
        risk_bps = (
            shifted_std.rolling(config.lookback_windows, min_periods=config.min_periods).mean()
            * 10_000.0
            * config.risk_multiplier
        )
        sample_count = shifted_count.rolling(
            config.lookback_windows,
            min_periods=config.min_periods,
        ).sum()
        calibrated = edge_bps.notna()
        current["score_bucket"] = int(bucket)
        current["expected_edge_bps"] = edge_bps.where(
            calibrated,
            config.default_edge_bps,
        )
        current["risk_penalty_bps"] = risk_bps.where(
            calibrated,
            config.default_risk_bps,
        ).clip(lower=0.0)
        current["calibration_window_count"] = window_count.fillna(0).astype(int)
        current["calibration_observation_count"] = sample_count.fillna(0).astype(int)
        current["forecast_calibration_reason"] = "warmup"
        current.loc[calibrated, "forecast_calibration_reason"] = "calibrated"
        schedules.append(
            current.loc[
                :,
                [
                    "timestamp",
                    "score_bucket",
                    "expected_edge_bps",
                    "risk_penalty_bps",
                    "calibration_window_count",
                    "calibration_observation_count",
                    "forecast_calibration_reason",
                ],
            ]
        )
    return (
        pd.concat(schedules, ignore_index=True)
        .sort_values(["timestamp", "score_bucket"])
        .reset_index(drop=True)
    )


def apply_score_forecast_calibration(
    scores: pd.DataFrame,
    calibration: pd.DataFrame,
    config: ScoreForecastCalibrationConfig,
) -> pd.DataFrame:
    """Attach calibrated optimizer forecast fields to score rows."""

    _require_columns(scores, ("timestamp", "instrument_id", "score"))
    output = scores.copy()
    if output.empty:
        return output.assign(
            forecast_calibration_bucket=pd.Series(dtype="int64"),
            expected_edge_bps=pd.Series(dtype="float64"),
            risk_penalty_bps=pd.Series(dtype="float64"),
            forecast_calibration_reason=pd.Series(dtype="object"),
        )
    output["forecast_calibration_bucket"] = _score_buckets(
        output,
        bucket_count=config.bucket_count,
    )
    if calibration.empty:
        output["expected_edge_bps"] = config.default_edge_bps
        output["risk_penalty_bps"] = config.default_risk_bps
        output["forecast_calibration_reason"] = "missing_calibration"
        return output
    _require_columns(
        calibration,
        (
            "timestamp",
            "score_bucket",
            "expected_edge_bps",
            "risk_penalty_bps",
            "forecast_calibration_reason",
        ),
    )
    calibrated = output.merge(
        calibration.loc[
            :,
            [
                "timestamp",
                "score_bucket",
                "expected_edge_bps",
                "risk_penalty_bps",
                "forecast_calibration_reason",
                "calibration_window_count",
                "calibration_observation_count",
            ],
        ].rename(columns={"score_bucket": "forecast_calibration_bucket"}),
        on=["timestamp", "forecast_calibration_bucket"],
        how="left",
    )
    calibrated["expected_edge_bps"] = calibrated["expected_edge_bps"].fillna(
        config.default_edge_bps
    )
    calibrated["risk_penalty_bps"] = calibrated["risk_penalty_bps"].fillna(
        config.default_risk_bps
    )
    calibrated["forecast_calibration_reason"] = calibrated[
        "forecast_calibration_reason"
    ].fillna("missing_calibration")
    if "calibration_window_count" in calibrated.columns:
        calibrated["calibration_window_count"] = (
            calibrated["calibration_window_count"].fillna(0).astype(int)
        )
    if "calibration_observation_count" in calibrated.columns:
        calibrated["calibration_observation_count"] = (
            calibrated["calibration_observation_count"].fillna(0).astype(int)
        )
    return calibrated


def _score_buckets(frame: pd.DataFrame, *, bucket_count: int) -> pd.Series:
    ranks = frame.groupby("timestamp", sort=False)["score"].rank(
        method="average",
        pct=True,
    )
    buckets = (ranks * bucket_count).apply(np.ceil)
    buckets = buckets.clip(lower=1, upper=bucket_count).fillna(1).astype(int)
    return buckets


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "score_bucket",
            "bucket_mean_label",
            "bucket_std_label",
            "bucket_observation_count",
        ]
    )


def _empty_calibration() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "score_bucket",
            "expected_edge_bps",
            "risk_penalty_bps",
            "calibration_window_count",
            "calibration_observation_count",
            "forecast_calibration_reason",
        ]
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
