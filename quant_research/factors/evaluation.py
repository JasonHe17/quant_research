"""Single-factor evaluation utilities for alpha datasets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from quant_research.models import infer_feature_columns


@dataclass(frozen=True, slots=True)
class SingleFactorEvaluationConfig:
    """Configuration for cross-sectional single-factor diagnostics."""

    label_column: str = "forward_return"
    feature_columns: tuple[str, ...] = ()
    top_n: int = 50
    quantiles: int = 5
    correlation_method: str = "spearman"

    def __post_init__(self) -> None:
        if not self.label_column:
            raise ValueError("label_column must be non-empty")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.quantiles <= 1:
            raise ValueError("quantiles must be greater than 1")
        if self.correlation_method not in {"pearson", "spearman"}:
            raise ValueError("correlation_method must be pearson or spearman")


@dataclass(frozen=True, slots=True)
class SingleFactorEvaluationResult:
    """Single-factor evaluation tables."""

    summary: pd.DataFrame
    by_timestamp: pd.DataFrame
    quantile_by_timestamp: pd.DataFrame
    quantile_returns: pd.DataFrame
    feature_correlation: pd.DataFrame


def evaluate_single_factors(
    frame: pd.DataFrame,
    config: SingleFactorEvaluationConfig | None = None,
) -> SingleFactorEvaluationResult:
    """Evaluate alpha columns against a forward-return label."""

    config = config or SingleFactorEvaluationConfig()
    _require_columns(frame, ("timestamp", "instrument_id", config.label_column))
    feature_columns = (
        config.feature_columns
        if config.feature_columns
        else infer_feature_columns(frame, label_column=config.label_column)
    )
    _require_columns(frame, feature_columns)
    summary_rows: list[dict[str, object]] = []
    timestamp_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []
    for feature in feature_columns:
        by_timestamp = _evaluate_factor_by_timestamp(
            frame,
            feature_column=feature,
            label_column=config.label_column,
            top_n=config.top_n,
        )
        by_timestamp.insert(0, "feature", feature)
        timestamp_frames.append(by_timestamp)
        quantiles = _evaluate_factor_quantiles(
            frame,
            feature_column=feature,
            label_column=config.label_column,
            quantiles=config.quantiles,
        )
        quantiles_by_timestamp = quantiles.copy()
        quantiles_by_timestamp.insert(0, "feature", feature)
        quantile_frames.append(quantiles_by_timestamp)
        valid = frame.loc[frame[[feature, config.label_column]].notna().all(axis=1)]
        summary_rows.append(
            _summary_row(
                feature=feature,
                valid=valid,
                total_rows=len(frame),
                by_timestamp=by_timestamp,
            )
        )
    quantile_by_timestamp = (
        pd.concat(quantile_frames, ignore_index=True)
        if quantile_frames
        else pd.DataFrame()
    )
    feature_correlation = frame.loc[:, list(feature_columns)].corr(
        method=config.correlation_method
    )
    return SingleFactorEvaluationResult(
        summary=pd.DataFrame(summary_rows).sort_values(
            "spearman_rank_ic_mean",
            ascending=False,
            na_position="last",
        ),
        by_timestamp=pd.concat(timestamp_frames, ignore_index=True)
        if timestamp_frames
        else pd.DataFrame(),
        quantile_by_timestamp=quantile_by_timestamp,
        quantile_returns=_summarize_quantiles(quantile_by_timestamp),
        feature_correlation=feature_correlation,
    )


def _evaluate_factor_by_timestamp(
    frame: pd.DataFrame,
    *,
    feature_column: str,
    label_column: str,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    turnover_by_timestamp = _top_n_turnover_by_timestamp(
        frame,
        feature_column=feature_column,
        top_n=top_n,
    )
    for timestamp, group in frame.groupby("timestamp", sort=True):
        valid = group.loc[group[[feature_column, label_column]].notna().all(axis=1)]
        if valid.empty:
            continue
        top = valid.nlargest(min(top_n, len(valid)), feature_column)
        bottom = valid.nsmallest(min(top_n, len(valid)), feature_column)
        rows.append(
            {
                "timestamp": timestamp,
                "sample_count": len(valid),
                "pearson_ic": valid[feature_column].corr(valid[label_column]),
                "spearman_rank_ic": valid[feature_column].corr(
                    valid[label_column],
                    method="spearman",
                ),
                "top_n_mean_label": top[label_column].mean(),
                "bottom_n_mean_label": bottom[label_column].mean(),
                "top_minus_bottom_label": top[label_column].mean()
                - bottom[label_column].mean(),
                "top_n_turnover": turnover_by_timestamp.get(timestamp),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_factor_quantiles(
    frame: pd.DataFrame,
    *,
    feature_column: str,
    label_column: str,
    quantiles: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        valid = group.loc[group[[feature_column, label_column]].notna().all(axis=1)]
        if valid.empty:
            continue
        buckets = _rank_quantile_buckets(valid[feature_column], quantiles=quantiles)
        bucketed = valid.assign(quantile=buckets)
        for quantile, quantile_group in bucketed.groupby("quantile", sort=True):
            rows.append(
                {
                    "timestamp": timestamp,
                    "quantile": int(quantile),
                    "sample_count": len(quantile_group),
                    "mean_label": quantile_group[label_column].mean(),
                }
            )
    return pd.DataFrame(rows)


def _summarize_quantiles(quantile_by_timestamp: pd.DataFrame) -> pd.DataFrame:
    if quantile_by_timestamp.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "quantile",
                "timestamp_count",
                "sample_count",
                "mean_label",
            ]
        )
    return (
        quantile_by_timestamp.groupby(["feature", "quantile"], as_index=False)
        .agg(
            timestamp_count=("timestamp", "nunique"),
            sample_count=("sample_count", "sum"),
            mean_label=("mean_label", "mean"),
        )
        .sort_values("quantile")
        .reset_index(drop=True)
    )


def _rank_quantile_buckets(values: pd.Series, *, quantiles: int) -> pd.Series:
    ranks = values.rank(method="first", pct=True)
    buckets = (ranks * quantiles).apply(lambda value: min(math.ceil(value), quantiles))
    return buckets.astype(int)


def _top_n_turnover_by_timestamp(
    frame: pd.DataFrame,
    *,
    feature_column: str,
    top_n: int,
) -> dict[object, float | None]:
    previous: set[str] | None = None
    turnovers: dict[object, float | None] = {}
    for timestamp, group in frame.groupby("timestamp", sort=True):
        valid = group.loc[group[feature_column].notna()]
        if valid.empty:
            continue
        selected = set(
            valid.nlargest(min(top_n, len(valid)), feature_column)[
                "instrument_id"
            ].astype(str)
        )
        turnovers[timestamp] = (
            None
            if previous is None or not selected
            else 1.0 - len(selected & previous) / len(selected)
        )
        previous = selected
    return turnovers


def _summary_row(
    *,
    feature: str,
    valid: pd.DataFrame,
    total_rows: int,
    by_timestamp: pd.DataFrame,
) -> dict[str, object]:
    coverage = len(valid) / total_rows if total_rows else 0.0
    return {
        "feature": feature,
        "sample_count": len(valid),
        "coverage": coverage,
        "timestamp_count": int(by_timestamp["timestamp"].nunique())
        if not by_timestamp.empty
        else 0,
        "pearson_ic_mean": _nullable_float(by_timestamp["pearson_ic"].mean())
        if not by_timestamp.empty
        else None,
        "spearman_rank_ic_mean": _nullable_float(
            by_timestamp["spearman_rank_ic"].mean()
        )
        if not by_timestamp.empty
        else None,
        "top_n_mean_label": _nullable_float(by_timestamp["top_n_mean_label"].mean())
        if not by_timestamp.empty
        else None,
        "bottom_n_mean_label": _nullable_float(
            by_timestamp["bottom_n_mean_label"].mean()
        )
        if not by_timestamp.empty
        else None,
        "top_minus_bottom_label": _nullable_float(
            by_timestamp["top_minus_bottom_label"].mean()
        )
        if not by_timestamp.empty
        else None,
        "top_n_turnover": _nullable_float(by_timestamp["top_n_turnover"].mean())
        if not by_timestamp.empty
        else None,
    }


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
