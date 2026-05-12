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
    include_feature_correlation: bool = True

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
    timestamp_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    previous_top_by_feature: dict[str, set[str] | None] = {
        feature: None for feature in feature_columns
    }
    for timestamp, group in frame.groupby("timestamp", sort=True):
        labels = group[config.label_column]
        for feature in feature_columns:
            valid_mask = group[feature].notna() & labels.notna()
            if not valid_mask.any():
                continue
            valid = group.loc[
                valid_mask,
                ["instrument_id", feature, config.label_column],
            ]
            feature_values = valid[feature]
            label_values = valid[config.label_column]
            top = valid.nlargest(min(config.top_n, len(valid)), feature)
            bottom = valid.nsmallest(min(config.top_n, len(valid)), feature)
            selected = set(top["instrument_id"].astype(str))
            previous = previous_top_by_feature[feature]
            turnover = (
                None
                if previous is None or not selected
                else 1.0 - len(selected & previous) / len(selected)
            )
            previous_top_by_feature[feature] = selected
            timestamp_rows.append(
                {
                    "feature": feature,
                    "timestamp": timestamp,
                    "sample_count": len(valid),
                    "pearson_ic": feature_values.corr(label_values),
                    "spearman_rank_ic": feature_values.corr(
                        label_values,
                        method="spearman",
                    ),
                    "top_n_mean_label": top[config.label_column].mean(),
                    "bottom_n_mean_label": bottom[config.label_column].mean(),
                    "top_minus_bottom_label": top[config.label_column].mean()
                    - bottom[config.label_column].mean(),
                    "top_n_turnover": turnover,
                }
            )
            buckets = _rank_quantile_buckets(feature_values, quantiles=config.quantiles)
            bucketed = valid.assign(quantile=buckets)
            for quantile, quantile_group in bucketed.groupby("quantile", sort=True):
                quantile_rows.append(
                    {
                        "feature": feature,
                        "timestamp": timestamp,
                        "quantile": int(quantile),
                        "sample_count": len(quantile_group),
                        "mean_label": quantile_group[config.label_column].mean(),
                    }
                )
    by_timestamp = pd.DataFrame(timestamp_rows)
    quantile_by_timestamp = pd.DataFrame(quantile_rows)
    feature_correlation = (
        frame.loc[:, list(feature_columns)].corr(method=config.correlation_method)
        if config.include_feature_correlation
        else pd.DataFrame(index=feature_columns, columns=feature_columns)
    )
    return SingleFactorEvaluationResult(
        summary=_summarize_by_feature(by_timestamp, total_rows=len(frame)),
        by_timestamp=by_timestamp,
        quantile_by_timestamp=quantile_by_timestamp,
        quantile_returns=_summarize_quantiles(quantile_by_timestamp),
        feature_correlation=feature_correlation,
    )


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


def _summarize_by_feature(by_timestamp: pd.DataFrame, *, total_rows: int) -> pd.DataFrame:
    if by_timestamp.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "sample_count",
                "coverage",
                "timestamp_count",
                "pearson_ic_mean",
                "spearman_rank_ic_mean",
                "top_n_mean_label",
                "bottom_n_mean_label",
                "top_minus_bottom_label",
                "top_n_turnover",
            ]
        )
    rows: list[dict[str, object]] = []
    for feature, group in by_timestamp.groupby("feature", sort=True):
        sample_count = int(group["sample_count"].sum())
        rows.append(
            {
                "feature": feature,
                "sample_count": sample_count,
                "coverage": sample_count / total_rows if total_rows else 0.0,
                "timestamp_count": int(group["timestamp"].nunique()),
                "pearson_ic_mean": _nullable_float(group["pearson_ic"].mean()),
                "spearman_rank_ic_mean": _nullable_float(
                    group["spearman_rank_ic"].mean()
                ),
                "top_n_mean_label": _nullable_float(group["top_n_mean_label"].mean()),
                "bottom_n_mean_label": _nullable_float(
                    group["bottom_n_mean_label"].mean()
                ),
                "top_minus_bottom_label": _nullable_float(
                    group["top_minus_bottom_label"].mean()
                ),
                "top_n_turnover": _nullable_float(group["top_n_turnover"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "spearman_rank_ic_mean",
        ascending=False,
        na_position="last",
    )


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
