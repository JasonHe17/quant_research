"""Single-factor evaluation utilities for alpha datasets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from quant_research.models import infer_feature_columns


@dataclass(frozen=True, slots=True)
class SingleFactorEvaluationConfig:
    """Configuration for cross-sectional single-factor diagnostics."""

    label_column: str = "forward_return"
    horizon_label_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    top_n: int = 50
    quantiles: int = 5
    correlation_method: str = "spearman"
    include_feature_correlation: bool = True
    group_columns: tuple[str, ...] = ()
    liquidity_columns: tuple[str, ...] = ("turnover", "volume")
    cost_bps: float = 0.0
    multiple_testing_alpha: float = 0.05

    def __post_init__(self) -> None:
        if not self.label_column:
            raise ValueError("label_column must be non-empty")
        if any(not column for column in self.horizon_label_columns):
            raise ValueError("horizon_label_columns values must be non-empty")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.quantiles <= 1:
            raise ValueError("quantiles must be greater than 1")
        if self.correlation_method not in {"pearson", "spearman"}:
            raise ValueError("correlation_method must be pearson or spearman")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if not 0 < self.multiple_testing_alpha < 1:
            raise ValueError("multiple_testing_alpha must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class SingleFactorEvaluationResult:
    """Single-factor evaluation tables."""

    summary: pd.DataFrame
    by_timestamp: pd.DataFrame
    quantile_by_timestamp: pd.DataFrame
    quantile_returns: pd.DataFrame
    feature_correlation: pd.DataFrame
    decay_by_label: pd.DataFrame
    group_summary: pd.DataFrame
    multiple_testing: pd.DataFrame


def evaluate_single_factors(
    frame: pd.DataFrame,
    config: SingleFactorEvaluationConfig | None = None,
) -> SingleFactorEvaluationResult:
    """Evaluate alpha columns against a forward-return label."""

    config = config or SingleFactorEvaluationConfig()
    _require_columns(frame, ("timestamp", "instrument_id", config.label_column))
    label_columns = _label_columns(config)
    feature_columns = (
        config.feature_columns
        if config.feature_columns
        else infer_feature_columns(
            frame,
            label_column=config.label_column,
            exclude_columns=_label_metadata_columns(label_columns),
        )
    )
    _require_columns(frame, feature_columns)
    timestamp_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    label_ic_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    previous_top_by_feature: dict[str, set[str] | None] = {
        feature: None for feature in feature_columns
    }
    previous_rank_by_feature: dict[str, pd.Series | None] = {
        feature: None for feature in feature_columns
    }
    _require_columns(frame, tuple(label_columns))
    secondary_label_columns = tuple(
        label_column
        for label_column in label_columns
        if label_column != config.label_column
    )
    group_columns = tuple(column for column in config.group_columns if column in frame.columns)
    liquidity_columns = tuple(
        column for column in config.liquidity_columns if column in frame.columns
    )
    for timestamp, group in frame.groupby("timestamp", sort=True):
        labels = group[config.label_column]
        for feature in feature_columns:
            valid_mask = group[feature].notna() & labels.notna()
            if not valid_mask.any():
                continue
            selected_columns = [
                "instrument_id",
                feature,
                config.label_column,
                *group_columns,
                *liquidity_columns,
            ]
            valid = group.loc[
                valid_mask,
                list(dict.fromkeys(selected_columns)),
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
            rank_autocorrelation = _rank_autocorrelation(
                valid,
                feature=feature,
                previous=previous_rank_by_feature[feature],
            )
            previous_rank_by_feature[feature] = valid.set_index("instrument_id")[
                feature
            ].rank(method="average")
            top_minus_bottom = top[config.label_column].mean() - bottom[
                config.label_column
            ].mean()
            cost_adjusted = (
                None
                if turnover is None
                else top_minus_bottom - turnover * config.cost_bps / 10_000.0
            )
            liquidity_summary = _liquidity_timestamp_summary(
                valid,
                top=top,
                liquidity_columns=liquidity_columns,
            )
            pearson_ic = _correlation(
                feature_values,
                label_values,
                method="pearson",
            )
            spearman_rank_ic = _correlation(
                feature_values,
                label_values,
                method="spearman",
            )
            timestamp_rows.append(
                {
                    "feature": feature,
                    "timestamp": timestamp,
                    "sample_count": len(valid),
                    "pearson_ic": pearson_ic,
                    "spearman_rank_ic": spearman_rank_ic,
                    "top_n_mean_label": top[config.label_column].mean(),
                    "bottom_n_mean_label": bottom[config.label_column].mean(),
                    "top_minus_bottom_label": top_minus_bottom,
                    "cost_adjusted_top_minus_bottom_label": cost_adjusted,
                    "top_n_turnover": turnover,
                    "rank_autocorrelation": rank_autocorrelation,
                    **liquidity_summary,
                }
            )
            label_ic_rows.append(
                {
                    "feature": feature,
                    "label_column": config.label_column,
                    "timestamp": timestamp,
                    "sample_count": len(valid),
                    "pearson_ic": pearson_ic,
                    "spearman_rank_ic": spearman_rank_ic,
                }
            )
            label_ic_rows.extend(
                _label_ic_rows(
                    group,
                    timestamp=timestamp,
                    feature=feature,
                    label_columns=secondary_label_columns,
                )
            )
            group_rows.extend(
                _group_ic_rows(
                    valid,
                    timestamp=timestamp,
                    feature=feature,
                    label_column=config.label_column,
                    group_columns=group_columns,
                )
            )
            buckets = _rank_quantile_buckets(feature_values, quantiles=config.quantiles)
            bucket_stats = label_values.groupby(buckets, sort=True).agg(["size", "mean"])
            for quantile, stats in bucket_stats.iterrows():
                quantile_rows.append(
                    {
                        "feature": feature,
                        "timestamp": timestamp,
                        "quantile": int(quantile),
                        "sample_count": int(stats["size"]),
                        "mean_label": stats["mean"],
                    }
                )
    by_timestamp = pd.DataFrame(timestamp_rows)
    quantile_by_timestamp = pd.DataFrame(quantile_rows)
    feature_correlation = _feature_correlation(
        frame,
        feature_columns=feature_columns,
        method=config.correlation_method,
        enabled=config.include_feature_correlation,
    )
    decay_by_label = _summarize_label_ic(pd.DataFrame(label_ic_rows))
    group_summary = _summarize_group_ic(pd.DataFrame(group_rows))
    summary = _summarize_by_feature(
        by_timestamp,
        total_rows=len(frame),
        cost_bps=config.cost_bps,
    )
    multiple_testing = _multiple_testing(summary, alpha=config.multiple_testing_alpha)
    return SingleFactorEvaluationResult(
        summary=summary,
        by_timestamp=by_timestamp,
        quantile_by_timestamp=quantile_by_timestamp,
        quantile_returns=_summarize_quantiles(quantile_by_timestamp),
        feature_correlation=feature_correlation,
        decay_by_label=decay_by_label,
        group_summary=group_summary,
        multiple_testing=multiple_testing,
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
    summary = (
        quantile_by_timestamp.groupby(["feature", "quantile"], as_index=False)
        .agg(
            timestamp_count=("timestamp", "nunique"),
            sample_count=("sample_count", "sum"),
            mean_label=("mean_label", "mean"),
        )
        .sort_values("quantile")
        .reset_index(drop=True)
    )
    spreads: list[dict[str, object]] = []
    for feature, group in summary.groupby("feature", sort=True):
        first = group.loc[group["quantile"] == group["quantile"].min(), "mean_label"]
        last = group.loc[group["quantile"] == group["quantile"].max(), "mean_label"]
        if first.empty or last.empty:
            continue
        spreads.append(
            {
                "feature": feature,
                "quantile": "long_short",
                "timestamp_count": int(group["timestamp_count"].max()),
                "sample_count": int(group["sample_count"].sum()),
                "mean_label": float(last.iloc[0] - first.iloc[0]),
            }
        )
    if spreads:
        summary = pd.concat([summary, pd.DataFrame(spreads)], ignore_index=True)
    summary["quantile"] = summary["quantile"].astype(str)
    return summary


def _rank_quantile_buckets(values: pd.Series, *, quantiles: int) -> pd.Series:
    ranks = values.rank(method="first", pct=True)
    buckets = (ranks * quantiles).apply(lambda value: min(math.ceil(value), quantiles))
    return buckets.astype(int)


def _summarize_by_feature(
    by_timestamp: pd.DataFrame,
    *,
    total_rows: int,
    cost_bps: float,
) -> pd.DataFrame:
    if by_timestamp.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "sample_count",
                "coverage",
                "timestamp_count",
                "pearson_ic_mean",
                "spearman_rank_ic_mean",
                "spearman_rank_ic_std",
                "spearman_rank_ic_standard_error",
                "spearman_rank_ic_t_stat",
                "spearman_rank_ic_ir",
                "spearman_rank_ic_positive_rate",
                "top_n_mean_label",
                "bottom_n_mean_label",
                "top_minus_bottom_label",
                "cost_adjusted_top_minus_bottom_label",
                "top_n_turnover",
                "rank_autocorrelation",
            ]
        )
    rows: list[dict[str, object]] = []
    for feature, group in by_timestamp.groupby("feature", sort=True):
        sample_count = int(group["sample_count"].sum())
        rank_ic = group["spearman_rank_ic"].dropna()
        rank_ic_std = rank_ic.std(ddof=1)
        rank_ic_se = rank_ic_std / math.sqrt(len(rank_ic)) if len(rank_ic) else None
        rank_ic_mean = rank_ic.mean()
        rank_ic_t = rank_ic_mean / rank_ic_se if rank_ic_se else None
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std else None
        top_turnover = group["top_n_turnover"].mean()
        top_minus_bottom = group["top_minus_bottom_label"].mean()
        rows.append(
            {
                "feature": feature,
                "sample_count": sample_count,
                "coverage": sample_count / total_rows if total_rows else 0.0,
                "timestamp_count": int(group["timestamp"].nunique()),
                "pearson_ic_mean": _nullable_float(group["pearson_ic"].mean()),
                "spearman_rank_ic_mean": _nullable_float(rank_ic_mean),
                "spearman_rank_ic_std": _nullable_float(rank_ic_std),
                "spearman_rank_ic_standard_error": _nullable_float(rank_ic_se),
                "spearman_rank_ic_t_stat": _nullable_float(rank_ic_t),
                "spearman_rank_ic_ir": _nullable_float(rank_ic_ir),
                "spearman_rank_ic_positive_rate": _nullable_float(
                    (rank_ic > 0).mean()
                ),
                "top_n_mean_label": _nullable_float(group["top_n_mean_label"].mean()),
                "bottom_n_mean_label": _nullable_float(
                    group["bottom_n_mean_label"].mean()
                ),
                "top_minus_bottom_label": _nullable_float(top_minus_bottom),
                "cost_adjusted_top_minus_bottom_label": _nullable_float(
                    top_minus_bottom - top_turnover * cost_bps / 10_000.0
                    if pd.notna(top_turnover)
                    else None
                ),
                "top_n_turnover": _nullable_float(top_turnover),
                "rank_autocorrelation": _nullable_float(
                    group["rank_autocorrelation"].mean()
                ),
                **_aggregate_optional_means(group),
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


def _label_columns(config: SingleFactorEvaluationConfig) -> tuple[str, ...]:
    return tuple(dict.fromkeys((config.label_column, *config.horizon_label_columns)))


def _label_metadata_columns(label_columns: tuple[str, ...]) -> tuple[str, ...]:
    columns: list[str] = []
    for label_column in label_columns:
        columns.extend(
            [
                label_column,
                f"{label_column}_rank",
                f"{label_column}_exit_timestamp",
                f"{label_column}_exit_price",
                f"{label_column}_exit_tradable_bar",
                f"{label_column}_exit_limit_up_open",
                f"{label_column}_exit_limit_down_open",
            ]
        )
    return tuple(dict.fromkeys(columns))


def _liquidity_timestamp_summary(
    valid: pd.DataFrame,
    *,
    top: pd.DataFrame,
    liquidity_columns: tuple[str, ...],
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for column in liquidity_columns:
        values[f"all_mean_{column}"] = _nullable_float(valid[column].mean())
        values[f"top_n_mean_{column}"] = _nullable_float(top[column].mean())
    return values


def _rank_autocorrelation(
    valid: pd.DataFrame,
    *,
    feature: str,
    previous: pd.Series | None,
) -> float | None:
    if previous is None:
        return None
    current = valid.set_index("instrument_id")[feature].rank(method="average")
    shared = current.index.intersection(previous.index)
    if len(shared) < 2:
        return None
    return _correlation(current.loc[shared], previous.loc[shared], method="pearson")


def _label_ic_rows(
    group: pd.DataFrame,
    *,
    timestamp: object,
    feature: str,
    label_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label_column in label_columns:
        valid = group.loc[group[feature].notna() & group[label_column].notna()]
        if valid.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "label_column": label_column,
                "timestamp": timestamp,
                "sample_count": len(valid),
                "pearson_ic": _correlation(
                    valid[feature],
                    valid[label_column],
                    method="pearson",
                ),
                "spearman_rank_ic": _correlation(
                    valid[feature],
                    valid[label_column],
                    method="spearman",
                ),
            }
        )
    return rows


def _group_ic_rows(
    valid: pd.DataFrame,
    *,
    timestamp: object,
    feature: str,
    label_column: str,
    group_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_column in group_columns:
        for group_value, group_frame in valid.groupby(group_column, sort=True):
            if len(group_frame) < 2:
                continue
            rows.append(
                {
                    "feature": feature,
                    "timestamp": timestamp,
                    "group_column": group_column,
                    "group_value": group_value,
                    "sample_count": len(group_frame),
                    "pearson_ic": _correlation(
                        group_frame[feature],
                        group_frame[label_column],
                        method="pearson",
                    ),
                    "spearman_rank_ic": _correlation(
                        group_frame[feature],
                        group_frame[label_column],
                        method="spearman",
                    ),
                }
            )
    return rows


def _feature_correlation(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    method: str,
    enabled: bool,
) -> pd.DataFrame:
    if not enabled:
        return pd.DataFrame(index=feature_columns, columns=feature_columns)
    features = frame.loc[:, list(feature_columns)]
    if method == "spearman":
        return features.rank(method="average").corr(method="pearson")
    return features.corr(method="pearson")


def _summarize_label_ic(label_ic: pd.DataFrame) -> pd.DataFrame:
    if label_ic.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "label_column",
                "timestamp_count",
                "sample_count",
                "pearson_ic_mean",
                "spearman_rank_ic_mean",
                "spearman_rank_ic_ir",
            ]
        )
    rows: list[dict[str, object]] = []
    for (feature, label_column), group in label_ic.groupby(
        ["feature", "label_column"], sort=True
    ):
        rank_ic = group["spearman_rank_ic"].dropna()
        rank_ic_std = rank_ic.std(ddof=1)
        rows.append(
            {
                "feature": feature,
                "label_column": label_column,
                "timestamp_count": int(group["timestamp"].nunique()),
                "sample_count": int(group["sample_count"].sum()),
                "pearson_ic_mean": _nullable_float(group["pearson_ic"].mean()),
                "spearman_rank_ic_mean": _nullable_float(rank_ic.mean()),
                "spearman_rank_ic_ir": _nullable_float(
                    rank_ic.mean() / rank_ic_std if rank_ic_std else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_group_ic(group_ic: pd.DataFrame) -> pd.DataFrame:
    if group_ic.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "group_column",
                "group_value",
                "timestamp_count",
                "sample_count",
                "pearson_ic_mean",
                "spearman_rank_ic_mean",
            ]
        )
    return (
        group_ic.groupby(["feature", "group_column", "group_value"], as_index=False)
        .agg(
            timestamp_count=("timestamp", "nunique"),
            sample_count=("sample_count", "sum"),
            pearson_ic_mean=("pearson_ic", "mean"),
            spearman_rank_ic_mean=("spearman_rank_ic", "mean"),
        )
        .reset_index(drop=True)
    )


def _multiple_testing(summary: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    columns = [
        "feature",
        "spearman_rank_ic_t_stat",
        "p_value",
        "q_value_bh",
        "significant_bh",
    ]
    if summary.empty or "spearman_rank_ic_t_stat" not in summary.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in summary.itertuples(index=False):
        t_stat = getattr(row, "spearman_rank_ic_t_stat")
        p_value = _normal_two_sided_p_value(t_stat)
        rows.append(
            {
                "feature": getattr(row, "feature"),
                "spearman_rank_ic_t_stat": t_stat,
                "p_value": p_value,
            }
        )
    output = pd.DataFrame(rows)
    output["q_value_bh"] = _benjamini_hochberg_q_values(output["p_value"])
    output["significant_bh"] = output["q_value_bh"] <= alpha
    return output.loc[:, columns]


def _normal_two_sided_p_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return math.erfc(abs(float(value)) / math.sqrt(2.0))


def _benjamini_hochberg_q_values(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().astype(float).sort_values(ascending=False)
    q_values = pd.Series(pd.NA, index=p_values.index, dtype="Float64")
    if valid.empty:
        return q_values
    total = len(valid)
    running_min = 1.0
    for rank_from_end, (index, p_value) in enumerate(valid.items(), start=1):
        rank = total - rank_from_end + 1
        running_min = min(running_min, p_value * total / rank)
        q_values.loc[index] = running_min
    return q_values.astype(float)


def _aggregate_optional_means(group: pd.DataFrame) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for column in group.columns:
        if column.startswith("all_mean_") or column.startswith("top_n_mean_"):
            output[column] = _nullable_float(group[column].mean())
    return output


def _correlation(left: pd.Series, right: pd.Series, *, method: str) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    if method == "spearman":
        left_values = left.rank(method="average")
        right_values = right.rank(method="average")
    else:
        left_values = left.astype(float)
        right_values = right.astype(float)
    left_array, right_array = _aligned_float_arrays(left_values, right_values)
    if len(left_array) < 2:
        return None
    if (
        float(np.min(left_array)) == float(np.max(left_array))
        or float(np.min(right_array)) == float(np.max(right_array))
    ):
        return None
    left_centered = left_array - float(np.mean(left_array))
    right_centered = right_array - float(np.mean(right_array))
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0:
        return None
    return _nullable_float(float(np.dot(left_centered, right_centered) / denominator))


def _aligned_float_arrays(
    left: pd.Series,
    right: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    if left.index.equals(right.index) and not left.hasnans and not right.hasnans:
        return (
            left.to_numpy(dtype=float, copy=False),
            right.to_numpy(dtype=float, copy=False),
        )
    paired = pd.concat([left, right], axis=1).dropna()
    if paired.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    values = paired.to_numpy(dtype=float, copy=False)
    return values[:, 0], values[:, 1]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
