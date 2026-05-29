"""Tree model baselines for supervised alpha datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class TreeBaselineConfig:
    """Configuration for a LightGBM regression baseline."""

    label_column: str = "forward_return"
    feature_columns: tuple[str, ...] = ()
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 200
    num_boost_round: int = 200
    early_stopping_rounds: int = 25
    seed: int = 42
    num_threads: int = 4
    sample_weight_column: str | None = None
    extra_params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label_column:
            raise ValueError("label_column must be non-empty")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_leaves <= 1:
            raise ValueError("num_leaves must be greater than 1")
        if self.min_data_in_leaf <= 0:
            raise ValueError("min_data_in_leaf must be positive")
        if self.num_boost_round <= 0:
            raise ValueError("num_boost_round must be positive")
        if self.early_stopping_rounds < 0:
            raise ValueError("early_stopping_rounds must be non-negative")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")


def load_supervised_partitions(paths: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate supervised parquet partitions."""

    if not paths:
        raise ValueError("at least one dataset partition is required")
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    _require_columns(frame, ("timestamp", "instrument_id"))
    return frame.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True)


def infer_feature_columns(
    frame: pd.DataFrame,
    *,
    label_column: str = "forward_return",
    exclude_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Infer numeric feature columns from a supervised alpha dataset."""

    blocked = {
        "timestamp",
        "instrument_id",
        "canonical_code",
        label_column,
        f"{label_column}_rank",
        f"{label_column}_entry_timestamp",
        f"{label_column}_exit_timestamp",
        f"{label_column}_entry_price",
        f"{label_column}_exit_price",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "entry_tradable_bar",
        "entry_limit_up_open",
        "entry_limit_down_open",
        "tradable_bar",
        "buyable_bar",
        "sellable_bar",
        "suspended_bar",
        "limit_up_open",
        "limit_down_open",
        "is_st",
        "previous_close",
        "trade_date",
        "sample_count",
        "rank",
        "diagnostic",
        *exclude_columns,
    }
    features: list[str] = []
    for column in frame.columns:
        if column in blocked:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            features.append(column)
    if not features:
        raise ValueError("no numeric feature columns found")
    return tuple(features)


def time_split(
    frame: pd.DataFrame,
    *,
    train_end: str,
    valid_start: str | None = None,
    valid_end: str | None = None,
    test_start: str,
    test_end: str | None = None,
    label_end_column: str = "exit_timestamp",
    embargo: str | pd.Timedelta | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a supervised dataset by timestamp boundaries.

    When ``label_end_column`` is present, training rows whose labels mature in
    the validation/test window are purged. This keeps the legacy tree baseline
    API aligned with the purged dataset split utilities used by factor research.
    """

    _require_columns(frame, ("timestamp",))
    timestamp = _to_datetime(frame["timestamp"], column="timestamp")
    train_end_at = _timestamp(train_end, field_name="train_end")
    test_start_at = _timestamp(test_start, field_name="test_start")
    eval_start = min(
        value
        for value in (
            _timestamp(valid_start, field_name="valid_start")
            if valid_start is not None
            else None,
            test_start_at,
        )
        if value is not None
    )
    train_mask = timestamp <= train_end_at
    if label_end_column in frame.columns:
        label_end = _to_datetime(frame[label_end_column], column=label_end_column)
        train_mask = train_mask & (label_end < eval_start - _embargo_delta(embargo))
    train = frame.loc[train_mask].copy()
    if valid_start is not None:
        valid_mask = timestamp >= _timestamp(valid_start, field_name="valid_start")
        if valid_end is not None:
            valid_mask = valid_mask & (
                timestamp <= _timestamp(valid_end, field_name="valid_end")
            )
        valid = frame.loc[valid_mask].copy()
    else:
        valid = pd.DataFrame(columns=frame.columns)
    test_mask = timestamp >= test_start_at
    if test_end is not None:
        test_mask = test_mask & (
            timestamp <= _timestamp(test_end, field_name="test_end")
        )
    test = frame.loc[test_mask].copy()
    return {"train": train, "valid": valid, "test": test}


def train_lightgbm_regressor(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: TreeBaselineConfig,
) -> Any:
    """Train a LightGBM regressor with a lazy optional dependency import."""

    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is required for tree baseline training; install it in the "
            "quant environment with `python -m pip install lightgbm`"
        ) from exc
    _require_columns(train, (config.label_column,))
    if train.empty:
        raise ValueError("train split is empty")
    feature_columns = _feature_columns_or_infer(train, config)
    train_set = lgb.Dataset(
        train.loc[:, feature_columns],
        label=train[config.label_column].astype(float),
        weight=_sample_weight_values(train, config),
        feature_name=list(feature_columns),
        free_raw_data=False,
    )
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks: list[object] = []
    if not valid.empty:
        _require_columns(valid, (config.label_column,))
        valid_sets.append(
            lgb.Dataset(
                valid.loc[:, feature_columns],
                label=valid[config.label_column].astype(float),
                weight=_sample_weight_values(valid, config),
                reference=train_set,
                feature_name=list(feature_columns),
                free_raw_data=False,
            )
        )
        valid_names.append("valid")
        if config.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(config.early_stopping_rounds))
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "seed": config.seed,
        "num_threads": config.num_threads,
        "verbosity": -1,
        **config.extra_params,
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def evaluate_cross_sectional_predictions(
    predictions: pd.DataFrame,
    *,
    label_column: str = "forward_return",
    score_column: str = "score",
    top_n: int = 50,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate score quality by timestamp-wise IC and top-N label returns."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    _require_columns(predictions, ("timestamp", "instrument_id", label_column, score_column))
    rows: list[dict[str, object]] = []
    for timestamp, group in predictions.groupby("timestamp", sort=True):
        valid = group.loc[group[[label_column, score_column]].notna().all(axis=1)]
        if valid.empty:
            continue
        top = valid.nlargest(min(top_n, len(valid)), score_column)
        bottom = valid.nsmallest(min(top_n, len(valid)), score_column)
        rows.append(
            {
                "timestamp": timestamp,
                "sample_count": len(valid),
                "pearson_ic": _correlation(
                    valid[score_column],
                    valid[label_column],
                    method="pearson",
                ),
                "spearman_rank_ic": _correlation(
                    valid[score_column],
                    valid[label_column],
                    method="spearman",
                ),
                "all_mean_label": valid[label_column].mean(),
                "top_n_mean_label": top[label_column].mean(),
                "bottom_n_mean_label": bottom[label_column].mean(),
                "top_minus_bottom_label": top[label_column].mean()
                - bottom[label_column].mean(),
            }
        )
    by_timestamp = pd.DataFrame(rows)
    if by_timestamp.empty:
        return {
            "timestamp_count": 0,
            "sample_count": 0,
        }, by_timestamp
    metrics = {
        "timestamp_count": int(len(by_timestamp)),
        "sample_count": int(predictions[[label_column, score_column]].dropna().shape[0]),
        "pearson_ic_mean": _nullable_float(by_timestamp["pearson_ic"].mean()),
        "spearman_rank_ic_mean": _nullable_float(
            by_timestamp["spearman_rank_ic"].mean()
        ),
        "all_mean_label": _nullable_float(by_timestamp["all_mean_label"].mean()),
        "top_n_mean_label": _nullable_float(by_timestamp["top_n_mean_label"].mean()),
        "bottom_n_mean_label": _nullable_float(
            by_timestamp["bottom_n_mean_label"].mean()
        ),
        "top_minus_bottom_label": _nullable_float(
            by_timestamp["top_minus_bottom_label"].mean()
        ),
    }
    return metrics, by_timestamp


def _feature_columns_or_infer(
    frame: pd.DataFrame,
    config: TreeBaselineConfig,
) -> tuple[str, ...]:
    if config.feature_columns:
        _require_columns(frame, config.feature_columns)
        return config.feature_columns
    exclude_columns = (
        (config.sample_weight_column,) if config.sample_weight_column else ()
    )
    return infer_feature_columns(
        frame,
        label_column=config.label_column,
        exclude_columns=exclude_columns,
    )


def _sample_weight_values(
    frame: pd.DataFrame,
    config: TreeBaselineConfig,
) -> pd.Series | None:
    if not config.sample_weight_column:
        return None
    _require_columns(frame, (config.sample_weight_column,))
    weights = pd.to_numeric(frame[config.sample_weight_column], errors="coerce")
    if weights.isna().any():
        raise ValueError("sample weights must be numeric and non-null")
    if (weights < 0).any():
        raise ValueError("sample weights must be non-negative")
    return weights.astype(float)


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _to_datetime(series: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{column} contains values that cannot be parsed as timestamps")
    return parsed


def _timestamp(value: str, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a parseable timestamp") from exc
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _embargo_delta(value: str | pd.Timedelta | None) -> pd.Timedelta:
    if value is None:
        return pd.Timedelta(0)
    if isinstance(value, pd.Timedelta):
        return value
    return pd.Timedelta(value)


def _correlation(left: pd.Series, right: pd.Series, *, method: str) -> float:
    if method == "spearman":
        return float(left.rank(method="average").corr(right.rank(method="average")))
    return float(left.corr(right))


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
