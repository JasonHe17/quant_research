"""Time split utilities for supervised alpha datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class PurgedTimeSplitConfig:
    """Configuration for purged train/valid/test splits."""

    train_end: str
    test_start: str
    valid_start: str | None = None
    valid_end: str | None = None
    test_end: str | None = None
    timestamp_column: str = "timestamp"
    label_end_column: str = "exit_timestamp"
    embargo: str | pd.Timedelta | None = None

    def __post_init__(self) -> None:
        if not self.train_end:
            raise ValueError("train_end is required")
        if not self.test_start:
            raise ValueError("test_start is required")
        if not self.timestamp_column:
            raise ValueError("timestamp_column must be non-empty")
        if not self.label_end_column:
            raise ValueError("label_end_column must be non-empty")


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One explicit walk-forward split window."""

    name: str
    train_start: str | None
    train_end: str
    test_start: str
    test_end: str | None = None
    valid_start: str | None = None
    valid_end: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("window name is required")
        if not self.train_end:
            raise ValueError("train_end is required")
        if not self.test_start:
            raise ValueError("test_start is required")


def purged_time_split(
    frame: pd.DataFrame,
    config: PurgedTimeSplitConfig,
) -> dict[str, pd.DataFrame]:
    """Split by time and purge training labels that overlap evaluation windows."""

    _require_columns(frame, (config.timestamp_column, config.label_end_column))
    timestamp = _to_datetime(frame[config.timestamp_column])
    label_end = _to_datetime(frame[config.label_end_column])
    train_end = _timestamp(config.train_end)
    test_start = _timestamp(config.test_start)
    eval_start = min(
        value
        for value in (
            _timestamp(config.valid_start) if config.valid_start is not None else None,
            test_start,
        )
        if value is not None
    )
    embargo = _embargo_delta(config.embargo)
    purge_cutoff = eval_start - embargo
    train_mask = (timestamp <= train_end) & (label_end < purge_cutoff)
    if config.valid_start is not None:
        valid_mask = timestamp >= _timestamp(config.valid_start)
        if config.valid_end is not None:
            valid_mask = valid_mask & (timestamp <= _timestamp(config.valid_end))
    else:
        valid_mask = pd.Series(False, index=frame.index)
    test_mask = timestamp >= test_start
    if config.test_end is not None:
        test_mask = test_mask & (timestamp <= _timestamp(config.test_end))
    return {
        "train": frame.loc[train_mask].copy().reset_index(drop=True),
        "valid": frame.loc[valid_mask].copy().reset_index(drop=True),
        "test": frame.loc[test_mask].copy().reset_index(drop=True),
    }


def walk_forward_time_splits(
    frame: pd.DataFrame,
    windows: tuple[WalkForwardWindow, ...],
    *,
    timestamp_column: str = "timestamp",
    label_end_column: str = "exit_timestamp",
    embargo: str | pd.Timedelta | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build named purged splits from explicit walk-forward windows."""

    if not windows:
        raise ValueError("at least one walk-forward window is required")
    return {
        window.name: purged_time_split(
            _filter_train_start(
                frame,
                train_start=window.train_start,
                timestamp_column=timestamp_column,
            ),
            PurgedTimeSplitConfig(
                train_end=window.train_end,
                valid_start=window.valid_start,
                valid_end=window.valid_end,
                test_start=window.test_start,
                test_end=window.test_end,
                timestamp_column=timestamp_column,
                label_end_column=label_end_column,
                embargo=embargo,
            ),
        )
        for window in windows
    }


def _filter_train_start(
    frame: pd.DataFrame,
    *,
    train_start: str | None,
    timestamp_column: str,
) -> pd.DataFrame:
    if train_start is None:
        return frame
    _require_columns(frame, (timestamp_column,))
    timestamp = _to_datetime(frame[timestamp_column])
    return frame.loc[timestamp >= _timestamp(train_start)].copy()


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _embargo_delta(value: str | pd.Timedelta | None) -> pd.Timedelta:
    if value is None:
        return pd.Timedelta(0)
    if isinstance(value, pd.Timedelta):
        return value
    return pd.Timedelta(value)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
