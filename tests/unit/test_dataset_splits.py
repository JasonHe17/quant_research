from __future__ import annotations

import pandas as pd
import pytest

from quant_research.datasets import (
    PurgedTimeSplitConfig,
    WalkForwardWindow,
    purged_time_split,
    walk_forward_time_splits,
)


def test_purged_time_split_removes_overlapping_training_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-01",
                "exit_timestamp": "2025-01-04",
                "value": "safe-train",
            },
            {
                "timestamp": "2025-01-02",
                "exit_timestamp": "2025-01-06",
                "value": "overlap-train",
            },
            {
                "timestamp": "2025-01-05",
                "exit_timestamp": "2025-01-07",
                "value": "test",
            },
        ]
    )

    splits = purged_time_split(
        frame,
        PurgedTimeSplitConfig(
            train_end="2025-01-04",
            test_start="2025-01-05",
        ),
    )

    assert splits["train"]["value"].tolist() == ["safe-train"]
    assert splits["test"]["value"].tolist() == ["test"]


def test_purged_time_split_applies_embargo() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-01",
                "exit_timestamp": "2025-01-04",
                "value": "purged-by-embargo",
            },
            {
                "timestamp": "2025-01-05",
                "exit_timestamp": "2025-01-06",
                "value": "test",
            },
        ]
    )

    splits = purged_time_split(
        frame,
        PurgedTimeSplitConfig(
            train_end="2025-01-04",
            test_start="2025-01-05",
            embargo="2D",
        ),
    )

    assert splits["train"].empty


def test_walk_forward_time_splits_build_named_windows() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "2025-01-01", "exit_timestamp": "2025-01-02", "x": 1},
            {"timestamp": "2025-01-03", "exit_timestamp": "2025-01-04", "x": 2},
            {"timestamp": "2025-01-05", "exit_timestamp": "2025-01-06", "x": 3},
        ]
    )

    splits = walk_forward_time_splits(
        frame,
        (
            WalkForwardWindow(
                name="wf1",
                train_start="2025-01-01",
                train_end="2025-01-03",
                test_start="2025-01-05",
                test_end="2025-01-05",
            ),
        ),
    )

    assert set(splits) == {"wf1"}
    assert splits["wf1"]["test"]["x"].tolist() == [3]


def test_purged_time_split_requires_label_end_column() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        purged_time_split(
            pd.DataFrame({"timestamp": ["2025-01-01"]}),
            PurgedTimeSplitConfig(train_end="2025-01-01", test_start="2025-01-02"),
        )
