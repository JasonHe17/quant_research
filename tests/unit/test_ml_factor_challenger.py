from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_module():
    path = Path("examples/run_ml_factor_challenger.py")
    spec = importlib.util.spec_from_file_location("run_ml_factor_challenger", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_fold_accepts_explicit_walk_forward_window() -> None:
    module = _load_module()

    fold = module._parse_fold(
        "y2025:train_start=2023-01-01,train_end=2024-12-31,"
        "valid_start=2024-01-01,valid_end=2024-12-31,"
        "test_start=2025-01-01,test_end=2025-12-31"
    )

    assert fold.name == "y2025"
    assert fold.train_start == "2023-01-01"
    assert fold.train_end == "2024-12-31"
    assert fold.valid_start == "2024-01-01"
    assert fold.valid_end == "2024-12-31"
    assert fold.test_start == "2025-01-01"
    assert fold.test_end == "2025-12-31"


def test_prepare_supervised_frame_applies_cross_sectional_rank_and_direction() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T09:35:00+08:00",
                "instrument_id": "a",
                "forward_return_48b": 0.01,
                "alpha_long": 1.0,
                "alpha_invert": 1.0,
            },
            {
                "timestamp": "2024-01-01T09:35:00+08:00",
                "instrument_id": "b",
                "forward_return_48b": 0.02,
                "alpha_long": 2.0,
                "alpha_invert": 2.0,
            },
        ]
    )

    prepared = module._prepare_supervised_frame(
        frame,
        feature_columns=("alpha_long", "alpha_invert"),
        directions={"alpha_long": 1, "alpha_invert": -1},
        label_column="forward_return_48b",
        score_transform="rank",
        drop_missing_label=True,
    )

    output = prepared.frame.sort_values("instrument_id")
    assert output["alpha_long"].tolist() == pytest.approx([0.0, 0.5])
    assert output["alpha_invert"].tolist() == pytest.approx([-0.0, -0.5])


def test_high_correlation_pairs_and_drop_suggestions_prefer_lower_gain() -> None:
    module = _load_module()
    correlation = pd.DataFrame(
        {
            "strong": {"strong": 1.0, "duplicate": 0.95, "weak": 0.1},
            "duplicate": {"strong": 0.95, "duplicate": 1.0, "weak": 0.2},
            "weak": {"strong": 0.1, "duplicate": 0.2, "weak": 1.0},
        }
    )
    pairs = module._high_correlation_pairs(correlation, threshold=0.9)
    importance = pd.DataFrame(
        {
            "feature": ["strong", "duplicate", "weak"],
            "mean_gain_importance": [10.0, 2.0, 0.1],
            "fold_count": [2, 2, 2],
        }
    )

    suggestions = module._drop_suggestions(
        importance,
        pairs,
        low_importance_quantile=0.2,
    )

    by_feature = {row["feature"]: row for row in suggestions}
    assert by_feature["duplicate"]["reason"] == "redundant_lower_gain_pair"
    assert by_feature["duplicate"]["reference_feature"] == "strong"
    assert by_feature["weak"]["reason"] == "low_mean_gain_importance"


def test_score_timestamp_strings_match_existing_score_format() -> None:
    module = _load_module()
    timestamps = pd.Series(pd.to_datetime(["2024-01-02T01:35:00Z"], utc=True))

    output = module._score_timestamp_strings(timestamps)

    assert output.tolist() == ["2024-01-02T09:35:00+08:00"]


def test_apply_sample_weights_emphasizes_cross_sectional_tails() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "forward_return_48b": -0.03},
            {"timestamp": "t0", "instrument_id": "b", "forward_return_48b": 0.00},
            {"timestamp": "t0", "instrument_id": "c", "forward_return_48b": 0.01},
            {"timestamp": "t0", "instrument_id": "d", "forward_return_48b": 0.04},
        ]
    )

    output = module._apply_sample_weights(
        frame,
        label_column="forward_return_48b",
        mode="top_bottom",
        top_quantile=0.25,
        multiplier=3.0,
    ).sort_values("instrument_id")

    assert output["sample_weight"].tolist() == pytest.approx([3.0, 1.0, 1.0, 3.0])


def test_filter_label_derived_candidates_drops_automatic_feature_leakage() -> None:
    module = _load_module()
    candidates = (
        module.CandidateFactor("alpha_safe", 1, 0.1),
        module.CandidateFactor("forward_return_48b_exit_price", 1, 0.2),
        module.CandidateFactor("forward_return_48b_rank", 1, 0.3),
    )

    filtered, excluded = module._filter_label_derived_candidates(
        candidates,
        label_column="forward_return_48b",
        include_features=(),
        allow_label_derived_features=False,
    )

    assert [candidate.feature for candidate in filtered] == ["alpha_safe"]
    assert excluded == ["forward_return_48b_exit_price", "forward_return_48b_rank"]


def test_filter_label_derived_candidates_rejects_explicit_leakage() -> None:
    module = _load_module()
    candidates = (
        module.CandidateFactor("forward_return_48b_exit_price", 1, 0.2),
    )

    with pytest.raises(ValueError, match="explicit include_features"):
        module._filter_label_derived_candidates(
            candidates,
            label_column="forward_return_48b",
            include_features=("forward_return_48b_exit_price",),
            allow_label_derived_features=False,
        )


def test_validate_dataset_columns_reports_missing_partition_columns(tmp_path) -> None:
    module = _load_module()
    good = tmp_path / "dataset_2024_01.parquet"
    bad = tmp_path / "dataset_2024_02.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T09:35:00+08:00",
                "instrument_id": "a",
                "forward_return_48b": 0.01,
                "forward_return_48b_exit_timestamp": "2024-01-03T09:35:00+08:00",
                "alpha": 1.0,
            }
        ]
    ).to_parquet(good, index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2024-02-01T09:35:00+08:00",
                "instrument_id": "a",
                "forward_return_48b": 0.01,
                "forward_return_48b_exit_timestamp": "2024-02-03T09:35:00+08:00",
            }
        ]
    ).to_parquet(bad, index=False)

    with pytest.raises(ValueError, match="missing required model columns"):
        module._validate_dataset_columns(
            [good, bad],
            required_columns=(
                "timestamp",
                "instrument_id",
                "forward_return_48b",
                "forward_return_48b_exit_timestamp",
                "alpha",
            ),
        )


def test_purge_train_uses_label_maturity_not_signal_timestamp() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T09:35:00+08:00",
                "forward_return_48b_exit_timestamp": "2024-01-03T09:35:00+08:00",
                "instrument_id": "a",
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "forward_return_48b_exit_timestamp": "2024-01-04T00:00:00+08:00",
                "instrument_id": "b",
            },
            {
                "timestamp": "2024-01-03T09:35:00+08:00",
                "forward_return_48b_exit_timestamp": "2024-01-06T09:35:00+08:00",
                "instrument_id": "c",
            },
        ]
    )

    purged = module._purge_train(
        frame,
        eval_start=pd.Timestamp("2024-01-05T00:00:00+08:00"),
        embargo="1D",
        label_end_column="forward_return_48b_exit_timestamp",
    )

    assert purged["instrument_id"].tolist() == ["a"]


def test_label_exit_timestamp_column_matches_dataset_conventions() -> None:
    module = _load_module()

    assert module._label_exit_timestamp_column("forward_return") == "exit_timestamp"
    assert (
        module._label_exit_timestamp_column("forward_return_48b")
        == "forward_return_48b_exit_timestamp"
    )


def test_primary_pool_rerank_keeps_only_primary_top_ranked_names(tmp_path) -> None:
    module = _load_module()
    primary_path = tmp_path / "score_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
                "score": 0.1,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "b",
                "score": 0.9,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "c",
                "score": 0.2,
            },
        ]
    ).to_parquet(primary_path, index=False)
    predictions = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-02T01:35:00Z"),
                "instrument_id": "a",
                "forward_return_48b": 0.01,
                "score": 0.8,
            },
            {
                "timestamp": pd.Timestamp("2024-01-02T01:35:00Z"),
                "instrument_id": "b",
                "forward_return_48b": 0.02,
                "score": 0.7,
            },
            {
                "timestamp": pd.Timestamp("2024-01-02T01:35:00Z"),
                "instrument_id": "c",
                "forward_return_48b": -0.01,
                "score": 0.9,
            },
        ]
    )

    output = module._apply_primary_pool_rerank(
        predictions,
        primary_path,
        label_column="forward_return_48b",
        pool_rank=2,
        primary_score_column="score",
    )

    assert output["instrument_id"].tolist() == ["b", "c"]
    assert output["score"].tolist() == pytest.approx([0.7, 0.9])


def test_primary_pool_rerank_can_blend_primary_and_ml_rank(tmp_path) -> None:
    module = _load_module()
    primary_path = tmp_path / "score_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "a",
                "score": 0.9,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "b",
                "score": 0.1,
            },
        ]
    ).to_parquet(primary_path, index=False)
    predictions = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-02T01:35:00Z"),
                "instrument_id": "a",
                "forward_return_48b": 0.01,
                "score": 0.2,
            },
            {
                "timestamp": pd.Timestamp("2024-01-02T01:35:00Z"),
                "instrument_id": "b",
                "forward_return_48b": 0.02,
                "score": 0.8,
            },
        ]
    )

    output = module._apply_primary_pool_rerank(
        predictions,
        primary_path,
        label_column="forward_return_48b",
        pool_rank=2,
        primary_score_column="score",
        primary_blend_weight=0.75,
    ).sort_values("instrument_id")

    assert output["score"].tolist() == pytest.approx([0.875, 0.625])
