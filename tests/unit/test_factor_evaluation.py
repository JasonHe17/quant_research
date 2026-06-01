from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from examples.evaluate_alpha_dataset import (
    _effective_workers,
    _evaluate_dataset_path,
    _infer_feature_columns_from_path,
    SingleFactorEvaluationConfig,
)
from quant_research.artifacts import ArtifactStore
from quant_research.factors import (
    evaluate_single_factors,
)


def test_evaluate_single_factors_reports_summary_and_quantiles() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "alpha": 0.9,
                "forward_return": 0.03,
                "forward_return_2": 0.05,
                "sector": "bank",
                "turnover": 1000.0,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha": 0.1,
                "forward_return": -0.01,
                "forward_return_2": -0.02,
                "sector": "bank",
                "turnover": 100.0,
            },
            {
                "timestamp": "t1",
                "instrument_id": "a",
                "alpha": 0.8,
                "forward_return": 0.02,
                "forward_return_2": 0.04,
                "sector": "bank",
                "turnover": 900.0,
            },
            {
                "timestamp": "t1",
                "instrument_id": "b",
                "alpha": 0.2,
                "forward_return": 0.00,
                "forward_return_2": -0.01,
                "sector": "bank",
                "turnover": 200.0,
            },
        ]
    )

    result = evaluate_single_factors(
        frame,
        SingleFactorEvaluationConfig(
            feature_columns=("alpha",),
            horizon_label_columns=("forward_return_2",),
            group_columns=("sector",),
            top_n=1,
            quantiles=2,
            cost_bps=10.0,
        ),
    )

    summary = result.summary.iloc[0]
    assert summary["feature"] == "alpha"
    assert summary["coverage"] == pytest.approx(1.0)
    assert summary["spearman_rank_ic_mean"] == pytest.approx(1.0)
    assert summary["spearman_rank_ic_positive_rate"] == pytest.approx(1.0)
    assert summary["rank_autocorrelation"] == pytest.approx(1.0)
    assert summary["top_minus_bottom_label"] == pytest.approx(0.03)
    assert pd.notna(summary["cost_adjusted_top_minus_bottom_label"])
    assert result.quantile_returns["quantile"].tolist() == ["1", "2", "long_short"]
    assert result.quantile_returns.loc[1, "mean_label"] == pytest.approx(0.025)
    assert result.decay_by_label["label_column"].tolist() == [
        "forward_return",
        "forward_return_2",
    ]
    assert not result.group_summary.empty
    assert result.multiple_testing.loc[0, "feature"] == "alpha"


def test_evaluate_single_factors_infers_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["t0", "t0"],
            "instrument_id": ["a", "b"],
            "alpha_a": [0.1, 0.2],
            "alpha_b": [1.0, 0.0],
            "forward_return": [0.0, 0.1],
            "forward_return_5d": [0.02, 0.03],
            "forward_return_5d_rank": [0.5, 1.0],
            "forward_return_5d_exit_price": [12.0, 13.0],
            "entry_price": [10.0, 10.0],
            "exit_price": [11.0, 12.0],
        }
    )

    result = evaluate_single_factors(
        frame,
        SingleFactorEvaluationConfig(horizon_label_columns=("forward_return_5d",)),
    )

    assert result.summary["feature"].tolist() == ["alpha_a", "alpha_b"]
    assert list(result.feature_correlation.columns) == ["alpha_a", "alpha_b"]


def test_evaluate_single_factors_returns_null_ic_for_constant_cross_section() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["t0", "t0", "t0"],
            "instrument_id": ["a", "b", "c"],
            "alpha": [1.0, 1.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )

    result = evaluate_single_factors(
        frame,
        SingleFactorEvaluationConfig(feature_columns=("alpha",), top_n=1),
    )

    row = result.by_timestamp.iloc[0]
    assert pd.isna(row["pearson_ic"])
    assert pd.isna(row["spearman_rank_ic"])


def test_single_factor_evaluation_rejects_missing_label() -> None:
    frame = pd.DataFrame({"timestamp": ["t0"], "instrument_id": ["a"], "alpha": [1.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_single_factors(frame)


def test_partition_evaluation_writes_artifacts_without_returning_frames(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    partition_dir = tmp_path / "partitions"
    partition_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "alpha": 0.9,
                "forward_return": 0.03,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha": 0.1,
                "forward_return": -0.01,
            },
        ]
    ).to_parquet(dataset_path, index=False)

    partition = _evaluate_dataset_path(
        dataset_path,
        SingleFactorEvaluationConfig(feature_columns=("alpha",), top_n=1),
        partition_dir=partition_dir,
        skip_feature_correlation=True,
    )

    assert partition.row_count == 2
    assert partition.by_timestamp_path.exists()
    assert partition.quantile_by_timestamp_path.exists()
    assert not hasattr(partition, "result")


def test_partition_evaluation_reuses_matching_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    partition_dir = tmp_path / "partitions"
    partition_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "alpha": 0.9,
                "forward_return": 0.03,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha": 0.1,
                "forward_return": -0.01,
            },
        ]
    ).to_parquet(dataset_path, index=False)
    config = SingleFactorEvaluationConfig(feature_columns=("alpha",), top_n=1)

    first = _evaluate_dataset_path(
        dataset_path,
        config,
        partition_dir=partition_dir,
        skip_feature_correlation=False,
        resume_existing=True,
    )
    second = _evaluate_dataset_path(
        dataset_path,
        config,
        partition_dir=partition_dir,
        skip_feature_correlation=False,
        resume_existing=True,
    )

    assert first.row_count == second.row_count == 2
    assert second.correlation is not None
    pd.testing.assert_frame_equal(
        first.correlation.to_frame(),
        second.correlation.to_frame(),
    )


def test_factor_evaluation_infers_features_from_parquet_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        {
            "timestamp": ["t0"],
            "instrument_id": ["a"],
            "alpha": [1.0],
            "forward_return": [0.01],
            "forward_return_rank": [1.0],
            "entry_price": [10.0],
        }
    ).to_parquet(dataset_path, index=False)

    assert _infer_feature_columns_from_path(
        dataset_path,
        label_column="forward_return",
    ) == ("alpha",)


def test_factor_evaluation_memory_budget_reduces_workers() -> None:
    assert _effective_workers(
        requested_workers=6,
        worker_memory_estimate_gb=5.0,
        memory_budget_gb=12.0,
    ) == 2
    assert _effective_workers(
        requested_workers=6,
        worker_memory_estimate_gb=5.0,
        memory_budget_gb=3.0,
    ) == 1


def test_factor_evaluation_persists_artifacts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha": 0.9, "forward_return": 0.03},
            {"timestamp": "t0", "instrument_id": "b", "alpha": 0.1, "forward_return": -0.01},
        ]
    )
    result = evaluate_single_factors(
        frame,
        SingleFactorEvaluationConfig(feature_columns=("alpha",), top_n=1),
    )
    store = ArtifactStore.from_path(tmp_path)

    artifacts = store.write_factor_evaluation("alpha_eval", result)
    manifest = store.read_artifact_manifest(artifacts["summary"])

    assert "summary" in artifacts
    assert artifacts["summary"].endswith(".parquet")
    assert manifest["artifact_type"] == "factor_evaluation"
    assert store.read_factor_evaluation_artifact("alpha_eval", "summary").equals(
        result.summary
    )
