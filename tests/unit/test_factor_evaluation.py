from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from examples.evaluate_alpha_dataset import (
    _evaluate_dataset_path,
    SingleFactorEvaluationConfig,
)
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
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha": 0.1,
                "forward_return": -0.01,
            },
            {
                "timestamp": "t1",
                "instrument_id": "a",
                "alpha": 0.8,
                "forward_return": 0.02,
            },
            {
                "timestamp": "t1",
                "instrument_id": "b",
                "alpha": 0.2,
                "forward_return": 0.00,
            },
        ]
    )

    result = evaluate_single_factors(
        frame,
        SingleFactorEvaluationConfig(
            feature_columns=("alpha",),
            top_n=1,
            quantiles=2,
        ),
    )

    summary = result.summary.iloc[0]
    assert summary["feature"] == "alpha"
    assert summary["coverage"] == pytest.approx(1.0)
    assert summary["spearman_rank_ic_mean"] == pytest.approx(1.0)
    assert summary["top_minus_bottom_label"] == pytest.approx(0.03)
    assert result.quantile_returns["quantile"].tolist() == [1, 2]
    assert result.quantile_returns.loc[1, "mean_label"] == pytest.approx(0.025)


def test_evaluate_single_factors_infers_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["t0", "t0"],
            "instrument_id": ["a", "b"],
            "alpha_a": [0.1, 0.2],
            "alpha_b": [1.0, 0.0],
            "forward_return": [0.0, 0.1],
            "entry_price": [10.0, 10.0],
            "exit_price": [11.0, 12.0],
        }
    )

    result = evaluate_single_factors(frame)

    assert result.summary["feature"].tolist() == ["alpha_a", "alpha_b"]
    assert list(result.feature_correlation.columns) == ["alpha_a", "alpha_b"]


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
