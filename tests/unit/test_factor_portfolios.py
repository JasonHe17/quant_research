from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from examples.run_candidate_factor_portfolios import _dataset_paths
from quant_research.portfolio import (
    CandidateFactor,
    build_composite_scores,
    factor_combination_weights,
    load_candidate_factors,
    write_score_partitions,
)


def test_load_candidate_factors_uses_admission_direction(tmp_path: Path) -> None:
    path = tmp_path / "admission.json"
    path.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "feature": "alpha_a",
                        "admission_status": "candidate",
                        "direction": "invert",
                        "spearman_rank_ic_mean": -0.02,
                    },
                    {
                        "feature": "alpha_b",
                        "admission_status": "watchlist",
                        "direction": "long",
                        "spearman_rank_ic_mean": 0.01,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    factors = load_candidate_factors(path)

    assert factors == (CandidateFactor("alpha_a", -1, -0.02),)


def test_factor_combination_weights_support_methods() -> None:
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", -1, -0.01),
    )
    correlation = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]],
        index=["alpha_a", "alpha_b"],
        columns=["alpha_a", "alpha_b"],
    )

    equal = factor_combination_weights(factors, method="equal")
    ic_weighted = factor_combination_weights(factors, method="ic_weighted")
    decorrelated = factor_combination_weights(
        factors,
        method="decorrelated",
        correlation=correlation,
    )

    assert equal == {"alpha_a": 0.5, "alpha_b": 0.5}
    assert ic_weighted["alpha_a"] == pytest.approx(2 / 3)
    assert sum(decorrelated.values()) == pytest.approx(1.0)


def test_build_composite_scores_ranks_and_orients_cross_sectionally() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0, "alpha_b": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0, "alpha_b": 0.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0, "alpha_b": -1.0},
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", -1, -0.01),
    )

    scores = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.5, "alpha_b": 0.5},
    )

    assert scores.iloc[0]["instrument_id"] == "c"
    assert scores.iloc[0]["score"] > scores.iloc[-1]["score"]


def test_write_score_partitions_writes_one_partition_per_method(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0},
        ]
    ).to_parquet(dataset_path, index=False)

    summary = write_score_partitions(
        [dataset_path],
        output_dir=tmp_path / "scores",
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        weights_by_method={"equal": {"alpha_a": 1.0}},
    )

    assert summary["methods"]["equal"]["row_count"] == 2
    assert Path(tmp_path / "scores" / "equal" / "score_2024_01.parquet").exists()


def test_candidate_factor_script_filters_dataset_partitions(tmp_path: Path) -> None:
    for partition in ("2023_01", "2023_02", "2023_03", "2023_04"):
        (tmp_path / f"dataset_{partition}.parquet").touch()

    args = type(
        "Args",
        (),
        {
            "dataset_dir": str(tmp_path),
            "partition_start": "2023_02",
            "partition_end": "2023_03",
            "max_partitions": None,
        },
    )()

    assert [path.name for path in _dataset_paths(args)] == [
        "dataset_2023_02.parquet",
        "dataset_2023_03.parquet",
    ]
