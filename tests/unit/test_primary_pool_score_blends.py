from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_module():
    path = Path("examples/build_primary_pool_score_blends.py")
    spec = importlib.util.spec_from_file_location("build_primary_pool_score_blends", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_blend_partition_rank_blends_inside_existing_ml_pool(tmp_path) -> None:
    module = _load_module()
    primary_path = tmp_path / "primary.parquet"
    ml_path = tmp_path / "ml.parquet"
    pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "score": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "score": 3.0},
            {"timestamp": "t0", "instrument_id": "c", "score": 2.0},
        ]
    ).to_parquet(primary_path, index=False)
    pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "score": 0.9},
            {"timestamp": "t0", "instrument_id": "c", "score": 0.1},
        ]
    ).to_parquet(ml_path, index=False)

    outputs = module._blend_partition(
        primary_path,
        ml_path,
        primary_blend_weights=(0.75,),
    )

    output = outputs[0.75].sort_values("instrument_id")
    assert output["instrument_id"].tolist() == ["a", "c"]
    assert output["score"].tolist() == pytest.approx([0.625, 0.875])


def test_blend_partition_can_apply_stricter_primary_pool_rank(tmp_path) -> None:
    module = _load_module()
    primary_path = tmp_path / "primary.parquet"
    ml_path = tmp_path / "ml.parquet"
    pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "score": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "score": 3.0},
            {"timestamp": "t0", "instrument_id": "c", "score": 2.0},
        ]
    ).to_parquet(primary_path, index=False)
    pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "score": 0.9},
            {"timestamp": "t0", "instrument_id": "b", "score": 0.1},
            {"timestamp": "t0", "instrument_id": "c", "score": 0.8},
        ]
    ).to_parquet(ml_path, index=False)

    outputs = module._blend_partition(
        primary_path,
        ml_path,
        primary_blend_weights=(0.5,),
        primary_pool_rank=2,
    )

    assert sorted(outputs[0.5]["instrument_id"].tolist()) == ["b", "c"]
