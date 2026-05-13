from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research.datasets import (
    DatasetPartitionManifest,
    read_dataset_manifest,
    write_dataset_manifest,
)


def test_dataset_partition_manifest_round_trips(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2025_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "inst-1",
                "alpha": 1.0,
                "forward_return": 0.01,
            }
        ]
    ).to_parquet(dataset_path, index=False)
    manifest = DatasetPartitionManifest.create(
        name="baseline-a",
        partition="2025_01",
        dataset_path=dataset_path,
        row_count=1,
        feature_columns=("alpha",),
        label_columns=("forward_return",),
        parameters={"horizon_bars": 48},
        data_snapshot="2026-05-13",
        catalog_reference="catalog-sha256:test",
    )

    manifest_path = write_dataset_manifest(manifest, tmp_path / "manifest.json")
    loaded = read_dataset_manifest(manifest_path)

    assert loaded == manifest
    assert loaded.dataset_sha256 is not None
    assert loaded.source_artifact_sha256 == {}
