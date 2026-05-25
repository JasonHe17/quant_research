from __future__ import annotations

import pandas as pd

from examples.build_joined_alpha_dataset import JoinSource, build_joined_alpha_dataset


def test_build_joined_alpha_dataset_left_joins_source_features(tmp_path) -> None:
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "joined"
    base_dir.mkdir()
    source_dir.mkdir()
    output_dir.mkdir()
    base = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:35", "2024-01-02 09:35"],
            "instrument_id": ["000001", "000002"],
            "base_feature": [1.0, 2.0],
            "forward_return": [0.01, -0.02],
        }
    )
    source = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:35"],
            "instrument_id": ["000001"],
            "extra_feature": [3.0],
        }
    )
    base.to_parquet(base_dir / "dataset_2024_01.parquet", index=False)
    source.to_parquet(source_dir / "dataset_2024_01.parquet", index=False)

    rows = build_joined_alpha_dataset(
        base_dataset_dir=base_dir,
        output_dir=output_dir,
        sources=(JoinSource(source_dir, ("extra_feature",)),),
    )

    joined = pd.read_parquet(output_dir / "dataset_2024_01.parquet")
    assert len(joined) == 2
    assert joined["extra_feature"].notna().sum() == 1
    assert rows[0]["extra_feature_coverage"] == 0.5
    assert (output_dir / "dataset_2024_01.manifest.json").exists()


def test_build_joined_alpha_dataset_rejects_duplicate_source_keys(tmp_path) -> None:
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "joined"
    base_dir.mkdir()
    source_dir.mkdir()
    output_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:35"],
            "instrument_id": ["000001"],
            "base_feature": [1.0],
        }
    ).to_parquet(base_dir / "dataset_2024_01.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:35", "2024-01-02 09:35"],
            "instrument_id": ["000001", "000001"],
            "extra_feature": [3.0, 4.0],
        }
    ).to_parquet(source_dir / "dataset_2024_01.parquet", index=False)

    try:
        build_joined_alpha_dataset(
            base_dataset_dir=base_dir,
            output_dir=output_dir,
            sources=(JoinSource(source_dir, ("extra_feature",)),),
        )
    except ValueError as exc:
        assert "duplicate keys" in str(exc)
    else:
        raise AssertionError("expected duplicate source keys to fail")
