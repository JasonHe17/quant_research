from __future__ import annotations

from pathlib import Path

from quant_research.data import CacheManifest, CacheManifestStore, CachePolicy


def test_cache_manifest_id_is_stable_for_rebuild_inputs() -> None:
    first = CacheManifest.create(
        dataset="minute_bars",
        parameters={"symbols": ["600000.SH"], "frequency": "1m"},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
        artifact_path="/tmp/a.parquet",
        row_count=10,
    )
    second = CacheManifest.create(
        dataset="minute_bars",
        parameters={"frequency": "1m", "symbols": ["600000.SH"]},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
        artifact_path="/tmp/b.parquet",
        row_count=99,
    )

    assert first.manifest_id == second.manifest_id


def test_cache_manifest_store_round_trips_json(tmp_path: Path) -> None:
    store = CacheManifestStore(root=tmp_path)
    manifest = CacheManifest.create(
        dataset="fundamentals",
        parameters={"symbols": ["600000.SH"], "dataset": "profit"},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:def",
        artifact_path=tmp_path / "snapshots/2026-05-09/fundamentals/profit.parquet",
        row_count=3,
        schema_fingerprint="schema-sha256:123",
    )

    path = store.write(manifest)
    loaded = store.read(snapshot=manifest.snapshot, manifest_id=manifest.manifest_id)

    assert path == store.path_for(manifest)
    assert loaded == manifest
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_cache_manifest_store_finds_by_rebuild_inputs(tmp_path: Path) -> None:
    store = CacheManifestStore(root=tmp_path)
    manifest = CacheManifest.create(
        dataset="minute_bars",
        parameters={"symbols": ["600000.SH"]},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
        artifact_path=tmp_path / "sample.parquet",
        row_count=10,
    )
    _ = store.write(manifest)

    found = store.find(
        dataset="minute_bars",
        parameters={"symbols": ["600000.SH"]},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
    )
    missing = store.find(
        dataset="minute_bars",
        parameters={"symbols": ["000001.SZ"]},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
    )

    assert found == manifest
    assert missing is None


def test_cache_manifest_store_lists_by_snapshot_and_dataset(tmp_path: Path) -> None:
    store = CacheManifestStore(root=tmp_path)
    first = CacheManifest.create(
        dataset="minute_bars",
        parameters={"symbols": ["600000.SH"]},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
        artifact_path=tmp_path / "bars.parquet",
        row_count=10,
    )
    second = CacheManifest.create(
        dataset="profit",
        parameters={"symbols": ["600000.SH"]},
        snapshot="2026-05-09",
        catalog_reference="catalog-sha256:abc",
        artifact_path=tmp_path / "profit.parquet",
        row_count=1,
    )
    _ = store.write(first)
    _ = store.write(second)

    assert store.list(snapshot="2026-05-09", dataset="minute_bars") == (first,)
    assert store.list(snapshot="2026-05-09") == (first, second)


def test_cache_policy_declares_manifest_root() -> None:
    policy = CachePolicy(root=Path("/ssd/quant_cache"))
    assert policy.manifest_root("2026-05-09") == Path(
        "/ssd/quant_cache/snapshots/2026-05-09/manifests"
    )
