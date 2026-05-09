"""Local research cache policy models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import pandas as pd

from quant_research.data.manifests import CacheManifest, CacheManifestStore


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Configuration for rebuildable hot-data cache behavior."""

    root: Path
    enabled: bool = True
    artifact_format: str = "pickle"

    @classmethod
    def disabled(cls) -> "CachePolicy":
        return cls(root=Path(".cache/quant_research"), enabled=False)

    def snapshot_root(self, snapshot: str) -> Path:
        return self.root / "snapshots" / snapshot

    def manifest_root(self, snapshot: str) -> Path:
        return self.snapshot_root(snapshot) / "manifests"

    def artifact_root(self, snapshot: str, dataset: str) -> Path:
        return self.snapshot_root(snapshot) / _safe_path_component(dataset)

    def artifact_path(self, *, snapshot: str, dataset: str, manifest_id: str) -> Path:
        suffix = _artifact_suffix(self.artifact_format)
        return self.artifact_root(snapshot, dataset) / f"{manifest_id}.{suffix}"


class DataFrameCache:
    """Rebuildable local cache for DataPortal DataFrame results."""

    def __init__(self, *, policy: CachePolicy) -> None:
        self.policy = policy
        self.manifests = CacheManifestStore(root=policy.root)

    def get_or_compute(
        self,
        *,
        dataset: str,
        parameters: dict[str, object],
        snapshot: str,
        catalog_reference: str,
        compute: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        if not self.policy.enabled:
            return compute()
        found = self.manifests.find(
            dataset=dataset,
            parameters=parameters,
            snapshot=snapshot,
            catalog_reference=catalog_reference,
        )
        if found is not None and found.artifact_path.exists():
            return self.read(found)

        frame = compute()
        probe = CacheManifest.create(
            dataset=dataset,
            parameters=parameters,
            snapshot=snapshot,
            catalog_reference=catalog_reference,
            artifact_path=".",
            row_count=len(frame),
        )
        artifact_path = self.policy.artifact_path(
            snapshot=snapshot,
            dataset=dataset,
            manifest_id=probe.manifest_id,
        )
        manifest = CacheManifest.create(
            dataset=dataset,
            parameters=parameters,
            snapshot=snapshot,
            catalog_reference=catalog_reference,
            artifact_path=artifact_path,
            row_count=len(frame),
        )
        self.write(frame, manifest)
        return frame

    def read(self, manifest: CacheManifest) -> pd.DataFrame:
        if self.policy.artifact_format != "pickle":
            raise ValueError(f"Unsupported cache format: {self.policy.artifact_format}")
        return pd.read_pickle(manifest.artifact_path)

    def write(self, frame: pd.DataFrame, manifest: CacheManifest) -> Path:
        if self.policy.artifact_format != "pickle":
            raise ValueError(f"Unsupported cache format: {self.policy.artifact_format}")
        manifest.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(manifest.artifact_path)
        return self.manifests.write(manifest)


def catalog_reference_for_path(path: str | Path) -> str:
    """Return a lightweight local catalog identity for cache invalidation."""

    catalog = Path(path)
    if not catalog.exists():
        return f"catalog-missing:{catalog}"
    stat = catalog.stat()
    payload = f"{catalog.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"catalog-stat-sha256:{digest}"


def _artifact_suffix(format_name: str) -> str:
    if format_name == "pickle":
        return "pkl"
    raise ValueError(f"Unsupported cache format: {format_name}")


def _safe_path_component(value: str) -> str:
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(allowed).strip("_") or "dataset"
