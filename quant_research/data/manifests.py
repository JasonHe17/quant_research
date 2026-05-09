"""Cache manifest models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheManifest:
    """Metadata for one rebuildable cache artifact."""

    dataset: str
    parameters: dict[str, Any]
    snapshot: str
    catalog_reference: str
    artifact_path: Path
    created_at: datetime
    row_count: int
    schema_fingerprint: str | None = None

    @property
    def manifest_id(self) -> str:
        """Stable id derived from rebuild inputs, not from creation time."""

        payload = {
            "dataset": self.dataset,
            "parameters": _jsonable(self.parameters),
            "snapshot": self.snapshot,
            "catalog_reference": self.catalog_reference,
            "schema_fingerprint": self.schema_fingerprint,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        dataset: str,
        parameters: dict[str, Any],
        snapshot: str,
        catalog_reference: str,
        artifact_path: str | Path,
        row_count: int,
        schema_fingerprint: str | None = None,
    ) -> "CacheManifest":
        return cls(
            dataset=dataset,
            parameters=dict(parameters),
            snapshot=snapshot,
            catalog_reference=catalog_reference,
            artifact_path=Path(artifact_path),
            created_at=datetime.now(timezone.utc),
            row_count=row_count,
            schema_fingerprint=schema_fingerprint,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CacheManifest":
        return cls(
            dataset=str(payload["dataset"]),
            parameters=dict(payload["parameters"]),
            snapshot=str(payload["snapshot"]),
            catalog_reference=str(payload["catalog_reference"]),
            artifact_path=Path(str(payload["artifact_path"])),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            row_count=int(payload["row_count"]),
            schema_fingerprint=str(payload["schema_fingerprint"])
            if payload.get("schema_fingerprint") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "dataset": self.dataset,
            "parameters": _jsonable(self.parameters),
            "snapshot": self.snapshot,
            "catalog_reference": self.catalog_reference,
            "artifact_path": str(self.artifact_path),
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "schema_fingerprint": self.schema_fingerprint,
        }


class CacheManifestStore:
    """JSON manifest store for rebuildable local cache artifacts."""

    def __init__(self, *, root: str | Path) -> None:
        self.root = Path(root)

    def manifest_root(self, snapshot: str) -> Path:
        return self.root / "snapshots" / snapshot / "manifests"

    def path_for(self, manifest: CacheManifest) -> Path:
        return self.manifest_root(manifest.snapshot) / f"{manifest.manifest_id}.json"

    def write(self, manifest: CacheManifest) -> Path:
        path = self.path_for(manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return path

    def read_path(self, path: str | Path) -> CacheManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return CacheManifest.from_dict(payload)

    def read(self, *, snapshot: str, manifest_id: str) -> CacheManifest:
        return self.read_path(self.manifest_root(snapshot) / f"{manifest_id}.json")

    def find(
        self,
        *,
        dataset: str,
        parameters: dict[str, Any],
        snapshot: str,
        catalog_reference: str,
        schema_fingerprint: str | None = None,
    ) -> CacheManifest | None:
        probe = CacheManifest.create(
            dataset=dataset,
            parameters=parameters,
            snapshot=snapshot,
            catalog_reference=catalog_reference,
            artifact_path=".",
            row_count=0,
            schema_fingerprint=schema_fingerprint,
        )
        path = self.path_for(probe)
        if not path.exists():
            return None
        return self.read_path(path)

    def list(
        self, *, snapshot: str | None = None, dataset: str | None = None
    ) -> tuple[CacheManifest, ...]:
        roots = (
            [self.manifest_root(snapshot)]
            if snapshot is not None
            else sorted((self.root / "snapshots").glob("*/manifests"))
        )
        manifests: list[CacheManifest] = []
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.json")):
                manifest = self.read_path(path)
                if dataset is None or manifest.dataset == dataset:
                    manifests.append(manifest)
        return tuple(
            sorted(
                manifests,
                key=lambda item: (item.snapshot, item.dataset, item.manifest_id),
            )
        )


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
