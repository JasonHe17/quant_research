"""Dataset artifact manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetPartitionManifest:
    """Manifest for one materialized model-research dataset partition."""

    name: str
    partition: str
    dataset_path: str
    row_count: int
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    data_snapshot: str | None = None
    catalog_reference: str | None = None
    source_artifacts: dict[str, str] = field(default_factory=dict)
    source_artifact_sha256: dict[str, str] = field(default_factory=dict)
    dataset_sha256: str | None = None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        partition: str,
        dataset_path: str | Path,
        row_count: int,
        feature_columns: tuple[str, ...],
        label_columns: tuple[str, ...],
        parameters: dict[str, Any] | None = None,
        data_snapshot: str | None = None,
        catalog_reference: str | None = None,
        source_artifacts: dict[str, str] | None = None,
        source_artifact_sha256: dict[str, str] | None = None,
        hash_dataset: bool = True,
    ) -> "DatasetPartitionManifest":
        path = Path(dataset_path)
        artifacts = dict(source_artifacts or {})
        return cls(
            name=name,
            partition=partition,
            dataset_path=str(path),
            row_count=row_count,
            feature_columns=tuple(feature_columns),
            label_columns=tuple(label_columns),
            parameters=dict(parameters or {}),
            data_snapshot=data_snapshot,
            catalog_reference=catalog_reference,
            source_artifacts=artifacts,
            source_artifact_sha256=dict(source_artifact_sha256 or _hash_artifacts(artifacts)),
            dataset_sha256=file_sha256(path) if hash_dataset and path.exists() else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_columns"] = list(self.feature_columns)
        payload["label_columns"] = list(self.label_columns)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetPartitionManifest":
        return cls(
            name=str(payload["name"]),
            partition=str(payload["partition"]),
            dataset_path=str(payload["dataset_path"]),
            row_count=int(payload["row_count"]),
            feature_columns=tuple(str(item) for item in payload["feature_columns"]),
            label_columns=tuple(str(item) for item in payload["label_columns"]),
            parameters=dict(payload.get("parameters", {})),
            data_snapshot=str(payload["data_snapshot"])
            if payload.get("data_snapshot") is not None
            else None,
            catalog_reference=str(payload["catalog_reference"])
            if payload.get("catalog_reference") is not None
            else None,
            source_artifacts={
                str(key): str(value)
                for key, value in payload.get("source_artifacts", {}).items()
            },
            source_artifact_sha256={
                str(key): str(value)
                for key, value in payload.get("source_artifact_sha256", {}).items()
            },
            dataset_sha256=str(payload["dataset_sha256"])
            if payload.get("dataset_sha256") is not None
            else None,
        )


def write_dataset_manifest(
    manifest: DatasetPartitionManifest,
    path: str | Path,
) -> Path:
    """Write a dataset partition manifest as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output_path


def read_dataset_manifest(path: str | Path) -> DatasetPartitionManifest:
    """Read a dataset partition manifest from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetPartitionManifest.from_dict(payload)


def file_sha256(path: str | Path) -> str:
    """Return a SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_artifacts(artifacts: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, artifact_path in artifacts.items():
        path = Path(artifact_path)
        if path.exists():
            hashes[name] = file_sha256(path)
    return hashes
