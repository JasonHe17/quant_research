"""Cache manifest models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "parameters": dict(self.parameters),
            "snapshot": self.snapshot,
            "catalog_reference": self.catalog_reference,
            "artifact_path": str(self.artifact_path),
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "schema_fingerprint": self.schema_fingerprint,
        }
