"""Experiment run metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from quant_research.experiments.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Persistent metadata for one reproducible experiment run."""

    run_id: str
    config: ExperimentConfig
    started_at: datetime
    status: str = "created"
    finished_at: datetime | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    cache_manifest_ids: tuple[str, ...] = ()
    notes: str | None = None

    @classmethod
    def create(
        cls,
        *,
        config: ExperimentConfig,
        run_id: str | None = None,
        started_at: datetime | None = None,
        status: str = "created",
        artifacts: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        cache_manifest_ids: tuple[str, ...] = (),
        notes: str | None = None,
    ) -> "ExperimentRun":
        started = started_at or datetime.now(timezone.utc)
        return cls(
            run_id=run_id or _make_run_id(config=config, started_at=started),
            config=config,
            started_at=started,
            status=status,
            artifacts=dict(artifacts or {}),
            metrics=dict(metrics or {}),
            cache_manifest_ids=tuple(cache_manifest_ids),
            notes=notes,
        )

    def complete(
        self,
        *,
        artifacts: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        cache_manifest_ids: tuple[str, ...] | None = None,
        finished_at: datetime | None = None,
    ) -> "ExperimentRun":
        return ExperimentRun(
            run_id=self.run_id,
            config=self.config,
            started_at=self.started_at,
            status="completed",
            finished_at=finished_at or datetime.now(timezone.utc),
            artifacts={**self.artifacts, **dict(artifacts or {})},
            metrics={**self.metrics, **dict(metrics or {})},
            cache_manifest_ids=cache_manifest_ids
            if cache_manifest_ids is not None
            else self.cache_manifest_ids,
            notes=self.notes,
        )

    def fail(
        self, *, notes: str, finished_at: datetime | None = None
    ) -> "ExperimentRun":
        return ExperimentRun(
            run_id=self.run_id,
            config=self.config,
            started_at=self.started_at,
            status="failed",
            finished_at=finished_at or datetime.now(timezone.utc),
            artifacts=dict(self.artifacts),
            metrics=dict(self.metrics),
            cache_manifest_ids=tuple(self.cache_manifest_ids),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "started_at": self.started_at.isoformat(),
            "status": self.status,
            "finished_at": self.finished_at.isoformat()
            if self.finished_at is not None
            else None,
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
            "cache_manifest_ids": list(self.cache_manifest_ids),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentRun":
        return cls(
            run_id=str(payload["run_id"]),
            config=ExperimentConfig.from_dict(dict(payload["config"])),
            started_at=datetime.fromisoformat(str(payload["started_at"])),
            status=str(payload["status"]),
            finished_at=datetime.fromisoformat(str(payload["finished_at"]))
            if payload.get("finished_at") is not None
            else None,
            artifacts={
                str(key): str(value)
                for key, value in payload.get("artifacts", {}).items()
            },
            metrics={
                str(key): float(value)
                for key, value in payload.get("metrics", {}).items()
            },
            cache_manifest_ids=tuple(
                str(item) for item in payload.get("cache_manifest_ids", ())
            ),
            notes=str(payload["notes"]) if payload.get("notes") is not None else None,
        )


class ExperimentRunStore:
    """JSON store for experiment run metadata."""

    def __init__(self, *, root: str | Path) -> None:
        self.root = Path(root)

    def run_path(self, run_id: str) -> Path:
        return self.root / "experiments" / f"{_safe_path_component(run_id)}.json"

    def write(self, run: ExperimentRun) -> Path:
        path = self.run_path(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(run.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return path

    def read(self, run_id: str) -> ExperimentRun:
        payload = json.loads(self.run_path(run_id).read_text(encoding="utf-8"))
        return ExperimentRun.from_dict(payload)

    def list(self, *, status: str | None = None) -> tuple[ExperimentRun, ...]:
        root = self.root / "experiments"
        if not root.exists():
            return ()
        runs = [
            ExperimentRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(root.glob("*.json"))
        ]
        if status is not None:
            runs = [run for run in runs if run.status == status]
        return tuple(sorted(runs, key=lambda run: (run.started_at, run.run_id)))


def _make_run_id(*, config: ExperimentConfig, started_at: datetime) -> str:
    payload = {
        "name": config.name,
        "data_snapshot": config.data_snapshot,
        "parameters": config.parameters,
        "started_at": started_at.isoformat(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{_safe_path_component(config.name)}-{digest}"


def _safe_path_component(value: str) -> str:
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(allowed).strip("_") or "run"
