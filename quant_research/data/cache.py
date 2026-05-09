"""Local research cache policy models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Configuration for rebuildable hot-data cache behavior."""

    root: Path
    enabled: bool = True

    @classmethod
    def disabled(cls) -> "CachePolicy":
        return cls(root=Path(".cache/quant_research"), enabled=False)

    def snapshot_root(self, snapshot: str) -> Path:
        return self.root / "snapshots" / snapshot
