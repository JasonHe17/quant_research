"""Research artifact store scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant_research.factors import FactorResult


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """Storage root for research outputs."""

    root: Path

    @classmethod
    def from_path(cls, root: str | Path) -> "ArtifactStore":
        return cls(root=Path(root))

    def factor_path(self, factor_name: str) -> Path:
        return self.root / "factors" / f"{_safe_path_component(factor_name)}.pkl"

    def write_factor(self, result: "FactorResult") -> Path:
        path = self.factor_path(result.factor_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.frame.to_pickle(path)
        return path

    def read_factor(self, factor_name: str) -> pd.DataFrame:
        return pd.read_pickle(self.factor_path(factor_name))


def _safe_path_component(value: str) -> str:
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(allowed).strip("_") or "artifact"
