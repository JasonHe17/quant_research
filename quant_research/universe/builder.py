"""Universe construction scaffolding."""

from __future__ import annotations

import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.universe.models import Universe, UniverseSpec


class UniverseBuilder:
    """Builds standard universe membership tables."""

    def __init__(self, *, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def build(
        self,
        spec: UniverseSpec,
        *,
        data: object | None = None,
        persist: bool = False,
    ) -> Universe:
        members = _members_from_spec(spec, data=data)
        diagnostics = pd.DataFrame(
            [
                {
                    "universe_name": spec.name,
                    "member_count": len(members),
                    "market": spec.market,
                    "asset_type": spec.asset_type,
                }
            ]
        )
        universe = Universe(spec=spec, members=members, diagnostics=diagnostics)
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            return universe.with_artifacts(self.artifact_store.write_universe(universe))
        return universe


def _members_from_spec(spec: UniverseSpec, *, data: object | None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "instrument_id": symbol,
                    "market": spec.market,
                    "asset_type": spec.asset_type,
                    "effective_from": spec.start,
                    "effective_to": spec.end,
                }
                for symbol in spec.symbols
            ]
        )
    resolved = data.resolve_instruments(
        list(spec.symbols),
        market=spec.market,
        asset_type=spec.asset_type,
    )
    _require_columns(
        resolved,
        ("instrument_id", "canonical_code", "market", "asset_type"),
    )
    members = resolved.loc[
        :, ["canonical_code", "instrument_id", "market", "asset_type"]
    ].copy()
    members = members.rename(columns={"canonical_code": "symbol"})
    members["effective_from"] = spec.start
    members["effective_to"] = spec.end
    return members.loc[
        :,
        [
            "symbol",
            "instrument_id",
            "market",
            "asset_type",
            "effective_from",
            "effective_to",
        ],
    ]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
