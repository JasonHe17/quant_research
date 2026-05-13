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
        diagnostics = _diagnostics_from_members(spec, members)
        universe = Universe(spec=spec, members=members, diagnostics=diagnostics)
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            return universe.with_artifacts(self.artifact_store.write_universe(universe))
        return universe


def _members_from_spec(spec: UniverseSpec, *, data: object | None) -> pd.DataFrame:
    if data is None:
        if spec.symbols == ("*",):
            raise ValueError("wildcard universe requires a DataPortal")
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
    if spec.symbols == ("*",):
        listed = data.list_instruments(
            market=spec.market,
            asset_type=spec.asset_type,
            as_of=spec.end or spec.start,
        )
        _require_columns(
            listed,
            ("instrument_id", "canonical_code", "market", "asset_type"),
        )
        members = listed.loc[
            :, ["canonical_code", "instrument_id", "market", "asset_type"]
        ].copy()
        members = members.rename(columns={"canonical_code": "symbol"})
        members["effective_from"] = (
            listed["effective_from"] if "effective_from" in listed.columns else spec.start
        )
        members["effective_to"] = (
            listed["effective_to"] if "effective_to" in listed.columns else spec.end
        )
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
        ].sort_values("symbol").reset_index(drop=True)
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


def _diagnostics_from_members(spec: UniverseSpec, members: pd.DataFrame) -> pd.DataFrame:
    effective_from_missing = (
        int(members["effective_from"].isna().sum())
        if "effective_from" in members.columns
        else len(members)
    )
    effective_to_missing = (
        int(members["effective_to"].isna().sum())
        if "effective_to" in members.columns
        else len(members)
    )
    return pd.DataFrame(
        [
            {
                "universe_name": spec.name,
                "member_count": len(members),
                "unique_symbol_count": int(members["symbol"].nunique())
                if "symbol" in members.columns
                else 0,
                "market": spec.market,
                "asset_type": spec.asset_type,
                "start": spec.start,
                "end": spec.end,
                "source": "data_portal" if spec.symbols == ("*",) else "symbols",
                "effective_from_missing_count": effective_from_missing,
                "effective_to_missing_count": effective_to_missing,
                "open_ended_member_count": effective_to_missing,
            }
        ]
    )
