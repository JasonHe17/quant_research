from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.universe import (
    UniverseBuilder,
    UniverseSpec,
    active_on,
    cn_main_board,
    is_cn_main_board_symbol,
)


def test_universe_builder_builds_static_members() -> None:
    universe = UniverseBuilder().build(
        UniverseSpec(
            name="static-cn",
            symbols=("600000.SH", "000001.SZ"),
            market="CN",
            asset_type="equity",
            start="2024-01-01",
            end="2024-12-31",
        )
    )

    assert universe.members["instrument_id"].tolist() == ["600000.SH", "000001.SZ"]
    assert universe.members["effective_from"].unique().tolist() == ["2024-01-01"]
    assert universe.diagnostics.loc[0, "member_count"] == 2
    assert universe.diagnostics.loc[0, "source"] == "symbols"


def test_universe_builder_resolves_members_with_data_portal() -> None:
    universe = UniverseBuilder().build(
        UniverseSpec(
            name="resolved-cn",
            symbols=("600000.SH",),
            market="CN",
            asset_type="equity",
        ),
        data=_FakeDataPortal(),
    )

    assert universe.members.loc[0, "symbol"] == "600000.SH"
    assert universe.members.loc[0, "instrument_id"] == "inst-600000"


def test_universe_builder_builds_wildcard_members_from_data_portal() -> None:
    universe = UniverseBuilder().build(
        UniverseSpec(
            name="all-cn-equity",
            symbols=("*",),
            market="CN",
            asset_type="equity",
            start="2024-01-01",
            end="2024-12-31",
        ),
        data=_FakeDataPortal(),
    )

    filtered = cn_main_board(universe)

    assert universe.members["symbol"].tolist() == ["300750.SZ", "600000.SH"]
    assert filtered.members["symbol"].tolist() == ["600000.SH"]
    assert universe.diagnostics.loc[0, "source"] == "data_portal"
    assert universe.diagnostics.loc[0, "effective_from_missing_count"] == 0
    assert filtered.diagnostics.iloc[-1]["filter"] == "cn_main_board"


def test_universe_active_on_filters_effective_dates() -> None:
    universe = UniverseBuilder().build(
        UniverseSpec(
            name="static-cn",
            symbols=("600000.SH", "000001.SZ"),
            market="CN",
            start="2024-01-01",
            end="2024-01-31",
        )
    )

    active = active_on(universe, "2024-01-15")
    inactive = active_on(universe, "2024-02-01")

    assert len(active.members) == 2
    assert len(inactive.members) == 0
    assert active.diagnostics.iloc[-1]["filter"] == "active_on"


def test_cn_main_board_filters_to_shenzhen_and_shanghai_main_board() -> None:
    universe = UniverseBuilder().build(
        UniverseSpec(
            name="cn-all",
            symbols=(
                "600000.SH",
                "601318.SH",
                "603000.SH",
                "605000.SH",
                "000001.SZ",
                "001979.SZ",
                "002415.SZ",
                "003816.SZ",
                "300750.SZ",
                "688001.SH",
                "920010.BJ",
            ),
            market="CN",
            asset_type="equity",
        )
    )

    filtered = cn_main_board(universe)

    assert filtered.members["symbol"].tolist() == [
        "600000.SH",
        "601318.SH",
        "603000.SH",
        "605000.SH",
        "000001.SZ",
        "001979.SZ",
        "002415.SZ",
        "003816.SZ",
    ]
    assert filtered.diagnostics.iloc[-1]["filter"] == "cn_main_board"


def test_is_cn_main_board_symbol_rejects_non_main_board_markets() -> None:
    assert is_cn_main_board_symbol("600000.SH")
    assert not is_cn_main_board_symbol("300750.SZ")
    assert not is_cn_main_board_symbol("688001.SH")
    assert not is_cn_main_board_symbol("920010.BJ")
    assert not is_cn_main_board_symbol("00001.HK", market="HK")
    assert not is_cn_main_board_symbol("510050.SH", asset_type="fund")


def test_universe_builder_persists_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore.from_path(tmp_path)
    universe = UniverseBuilder(artifact_store=store).build(
        UniverseSpec(name="persisted", symbols=("600000.SH",), market="CN"),
        persist=True,
    )

    assert set(universe.artifacts) == {"members", "diagnostics"}
    assert store.read_universe_artifact("persisted", "members").equals(
        universe.members
    )


def test_universe_spec_validates_inputs() -> None:
    with pytest.raises(ValueError, match="universe name"):
        UniverseSpec(name="", symbols=("600000.SH",))

    with pytest.raises(ValueError, match="symbols"):
        UniverseSpec(name="empty", symbols=())

    with pytest.raises(ValueError, match="artifact_store"):
        UniverseBuilder().build(
            UniverseSpec(name="static-cn", symbols=("600000.SH",)),
            persist=True,
        )


class _FakeDataPortal:
    def list_instruments(
        self,
        *,
        market: str | None = None,
        asset_type: str | None = None,
        as_of: str | None = None,
    ) -> pd.DataFrame:
        _ = as_of
        return pd.DataFrame(
            [
                {
                    "canonical_code": "600000.SH",
                    "instrument_id": "inst-600000",
                    "market": market,
                    "asset_type": asset_type,
                    "effective_from": "2024-01-01",
                    "effective_to": "2024-12-31",
                },
                {
                    "canonical_code": "300750.SZ",
                    "instrument_id": "inst-300750",
                    "market": market,
                    "asset_type": asset_type,
                    "effective_from": "2024-01-01",
                    "effective_to": "2024-12-31",
                },
            ]
        )

    def resolve_instruments(
        self,
        symbols: list[str],
        *,
        market: str | None = None,
        asset_type: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical_code": symbol,
                    "instrument_id": "inst-600000",
                    "market": market,
                    "asset_type": asset_type,
                }
                for symbol in symbols
            ]
        )
