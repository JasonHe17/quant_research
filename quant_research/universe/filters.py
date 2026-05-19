"""Universe filter placeholders."""

from __future__ import annotations

import pandas as pd

from quant_research.universe.models import Universe


def identity_filter(universe: Universe) -> Universe:
    """Return a universe unchanged."""

    return universe


def active_on(universe: Universe, date: str) -> Universe:
    """Filter members whose effective interval contains ``date``."""

    members = universe.members.copy()
    start_ok = members["effective_from"].isna() | (members["effective_from"] <= date)
    end_ok = members["effective_to"].isna() | (members["effective_to"] >= date)
    filtered = members.loc[start_ok & end_ok].reset_index(drop=True)
    diagnostics = pd.concat(
        [
            universe.diagnostics,
            pd.DataFrame(
                [
                    {
                        "universe_name": universe.spec.name,
                        "filter": "active_on",
                        "date": date,
                        "member_count": len(filtered),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return Universe(
        spec=universe.spec,
        members=filtered,
        diagnostics=diagnostics,
        artifacts=dict(universe.artifacts),
    )


def cn_main_board(universe: Universe) -> Universe:
    """Keep Shanghai/Shenzhen main-board A-share members by canonical code."""

    members = universe.members.copy()
    _require_columns(members, ("symbol", "market", "asset_type"))
    mask = cn_main_board_symbol_mask(
        members["symbol"],
        market=members["market"],
        asset_type=members["asset_type"],
    )
    filtered = members.loc[mask].reset_index(drop=True)
    diagnostics = pd.concat(
        [
            universe.diagnostics,
            pd.DataFrame(
                [
                    {
                        "universe_name": universe.spec.name,
                        "filter": "cn_main_board",
                        "member_count": len(filtered),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return Universe(
        spec=universe.spec,
        members=filtered,
        diagnostics=diagnostics,
        artifacts=dict(universe.artifacts),
    )


def is_cn_main_board_symbol(
    symbol: str,
    *,
    market: str | None = "CN",
    asset_type: str | None = "equity",
) -> bool:
    """Return whether a canonical code is a Shanghai/Shenzhen main-board A share."""

    if market is not None and market.upper() != "CN":
        return False
    if asset_type is not None and asset_type.lower() != "equity":
        return False
    code, suffix = _split_canonical_symbol(symbol)
    if suffix == "SH":
        return code.startswith(("600", "601", "603", "605"))
    if suffix == "SZ":
        return code.startswith(("000", "001", "002", "003"))
    return False


def cn_main_board_symbol_mask(
    symbols: pd.Series,
    *,
    market: pd.Series | str | None = "CN",
    asset_type: pd.Series | str | None = "equity",
) -> pd.Series:
    """Vectorized main-board A-share mask matching ``is_cn_main_board_symbol``."""

    codes, suffixes = _split_canonical_symbol_series(symbols)
    market_ok = _optional_text_filter(market, expected="CN", normalize="upper")
    asset_type_ok = _optional_text_filter(
        asset_type,
        expected="equity",
        normalize="lower",
    )
    sh_main = suffixes.eq("SH") & codes.str.startswith(("600", "601", "603", "605"))
    sz_main = suffixes.eq("SZ") & codes.str.startswith(("000", "001", "002", "003"))
    return (market_ok & asset_type_ok & (sh_main | sz_main)).astype(bool)


def _split_canonical_symbol(symbol: str) -> tuple[str, str]:
    parts = symbol.strip().upper().split(".", maxsplit=1)
    if len(parts) != 2:
        return "", ""
    code, suffix = parts
    return code, suffix


def _split_canonical_symbol_series(symbols: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = symbols.fillna("").astype(str).str.strip().str.upper().str.split(".", n=1)
    codes = parts.str[0].fillna("")
    suffixes = parts.str[1].fillna("")
    return codes, suffixes


def _optional_text_filter(
    values: pd.Series | str | None,
    *,
    expected: str,
    normalize: str,
) -> pd.Series | bool:
    if values is None:
        return True
    if isinstance(values, pd.Series):
        text = values.astype("string")
        if normalize == "upper":
            normalized = text.str.upper()
        else:
            normalized = text.str.lower()
        return text.isna() | normalized.eq(expected)
    value = str(values)
    normalized_value = value.upper() if normalize == "upper" else value.lower()
    return normalized_value == expected


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
