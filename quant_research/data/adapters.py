"""Adapters for stable data-layer SDKs."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator

import pandas as pd


class QuantDbAdapter:
    """Adapter boundary for quantdb.sdk integration."""

    def __init__(
        self,
        *,
        quant_dataset_root: str | Path,
        sdk_module: ModuleType | object | None = None,
    ) -> None:
        self.quant_dataset_root = Path(quant_dataset_root).resolve()
        self._sdk_module = sdk_module
        self._external_sdk = sdk_module is not None
        self._loaded = sdk_module is not None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def resolve_instruments(
        self,
        symbols: list[str],
        *,
        market: str | None,
        asset_type: str | None,
    ) -> pd.DataFrame:
        if market is None:
            raise ValueError("market is required for instrument resolution")
        sdk = self._sdk()
        with self._runtime_context():
            rows = [
                sdk.resolve_instrument(symbol, market=market, asset_type=asset_type)
                for symbol in symbols
            ]
        return rows_to_frame(rows)

    def list_instruments(
        self,
        *,
        market: str | None,
        asset_type: str | None,
        as_of: str | None,
    ) -> pd.DataFrame:
        sdk = self._sdk()
        with self._runtime_context():
            rows = sdk.list_instruments(
                market=market,
                asset_type=asset_type,
                as_of=as_of,
            )
        return rows_to_frame(rows)

    def get_trading_calendar(
        self,
        *,
        market: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        sdk = self._sdk()
        with self._runtime_context():
            rows = sdk.get_trading_calendar(market=market, start=start, end=end)
        return rows_to_frame(rows)

    def get_bars(
        self,
        symbols: list[str],
        *,
        start: str,
        end: str,
        frequency: str,
        adjustment: str,
        market: str | None,
        asset_type: str | None,
    ) -> pd.DataFrame:
        instruments = self.resolve_instruments(
            symbols, market=market, asset_type=asset_type
        )
        sdk = self._sdk()
        frames: list[pd.DataFrame] = []
        with self._runtime_context():
            for instrument in instruments.to_dict("records"):
                if adjustment.lower() == "raw":
                    rows = sdk.get_minute_bars(
                        instrument["instrument_id"],
                        market=instrument["market"],
                        asset_type=instrument["asset_type"],
                        frequency=frequency,
                        adjustment="raw",
                        start=start,
                        end=end,
                    )
                else:
                    rows = sdk.get_adjusted_bars(
                        instrument["instrument_id"],
                        market=instrument["market"],
                        asset_type=instrument["asset_type"],
                        frequency=frequency,
                        adjustment=adjustment,
                        start=start,
                        end=end,
                    )
                frame = rows_to_frame(rows)
                if not frame.empty:
                    frame.insert(0, "query_symbol", instrument["canonical_code"])
                frames.append(frame)
        return concat_frames(frames)

    def get_fundamentals_asof(
        self,
        symbols: list[str],
        *,
        dataset: str,
        as_of: str,
        market: str | None,
        asset_type: str | None,
    ) -> pd.DataFrame:
        instruments = self.resolve_instruments(
            symbols, market=market, asset_type=asset_type
        )
        sdk = self._sdk()
        frames: list[pd.DataFrame] = []
        with self._runtime_context():
            for instrument in instruments.to_dict("records"):
                rows = sdk.get_fundamentals_asof(
                    instrument["instrument_id"],
                    dataset=dataset,
                    as_of=as_of,
                    market=instrument["market"],
                    asset_type=instrument["asset_type"],
                )
                frame = rows_to_frame(rows)
                if not frame.empty:
                    frame.insert(0, "query_symbol", instrument["canonical_code"])
                frames.append(frame)
        return concat_frames(frames)

    def get_market_indicators(
        self,
        *,
        names: list[str] | None,
        start: str | None,
        end: str | None,
        frequency: str,
        market: str,
        indicator_type: str | None,
    ) -> pd.DataFrame:
        sdk = self._sdk()
        with self._runtime_context():
            if names is None:
                rows = sdk.get_market_indicators(
                    market=market, indicator_type=indicator_type
                )
                return rows_to_frame(rows)
            frames = [
                rows_to_frame(
                    sdk.get_market_indicator_minutes(
                        name,
                        indicator_type=indicator_type,
                        market=market,
                        frequency=frequency,
                        start=start,
                        end=end,
                    )
                )
                for name in names
            ]
        return concat_frames(frames)

    def list_available_datasets(self, *, domain: str | None = None) -> pd.DataFrame:
        sdk = self._sdk()
        with self._runtime_context():
            rows = sdk.list_available_datasets(domain=domain)
        return rows_to_frame(rows)

    def _sdk(self) -> Any:
        if self._sdk_module is not None:
            return self._sdk_module
        with self._quant_dataset_context():
            self._sdk_module = importlib.import_module("quantdb.sdk")
        self._loaded = True
        return self._sdk_module

    def _runtime_context(self) -> contextmanager[None]:
        if self._external_sdk:
            return nullcontext()
        return self._quant_dataset_context()

    @contextmanager
    def _quant_dataset_context(self) -> Iterator[None]:
        root_text = str(self.quant_dataset_root)
        previous_cwd = Path.cwd()
        inserted = False
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
            inserted = True
        os.chdir(self.quant_dataset_root)
        try:
            yield
        finally:
            os.chdir(previous_cwd)
            if inserted:
                try:
                    sys.path.remove(root_text)
                except ValueError:
                    pass


def rows_to_frame(rows: object) -> pd.DataFrame:
    """Convert SDK dataclass/object rows into a DataFrame."""

    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, tuple | list):
        normalized = [_row_to_dict(row) for row in rows]
        return pd.DataFrame(normalized)
    return pd.DataFrame([_row_to_dict(rows)])


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def select_fields(frame: pd.DataFrame, fields: list[str] | None) -> pd.DataFrame:
    if fields is None:
        return frame
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise KeyError(f"Requested fields are not present: {missing}")
    return frame.loc[:, fields]


def _row_to_dict(row: object) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if is_dataclass(row):
        return asdict(row)
    slots = getattr(row, "__slots__", None)
    if slots is not None:
        return {name: getattr(row, name) for name in slots}
    values = getattr(row, "__dict__", None)
    if values is not None:
        return dict(values)
    raise TypeError(f"Cannot convert row of type {type(row)!r} to a mapping")
