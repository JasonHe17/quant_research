from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.factors import Factor, FactorContext, FactorEngine


class CloseReturnFactor(Factor):
    def compute(self, context: FactorContext) -> pd.DataFrame:
        bars = context.data.get_bars(
            list(context.symbols),
            start=context.start,
            end=context.end,
            frequency=context.frequency,
            adjustment="raw",
            market=context.market,
            asset_type=context.asset_type,
            fields=["instrument_id", "bar_end_time", "close_price"],
            cache=False,
        )
        result = bars.copy()
        result["factor_value"] = result["close_price"].pct_change().fillna(0.0)
        return result[["instrument_id", "bar_end_time", "factor_value"]]


class BadFactor(Factor):
    def compute(self, context: FactorContext) -> object:
        return {"not": "a dataframe"}


class MismatchedFactorName(Factor):
    def compute(self, context: FactorContext) -> pd.DataFrame:
        return pd.DataFrame([{"factor_name": "other", "factor_value": 1.0}])


def test_factor_engine_computes_dataframe_result_with_metadata() -> None:
    data = _FakeDataPortal()
    context = FactorContext(
        data=data,
        start="2024-01-01T09:31:00+08:00",
        end="2024-01-01T09:32:00+08:00",
        symbols=("600000.SH",),
        market="CN",
        asset_type="equity",
        snapshot="2026-05-09",
    )
    factor = CloseReturnFactor(name="close_return_1m", inputs=("close_price",))

    result = FactorEngine().compute(factor, context)

    assert result.factor_name == "close_return_1m"
    assert result.metadata["snapshot"] == "2026-05-09"
    assert list(result.frame.columns) == [
        "factor_name",
        "instrument_id",
        "bar_end_time",
        "factor_value",
    ]
    assert result.frame["factor_value"].tolist() == pytest.approx([0.0, 0.1])
    assert data.calls == 1


def test_factor_engine_persists_factor_result(tmp_path: Path) -> None:
    context = FactorContext(
        data=_FakeDataPortal(),
        start="2024-01-01T09:31:00+08:00",
        end="2024-01-01T09:32:00+08:00",
        symbols=("600000.SH",),
        market="CN",
        asset_type="equity",
    )
    factor = CloseReturnFactor(name="close_return_1m", inputs=("close_price",))
    store = ArtifactStore.from_path(tmp_path)

    result = FactorEngine(artifact_store=store).compute(
        factor, context, persist=True
    )
    loaded = store.read_factor("close_return_1m")
    manifest = store.read_artifact_manifest(store.factor_path("close_return_1m"))

    assert store.factor_path("close_return_1m").exists()
    assert store.factor_path("close_return_1m").suffix == ".parquet"
    assert manifest["format"] == "parquet"
    assert manifest["row_count"] == len(result.frame)
    assert loaded.equals(result.frame)


def test_factor_engine_rejects_non_dataframe_outputs() -> None:
    context = FactorContext(
        data=_FakeDataPortal(),
        start="2024-01-01",
        end="2024-01-02",
        symbols=("600000.SH",),
    )

    with pytest.raises(TypeError, match="pandas DataFrame"):
        FactorEngine().compute(BadFactor(name="bad", inputs=()), context)


def test_factor_engine_rejects_mismatched_factor_name_column() -> None:
    context = FactorContext(
        data=_FakeDataPortal(),
        start="2024-01-01",
        end="2024-01-02",
        symbols=("600000.SH",),
    )

    with pytest.raises(ValueError, match="factor_name"):
        FactorEngine().compute(
            MismatchedFactorName(name="expected", inputs=()), context
        )


class _FakeDataPortal:
    def __init__(self) -> None:
        self.calls = 0

    def get_bars(self, *args: object, **kwargs: object) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame(
            [
                {
                    "instrument_id": "inst-600000",
                    "bar_end_time": "2024-01-01T09:31:00+08:00",
                    "close_price": 10.0,
                },
                {
                    "instrument_id": "inst-600000",
                    "bar_end_time": "2024-01-01T09:32:00+08:00",
                    "close_price": 11.0,
                },
            ]
        )
