from __future__ import annotations

import pandas as pd
import pytest

from quant_research.schemas import validate_standard_table


def test_standard_table_schema_accepts_valid_backtest_trades() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-03T09:35:00+08:00",
                "instrument_id": "inst-1",
                "quantity": 100,
                "price": 10.0,
            }
        ]
    )

    validate_standard_table("backtest_trades", frame)


def test_standard_table_schema_rejects_missing_columns() -> None:
    frame = pd.DataFrame([{"timestamp": "2025-01-03T09:35:00+08:00"}])

    with pytest.raises(ValueError, match="backtest_trades"):
        validate_standard_table("backtest_trades", frame)


def test_standard_table_schema_rejects_bad_numeric_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-03T09:35:00+08:00",
                "instrument_id": "inst-1",
                "quantity": "not-a-number",
                "price": 10.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="quantity.*numeric"):
        validate_standard_table("backtest_trades", frame)


def test_standard_table_schema_rejects_null_required_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-03T09:35:00+08:00",
                "instrument_id": None,
                "quantity": 100,
                "price": 10.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="instrument_id.*nulls"):
        validate_standard_table("backtest_trades", frame)
