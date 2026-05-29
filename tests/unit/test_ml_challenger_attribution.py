from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_module():
    path = Path("examples/analyze_ml_challenger_attribution.py")
    spec = importlib.util.spec_from_file_location("analyze_ml_challenger_attribution", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_top_n_scores_keeps_timestamp_top_names() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        [
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "a", "score": 0.1},
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "b", "score": 0.3},
            {"timestamp": "2024-01-02T09:35:00+08:00", "instrument_id": "c", "score": 0.2},
            {"timestamp": "2024-01-02T09:40:00+08:00", "instrument_id": "a", "score": 0.4},
        ]
    )

    output = module._top_n_scores(frame, top_n=2)

    assert output["instrument_id"].tolist() == ["b", "c", "a"]


def test_monthly_switch_rules_rank_lagged_observable_rules() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03", "2024-04"],
            "baseline_return": [0.0, 0.0, 0.0, 0.0],
            "challenger_return": [0.0, 0.1, -0.1, 0.1],
            "return_delta": [0.0, 0.1, -0.1, 0.1],
            "market_state_return_5m_sum": [-3.0, -2.0, 1.0, -1.0],
        }
    )

    output = module._monthly_switch_rules(frame)

    assert output.iloc[0]["compound_return"] >= output.iloc[-1]["compound_return"]
    assert "always_baseline" in set(output["rule"])
    assert "always_challenger" in set(output["rule"])
    assert output["compound_return"].notna().all()


def test_compound_return_multiplies_monthly_returns() -> None:
    module = _load_module()

    assert module._compound_return(pd.Series([0.1, -0.1])) == pytest.approx(-0.01)
