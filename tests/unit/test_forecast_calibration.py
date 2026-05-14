from __future__ import annotations

import pandas as pd
import pytest

from quant_research.portfolio import (
    ScoreForecastCalibrationConfig,
    apply_score_forecast_calibration,
    build_score_forecast_calibration,
)


def test_score_forecast_calibration_uses_lagged_bucket_labels() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "score": 0.9, "forward_return": 0.02},
            {"timestamp": "t0", "instrument_id": "b", "score": 0.1, "forward_return": -0.01},
            {"timestamp": "t1", "instrument_id": "a", "score": 0.8, "forward_return": -0.03},
            {"timestamp": "t1", "instrument_id": "b", "score": 0.2, "forward_return": 0.01},
            {"timestamp": "t2", "instrument_id": "a", "score": 0.7, "forward_return": 0.04},
            {"timestamp": "t2", "instrument_id": "b", "score": 0.3, "forward_return": -0.02},
        ]
    )
    config = ScoreForecastCalibrationConfig(
        lookback_windows=1,
        min_periods=1,
        label_lag_windows=1,
        bucket_count=2,
    )

    calibration = build_score_forecast_calibration(frame, config)
    top_bucket = calibration[calibration["score_bucket"] == 2].set_index("timestamp")

    assert top_bucket.loc["t0", "forecast_calibration_reason"] == "warmup"
    assert top_bucket.loc["t1", "expected_edge_bps"] == pytest.approx(200.0)
    assert top_bucket.loc["t2", "expected_edge_bps"] == pytest.approx(-300.0)


def test_apply_score_forecast_calibration_attaches_bucket_forecasts() -> None:
    scores = pd.DataFrame(
        [
            {"timestamp": "t1", "instrument_id": "a", "score": 0.9},
            {"timestamp": "t1", "instrument_id": "b", "score": 0.1},
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "timestamp": "t1",
                "score_bucket": 1,
                "expected_edge_bps": -50.0,
                "risk_penalty_bps": 10.0,
                "calibration_window_count": 3,
                "calibration_observation_count": 30,
                "forecast_calibration_reason": "calibrated",
            },
            {
                "timestamp": "t1",
                "score_bucket": 2,
                "expected_edge_bps": 100.0,
                "risk_penalty_bps": 20.0,
                "calibration_window_count": 3,
                "calibration_observation_count": 30,
                "forecast_calibration_reason": "calibrated",
            },
        ]
    )

    result = apply_score_forecast_calibration(
        scores,
        calibration,
        ScoreForecastCalibrationConfig(bucket_count=2),
    ).set_index("instrument_id")

    assert result.loc["a", "forecast_calibration_bucket"] == 2
    assert result.loc["a", "expected_edge_bps"] == pytest.approx(100.0)
    assert result.loc["b", "risk_penalty_bps"] == pytest.approx(10.0)
