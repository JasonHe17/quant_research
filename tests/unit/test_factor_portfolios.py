from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from examples.run_candidate_factor_portfolios import (
    BacktestPolicySpec,
    _build_factor_health_schedule,
    _backtest_command,
    _backtest_jobs,
    _backtest_policy_specs,
    _backtest_summary_rows,
    _dataset_paths,
    _default_label_lag_windows,
    _effective_backtest_memory_budget_gb,
    _factor_weight_scale_schedule,
    _parse_factor_health_ensemble_lookbacks,
    _registry_filter_summary,
    _reused_scores_summary,
    _summary_params,
)
from quant_research.portfolio import (
    CandidateFactor,
    FactorHealthConfig,
    ScoreForecastCalibrationConfig,
    build_composite_scores,
    build_factor_health_ensemble_schedule,
    build_factor_health_schedule,
    build_state_conditioned_factor_health_schedule,
    build_state_conditioned_factor_health_schedule_from_partitions,
    cap_factor_weights,
    factor_contribution_diagnostics,
    factor_combination_weights,
    load_candidate_factors,
    write_score_partitions,
)


def test_load_candidate_factors_uses_admission_direction(tmp_path: Path) -> None:
    path = tmp_path / "admission.json"
    path.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "feature": "alpha_a",
                        "admission_status": "candidate",
                        "direction": "invert",
                        "spearman_rank_ic_mean": -0.02,
                    },
                    {
                        "feature": "alpha_b",
                        "admission_status": "watchlist",
                        "direction": "long",
                        "spearman_rank_ic_mean": 0.01,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    factors = load_candidate_factors(path)

    assert factors == (CandidateFactor("alpha_a", -1, -0.02),)


def test_load_candidate_factors_can_filter_shared_admission_features(
    tmp_path: Path,
) -> None:
    path = tmp_path / "admission.json"
    path.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "feature": "alpha_a",
                        "admission_status": "candidate",
                        "direction": "long",
                        "spearman_rank_ic_mean": 0.02,
                    },
                    {
                        "feature": "alpha_b",
                        "admission_status": "candidate",
                        "direction": "invert",
                        "spearman_rank_ic_mean": -0.01,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    factors = load_candidate_factors(path, include_features=("alpha_b",))

    assert factors == (CandidateFactor("alpha_b", -1, -0.01),)


def test_candidate_factor_registry_filter_excludes_unregistered_and_rejected(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "test",
                "version": 1,
                "entries": [
                    {
                        "factor_id": "alpha_a",
                        "display_name": "alpha_a",
                        "family": "momentum",
                        "status": "candidate",
                        "expected_direction": "long",
                        "feature_columns": ["alpha_a"],
                        "required_inputs": ["close_price"],
                        "frequency": "5m",
                        "description": "test",
                        "hypothesis": "test",
                    },
                    {
                        "factor_id": "alpha_b",
                        "display_name": "alpha_b",
                        "family": "momentum",
                        "status": "reject",
                        "expected_direction": "long",
                        "feature_columns": ["alpha_b"],
                        "required_inputs": ["close_price"],
                        "frequency": "5m",
                        "description": "test",
                        "hypothesis": "test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "enforce_registry": True,
            "registry": str(registry_path),
            "registry_statuses": ["candidate", "promoted"],
        },
    )()

    summary = _registry_filter_summary(
        args,
        (
            CandidateFactor("alpha_a", 1, 0.02),
            CandidateFactor("alpha_b", 1, 0.01),
            CandidateFactor("alpha_unregistered", 1, 0.01),
        ),
    )

    assert summary["included_features"] == ["alpha_a"]
    assert summary["output_candidate_count"] == 1
    assert summary["excluded_features"] == [
        {"feature": "alpha_b", "reason": "registry_status=reject"},
        {"feature": "alpha_unregistered", "reason": "unregistered"},
    ]


def test_factor_combination_weights_support_methods() -> None:
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", -1, -0.01),
    )
    correlation = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]],
        index=["alpha_a", "alpha_b"],
        columns=["alpha_a", "alpha_b"],
    )

    equal = factor_combination_weights(factors, method="equal")
    ic_weighted = factor_combination_weights(factors, method="ic_weighted")
    decorrelated = factor_combination_weights(
        factors,
        method="decorrelated",
        correlation=correlation,
    )

    assert equal == {"alpha_a": 0.5, "alpha_b": 0.5}
    assert ic_weighted["alpha_a"] == pytest.approx(2 / 3)
    assert sum(decorrelated.values()) == pytest.approx(1.0)


def test_cap_factor_weights_limits_concentrated_static_weights() -> None:
    weights = cap_factor_weights(
        {"alpha_a": 0.8, "alpha_b": 0.15, "alpha_c": 0.05},
        max_weight=0.6,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["alpha_a"] <= 0.6000001
    assert weights["alpha_b"] > 0.15


def test_build_composite_scores_ranks_and_orients_cross_sectionally() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0, "alpha_b": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0, "alpha_b": 0.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0, "alpha_b": -1.0},
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", -1, -0.01),
    )

    scores = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.5, "alpha_b": 0.5},
    )

    assert scores.iloc[0]["instrument_id"] == "c"
    assert scores.iloc[0]["score"] > scores.iloc[-1]["score"]


def test_build_composite_scores_applies_factor_health_scales() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0, "alpha_b": 3.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0, "alpha_b": 2.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0, "alpha_b": 1.0},
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", 1, 0.01),
    )
    health = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "weight_scale": 1.0},
            {"timestamp": "t0", "feature": "alpha_b", "weight_scale": 0.0},
        ]
    )

    scores = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.5, "alpha_b": 0.5},
        factor_health=health,
    )

    assert scores.iloc[0]["instrument_id"] == "c"
    assert scores.iloc[-1]["instrument_id"] == "a"


def test_build_composite_scores_caps_row_level_factor_contributions() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0, "alpha_b": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0, "alpha_b": 1.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0, "alpha_b": 1.0},
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", 1, 0.01),
    )

    uncapped = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.9, "alpha_b": 0.1},
    )
    capped = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.9, "alpha_b": 0.1},
        max_factor_contribution_share=0.5,
    )

    assert capped.iloc[0]["score"] < uncapped.iloc[0]["score"]


def test_build_composite_scores_does_not_zero_single_factor_when_capped() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0},
        ]
    )

    scores = build_composite_scores(
        frame,
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        weights={"alpha_a": 1.0},
        max_factor_contribution_share=0.7,
    )

    assert scores["score"].abs().max() > 0.0
    assert scores.iloc[0]["instrument_id"] == "c"


def test_build_factor_health_schedule_uses_only_lagged_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index == 0 else -value,
                "forward_return_240b": 10.0 * (value if index == 0 else -value),
            }
            for index in range(4)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)

    schedule = build_factor_health_schedule(
        [dataset_path],
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        config=FactorHealthConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            min_scale=0.25,
            max_scale=1.0,
            rank_ic_floor=-1.0,
            rank_ic_ceiling=1.0,
            spread_floor=-1.0,
            spread_ceiling=1.0,
        ),
        top_n=1,
        label_column="forward_return_240b",
    )

    assert schedule.loc[0, "shrink_reason"] == "warmup"
    assert schedule.loc[0, "label_column"] == "forward_return_240b"
    assert schedule.loc[0, "health_state"] == "warmup"
    assert "recommended_weight_scale" in schedule.columns
    assert schedule.loc[1, "weight_scale"] == pytest.approx(1.0)
    assert schedule.loc[2, "weight_scale"] == pytest.approx(0.25)


def test_build_factor_health_schedule_can_monitor_without_shrinking(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index == 0 else -value,
            }
            for index in range(3)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)

    schedule = build_factor_health_schedule(
        [dataset_path],
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        config=FactorHealthConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            min_scale=0.25,
            max_scale=1.0,
            rank_ic_floor=-1.0,
            rank_ic_ceiling=1.0,
            spread_floor=-1.0,
            spread_ceiling=1.0,
        ),
        top_n=1,
        apply_shrink=False,
    )

    assert schedule["weight_scale"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert schedule.loc[2, "recommended_weight_scale"] == pytest.approx(0.25)
    assert schedule.loc[2, "shrink_reason"] == "monitor_only"
    assert schedule.loc[2, "health_state"] == "impaired"


def test_build_factor_health_ensemble_schedule_blends_lookbacks(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index == 0 else -value,
            }
            for index in range(4)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)

    schedule = build_factor_health_ensemble_schedule(
        [dataset_path],
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        configs=(
            FactorHealthConfig(
                lookback_windows=1,
                min_periods=1,
                label_lag_windows=1,
                min_scale=0.25,
                max_scale=1.0,
                rank_ic_floor=-1.0,
                rank_ic_ceiling=1.0,
                spread_floor=-1.0,
                spread_ceiling=1.0,
            ),
            FactorHealthConfig(
                lookback_windows=2,
                min_periods=1,
                label_lag_windows=1,
                min_scale=0.25,
                max_scale=1.0,
                rank_ic_floor=-1.0,
                rank_ic_ceiling=1.0,
                spread_floor=-1.0,
                spread_ceiling=1.0,
            ),
        ),
        top_n=1,
        combine_mode="mean",
    )

    assert schedule.loc[0, "shrink_reason"] == "warmup"
    assert schedule.loc[2, "weight_scale"] == pytest.approx(0.4375)
    assert schedule.loc[2, "shrink_reason"] == "ensemble_lagged_health_shrink"
    assert schedule.loc[2, "ensemble_lookback_windows"] == "1,2"


def test_factor_contribution_diagnostics_reports_top_concentration() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "alpha_a": 1.0,
                "alpha_b": 1.0,
                "forward_return": 0.01,
                "forward_return_240b": 0.10,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha_a": 2.0,
                "alpha_b": 1.0,
                "forward_return": 0.02,
                "forward_return_240b": 0.20,
            },
            {
                "timestamp": "t0",
                "instrument_id": "c",
                "alpha_a": 3.0,
                "alpha_b": 1.0,
                "forward_return": 0.03,
                "forward_return_240b": 0.30,
            },
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", 1, 0.01),
    )
    scores = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.8, "alpha_b": 0.2},
    )

    diagnostics = factor_contribution_diagnostics(
        frame,
        scores=scores,
        candidates=factors,
        weights={"alpha_a": 0.8, "alpha_b": 0.2},
        factor_health=None,
        max_factor_contribution_share=0.5,
        top_n=2,
        label_column="forward_return_240b",
    )

    assert diagnostics.loc[0, "label_column"] == "forward_return_240b"
    assert diagnostics.loc[0, "top_score_mean_label"] == pytest.approx(0.25)
    assert diagnostics.loc[0, "largest_contribution_feature"] == "alpha_a"
    assert diagnostics.loc[0, "largest_abs_contribution_share"] <= 0.5


def test_write_score_partitions_writes_one_partition_per_method(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "alpha_a": 1.0,
                "forward_return": 0.01,
                "forward_return_240b": 0.10,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha_a": 2.0,
                "forward_return": 0.02,
                "forward_return_240b": 0.20,
            },
        ]
    ).to_parquet(dataset_path, index=False)

    summary = write_score_partitions(
        [dataset_path],
        output_dir=tmp_path / "scores",
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        weights_by_method={"equal": {"alpha_a": 1.0}},
        diagnostics_top_n=1,
        diagnostics_label_column="forward_return_240b",
    )

    assert summary["methods"]["equal"]["row_count"] == 2
    assert Path(tmp_path / "scores" / "equal" / "score_2024_01.parquet").exists()
    assert Path(
        tmp_path
        / "scores"
        / "equal"
        / "diagnostics"
        / "factor_contribution_2024_01.csv"
    ).exists()
    diagnostics = pd.read_csv(
        tmp_path
        / "scores"
        / "equal"
        / "diagnostics"
        / "factor_contribution_2024_01.csv"
    )
    assert diagnostics.loc[0, "label_column"] == "forward_return_240b"


def test_write_score_partitions_can_attach_calibrated_forecasts(tmp_path: Path) -> None:
    for partition, rows in {
        "2024_01": [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 3.0, "forward_return": 0.02},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 1.0, "forward_return": -0.01},
            {"timestamp": "t1", "instrument_id": "a", "alpha_a": 3.0, "forward_return": -0.03},
            {"timestamp": "t1", "instrument_id": "b", "alpha_a": 1.0, "forward_return": 0.01},
        ],
        "2024_02": [
            {"timestamp": "t2", "instrument_id": "a", "alpha_a": 3.0, "forward_return": 0.04},
            {"timestamp": "t2", "instrument_id": "b", "alpha_a": 1.0, "forward_return": -0.02},
        ],
    }.items():
        pd.DataFrame(rows).to_parquet(tmp_path / f"dataset_{partition}.parquet", index=False)

    summary = write_score_partitions(
        [
            tmp_path / "dataset_2024_01.parquet",
            tmp_path / "dataset_2024_02.parquet",
        ],
        output_dir=tmp_path / "scores",
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        weights_by_method={"equal": {"alpha_a": 1.0}},
        forecast_calibration_config=ScoreForecastCalibrationConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            bucket_count=2,
        ),
    )

    scores = pd.read_parquet(tmp_path / "scores" / "equal" / "score_2024_02.parquet")
    top = scores.sort_values("score", ascending=False).iloc[0]
    assert "expected_edge_bps" in scores.columns
    assert top["expected_edge_bps"] == pytest.approx(-300.0)
    assert summary["methods"]["equal"]["score_forecast_calibration"]


def test_candidate_factor_builds_factor_health_schedule_when_enabled(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value,
            }
            for index in range(2)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)
    args = _portfolio_args(
        dataset_dir=str(tmp_path),
        factor_health_mode="shrink",
        factor_health_lookback_windows=1,
        factor_health_min_periods=1,
        factor_health_label_lag_windows=1,
        score_diagnostics_top_n=1,
    )

    schedule = _build_factor_health_schedule(
        args,
        [dataset_path],
        (CandidateFactor("alpha_a", 1, 0.02),),
    )

    assert schedule is not None
    assert len(schedule) == 2


def test_candidate_factor_builds_ensemble_factor_health_schedule(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index == 0 else -value,
            }
            for index in range(4)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)
    args = _portfolio_args(
        dataset_dir=str(tmp_path),
        factor_health_mode="shrink",
        factor_health_ensemble_lookbacks="1,2",
        factor_health_min_periods=1,
        factor_health_label_lag_windows=1,
        factor_health_rank_ic_floor=-1.0,
        factor_health_rank_ic_ceiling=1.0,
        factor_health_spread_floor=-1.0,
        factor_health_spread_ceiling=1.0,
        score_diagnostics_top_n=1,
    )

    schedule = _build_factor_health_schedule(
        args,
        [dataset_path],
        (CandidateFactor("alpha_a", 1, 0.02),),
    )

    assert schedule is not None
    assert schedule.loc[2, "ensemble_combine_mode"] == "mean"
    assert schedule.loc[2, "ensemble_lookback_windows"] == "1,2"


def test_factor_health_ensemble_reads_each_partition_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index == 0 else -value,
            }
            for index in range(4)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)
    read_calls: list[Path] = []
    original_read_parquet = pd.read_parquet

    def spy_read_parquet(path: Path, **kwargs: object) -> pd.DataFrame:
        read_calls.append(Path(path))
        return original_read_parquet(path, **kwargs)

    monkeypatch.setattr(
        "quant_research.portfolio.factor_portfolios.pd.read_parquet",
        spy_read_parquet,
    )

    schedule = build_factor_health_ensemble_schedule(
        [dataset_path],
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        configs=(
            FactorHealthConfig(
                lookback_windows=1,
                min_periods=1,
                label_lag_windows=1,
                rank_ic_floor=-1.0,
                rank_ic_ceiling=1.0,
                spread_floor=-1.0,
                spread_ceiling=1.0,
            ),
            FactorHealthConfig(
                lookback_windows=2,
                min_periods=1,
                label_lag_windows=1,
                rank_ic_floor=-1.0,
                rank_ic_ceiling=1.0,
                spread_floor=-1.0,
                spread_ceiling=1.0,
            ),
        ),
        top_n=1,
    )

    assert not schedule.empty
    assert read_calls == [dataset_path]


def test_build_state_conditioned_factor_health_schedule_selects_stress() -> None:
    normal = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "weight_scale": 0.9},
            {"timestamp": "t1", "feature": "alpha_a", "weight_scale": 0.8},
        ]
    )
    stress = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "weight_scale": 0.5},
            {"timestamp": "t1", "feature": "alpha_a", "weight_scale": 0.4},
        ]
    )
    regime = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "regime_alpha", "weight_scale": 1.0},
            {"timestamp": "t1", "feature": "regime_alpha", "weight_scale": 0.7},
        ]
    )

    schedule = build_state_conditioned_factor_health_schedule(
        normal,
        stress,
        regime,
        regime_feature="regime_alpha",
        mode="select",
        threshold=0.9,
    )

    assert schedule.loc[0, "weight_scale"] == pytest.approx(0.9)
    assert schedule.loc[1, "weight_scale"] == pytest.approx(0.4)
    assert schedule.loc[1, "regime_weight"] == pytest.approx(1.0)


def test_candidate_factor_builds_state_conditioned_factor_health_schedule(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index in {0, 1} else -value,
            }
            for index in range(4)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)
    regime_path = tmp_path / "regime.csv"
    pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "regime_alpha", "weight_scale": 1.0},
            {"timestamp": "t1", "feature": "regime_alpha", "weight_scale": 1.0},
            {"timestamp": "t2", "feature": "regime_alpha", "weight_scale": 0.5},
            {"timestamp": "t3", "feature": "regime_alpha", "weight_scale": 0.5},
        ]
    ).to_csv(regime_path, index=False)
    args = _portfolio_args(
        dataset_dir=str(tmp_path),
        factor_health_mode="shrink",
        factor_health_lookback_windows=1,
        factor_health_stress_lookback_windows=2,
        factor_health_min_periods=1,
        factor_health_label_lag_windows=1,
        factor_health_rank_ic_floor=-1.0,
        factor_health_rank_ic_ceiling=1.0,
        factor_health_spread_floor=-1.0,
        factor_health_spread_ceiling=1.0,
        factor_health_state_regime_mode="select",
        factor_health_state_regime_schedule=str(regime_path),
        factor_health_state_regime_feature="regime_alpha",
        score_diagnostics_top_n=1,
    )

    schedule = _build_factor_health_schedule(
        args,
        [dataset_path],
        (CandidateFactor("alpha_a", 1, 0.02),),
    )

    assert schedule is not None
    assert schedule.loc[2, "state_conditioned_mode"] == "select"
    assert schedule.loc[2, "regime_weight"] == pytest.approx(1.0)
    assert "normal_weight_scale" in schedule.columns
    assert "stress_weight_scale" in schedule.columns


def test_state_conditioned_factor_health_reads_each_partition_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index in {0, 1} else -value,
            }
            for index in range(4)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)
    regime = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "regime_alpha", "weight_scale": 1.0},
            {"timestamp": "t1", "feature": "regime_alpha", "weight_scale": 1.0},
            {"timestamp": "t2", "feature": "regime_alpha", "weight_scale": 0.5},
            {"timestamp": "t3", "feature": "regime_alpha", "weight_scale": 0.5},
        ]
    )
    read_calls: list[Path] = []
    original_read_parquet = pd.read_parquet

    def spy_read_parquet(path: Path, **kwargs: object) -> pd.DataFrame:
        read_calls.append(Path(path))
        return original_read_parquet(path, **kwargs)

    monkeypatch.setattr(
        "quant_research.portfolio.factor_portfolios.pd.read_parquet",
        spy_read_parquet,
    )

    schedule = build_state_conditioned_factor_health_schedule_from_partitions(
        [dataset_path],
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        normal_config=FactorHealthConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            rank_ic_floor=-1.0,
            rank_ic_ceiling=1.0,
            spread_floor=-1.0,
            spread_ceiling=1.0,
        ),
        stress_config=FactorHealthConfig(
            lookback_windows=2,
            min_periods=1,
            label_lag_windows=1,
            rank_ic_floor=-1.0,
            rank_ic_ceiling=1.0,
            spread_floor=-1.0,
            spread_ceiling=1.0,
        ),
        regime=regime,
        regime_feature="regime_alpha",
        top_n=1,
    )

    assert not schedule.empty
    assert schedule.loc[2, "regime_weight"] == pytest.approx(1.0)
    assert read_calls == [dataset_path]


def test_candidate_factor_monitor_mode_does_not_apply_health_scales(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index == 0 else -value,
            }
            for index in range(3)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)
    args = _portfolio_args(
        dataset_dir=str(tmp_path),
        factor_health_mode="monitor",
        factor_health_lookback_windows=1,
        factor_health_min_periods=1,
        factor_health_label_lag_windows=1,
        score_diagnostics_top_n=1,
    )

    schedule = _build_factor_health_schedule(
        args,
        [dataset_path],
        (CandidateFactor("alpha_a", 1, 0.02),),
    )

    assert schedule is not None
    assert schedule["weight_scale"].max() == pytest.approx(1.0)
    assert schedule["recommended_weight_scale"].min() < 1.0


def test_candidate_factor_combines_external_weight_scale_schedule(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "weight_scale.csv"
    pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "weight_scale": 0.4},
            {"timestamp": "t0", "feature": "alpha_b", "weight_scale": 0.8},
        ]
    ).to_csv(schedule_path, index=False)
    health = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "weight_scale": 0.5},
            {"timestamp": "t0", "feature": "alpha_b", "weight_scale": 1.0},
        ]
    )
    args = _portfolio_args(
        factor_weight_scale_schedule=str(schedule_path),
        factor_weight_scale_combine_mode="multiply",
    )

    schedule = _factor_weight_scale_schedule(args, health)

    assert schedule is not None
    by_feature = schedule.set_index("feature")["weight_scale"]
    assert by_feature["alpha_a"] == pytest.approx(0.2)
    assert by_feature["alpha_b"] == pytest.approx(0.8)
    assert "external_weight_scale" in schedule.columns


def test_candidate_factor_script_filters_dataset_partitions(tmp_path: Path) -> None:
    for partition in ("2023_01", "2023_02", "2023_03", "2023_04"):
        (tmp_path / f"dataset_{partition}.parquet").touch()

    args = type(
        "Args",
        (),
        {
            "dataset_dir": str(tmp_path),
            "partition_start": "2023_02",
            "partition_end": "2023_03",
            "max_partitions": None,
        },
    )()

    assert [path.name for path in _dataset_paths(args)] == [
        "dataset_2023_02.parquet",
        "dataset_2023_03.parquet",
    ]


def test_candidate_factor_policy_set_builds_standard_comparison_specs() -> None:
    args = _portfolio_args(
        backtest_policy_set="comparison",
        top_n=50,
        policy_no_trade_weight_band=0.002,
    )

    specs = _backtest_policy_specs(args)

    assert [spec.name for spec in specs] == [
        "naive_top_n_every_bar",
        "top_k_drop_daily",
        "entry_exit_buffer_every_bar",
        "entry_exit_buffer_daily",
        "partial_rebalance_daily",
        "cost_aware_optimizer_every_bar",
        "cost_aware_optimizer_daily",
    ]
    top_k_drop = specs[1]
    assert top_k_drop.trade_policy == "rank_buffer_drop"
    assert top_k_drop.rebalance_every_n_bars == 48
    assert top_k_drop.policy_entry_rank == 50
    assert top_k_drop.policy_exit_rank == 50
    assert top_k_drop.policy_max_entries_per_rebalance == 10
    assert top_k_drop.policy_max_exits_per_rebalance == 10
    assert top_k_drop.policy_estimated_cost_bps == pytest.approx(13.0)
    buffered = specs[3]
    assert buffered.policy_exit_rank == 150
    assert buffered.policy_no_trade_weight_band == pytest.approx(0.002)
    assert specs[4].policy_partial_rebalance_rate == pytest.approx(0.5)
    assert specs[5].trade_policy == "cost_aware_optimizer"
    assert specs[5].rebalance_every_n_bars == 1
    assert specs[5].optimizer_weighting == "utility"
    assert specs[6].trade_policy == "cost_aware_optimizer"
    assert specs[6].rebalance_every_n_bars == 48
    assert specs[6].policy_partial_rebalance_rate == pytest.approx(0.5)


def test_candidate_factor_policy_set_filters_named_specs() -> None:
    args = _portfolio_args(
        backtest_policy_set="comparison",
        backtest_policies=["top_k_drop_daily", "partial_rebalance_daily"],
    )

    specs = _backtest_policy_specs(args)

    assert [spec.name for spec in specs] == [
        "top_k_drop_daily",
        "partial_rebalance_daily",
    ]


def test_candidate_factor_policy_set_rejects_unknown_filter() -> None:
    args = _portfolio_args(
        backtest_policy_set="comparison",
        backtest_policies=["missing_policy"],
    )

    with pytest.raises(ValueError, match="unknown backtest policies"):
        _backtest_policy_specs(args)


def test_candidate_factor_backtest_command_includes_policy_args(tmp_path: Path) -> None:
    args = _portfolio_args(
        max_bar_turnover_participation=0.05,
        allow_same_bar_capacity=True,
        policy_total_gross_turnover_budget=120.0,
        policy_turnover_budget_period="month",
        policy_turnover_budget_pacing=1.25,
    )
    spec = BacktestPolicySpec(
        name="entry_exit_buffer_daily",
        trade_policy="rank_buffer_drop",
        rebalance_every_n_bars=48,
        policy_entry_rank=50,
        policy_exit_rank=150,
        policy_max_entries_per_rebalance=10,
        policy_max_exits_per_rebalance=10,
        policy_no_trade_weight_band=0.002,
        policy_partial_rebalance_rate=0.5,
        policy_gross_exposure_scale=0.75,
        policy_gross_exposure_scale_path="gate.csv",
        policy_drawdown_brake_threshold=-0.07,
        policy_drawdown_brake_reduced_scale=0.4,
    )

    command = _backtest_command(args, "scores/*.parquet", tmp_path / "bt", spec)

    assert command[command.index("--trade-policy") + 1] == "rank_buffer_drop"
    assert command[command.index("--rebalance-every-n-bars") + 1] == "48"
    assert command[command.index("--policy-entry-rank") + 1] == "50"
    assert command[command.index("--policy-exit-rank") + 1] == "150"
    assert command[command.index("--policy-no-trade-weight-band") + 1] == "0.002"
    assert command[command.index("--policy-partial-rebalance-rate") + 1] == "0.5"
    assert command[command.index("--policy-gross-exposure-scale") + 1] == "0.75"
    assert command[command.index("--policy-gross-exposure-scale-path") + 1] == "gate.csv"
    assert command[command.index("--policy-drawdown-brake-threshold") + 1] == "-0.07"
    assert command[command.index("--policy-drawdown-brake-reduced-scale") + 1] == "0.4"
    assert command[command.index("--policy-total-gross-turnover-budget") + 1] == "120.0"
    assert command[command.index("--policy-turnover-budget-period") + 1] == "month"
    assert command[command.index("--policy-turnover-budget-pacing") + 1] == "1.25"
    assert command[command.index("--max-bar-turnover-participation") + 1] == "0.05"
    assert "--allow-same-bar-capacity" in command


def test_candidate_factor_backtest_jobs_use_nested_policy_paths(tmp_path: Path) -> None:
    args = _portfolio_args(
        output_dir=str(tmp_path),
        backtest_policy_set="comparison",
        backtest_policies=["top_k_drop_daily", "partial_rebalance_daily"],
        backtest_memory_estimate_gb=4.5,
    )
    scores_summary = {
        "methods": {
            "decorrelated": {"path": "scores/decorrelated/*.parquet"},
        }
    }

    jobs = _backtest_jobs(args, scores_summary=scores_summary)

    assert len(jobs) == 2
    assert jobs[0].summary_path == (
        tmp_path
        / "backtests"
        / "decorrelated"
        / "top_k_drop_daily"
        / "summary.json"
    )
    assert jobs[0].log_path == (
        tmp_path / "logs" / "backtest_decorrelated_top_k_drop_daily.log"
    )
    assert jobs[0].memory_estimate_gb == pytest.approx(4.5)


def test_candidate_factor_reused_scores_summary_keeps_existing_score_paths(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "source" / "scores" / "equal"
    score_dir.mkdir(parents=True)
    score_path = score_dir / "score_2024_01.parquet"
    score_path.touch()
    diagnostics_path = score_dir / "diagnostics" / "factor_contribution_2024_01.csv"
    diagnostics_path.parent.mkdir()
    diagnostics_path.touch()
    source_summary = {
        "candidate_features": ["alpha_a"],
        "methods": {
            "equal": {
                "path": str(score_dir / "*.parquet"),
                "partition_count": 1,
                "factor_contribution_diagnostics": [str(diagnostics_path)],
            },
            "decorrelated": {
                "path": str(tmp_path / "missing" / "*.parquet"),
            },
        },
    }
    args = _portfolio_args(methods=["equal"], reuse_scores_from=str(tmp_path / "source"))

    summary = _reused_scores_summary(args, source_summary)

    assert summary["candidate_features"] == ["alpha_a"]
    assert summary["methods"]["equal"]["path"] == str(score_dir / "*.parquet")  # type: ignore[index]


def test_candidate_factor_reused_scores_summary_requires_requested_method(
    tmp_path: Path,
) -> None:
    args = _portfolio_args(methods=["decorrelated"], reuse_scores_from=str(tmp_path))

    with pytest.raises(ValueError, match="missing method"):
        _reused_scores_summary(args, {"methods": {"equal": {"path": "scores/*.parquet"}}})


def test_candidate_factor_backtest_summary_rows_flatten_nested_results() -> None:
    rows = _backtest_summary_rows(
        {
            "decorrelated": {
                "entry_exit_buffer_daily": {
                    "params": {
                        "trade_policy": "rank_buffer_drop",
                        "rebalance_every_n_bars": 48,
                        "policy_entry_rank": 50,
                        "policy_exit_rank": 150,
                        "policy_max_entries_per_rebalance": 10,
                        "policy_max_exits_per_rebalance": 10,
                        "policy_no_trade_weight_band": 0.002,
                        "policy_partial_rebalance_rate": 1.0,
                        "policy_total_gross_turnover_budget": 120.0,
                        "policy_turnover_budget_period": "month",
                        "policy_turnover_budget_pacing": 1.2,
                        "policy_gross_exposure_scale": 0.75,
                        "policy_gross_exposure_scale_path": "gate.csv",
                    },
                    "metrics": {
                        "total_return": 0.079,
                        "max_drawdown": -0.06,
                        "gross_turnover": 47.4,
                        "trade_count": 1148,
                        "total_transaction_cost": 33462,
                        "final_equity": 1_079_000,
                    },
                    "signal_count": 1323,
                    "execution_row_count": 904224,
                    "execution_constraint_counts": {
                        "capacity_limited_event_count": 7,
                        "capacity_capped_event_count": 5,
                        "capacity_zero_event_count": 2,
                        "capacity_desired_shares": 10_000,
                        "capacity_executable_shares": 8_000,
                        "capacity_unfilled_shares": 2_000,
                        "capacity_desired_notional": 100_000.0,
                        "capacity_executable_notional": 80_000.0,
                        "capacity_unfilled_notional": 20_000.0,
                    },
                    "policy_diagnostics": {
                        "planned_gross_turnover": 54.4,
                        "average_target_gross_exposure": 0.74,
                        "average_dynamic_turnover_cap": 0.4,
                        "turnover_budget_period_count": 12,
                        "turnover_path_budget_remaining": 65.6,
                        "gross_exposure_scaled_count": 96,
                        "risk_reduction_count": 24,
                        "order_intent_count": 1232,
                        "entry_count": 590,
                        "exit_count": 564,
                        "hold_count": 655,
                        "no_trade_count": 0,
                    },
                }
            }
        }
    )

    assert rows == [
        {
            "method": "decorrelated",
            "policy": "entry_exit_buffer_daily",
            "trade_policy": "rank_buffer_drop",
            "rebalance_every_n_bars": 48,
            "policy_entry_rank": 50,
            "policy_exit_rank": 150,
            "policy_max_entries_per_rebalance": 10,
            "policy_max_exits_per_rebalance": 10,
            "policy_no_trade_weight_band": 0.002,
            "policy_partial_rebalance_rate": 1.0,
            "policy_total_gross_turnover_budget": 120.0,
            "policy_turnover_budget_period": "month",
            "policy_turnover_budget_pacing": 1.2,
            "policy_gross_exposure_scale": 0.75,
            "policy_gross_exposure_scale_path": "gate.csv",
            "optimizer_candidate_rank": None,
            "optimizer_score_to_edge_bps": None,
            "optimizer_min_net_edge_bps": None,
            "optimizer_risk_penalty_multiplier": None,
            "optimizer_weighting": None,
            "optimizer_max_name_weight": None,
            "optimizer_max_gross_exposure_increase_per_rebalance": None,
            "total_return": 0.079,
            "max_drawdown": -0.06,
            "gross_turnover": 47.4,
            "trade_count": 1148,
            "total_transaction_cost": 33462,
            "final_equity": 1_079_000,
            "signal_count": 1323,
            "execution_row_count": 904224,
            "capacity_limited_event_count": 7,
            "capacity_capped_event_count": 5,
            "capacity_zero_event_count": 2,
            "capacity_desired_shares": 10_000,
            "capacity_executable_shares": 8_000,
            "capacity_unfilled_shares": 2_000,
            "capacity_desired_notional": 100_000.0,
            "capacity_executable_notional": 80_000.0,
            "capacity_unfilled_notional": 20_000.0,
            "planned_gross_turnover": 54.4,
            "average_target_gross_exposure": 0.74,
            "average_dynamic_turnover_cap": 0.4,
            "turnover_budget_period_count": 12,
            "turnover_path_budget_remaining": 65.6,
            "gross_exposure_scaled_count": 96,
            "risk_reduction_count": 24,
            "order_intent_count": 1232,
            "entry_count": 590,
            "exit_count": 564,
            "hold_count": 655,
            "no_trade_count": 0,
        }
    ]


def test_candidate_factor_backtest_memory_budget_auto_detects_available() -> None:
    args = _portfolio_args(
        backtest_memory_budget_gb=0.0,
        backtest_memory_estimate_gb=5.0,
    )

    assert _effective_backtest_memory_budget_gb(args) >= 5.0


def test_candidate_factor_summary_params_record_backtest_policy_set() -> None:
    args = _portfolio_args(
        run_backtests=True,
        backtest_policy_set="comparison",
        backtest_policies=["partial_rebalance_daily"],
        policy_set_exit_rank=150,
        backtest_workers=2,
    )

    params = _summary_params(args)

    assert params["run_backtests"] is True
    assert params["backtest"]["backtest_policy_set"] == "comparison"  # type: ignore[index]
    assert params["backtest"]["backtest_policies"] == ["partial_rebalance_daily"]  # type: ignore[index]
    assert params["backtest"]["policy_set_exit_rank"] == 150  # type: ignore[index]
    assert params["backtest"]["backtest_workers"] == 2  # type: ignore[index]


def test_candidate_factor_default_label_lag_follows_horizon_suffix() -> None:
    assert _default_label_lag_windows("forward_return") == 48
    assert _default_label_lag_windows("forward_return_240b") == 240


def test_parse_factor_health_ensemble_lookbacks() -> None:
    assert _parse_factor_health_ensemble_lookbacks(None) == ()
    assert _parse_factor_health_ensemble_lookbacks("16, 20") == (16, 20)
    with pytest.raises(ValueError, match="duplicates"):
        _parse_factor_health_ensemble_lookbacks("16,16")


def _portfolio_args(**overrides: object) -> object:
    defaults = {
        "dataset_dir": "dataset",
        "label_column": "forward_return",
        "admission_report": "admission.json",
        "registry": "configs/factors/factor_registry.json",
        "enforce_registry": True,
        "registry_statuses": ["candidate", "promoted"],
        "factor_correlation": "correlation.csv",
        "methods": ["decorrelated"],
        "statuses": ["candidate"],
        "include_features": [],
        "max_partitions": None,
        "partition_start": None,
        "partition_end": None,
        "run_backtests": False,
        "output_dir": "runs",
        "factor_max_weight": None,
        "factor_max_contribution_share": None,
        "factor_health_mode": "off",
        "factor_health_lookback_windows": 20,
        "factor_health_ensemble_lookbacks": None,
        "factor_health_ensemble_combine_mode": "mean",
        "factor_health_min_periods": 5,
        "factor_health_label_lag_windows": 48,
        "factor_health_min_scale": 0.25,
        "factor_health_max_scale": 1.0,
        "factor_health_stress_lookback_windows": None,
        "factor_health_stress_min_periods": None,
        "factor_health_stress_min_scale": None,
        "factor_health_stress_max_scale": None,
        "factor_health_state_regime_mode": "off",
        "factor_health_state_regime_schedule": None,
        "factor_health_state_regime_feature": "intraday_overnight_gap_5m",
        "factor_health_state_regime_threshold": 0.999,
        "factor_health_rank_ic_floor": -0.05,
        "factor_health_rank_ic_ceiling": 0.05,
        "factor_health_spread_floor": -0.001,
        "factor_health_spread_ceiling": 0.001,
        "factor_weight_scale_schedule": None,
        "factor_weight_scale_combine_mode": "min",
        "forecast_calibration_mode": "off",
        "forecast_calibration_lookback_windows": 20,
        "forecast_calibration_min_periods": 5,
        "forecast_calibration_label_lag_windows": 48,
        "forecast_calibration_bucket_count": 5,
        "forecast_calibration_risk_multiplier": 1.0,
        "forecast_calibration_max_abs_edge_bps": None,
        "score_diagnostics_top_n": None,
        "reuse_scores_from": None,
        "backtest_policy_set": "single",
        "backtest_policies": None,
        "trade_policy": "naive_top_n",
        "rebalance_every_n_bars": 1,
        "hold_rank_buffer": None,
        "policy_entry_rank": None,
        "policy_exit_rank": None,
        "policy_max_entries_per_rebalance": None,
        "policy_max_exits_per_rebalance": None,
        "policy_min_hold_bars": 0,
        "policy_min_expected_edge_bps": None,
        "policy_estimated_cost_bps": None,
        "policy_no_trade_weight_band": 0.0,
        "policy_partial_rebalance_rate": 1.0,
        "policy_max_gross_turnover_per_rebalance": None,
        "policy_total_gross_turnover_budget": None,
        "policy_turnover_budget_period": "path",
        "policy_turnover_budget_pacing": 0.0,
        "policy_gross_exposure_scale": 1.0,
        "policy_gross_exposure_scale_path": None,
        "policy_drawdown_brake_threshold": None,
        "policy_drawdown_brake_reduced_scale": 0.5,
        "optimizer_candidate_rank": None,
        "optimizer_score_to_edge_bps": 100.0,
        "optimizer_min_net_edge_bps": 0.0,
        "optimizer_risk_penalty_multiplier": 1.0,
        "optimizer_weighting": "utility",
        "optimizer_max_name_weight": None,
        "optimizer_max_gross_exposure_increase_per_rebalance": None,
        "policy_set_drop_count": 10,
        "policy_set_exit_rank": None,
        "policy_set_rebalance_every_n_bars": 48,
        "policy_set_partial_rebalance_rate": 0.5,
        "backtest_workers": 1,
        "backtest_memory_budget_gb": 0.0,
        "backtest_memory_estimate_gb": 5.0,
        "resume_existing": False,
        "catalog_path": "catalog.duckdb",
        "start": "2023-01-03T09:35:00+08:00",
        "end": "2023-03-31T15:00:00+08:00",
        "top_n": 50,
        "initial_cash": 1_000_000.0,
        "commission_bps": 3.0,
        "slippage_bps": 1.0,
        "sell_stamp_tax_bps": 5.0,
        "min_commission": 5.0,
        "lot_size": 100,
        "min_trade_weight": 0.0005,
        "exclude_st": True,
        "limit_up_bps": 980.0,
        "limit_down_bps": 980.0,
        "max_bar_turnover_participation": None,
        "allow_same_bar_capacity": False,
        "data_access_mode": "fast_parquet",
        "streaming_chunk": "month",
        "streaming_chunk_padding_days": 10,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()
