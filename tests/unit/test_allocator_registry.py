from __future__ import annotations

import json
from pathlib import Path

from quant_research.factors import (
    FactorRegistry,
    FactorRegistryEntry,
    load_factor_registry,
)
from quant_research.portfolio import (
    AllocatorRegistry,
    AllocatorRegistryEntry,
    load_allocator_registry,
    validate_allocator_registry,
)


def test_candidate_allocator_registry_is_governance_clean() -> None:
    allocator_registry = load_allocator_registry(
        "configs/allocators/candidate_allocator_registry.json"
    )
    factor_registry = load_factor_registry("configs/factors/factor_registry.json")

    report = validate_allocator_registry(
        allocator_registry,
        factor_registry=factor_registry,
        project_root=".",
    )

    assert report.status == "pass"
    assert report.summary["allocator_count"] >= 1
    assert report.summary["status_counts"]["candidate"] >= 1


def test_allocator_registry_rejects_duplicate_allocator_ids() -> None:
    entry = _allocator("alpha_allocator")
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(entry, entry),
    )

    report = validate_allocator_registry(registry)

    assert report.status == "fail"
    assert any(issue.code == "duplicate_allocator_id" for issue in report.issues)


def test_allocator_registry_requires_normalized_feature_weights() -> None:
    entry = _allocator(
        "alpha_allocator",
        features=(
            _feature("feature_a", weight=0.7),
            _feature("feature_b", weight=0.7),
        ),
    )
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(entry,),
    )

    report = validate_allocator_registry(registry)

    assert report.status == "fail"
    assert any(
        issue.code == "feature_weights_not_normalized" for issue in report.issues
    )


def test_allocator_registry_checks_factor_registry_feature_membership() -> None:
    allocator_registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(_allocator("alpha_allocator"),),
    )
    factor_registry = FactorRegistry(
        registry_name="factor_test",
        version=1,
        entries=(_factor("feature_b"),),
    )

    report = validate_allocator_registry(
        allocator_registry,
        factor_registry=factor_registry,
    )

    assert report.status == "fail"
    assert any(
        issue.code == "feature_missing_from_factor_registry" for issue in report.issues
    )


def test_allocator_registry_loader_reads_json() -> None:
    registry = load_allocator_registry(
        Path("configs/allocators/candidate_allocator_registry.json")
    )

    allocator = registry.get("event_limit_diffusion_complementary_health_shrink_48b")

    assert allocator.status == "candidate"
    assert allocator.validation["status"] == "pass"
    assert allocator.governance["capacity_monitoring"]["mode"] == "monitor_only"


def test_allocator_registry_warns_when_capacity_checked_without_monitoring() -> None:
    entry = _allocator("alpha_allocator")
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(entry,),
    )

    report = validate_allocator_registry(registry)

    assert report.status == "warn"
    assert any(issue.code == "missing_capacity_monitoring" for issue in report.issues)


def test_allocator_registry_warns_on_capacity_monitor_threshold_breach(
    tmp_path: Path,
) -> None:
    capacity_summary = tmp_path / "capacity_summary.json"
    capacity_summary.write_text(
        json.dumps(
            [
                {
                    "scenario": "capacity_2pct",
                    "total_return": -0.01,
                    "max_drawdown": -0.40,
                    "capacity_unfilled_vs_traded": 0.08,
                    "capacity_unfilled_vs_desired_capacity_events": 0.60,
                }
            ]
        ),
        encoding="utf-8",
    )
    entry = _allocator(
        "alpha_allocator",
        capacity_monitoring={
            "mode": "monitor_only",
            "diagnostic_summary": str(capacity_summary),
            "stress_scenarios": ["capacity_2pct"],
            "warning_thresholds": {
                "min_total_return": 0.0,
                "max_abs_drawdown": 0.35,
                "max_unfilled_vs_traded_notional": 0.05,
                "max_unfilled_vs_desired_capacity_events": 0.55,
            },
        },
    )
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(entry,),
    )

    report = validate_allocator_registry(registry)

    assert report.status == "warn"
    assert (
        sum(
            issue.code == "capacity_monitor_threshold_breach" for issue in report.issues
        )
        == 4
    )


def test_allocator_registry_accepts_score_overlay_allocator(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    satellite = tmp_path / "satellite"
    penalty = tmp_path / "penalty"
    primary.mkdir()
    satellite.mkdir()
    penalty.mkdir()
    (primary / "score_2023_01.parquet").touch()
    (satellite / "score_2023_01.parquet").touch()
    (penalty / "score_2023_01.parquet").touch()
    schedule = tmp_path / "condition.csv"
    schedule.write_text("timestamp,risk_state\n2023-01-03T15:00:00+08:00,reduced\n")
    entry = _score_overlay_allocator(
        "overlay_allocator",
        primary_score_dir=str(primary),
        satellite_score_dir=str(satellite),
        condition_schedule=str(schedule),
        postprocessors=[
            {
                "type": "optimizer_risk_penalty_join",
                "penalty_dir": str(penalty),
                "penalty_column": "optimizer_risk_penalty_bps",
                "fill_value": 0.0,
            }
        ],
    )
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(entry,),
    )

    report = validate_allocator_registry(registry, project_root=".")

    assert report.status == "pass"
    assert report.allocators[0]["score_construction"] == "score_overlay"


def test_allocator_registry_rejects_invalid_score_overlay_weight(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    satellite = tmp_path / "satellite"
    primary.mkdir()
    satellite.mkdir()
    (primary / "score_2023_01.parquet").touch()
    (satellite / "score_2023_01.parquet").touch()
    entry = _score_overlay_allocator(
        "overlay_allocator",
        primary_score_dir=str(primary),
        satellite_score_dir=str(satellite),
        overlay_weights=[1.2],
    )
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(entry,),
    )

    report = validate_allocator_registry(registry, project_root=".")

    assert report.status == "fail"
    assert any(issue.code == "invalid_overlay_weight" for issue in report.issues)


def test_allocator_registry_accepts_target_weight_cap_postprocessor(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    satellite = tmp_path / "satellite"
    cap = tmp_path / "cap"
    primary.mkdir()
    satellite.mkdir()
    cap.mkdir()
    (primary / "score_2023_01.parquet").touch()
    (satellite / "score_2023_01.parquet").touch()
    (cap / "score_2023_01.parquet").touch()
    entry = _score_overlay_allocator(
        "overlay_allocator",
        primary_score_dir=str(primary),
        satellite_score_dir=str(satellite),
        postprocessors=[
            {
                "type": "target_weight_cap_join",
                "cap_dir": str(cap),
                "cap_column": "max_target_weight",
                "fill_value": 1.0,
            }
        ],
    )
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(entry,),
    )

    report = validate_allocator_registry(registry, project_root=".")

    assert report.status == "pass"


def test_allocator_registry_rejects_invalid_optimizer_target_cap_mode() -> None:
    entry = _allocator("alpha_allocator")
    payload = entry.to_dict()
    payload["execution_policy"]["optimizer_target_cap_mode"] = "bad"
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(AllocatorRegistryEntry.from_dict(payload),),
    )

    report = validate_allocator_registry(registry)

    assert report.status == "fail"
    assert any(
        issue.code == "invalid_optimizer_target_cap_mode" for issue in report.issues
    )


def test_allocator_registry_accepts_replace_optimizer_target_cap_mode() -> None:
    entry = _allocator("alpha_allocator")
    payload = entry.to_dict()
    payload["execution_policy"]["optimizer_target_cap_mode"] = "replace"
    registry = AllocatorRegistry(
        registry_name="test_allocators",
        version=1,
        allocators=(AllocatorRegistryEntry.from_dict(payload),),
    )

    report = validate_allocator_registry(registry)

    assert report.status != "fail"
    assert not any(
        issue.code == "invalid_optimizer_target_cap_mode" for issue in report.issues
    )


def _allocator(
    allocator_id: str,
    *,
    features: tuple[dict[str, object], ...] | None = None,
    capacity_monitoring: dict[str, object] | None = None,
) -> AllocatorRegistryEntry:
    governance: dict[str, object] = {"decision": "candidate_allocator"}
    if capacity_monitoring is not None:
        governance["capacity_monitoring"] = capacity_monitoring
    return AllocatorRegistryEntry(
        allocator_id=allocator_id,
        display_name=allocator_id,
        status="candidate",
        description="A test allocator.",
        hypothesis="A test hypothesis.",
        score={
            "features": list(features or (_feature("feature_a", weight=1.0),)),
        },
        risk_controls={
            "event_state_gate": {
                "blocked_states": ["stress"],
                "schedule_path": "pyproject.toml",
            },
            "factor_health": {"mode": "lagged_shrink"},
        },
        execution_policy={
            "top_n": 50,
            "entry_rank": 50,
            "exit_rank": 150,
            "max_entries_per_rebalance": 10,
            "max_exits_per_rebalance": 10,
            "rebalance_every_n_bars": 48,
            "partial_rebalance_rate": 0.5,
            "no_trade_weight_band": 0.002,
        },
        cost_model={
            "commission_bps": 3.0,
            "slippage_bps": 5.0,
            "sell_stamp_tax_bps": 5.0,
            "min_commission": 5.0,
        },
        validation={
            "status": "pass",
            "standard_validation": "pyproject.toml",
            "robust_validation": "pyproject.toml",
            "capacity_2pct_summary": "pyproject.toml",
        },
        governance=governance,
        data={},
        references=("pyproject.toml",),
        tags=("capacity_checked",),
    )


def _feature(
    feature: str,
    *,
    weight: float,
    direction: str = "long",
) -> dict[str, object]:
    return {"feature": feature, "direction": direction, "weight": weight}


def _factor(feature: str, *, status: str = "candidate") -> FactorRegistryEntry:
    return FactorRegistryEntry(
        factor_id=feature,
        display_name=feature,
        family="momentum",
        status=status,
        expected_direction="long",
        feature_columns=(feature,),
        required_inputs=("close_price",),
        frequency="5m",
        description="Test factor.",
        hypothesis="Test hypothesis.",
        a_share_constraints={
            "long_only": True,
            "price_limit_aware": True,
            "st_aware": True,
            "t_plus_one_safe": True,
        },
        point_in_time_safe=True,
        live_available=True,
    )


def _score_overlay_allocator(
    allocator_id: str,
    *,
    primary_score_dir: str,
    satellite_score_dir: str,
    condition_schedule: str | None = None,
    overlay_weights: list[float] | None = None,
    postprocessors: list[dict[str, object]] | None = None,
) -> AllocatorRegistryEntry:
    score: dict[str, object] = {
        "construction": "score_overlay",
        "primary_score_dir": primary_score_dir,
        "satellite_score_dir": satellite_score_dir,
        "method_prefix": "overlay_test",
        "overlay_weights": overlay_weights or [0.1],
        "overlay_mode": "blend",
        "rank_normalize": True,
    }
    if postprocessors is not None:
        score["postprocessors"] = postprocessors
    if condition_schedule is not None:
        score["condition"] = {
            "schedule_path": condition_schedule,
            "column": "risk_state",
            "values": ["reduced"],
        }
    return AllocatorRegistryEntry(
        allocator_id=allocator_id,
        display_name=allocator_id,
        status="candidate",
        description="A score-overlay test allocator.",
        hypothesis="A test overlay hypothesis.",
        score=score,
        risk_controls={},
        execution_policy={
            "top_n": 50,
            "entry_rank": 50,
            "exit_rank": 150,
            "max_entries_per_rebalance": 10,
            "max_exits_per_rebalance": 10,
            "rebalance_every_n_bars": 48,
            "partial_rebalance_rate": 0.5,
            "no_trade_weight_band": 0.002,
        },
        cost_model={
            "commission_bps": 3.0,
            "slippage_bps": 1.0,
            "sell_stamp_tax_bps": 5.0,
            "min_commission": 5.0,
        },
        validation={
            "status": "pass",
            "standard_validation": "pyproject.toml",
            "robust_validation": "pyproject.toml",
        },
        governance={"decision": "candidate_allocator"},
        data={},
        references=("pyproject.toml",),
    )
