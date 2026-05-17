from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research.factors import (
    FactorRegistry,
    FactorRegistryEntry,
    build_factor_opportunity_map,
    render_factor_opportunity_map_markdown,
    write_factor_opportunity_map_outputs,
)


def test_factor_opportunity_map_classifies_top_bucket_health(tmp_path: Path) -> None:
    _write_batch_artifacts(
        tmp_path,
        batch="batch",
        admission_rows=[
            _admission_row("alpha_feature", direction="long", spread=0.001),
            _admission_row("avoid_feature", direction="long", spread=0.001),
            _admission_row("risk_feature", direction="invert", spread=-0.001),
        ],
        by_timestamp_rows=[
            _by_timestamp("alpha_feature", "2024-01-01", top=0.01, bottom=-0.01),
            _by_timestamp("alpha_feature", "2025-01-01", top=0.02, bottom=-0.01),
            _by_timestamp("avoid_feature", "2024-01-01", top=-0.01, bottom=-0.03),
            _by_timestamp("avoid_feature", "2025-01-01", top=-0.02, bottom=-0.04),
            _by_timestamp("risk_feature", "2024-01-01", top=-0.04, bottom=-0.01),
            _by_timestamp("risk_feature", "2025-01-01", top=-0.03, bottom=-0.02),
        ],
    )
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(
            _entry("alpha", "alpha_feature", family="momentum"),
            _entry("avoid", "avoid_feature", family="momentum"),
            _entry(
                "risk",
                "risk_feature",
                family="risk",
                evaluation={
                    "admission_report": "batch/factor_admission/factor_admission_report.json",
                    "admission_status": "watchlist",
                    "admission_direction": "invert",
                    "portfolio_validation_status": "risk_gate_validated",
                },
            ),
        ),
    )

    opportunity_map = build_factor_opportunity_map(registry, base_dir=tmp_path)
    classes = {
        row["factor_id"]: row["opportunity_class"]
        for row in opportunity_map["records"]
    }

    assert classes == {
        "alpha": "long_alpha_candidate",
        "avoid": "bottom_avoidance_only",
        "risk": "risk_gate_only",
    }
    alpha = next(row for row in opportunity_map["records"] if row["factor_id"] == "alpha")
    assert alpha["selected_positive_years"] == 2
    assert alpha["selected_mean_label"] == 0.015


def test_factor_opportunity_map_blocks_rejected_top_bucket_as_dead_zone(
    tmp_path: Path,
) -> None:
    _write_batch_artifacts(
        tmp_path,
        batch="batch",
        admission_rows=[
            {
                **_admission_row("alpha_feature", direction="long", spread=0.001),
                "admission_status": "reject",
                "failed_checks": ["directional_ic_hit_rate"],
            }
        ],
        by_timestamp_rows=[
            _by_timestamp("alpha_feature", "2024-01-01", top=0.01, bottom=-0.01),
            _by_timestamp("alpha_feature", "2025-01-01", top=0.02, bottom=-0.01),
        ],
    )
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(
            _entry(
                "alpha",
                "alpha_feature",
                evaluation={
                    "admission_report": "batch/factor_admission/factor_admission_report.json",
                    "admission_status": "reject",
                    "admission_direction": "long",
                },
            ),
        ),
    )

    opportunity_map = build_factor_opportunity_map(registry, base_dir=tmp_path)

    assert opportunity_map["records"][0]["opportunity_class"] == "dead_zone"


def test_factor_opportunity_map_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    _write_batch_artifacts(
        tmp_path,
        batch="batch",
        admission_rows=[_admission_row("alpha_feature", direction="long", spread=0.001)],
        by_timestamp_rows=[
            _by_timestamp("alpha_feature", "2024-01-01", top=0.01, bottom=-0.01),
            _by_timestamp("alpha_feature", "2025-01-01", top=0.02, bottom=-0.01),
        ],
    )
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(_entry("alpha", "alpha_feature"),),
    )
    opportunity_map = build_factor_opportunity_map(registry, base_dir=tmp_path)

    artifacts = write_factor_opportunity_map_outputs(
        opportunity_map,
        output_dir=tmp_path / "out",
    )

    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["csv"]).exists()
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    assert "# Factor Opportunity Map" in markdown
    assert "long_alpha_candidate" in markdown
    assert render_factor_opportunity_map_markdown(opportunity_map).startswith(
        "# Factor Opportunity Map"
    )


def _write_batch_artifacts(
    root: Path,
    *,
    batch: str,
    admission_rows: list[dict[str, object]],
    by_timestamp_rows: list[dict[str, object]],
) -> None:
    admission_dir = root / batch / "factor_admission"
    evaluation_dir = root / batch / "factor_evaluation"
    admission_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    (admission_dir / "factor_admission_report.json").write_text(
        json.dumps({"factors": admission_rows}),
        encoding="utf-8",
    )
    by_timestamp_path = evaluation_dir / "single_factor_by_timestamp.csv"
    pd.DataFrame(by_timestamp_rows).to_csv(by_timestamp_path, index=False)
    (evaluation_dir / "summary.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "by_timestamp": str(by_timestamp_path.relative_to(root)),
                }
            }
        ),
        encoding="utf-8",
    )


def _entry(
    factor_id: str,
    feature_column: str,
    *,
    family: str = "momentum",
    evaluation: dict[str, object] | None = None,
) -> FactorRegistryEntry:
    return FactorRegistryEntry(
        factor_id=factor_id,
        display_name=factor_id,
        family=family,
        status="candidate",
        expected_direction="long",
        feature_columns=(feature_column,),
        required_inputs=("close_price",),
        frequency="5m",
        description="test factor",
        hypothesis="test hypothesis",
        implementation={"module": "tests", "builder": "build"},
        evaluation=evaluation
        or {
            "admission_report": "batch/factor_admission/factor_admission_report.json",
            "admission_status": "candidate",
            "admission_direction": "long",
        },
        lookback_bars=48,
        label_lag_bars=48,
        point_in_time_safe=True,
        live_available=True,
        a_share_constraints={
            "long_only": True,
            "price_limit_aware": True,
            "st_aware": True,
            "t_plus_one_safe": True,
        },
    )


def _admission_row(
    feature: str,
    *,
    direction: str,
    spread: float,
) -> dict[str, object]:
    return {
        "feature": feature,
        "admission_status": "candidate",
        "direction": direction,
        "spearman_rank_ic_mean": 0.01 if direction == "long" else -0.01,
        "spearman_rank_ic_t_stat": 3.0,
        "directional_ic_hit_rate": 0.55,
        "stable_year_count": 2,
        "cost_adjusted_top_minus_bottom_label": spread,
        "top_n_turnover": 0.2,
        "failed_checks": [],
    }


def _by_timestamp(
    feature: str,
    timestamp: str,
    *,
    top: float,
    bottom: float,
) -> dict[str, object]:
    return {
        "feature": feature,
        "timestamp": timestamp,
        "top_n_mean_label": top,
        "bottom_n_mean_label": bottom,
        "top_minus_bottom_label": top - bottom,
    }
