from __future__ import annotations

import json
from pathlib import Path

from quant_research.factors import (
    FactorRegistry,
    FactorRegistryEntry,
    build_factor_failure_atlas,
    render_factor_failure_atlas_markdown,
    write_factor_failure_atlas_outputs,
)


def test_factor_failure_atlas_joins_registry_admission_and_portfolio(
    tmp_path: Path,
) -> None:
    admission_path = tmp_path / "admission.json"
    portfolio_path = tmp_path / "portfolio.json"
    admission_path.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "feature": "alpha_bad_feature",
                        "admission_status": "reject",
                        "direction": "invert",
                        "spearman_rank_ic_mean": -0.01,
                        "spearman_rank_ic_t_stat": -3.0,
                        "directional_ic_hit_rate": 0.49,
                        "stable_year_count": 1,
                        "cost_adjusted_top_minus_bottom_label": 0.001,
                        "top_n_turnover": 0.2,
                        "failed_checks": [
                            "directional_ic_hit_rate",
                            "stable_year_count",
                        ],
                        "yearly_spearman_rank_ic_mean": {
                            "2024": 0.01,
                            "2025": -0.02,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    portfolio_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "validation": {"overall_status": "fail"},
                "results": [
                    {
                        "scenario": "full_base",
                        "method": "decorrelated",
                        "total_return": -0.1,
                        "max_drawdown": -0.2,
                        "gross_turnover": 12.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(
            _entry(
                "alpha_bad",
                "alpha_bad_feature",
                status="reject",
                evaluation={
                    "admission_report": str(admission_path.relative_to(tmp_path)),
                    "admission_status": "reject",
                    "portfolio_validation": str(portfolio_path.relative_to(tmp_path)),
                    "portfolio_validation_status": "portfolio_negative",
                },
                research_memory={
                    "decision_reason": "weak_hit_rate",
                    "negative_findings": "Bad hit rate.",
                    "retry_conditions": "Retry only with a new state variable.",
                    "similar_to": ["alpha_parent"],
                    "evidence_artifacts": ["admission.json"],
                },
            ),
        ),
    )

    atlas = build_factor_failure_atlas(registry, base_dir=tmp_path)

    assert atlas["summary"]["factor_count"] == 1
    assert atlas["summary"]["admission_evidence_count"] == 1
    assert atlas["summary"]["portfolio_evidence_count"] == 1
    row = atlas["records"][0]
    assert row["factor_id"] == "alpha_bad"
    assert row["decision_reason"] == "weak_hit_rate"
    assert row["admission_failed_checks"] == [
        "directional_ic_hit_rate",
        "stable_year_count",
    ]
    assert row["portfolio_artifacts"][0]["overall_status"] == "fail"
    assert "weak_hit_rate" in row["failure_modes"]
    assert "portfolio_negative" in row["failure_modes"]


def test_factor_failure_atlas_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(_entry("alpha", "alpha_feature"),),
    )
    atlas = build_factor_failure_atlas(registry, base_dir=tmp_path)

    artifacts = write_factor_failure_atlas_outputs(atlas, output_dir=tmp_path / "out")

    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["csv"]).exists()
    assert Path(artifacts["modes_csv"]).exists()
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    assert "# Factor Failure Atlas" in markdown
    assert "Recommendations" in markdown
    assert render_factor_failure_atlas_markdown(atlas).startswith("# Factor Failure Atlas")


def _entry(
    factor_id: str,
    feature_column: str,
    *,
    status: str = "candidate",
    evaluation: dict[str, object] | None = None,
    research_memory: dict[str, object] | None = None,
) -> FactorRegistryEntry:
    return FactorRegistryEntry(
        factor_id=factor_id,
        display_name=factor_id,
        family="momentum",
        status=status,
        expected_direction="long",
        feature_columns=(feature_column,),
        required_inputs=("close_price",),
        frequency="5m",
        description="test factor",
        hypothesis="test hypothesis",
        implementation={"module": "tests", "builder": "build"},
        evaluation=evaluation or {},
        research_memory=research_memory or {},
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
