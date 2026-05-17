from __future__ import annotations

import json
from pathlib import Path

from quant_research.factors import (
    FactorRegistry,
    FactorRegistryEntry,
    build_candidate_alpha_queue_review,
    render_candidate_alpha_queue_review_markdown,
    write_candidate_alpha_queue_review_outputs,
)


def test_candidate_alpha_queue_review_identifies_validation_gaps(tmp_path: Path) -> None:
    _write_admission(
        tmp_path / "single" / "factor_admission" / "factor_admission_report.json",
        ["single_feature"],
    )
    _write_admission(
        tmp_path / "shared" / "factor_admission" / "factor_admission_report.json",
        ["shared_feature", "other_feature"],
    )
    portfolio_path = tmp_path / "portfolio" / "validation_summary.json"
    portfolio_path.parent.mkdir(parents=True)
    portfolio_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(
            _entry(
                "single",
                "single_feature",
                evaluation={
                    "admission_report": "single/factor_admission/factor_admission_report.json",
                    "admission_status": "candidate",
                    "portfolio_validation_status": "",
                },
            ),
            _entry(
                "shared",
                "shared_feature",
                evaluation={
                    "admission_report": "shared/factor_admission/factor_admission_report.json",
                    "admission_status": "candidate",
                    "portfolio_validation_status": "",
                },
            ),
            _entry(
                "validated",
                "validated_feature",
                evaluation={
                    "admission_report": "single/factor_admission/factor_admission_report.json",
                    "admission_status": "candidate",
                    "portfolio_validation": "portfolio/validation_summary.json",
                    "portfolio_validation_status": "standalone_validated_drawdown_watch",
                },
            ),
        ),
    )
    opportunity_map = {
        "source": {"registry_name": "test", "registry_version": 1, "base_dir": str(tmp_path)},
        "records": [
            _opportunity_record("single", selected=0.01),
            _opportunity_record("shared", selected=0.02),
            _opportunity_record("validated", selected=0.03),
        ],
    }

    review = build_candidate_alpha_queue_review(
        registry,
        base_dir=tmp_path,
        opportunity_map=opportunity_map,
    )
    statuses = {row["factor_id"]: row["queue_status"] for row in review["queue"]}

    assert statuses == {
        "single": "needs_portfolio_validation",
        "shared": "needs_shared_admission_filtered_validation",
        "validated": "portfolio_validated_watch",
    }
    single = next(row for row in review["queue"] if row["factor_id"] == "single")
    assert "--dataset-dir single/alpha_dataset" in single["recommended_command"]
    assert "--admission-report single/factor_admission/factor_admission_report.json" in (
        single["recommended_command"]
    )
    shared = next(row for row in review["queue"] if row["factor_id"] == "shared")
    assert shared["requires_include_feature_filter"] is True
    assert "--include-features shared_feature" in shared["recommended_command"]


def test_candidate_alpha_queue_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    _write_admission(
        tmp_path / "single" / "factor_admission" / "factor_admission_report.json",
        ["single_feature"],
    )
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(
            _entry(
                "single",
                "single_feature",
                evaluation={
                    "admission_report": "single/factor_admission/factor_admission_report.json",
                    "admission_status": "candidate",
                },
            ),
        ),
    )
    opportunity_map = {
        "source": {"registry_name": "test", "registry_version": 1, "base_dir": str(tmp_path)},
        "records": [_opportunity_record("single", selected=0.01)],
    }
    review = build_candidate_alpha_queue_review(
        registry,
        base_dir=tmp_path,
        opportunity_map=opportunity_map,
    )

    artifacts = write_candidate_alpha_queue_review_outputs(
        review,
        output_dir=tmp_path / "out",
    )

    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["csv"]).exists()
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    assert "# Candidate Alpha Queue Review" in markdown
    assert "needs_portfolio_validation" in markdown
    assert render_candidate_alpha_queue_review_markdown(review).startswith(
        "# Candidate Alpha Queue Review"
    )


def _write_admission(path: Path, features: list[str]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"factors": [{"feature": feature} for feature in features]}),
        encoding="utf-8",
    )


def _opportunity_record(factor_id: str, *, selected: float) -> dict[str, object]:
    return {
        "factor_id": factor_id,
        "opportunity_class": "long_alpha_candidate",
        "selected_mean_label": selected,
        "selected_positive_years": 3,
        "cost_adjusted_spread": 0.001,
    }


def _entry(
    factor_id: str,
    feature_column: str,
    *,
    evaluation: dict[str, object],
) -> FactorRegistryEntry:
    return FactorRegistryEntry(
        factor_id=factor_id,
        display_name=factor_id,
        family="momentum",
        status="candidate",
        expected_direction="long",
        feature_columns=(feature_column,),
        required_inputs=("close_price",),
        frequency="5m",
        description="test factor",
        hypothesis="test hypothesis",
        implementation={"module": "tests", "builder": "build"},
        evaluation=evaluation,
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
