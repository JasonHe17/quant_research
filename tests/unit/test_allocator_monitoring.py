from __future__ import annotations

from quant_research.factors import load_factor_registry
from quant_research.portfolio import (
    AllocatorMonitoringReport,
    allocator_monitoring_history_row,
    allocator_monitoring_history_status,
    append_allocator_monitoring_history,
    generate_allocator_monitoring_report,
    load_allocator_registry,
    render_allocator_monitoring_markdown,
)


def test_allocator_monitoring_report_reads_registered_evidence() -> None:
    registry = load_allocator_registry("configs/allocators/candidate_allocator_registry.json")
    factor_registry = load_factor_registry("configs/factors/factor_registry.json")

    report = generate_allocator_monitoring_report(
        registry,
        allocator_id="event_limit_diffusion_complementary_health_shrink_48b",
        factor_registry=factor_registry,
        project_root=".",
    )

    assert isinstance(report, AllocatorMonitoringReport)
    assert report.status == "warn"
    assert report.sections["registry"]["status"] == "pass"
    assert report.sections["validation"]["status"] == "pass"
    assert report.sections["capacity"]["status"] == "pass"
    assert report.sections["event_state_gate"]["status"] == "warn"
    assert report.sections["factor_health"]["status"] == "warn"
    assert report.sections["capacity"]["scenarios"][1]["scenario"] == "capacity_2pct"
    assert report.sections["capacity"]["scenarios"][1]["breaches"] == []
    assert report.sections["factor_health"]["latest_impaired_feature_count"] == 2


def test_allocator_monitoring_markdown_contains_key_sections() -> None:
    registry = load_allocator_registry("configs/allocators/candidate_allocator_registry.json")
    report = generate_allocator_monitoring_report(
        registry,
        allocator_id="event_limit_diffusion_complementary_health_shrink_48b",
        project_root=".",
    )

    markdown = render_allocator_monitoring_markdown(report)

    assert "# Allocator Monitoring Report" in markdown
    assert "## Capacity" in markdown
    assert "## Latest Factor Health" in markdown


def test_allocator_monitoring_history_tracks_sustained_warnings(
    tmp_path,
) -> None:
    registry = load_allocator_registry("configs/allocators/candidate_allocator_registry.json")
    report = generate_allocator_monitoring_report(
        registry,
        allocator_id="event_limit_diffusion_complementary_health_shrink_48b",
        project_root=".",
    )
    history_csv = tmp_path / "history.csv"

    row = allocator_monitoring_history_row(report)
    first = append_allocator_monitoring_history(report, history_csv=history_csv)
    second = append_allocator_monitoring_history(report, history_csv=history_csv)
    summary = allocator_monitoring_history_status(
        history_csv,
        sustained_warning_window=2,
    )

    assert row["status"] == "warn"
    assert row["capacity_status"] == "pass"
    assert first["row_count"] == 1
    assert second["row_count"] == 2
    assert summary["status"] == "warn"
    assert summary["sustained_warning"] is True
    assert summary["sustained_failure"] is False


def test_allocator_monitoring_history_can_replace_existing_run_id(
    tmp_path,
) -> None:
    registry = load_allocator_registry("configs/allocators/candidate_allocator_registry.json")
    report = generate_allocator_monitoring_report(
        registry,
        allocator_id="event_limit_diffusion_complementary_health_shrink_48b",
        project_root=".",
    )
    history_csv = tmp_path / "daily_history.csv"

    first = append_allocator_monitoring_history(
        report,
        history_csv=history_csv,
        extra_fields={"run_id": "2026-05-25", "mode": "paper"},
        replace_existing_on=("allocator_id", "run_id"),
    )
    second = append_allocator_monitoring_history(
        report,
        history_csv=history_csv,
        extra_fields={"run_id": "2026-05-25", "mode": "paper"},
        replace_existing_on=("allocator_id", "run_id"),
    )
    third = append_allocator_monitoring_history(
        report,
        history_csv=history_csv,
        extra_fields={"run_id": "2026-05-26", "mode": "paper"},
        replace_existing_on=("allocator_id", "run_id"),
    )

    assert first["row_count"] == 1
    assert first["replaced_count"] == 0
    assert second["row_count"] == 1
    assert second["replaced_count"] == 1
    assert third["row_count"] == 2
    assert third["replaced_count"] == 0
