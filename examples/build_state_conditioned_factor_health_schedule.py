"""Build a factor-weight schedule by selecting health memory by regime state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    args = _parse_args()
    normal = _load_schedule(Path(args.normal_schedule), label="normal")
    stress = _load_schedule(Path(args.stress_schedule), label="stress")
    regime = _load_regime_weight(
        Path(args.regime_schedule),
        feature=args.regime_feature,
        mode=args.regime_mode,
        threshold=args.regime_threshold,
    )
    output = build_state_conditioned_schedule(
        normal,
        stress,
        regime,
        mode=args.regime_mode,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "factor_health_schedule.csv"
    output.to_csv(schedule_path, index=False)
    summary = {
        "normal_schedule": args.normal_schedule,
        "stress_schedule": args.stress_schedule,
        "regime_schedule": args.regime_schedule,
        "regime_feature": args.regime_feature,
        "regime_mode": args.regime_mode,
        "regime_threshold": args.regime_threshold,
        "schedule_path": str(schedule_path),
        "row_count": int(len(output)),
        "feature_count": int(output["feature"].nunique()) if not output.empty else 0,
        "regime_weight_summary": _series_summary(output["regime_weight"]),
        "weight_scale_summary": _series_summary(output["weight_scale"]),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_state_conditioned_schedule(
    normal: pd.DataFrame,
    stress: pd.DataFrame,
    regime: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    """Return a schedule whose weight scale is selected/blended by regime weight."""

    keys = ["timestamp", "feature"]
    joined = normal.merge(stress, on=keys, how="outer", sort=False)
    joined = joined.merge(regime, on="timestamp", how="left", sort=False)
    joined["normal_weight_scale"] = joined["normal_weight_scale"].fillna(1.0)
    joined["stress_weight_scale"] = joined["stress_weight_scale"].fillna(1.0)
    joined["regime_weight"] = joined["regime_weight"].fillna(0.0).clip(0.0, 1.0)
    joined["weight_scale"] = (
        joined["normal_weight_scale"] * (1.0 - joined["regime_weight"])
        + joined["stress_weight_scale"] * joined["regime_weight"]
    ).clip(0.0, 1.0)
    joined["shrink_reason"] = "state_conditioned_health_memory"
    active = joined["regime_weight"] > 0.0
    joined.loc[active, "shrink_reason"] = (
        "state_conditioned_health_memory,stress_health_memory"
    )
    joined["state_conditioned_mode"] = mode
    columns = [
        "timestamp",
        "feature",
        "weight_scale",
        "shrink_reason",
        "normal_weight_scale",
        "stress_weight_scale",
        "regime_weight",
        "regime_selector_scale",
        "state_conditioned_mode",
    ]
    return joined.loc[:, columns].sort_values(keys).reset_index(drop=True)


def _load_schedule(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} schedule not found: {path}")
    frame = pd.read_csv(path, usecols=["timestamp", "feature", "weight_scale"])
    missing = {"timestamp", "feature", "weight_scale"} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} schedule missing columns: {sorted(missing)}")
    output = frame.loc[:, ["timestamp", "feature", "weight_scale"]].copy()
    output["timestamp"] = output["timestamp"].astype(str)
    output["feature"] = output["feature"].astype(str)
    output["weight_scale"] = pd.to_numeric(output["weight_scale"], errors="coerce")
    if output["weight_scale"].isna().any():
        raise ValueError(f"{label} schedule contains invalid weight_scale")
    if not output["weight_scale"].between(0.0, 1.0).all():
        raise ValueError(f"{label} schedule weight_scale values must be in [0, 1]")
    duplicates = output.duplicated(["timestamp", "feature"], keep=False)
    if bool(duplicates.any()):
        raise ValueError(f"{label} schedule has duplicate timestamp/feature rows")
    return output.rename(columns={"weight_scale": f"{label}_weight_scale"})


def _load_regime_weight(
    path: Path,
    *,
    feature: str,
    mode: str,
    threshold: float,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"regime schedule not found: {path}")
    if mode not in {"select", "blend"}:
        raise ValueError("mode must be select or blend")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    frame = pd.read_csv(path)
    missing = {"timestamp", "feature", "weight_scale"} - set(frame.columns)
    if missing:
        raise ValueError(f"regime schedule missing columns: {sorted(missing)}")
    selected = frame.loc[frame["feature"].astype(str) == feature].copy()
    if selected.empty:
        raise ValueError(f"regime schedule has no rows for feature: {feature}")
    selected["regime_selector_scale"] = pd.to_numeric(
        selected["weight_scale"],
        errors="coerce",
    )
    if selected["regime_selector_scale"].isna().any():
        raise ValueError("regime schedule contains invalid weight_scale")
    if mode == "select":
        selected["regime_weight"] = (
            selected["regime_selector_scale"] < threshold
        ).astype(float)
    else:
        selected["regime_weight"] = (1.0 - selected["regime_selector_scale"]).clip(
            0.0,
            1.0,
        )
    return selected.loc[:, ["timestamp", "regime_selector_scale", "regime_weight"]]


def _series_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-schedule", required=True)
    parser.add_argument("--stress-schedule", required=True)
    parser.add_argument("--regime-schedule", required=True)
    parser.add_argument(
        "--regime-feature",
        default="intraday_overnight_gap_5m",
        help="feature row in the regime schedule used as the observable state proxy",
    )
    parser.add_argument("--regime-mode", choices=("select", "blend"), default="select")
    parser.add_argument("--regime-threshold", type=float, default=0.999)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
