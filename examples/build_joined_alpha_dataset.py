"""Join selected feature columns from auxiliary alpha datasets into a base dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True, slots=True)
class JoinSource:
    """One auxiliary alpha dataset and the feature columns imported from it."""

    dataset_dir: Path
    feature_columns: tuple[str, ...]
    prefix: str | None = None

    @property
    def name(self) -> str:
        return self.prefix or self.dataset_dir.name


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = tuple(_parse_source(raw) for raw in args.source)
    rows = build_joined_alpha_dataset(
        base_dataset_dir=Path(args.base_dataset_dir),
        output_dir=output_dir,
        sources=sources,
        overwrite=args.overwrite,
    )
    summary = {
        "params": {
            "base_dataset_dir": args.base_dataset_dir,
            "sources": [
                {
                    "dataset_dir": str(source.dataset_dir),
                    "feature_columns": list(source.feature_columns),
                    "prefix": source.prefix,
                }
                for source in sources
            ],
            "overwrite": args.overwrite,
        },
        "partitions": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_joined_alpha_dataset(
    *,
    base_dataset_dir: Path,
    output_dir: Path,
    sources: tuple[JoinSource, ...],
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write joined monthly parquet partitions and return QA rows."""

    base_paths = sorted(base_dataset_dir.glob("dataset_*.parquet"))
    if not base_paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {base_dataset_dir}")
    if not sources:
        raise ValueError("at least one --source is required")
    rows: list[dict[str, Any]] = []
    for base_path in base_paths:
        partition = base_path.stem.removeprefix("dataset_")
        output_path = output_dir / base_path.name
        manifest_path = output_dir / f"dataset_{partition}.manifest.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output partition exists: {output_path}")
        base = pd.read_parquet(base_path)
        base_rows = len(base)
        joined = base
        row: dict[str, Any] = {
            "partition": partition,
            "base_path": str(base_path),
            "dataset_path": str(output_path),
            "manifest_path": str(manifest_path),
            "base_row_count": int(base_rows),
            "output_row_count": int(base_rows),
        }
        for source in sources:
            source_path = source.dataset_dir / base_path.name
            if not source_path.exists():
                raise FileNotFoundError(f"missing source partition: {source_path}")
            source_columns = ["timestamp", "instrument_id", *source.feature_columns]
            source_frame = pd.read_parquet(source_path, columns=source_columns)
            duplicate_count = int(
                source_frame.duplicated(["timestamp", "instrument_id"]).sum()
            )
            if duplicate_count:
                raise ValueError(
                    f"source partition has duplicate keys: {source_path}, "
                    f"duplicates={duplicate_count}"
                )
            rename_map = _rename_map(source)
            source_frame = source_frame.rename(columns=rename_map)
            output_features = tuple(rename_map.get(column, column) for column in source.feature_columns)
            overlap = [column for column in output_features if column in joined.columns]
            if overlap:
                raise ValueError(
                    f"joined output would overwrite existing columns for "
                    f"{source_path}: {overlap}"
                )
            before_columns = set(joined.columns)
            joined = joined.merge(
                source_frame,
                on=["timestamp", "instrument_id"],
                how="left",
                validate="many_to_one",
            )
            if len(joined) != base_rows:
                raise ValueError(
                    f"row count changed for {base_path}: before={base_rows}, "
                    f"after={len(joined)}"
                )
            added_columns = [column for column in joined.columns if column not in before_columns]
            source_key = _safe_key(source.name)
            row[f"{source_key}_source_path"] = str(source_path)
            row[f"{source_key}_row_count"] = int(len(source_frame))
            for column in added_columns:
                non_null = int(joined[column].notna().sum())
                row[f"{column}_non_null_count"] = non_null
                row[f"{column}_coverage"] = float(non_null / base_rows) if base_rows else None
        joined.to_parquet(output_path, index=False)
        _write_manifest(
            manifest_path,
            partition=partition,
            output_path=output_path,
            base_dataset_dir=base_dataset_dir,
            sources=sources,
            row=row,
            columns=tuple(joined.columns),
        )
        rows.append(row)
    return rows


def _write_manifest(
    path: Path,
    *,
    partition: str,
    output_path: Path,
    base_dataset_dir: Path,
    sources: tuple[JoinSource, ...],
    row: dict[str, Any],
    columns: tuple[str, ...],
) -> None:
    payload = {
        "name": "joined_alpha_dataset",
        "partition": partition,
        "dataset_path": str(output_path),
        "row_count": row["output_row_count"],
        "columns": list(columns),
        "parameters": {
            "base_dataset_dir": str(base_dataset_dir),
            "sources": [
                {
                    "dataset_dir": str(source.dataset_dir),
                    "feature_columns": list(source.feature_columns),
                    "prefix": source.prefix,
                }
                for source in sources
            ],
        },
        "qa": row,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_source(raw: str) -> JoinSource:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            "--source must be dataset_dir:feature[,feature...] or "
            "dataset_dir:feature[,feature...]:prefix"
        )
    dataset_dir = Path(parts[0])
    feature_columns = tuple(column for column in parts[1].split(",") if column)
    if not feature_columns:
        raise argparse.ArgumentTypeError("--source feature list cannot be empty")
    prefix = parts[2] or None if len(parts) == 3 else None
    return JoinSource(dataset_dir=dataset_dir, feature_columns=feature_columns, prefix=prefix)


def _rename_map(source: JoinSource) -> dict[str, str]:
    if not source.prefix:
        return {}
    return {
        column: f"{source.prefix}{column}"
        for column in source.feature_columns
    }


def _safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "auxiliary source as dataset_dir:feature[,feature...] or "
            "dataset_dir:feature[,feature...]:prefix; repeatable"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
