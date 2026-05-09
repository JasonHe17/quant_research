"""Local smoke check for the sibling quant_dataset repository.

This script intentionally reads only the dataset inventory exposed by
quantdb.sdk. It verifies the research/data boundary without pulling large
market data into memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_research import DataPortal


def main() -> int:
    args = _parse_args()
    workspace_root = args.workspace_root.resolve()
    quant_dataset_root = args.quant_dataset_root.resolve()
    canonical_root = quant_dataset_root / "canonical_store"
    catalog_path = canonical_root / "catalog" / "quant_research.duckdb"

    missing_paths = [
        path
        for path in (quant_dataset_root, canonical_root, catalog_path)
        if not path.exists()
    ]
    if missing_paths:
        print("real-data smoke check skipped; missing local paths:")
        for path in missing_paths:
            print(f"- {path}")
        return 2

    portal = DataPortal(
        canonical_root=canonical_root,
        catalog_path=catalog_path,
        snapshot=args.snapshot,
        quant_dataset_root=quant_dataset_root,
    )
    datasets = portal.list_available_datasets(domain=args.domain)

    print(f"workspace_root={workspace_root}")
    print(f"quant_dataset_root={quant_dataset_root}")
    print(f"snapshot={args.snapshot}")
    print(f"dataset_count={len(datasets)}")
    if datasets.empty:
        return 1

    columns = [
        column
        for column in ("dataset_name", "domain", "markets", "asset_types")
        if column in datasets.columns
    ]
    print(datasets[columns].to_string(index=False))
    return 0


def _parse_args() -> argparse.Namespace:
    default_workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Verify quant_research can see local quant_dataset inventory."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=default_workspace,
        help="Path containing sibling quant_dataset and quant_research repos.",
    )
    parser.add_argument(
        "--quant-dataset-root",
        type=Path,
        default=default_workspace / "quant_dataset",
        help="Path to the sibling quant_dataset repository.",
    )
    parser.add_argument(
        "--snapshot",
        default="2026-05-09",
        help="Research data snapshot label used for the smoke check.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional quantdb dataset domain filter, such as market or reference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
