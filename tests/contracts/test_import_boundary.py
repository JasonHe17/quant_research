from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_IMPORT_PREFIXES = (
    "_bootstrap_raw_to_canonical",
    "baostock_sync",
    "jqdata_probe",
    "tests",
)


def test_research_package_does_not_import_forbidden_data_internals() -> None:
    offenders: list[str] = []
    for path in Path("quant_research").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        offenders.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _is_forbidden(module):
                    offenders.append(f"{path}: from {module} import ...")
    assert not offenders, offenders


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )
