"""Experiment provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from importlib import metadata
from typing import Iterable

from quant_research.datasets.manifests import file_sha256
from quant_research.data.cache import catalog_reference_for_path


DEFAULT_PROVENANCE_PACKAGES = (
    "quant-research",
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "lightgbm",
    "scipy",
)


def collect_experiment_provenance(
    *,
    data_snapshot: str,
    catalog_path: str | Path | None = None,
    command_line: tuple[str, ...] | None = None,
    packages: Iterable[str] = DEFAULT_PROVENANCE_PACKAGES,
) -> dict[str, object]:
    """Collect lightweight environment metadata for one experiment run."""

    package_versions = _package_versions(packages)
    payload = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "command_line": list(command_line or sys.argv),
        "package_versions": package_versions,
        "code_version": _git_revision(Path.cwd()),
        "data_snapshot": data_snapshot,
        "catalog_reference": catalog_reference_for_path(catalog_path)
        if catalog_path is not None
        else None,
    }
    payload["environment_hash"] = _stable_hash(payload)
    return payload


def artifact_hashes(artifacts: dict[str, str]) -> dict[str, str]:
    """Return SHA-256 hashes for artifact paths that exist locally."""

    hashes: dict[str, str] = {}
    for name, artifact_path in artifacts.items():
        path = Path(artifact_path)
        if path.exists() and path.is_file():
            hashes[name] = file_sha256(path)
    return hashes


def _package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_revision(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
