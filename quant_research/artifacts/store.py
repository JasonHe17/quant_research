"""Research artifact store scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from quant_research.schemas import STANDARD_TABLE_SCHEMAS, validate_standard_table

if TYPE_CHECKING:
    from quant_research.backtest import BacktestResult
    from quant_research.factors import FactorResult, SingleFactorEvaluationResult
    from quant_research.metrics import MetricsReport
    from quant_research.portfolio import PortfolioConstructionResult
    from quant_research.signals import SignalResult
    from quant_research.universe import Universe


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """Storage root for research outputs."""

    root: Path

    @classmethod
    def from_path(cls, root: str | Path) -> "ArtifactStore":
        return cls(root=Path(root))

    def factor_path(self, factor_name: str) -> Path:
        return self.root / "factors" / f"{_safe_path_component(factor_name)}.parquet"

    def write_factor(self, result: "FactorResult") -> Path:
        path = self.factor_path(result.factor_name)
        self._write_table_artifact(
            path,
            result.frame,
            artifact_type="factor",
            artifact_name=result.factor_name,
        )
        return path

    def read_factor(self, factor_name: str) -> pd.DataFrame:
        return self._read_table_artifact(self.factor_path(factor_name))

    def factor_evaluation_root(self, name: str) -> Path:
        return self.root / "factor_evaluations" / _safe_path_component(name)

    def factor_evaluation_path(self, name: str, artifact_name: str) -> Path:
        return self.factor_evaluation_root(name) / f"{artifact_name}.parquet"

    def write_factor_evaluation(
        self,
        name: str,
        result: "SingleFactorEvaluationResult",
    ) -> dict[str, str]:
        paths = {
            "summary": self.factor_evaluation_path(name, "summary"),
            "by_timestamp": self.factor_evaluation_path(name, "by_timestamp"),
            "quantile_by_timestamp": self.factor_evaluation_path(
                name, "quantile_by_timestamp"
            ),
            "quantile_returns": self.factor_evaluation_path(
                name, "quantile_returns"
            ),
            "feature_correlation": self.factor_evaluation_path(
                name, "feature_correlation"
            ),
            "decay_by_label": self.factor_evaluation_path(name, "decay_by_label"),
            "group_summary": self.factor_evaluation_path(name, "group_summary"),
            "multiple_testing": self.factor_evaluation_path(name, "multiple_testing"),
        }
        frames = {
            "summary": result.summary,
            "by_timestamp": result.by_timestamp,
            "quantile_by_timestamp": result.quantile_by_timestamp,
            "quantile_returns": result.quantile_returns,
            "feature_correlation": result.feature_correlation,
            "decay_by_label": result.decay_by_label,
            "group_summary": result.group_summary,
            "multiple_testing": result.multiple_testing,
        }
        for name_, path in paths.items():
            self._write_table_artifact(
                path,
                frames[name_],
                artifact_type="factor_evaluation",
                artifact_name=f"{name}:{name_}",
            )
        return {artifact_name: str(path) for artifact_name, path in paths.items()}

    def read_factor_evaluation_artifact(
        self,
        name: str,
        artifact_name: str,
    ) -> pd.DataFrame:
        return self._read_table_artifact(
            self.factor_evaluation_path(name, artifact_name)
        )

    def signal_root(self, signal_name: str) -> Path:
        return self.root / "signals" / _safe_path_component(signal_name)

    def signal_path(self, signal_name: str, artifact_name: str) -> Path:
        return self.signal_root(signal_name) / f"{artifact_name}.parquet"

    def write_signal(self, result: "SignalResult") -> dict[str, str]:
        paths = {
            "signals": self.signal_path(result.spec.name, "signals"),
            "diagnostics": self.signal_path(result.spec.name, "diagnostics"),
        }
        self._write_table_artifact(
            paths["signals"],
            result.frame,
            artifact_type="signal",
            artifact_name=result.spec.name,
        )
        self._write_table_artifact(
            paths["diagnostics"],
            result.diagnostics,
            artifact_type="signal_diagnostics",
            artifact_name=result.spec.name,
        )
        return {name: str(path) for name, path in paths.items()}

    def read_signal_artifact(
        self, signal_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return self._read_table_artifact(self.signal_path(signal_name, artifact_name))

    def backtest_root(self, backtest_name: str) -> Path:
        return self.root / "backtests" / _safe_path_component(backtest_name)

    def backtest_path(self, backtest_name: str, artifact_name: str) -> Path:
        return self.backtest_root(backtest_name) / f"{artifact_name}.parquet"

    def write_backtest(self, result: "BacktestResult") -> dict[str, str]:
        paths = {
            "trades": self.backtest_path(result.config.name, "trades"),
            "positions": self.backtest_path(result.config.name, "positions"),
            "equity_curve": self.backtest_path(result.config.name, "equity_curve"),
            "diagnostics": self.backtest_path(result.config.name, "diagnostics"),
        }
        self._write_table_artifact(
            paths["trades"],
            result.trades,
            artifact_type="backtest_trades",
            artifact_name=result.config.name,
        )
        self._write_table_artifact(
            paths["positions"],
            result.positions,
            artifact_type="backtest_positions",
            artifact_name=result.config.name,
        )
        self._write_table_artifact(
            paths["equity_curve"],
            result.equity_curve,
            artifact_type="backtest_equity_curve",
            artifact_name=result.config.name,
        )
        self._write_table_artifact(
            paths["diagnostics"],
            result.diagnostics,
            artifact_type="backtest_diagnostics",
            artifact_name=result.config.name,
        )
        return {name: str(path) for name, path in paths.items()}

    def read_backtest_artifact(
        self, backtest_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return self._read_table_artifact(
            self.backtest_path(backtest_name, artifact_name)
        )

    def portfolio_root(self, portfolio_name: str) -> Path:
        return self.root / "portfolios" / _safe_path_component(portfolio_name)

    def portfolio_path(self, portfolio_name: str, artifact_name: str) -> Path:
        return self.portfolio_root(portfolio_name) / f"{artifact_name}.parquet"

    def write_portfolio(
        self, result: "PortfolioConstructionResult"
    ) -> dict[str, str]:
        paths = {
            "target_weights": self.portfolio_path(
                result.config.name, "target_weights"
            ),
            "rebalance_orders": self.portfolio_path(
                result.config.name, "rebalance_orders"
            ),
            "diagnostics": self.portfolio_path(result.config.name, "diagnostics"),
        }
        self._write_table_artifact(
            paths["target_weights"],
            result.target_weights,
            artifact_type="portfolio_target_weights",
            artifact_name=result.config.name,
        )
        self._write_table_artifact(
            paths["rebalance_orders"],
            result.rebalance_orders,
            artifact_type="portfolio_rebalance_orders",
            artifact_name=result.config.name,
        )
        self._write_table_artifact(
            paths["diagnostics"],
            result.diagnostics,
            artifact_type="portfolio_diagnostics",
            artifact_name=result.config.name,
        )
        return {name: str(path) for name, path in paths.items()}

    def read_portfolio_artifact(
        self, portfolio_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return self._read_table_artifact(
            self.portfolio_path(portfolio_name, artifact_name)
        )

    def universe_root(self, universe_name: str) -> Path:
        return self.root / "universes" / _safe_path_component(universe_name)

    def universe_path(self, universe_name: str, artifact_name: str) -> Path:
        return self.universe_root(universe_name) / f"{artifact_name}.parquet"

    def write_universe(self, universe: "Universe") -> dict[str, str]:
        paths = {
            "members": self.universe_path(universe.spec.name, "members"),
            "diagnostics": self.universe_path(universe.spec.name, "diagnostics"),
        }
        self._write_table_artifact(
            paths["members"],
            universe.members,
            artifact_type="universe_members",
            artifact_name=universe.spec.name,
        )
        self._write_table_artifact(
            paths["diagnostics"],
            universe.diagnostics,
            artifact_type="universe_diagnostics",
            artifact_name=universe.spec.name,
        )
        return {name: str(path) for name, path in paths.items()}

    def read_universe_artifact(
        self, universe_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return self._read_table_artifact(
            self.universe_path(universe_name, artifact_name)
        )

    def report_root(self, report_name: str) -> Path:
        return self.root / "reports" / _safe_path_component(report_name)

    def report_path(self, report_name: str) -> Path:
        return self.report_root(report_name) / "metrics.json"

    def write_metrics_report(self, report: "MetricsReport") -> dict[str, str]:
        path = self.report_path(report.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        self._write_artifact_manifest(
            path,
            artifact_type="metrics_report",
            artifact_name=report.name,
            format_name="json",
            row_count=None,
            columns=(),
            dtypes={},
        )
        return {"metrics_report": str(path)}

    def read_metrics_report(self, report_name: str) -> dict[str, object]:
        return json.loads(self.report_path(report_name).read_text(encoding="utf-8"))

    def artifact_manifest_path(self, artifact_path: str | Path) -> Path:
        """Return the JSON sidecar manifest path for an artifact."""

        path = Path(artifact_path)
        return path.with_name(f"{path.name}.manifest.json")

    def read_artifact_manifest(self, artifact_path: str | Path) -> dict[str, object]:
        """Read the JSON sidecar manifest for an artifact."""

        return json.loads(
            self.artifact_manifest_path(artifact_path).read_text(encoding="utf-8")
        )

    def _write_table_artifact(
        self,
        path: Path,
        frame: pd.DataFrame,
        *,
        artifact_type: str,
        artifact_name: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_type in STANDARD_TABLE_SCHEMAS:
            validate_standard_table(artifact_type, frame)
        output = _parquet_ready_frame(frame)
        output.to_parquet(path)
        self._write_artifact_manifest(
            path,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
            format_name="parquet",
            row_count=len(output),
            columns=tuple(str(column) for column in output.columns),
            dtypes={str(column): str(dtype) for column, dtype in output.dtypes.items()},
        )

    def _read_table_artifact(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        legacy_path = path.with_suffix(".pkl")
        if legacy_path.exists():
            return pd.read_pickle(legacy_path)
        raise FileNotFoundError(path)

    def _write_artifact_manifest(
        self,
        path: Path,
        *,
        artifact_type: str,
        artifact_name: str,
        format_name: str,
        row_count: int | None,
        columns: tuple[str, ...],
        dtypes: dict[str, str],
    ) -> Path:
        payload = {
            "artifact_name": artifact_name,
            "artifact_type": artifact_type,
            "columns": list(columns),
            "dtypes": dtypes,
            "format": format_name,
            "path": str(path),
            "row_count": row_count,
            "sha256": _file_sha256(path),
        }
        manifest_path = self.artifact_manifest_path(path)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path


def _safe_path_component(value: str) -> str:
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(allowed).strip("_") or "artifact"


def _parquet_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        series = output[column]
        if series.dtype != "object":
            continue
        inferred = pd.api.types.infer_dtype(series.dropna(), skipna=True)
        if inferred.startswith("mixed"):
            output[column] = series.astype("string")
    return output


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
