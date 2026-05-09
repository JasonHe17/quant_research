"""Research artifact store scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant_research.backtest import BacktestResult
    from quant_research.factors import FactorResult
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
        return self.root / "factors" / f"{_safe_path_component(factor_name)}.pkl"

    def write_factor(self, result: "FactorResult") -> Path:
        path = self.factor_path(result.factor_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.frame.to_pickle(path)
        return path

    def read_factor(self, factor_name: str) -> pd.DataFrame:
        return pd.read_pickle(self.factor_path(factor_name))

    def signal_root(self, signal_name: str) -> Path:
        return self.root / "signals" / _safe_path_component(signal_name)

    def signal_path(self, signal_name: str, artifact_name: str) -> Path:
        return self.signal_root(signal_name) / f"{artifact_name}.pkl"

    def write_signal(self, result: "SignalResult") -> dict[str, str]:
        paths = {
            "signals": self.signal_path(result.spec.name, "signals"),
            "diagnostics": self.signal_path(result.spec.name, "diagnostics"),
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        result.frame.to_pickle(paths["signals"])
        result.diagnostics.to_pickle(paths["diagnostics"])
        return {name: str(path) for name, path in paths.items()}

    def read_signal_artifact(
        self, signal_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return pd.read_pickle(self.signal_path(signal_name, artifact_name))

    def backtest_root(self, backtest_name: str) -> Path:
        return self.root / "backtests" / _safe_path_component(backtest_name)

    def backtest_path(self, backtest_name: str, artifact_name: str) -> Path:
        return self.backtest_root(backtest_name) / f"{artifact_name}.pkl"

    def write_backtest(self, result: "BacktestResult") -> dict[str, str]:
        paths = {
            "trades": self.backtest_path(result.config.name, "trades"),
            "positions": self.backtest_path(result.config.name, "positions"),
            "equity_curve": self.backtest_path(result.config.name, "equity_curve"),
            "diagnostics": self.backtest_path(result.config.name, "diagnostics"),
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        result.trades.to_pickle(paths["trades"])
        result.positions.to_pickle(paths["positions"])
        result.equity_curve.to_pickle(paths["equity_curve"])
        result.diagnostics.to_pickle(paths["diagnostics"])
        return {name: str(path) for name, path in paths.items()}

    def read_backtest_artifact(
        self, backtest_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return pd.read_pickle(self.backtest_path(backtest_name, artifact_name))

    def portfolio_root(self, portfolio_name: str) -> Path:
        return self.root / "portfolios" / _safe_path_component(portfolio_name)

    def portfolio_path(self, portfolio_name: str, artifact_name: str) -> Path:
        return self.portfolio_root(portfolio_name) / f"{artifact_name}.pkl"

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
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        result.target_weights.to_pickle(paths["target_weights"])
        result.rebalance_orders.to_pickle(paths["rebalance_orders"])
        result.diagnostics.to_pickle(paths["diagnostics"])
        return {name: str(path) for name, path in paths.items()}

    def read_portfolio_artifact(
        self, portfolio_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return pd.read_pickle(self.portfolio_path(portfolio_name, artifact_name))

    def universe_root(self, universe_name: str) -> Path:
        return self.root / "universes" / _safe_path_component(universe_name)

    def universe_path(self, universe_name: str, artifact_name: str) -> Path:
        return self.universe_root(universe_name) / f"{artifact_name}.pkl"

    def write_universe(self, universe: "Universe") -> dict[str, str]:
        paths = {
            "members": self.universe_path(universe.spec.name, "members"),
            "diagnostics": self.universe_path(universe.spec.name, "diagnostics"),
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        universe.members.to_pickle(paths["members"])
        universe.diagnostics.to_pickle(paths["diagnostics"])
        return {name: str(path) for name, path in paths.items()}

    def read_universe_artifact(
        self, universe_name: str, artifact_name: str
    ) -> pd.DataFrame:
        return pd.read_pickle(self.universe_path(universe_name, artifact_name))

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
        return {"metrics_report": str(path)}

    def read_metrics_report(self, report_name: str) -> dict[str, object]:
        return json.loads(self.report_path(report_name).read_text(encoding="utf-8"))


def _safe_path_component(value: str) -> str:
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(allowed).strip("_") or "artifact"
