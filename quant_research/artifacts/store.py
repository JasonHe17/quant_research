"""Research artifact store scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant_research.backtest import BacktestResult
    from quant_research.factors import FactorResult
    from quant_research.portfolio import PortfolioConstructionResult


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


def _safe_path_component(value: str) -> str:
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(allowed).strip("_") or "artifact"
