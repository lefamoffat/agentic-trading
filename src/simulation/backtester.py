"""Utility module for running offline back-tests on trained RL agents.

This implementation purposely stays **framework-agnostic** - all
interactions with MLflow, Stable-Baselines3, and the custom trading
environment are abstracted away behind small helper functions.  This
keeps public surface area minimal and therefore easier to test.

The primary entry-point is :class:`Backtester`.  Typical usage::

    backtester = Backtester(
        model_uri="models:/eurusd_ppo@Production",  # MLflow Model Registry URI
        data=df,                                     # pd.DataFrame with features
        initial_balance=10_000,
    )
    result = backtester.run()

The returned :class:`SimulationResult` contains the full equity curve,
trade log, and a dictionary of performance metrics ready for UI
consumption.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.environments.factory import environment_factory
from src.environments.types import Position
from src.utils.logger import get_logger

__all__ = ["SimulationResult", "Backtester"]

logger = get_logger(__name__)


@dataclass(slots=True)
class SimulationResult:
    """Container for all artefacts produced by a back-test."""

    equity_curve: pd.Series  # Indexed by step (int) - normalised PV
    trades: pd.DataFrame  # One row per closed trade
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, any]:  # type: ignore[override]
        """Return a JSON-serialisable representation (for Streamlit)."""
        return {
            "equity_curve": self.equity_curve.to_list(),
            "trades": self.trades.to_dict(orient="records"),
            "metrics": self.metrics,
        }


class Backtester:
    """Run a historical simulation using a trained SB3 agent.

    The *data* argument must be identical (column wise) to what the agent
    observed during training - i.e. **after** feature engineering.
    """

    def __init__(
        self,
        model_uri: str,
        data: pd.DataFrame,
        initial_balance: float = 10_000.0,
        trade_fee: float = 0.001,
    ) -> None:
        self.model_uri = model_uri
        self.data = data.copy().reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.trade_fee = float(trade_fee)

        # Lazily loaded objects
        self._model: Optional[mlflow.pyfunc.PyFuncModel] = None
        self._env = environment_factory.create_environment(
            name="default",
            data=self.data,
            initial_balance=self.initial_balance,
            trade_fee=self.trade_fee,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> mlflow.pyfunc.PyFuncModel:
        if self._model is None:
            logger.info("Downloading model from MLflow - this might take a moment…")
            self._model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info("Model loaded successfully.")
        return self._model

    def _calculate_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Compute simple performance metrics from the equity curve."""
        returns = equity_curve.pct_change().dropna()
        total_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
        trading_days = int(os.getenv("TRADING_DAYS_PER_YEAR", "252"))
        sharpe_ratio = (
            (returns.mean() / returns.std()) * np.sqrt(trading_days)
            if not returns.empty and returns.std() != 0
            else 0.0
        )
        max_drawdown = self._max_drawdown(equity_curve.values)

        return {
            "total_profit": round(float(total_profit), 4),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "num_trades": len(self._env.trade_history),
        }

    @staticmethod
    def _max_drawdown(values: np.ndarray) -> float:
        """Vectorised maximum drawdown calculation."""
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute the back-test and return a :class:`SimulationResult`."""
        model = self._load_model()
        observation, _ = self._env.reset()

        equity_vals: List[float] = [self.initial_balance]

        done = False
        while not done:
            # Build a single-row DataFrame - the wrapper handles feature → tensor
            current_step = self._env.current_step
            obs_df = self.data.iloc[current_step : current_step + 1]

            action = int(model.predict(obs_df)[0])  # type: ignore[index]
            _, _, terminated, truncated, _ = self._env.step(action)
            done = terminated or truncated

            equity_vals.append(self._env.portfolio_value)

        equity_curve = pd.Series(equity_vals, name="equity_curve")

        # Flatten trade history
        trades_df = self._to_dataframe(self._env.trade_history)

        metrics = self._calculate_metrics(equity_curve)

        return SimulationResult(equity_curve=equity_curve, trades=trades_df, metrics=metrics)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(trades: List) -> pd.DataFrame:  # type: ignore[override]
        if not trades:
            return pd.DataFrame(columns=[
                "entry_price",
                "exit_price",
                "position",
                "profit",
            ])

        records = [
            {
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "position": t.position.name if isinstance(t.position, Position) else str(t.position),
                "profit": t.profit,
            }
            for t in trades
        ]
        return pd.DataFrame.from_records(records)
