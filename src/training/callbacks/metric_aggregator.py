from __future__ import annotations
from typing import Dict, Any

class MetricAggregator:
    """Accumulates per-step metrics and produces rolling averages."""

    def __init__(self) -> None:
        self.reset()
        self._initial_balance_set = False
        self._initial_balance = 0.0

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def update(self, info: Dict[str, Any], reward: float) -> None:
        """Update rolling aggregates with latest step information."""
        self._reward_sum += float(reward)
        # Trade count: derive from cumulative total_trades field if provided
        total_trades = int(info.get("total_trades", 0))
        trades_delta = total_trades - self._last_total_trades
        if trades_delta < 0:
            trades_delta = 0  # Safeguard against resets
        self._trade_count += trades_delta
        self._last_total_trades = total_trades

        self._count += 1

        # Snapshot values (latest)
        self._last_portfolio_value = float(info.get("portfolio_value", self._last_portfolio_value))

        # Lazily set initial balance to compute return
        if not self._initial_balance_set and "balance" in info:
            self._initial_balance = float(info["balance"])
            self._initial_balance_set = True

        if self._initial_balance_set and self._initial_balance != 0:
            self._last_total_return = (self._last_portfolio_value / self._initial_balance) - 1.0

    def summary(self) -> Dict[str, Any]:
        if self._count == 0:
            return {}
        avg_reward = self._reward_sum / self._count
        avg_trades = self._trade_count / self._count if self._count else 0.0

        out = {
            "avg_reward": avg_reward,
            "avg_trades": avg_trades,
            "portfolio_value": self._last_portfolio_value,
            "total_return": self._last_total_return,
        }
        self.reset()  # prepare for next window
        return out

    def reset(self) -> None:
        self._reward_sum = 0.0
        self._trade_count = 0
        self._count = 0
        self._last_portfolio_value = 0.0
        self._last_total_return = 0.0
        self._last_total_trades = 0
        self._learning_rate = 0.0 