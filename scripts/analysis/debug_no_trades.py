import argparse
import logging
from pathlib import Path

import pandas as pd
import uvicorn

from src.simulation.visualisation import price_with_trades
from scripts.analysis.backend import run_simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_no_trades")


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Simple CLI parser."""
    p = argparse.ArgumentParser(description="Debug a trained agent that produced no trades.")
    p.add_argument("--model-uri", required=True, help="MLflow model URI or 'random' for random actions")
    p.add_argument("--data-path", required=True, help="CSV with feature columns used for training")
    p.add_argument("--initial-balance", type=float, default=10_000)
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(args.data_path)

    logger.info("Running simulation for debug…")
    if args.model_uri.lower() == "random":
        from src.simulation.backtester import Backtester
        import numpy as np

        df = pd.read_csv(args.data_path)
        backtester = Backtester(model_uri="random", data=df, initial_balance=args.initial_balance)

        env = backtester._env  # pylint: disable=protected-access
        obs, _ = env.reset()
        equity = [backtester.initial_balance]
        actions = []
        rewards = []

        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            actions.append(action)
            rewards.append(reward)
            equity.append(env.portfolio_value)
            step += 1

        trades_df = backtester._to_dataframe(env.trade_history)  # pylint: disable=protected-access
        equity_curve = pd.Series(equity)
        logger.info("Random agent produced %s trades", len(trades_df))

        pd.DataFrame({"action": actions, "reward": rewards}).to_csv("debug_actions.csv", index=False)
        trades_df.to_csv("debug_trades.csv", index=False)
        logger.info("Saved debug_actions.csv and debug_trades.csv")
        return

    result, _ = run_simulation(
        model_uri=args.model_uri,
        data_path=args.data_path,
        initial_balance=args.initial_balance,
    )

    trades = result.trades
    if trades.empty:
        logger.warning("❌ Simulation produced ZERO trades.")
    else:
        logger.info("✅ Simulation produced %s trades", len(trades))

    # Dump CSV for offline inspection
    trades.to_csv("debug_trades.csv", index=False)
    logger.info("Trade log saved to debug_trades.csv")


if __name__ == "__main__":
    main() 