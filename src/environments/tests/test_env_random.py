from __future__ import annotations

"""Diagnostic test for the default TradingEnv – ensures random policy yields trades."""

from typing import List

import numpy as np
import pandas as pd
import pytest

from src.environments.factory import environment_factory
from src.environments.trading_env import Trade


def _generate_dummy(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed=123)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0003, size=n))
    return pd.DataFrame({"close": close}).reset_index(drop=True)


def _run_random_episode(env) -> List[Trade]:
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["trade_history"]


@pytest.mark.slow
def test_random_policy_generates_trades():
    env = environment_factory.create_environment("default", data=_generate_dummy())
    trades = _run_random_episode(env)
    assert len(trades) > 5, "Random agent produced too few trades – environment logic suspect" 