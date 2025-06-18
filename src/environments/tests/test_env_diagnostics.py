from __future__ import annotations

"""Diagnostic tests for TradingEnv.

These are NOT strict unit-tests; they act as sanity-checks when debugging the
learning pipeline. The test is marked with ``pytest.mark.slow`` so the default
CI run (``pytest -m "not slow"``) will skip it. You can run it manually via
``pytest -m slow``.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.environments.factory import environment_factory
from src.environments.trading_env import Trade

# ---------------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------------

def _generate_dummy_data(n: int = 1000) -> pd.DataFrame:
    """Create dummy candle+feature data for deterministic diagnostics."""
    rng = np.random.default_rng(seed=42)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, size=n))  # small random walk

    df = pd.DataFrame(
        {
            "close": close,
            # Very naive feature examples (scaled close)
            "sma_5": pd.Series(close).rolling(window=5).mean().bfill(),
            "sma_10": pd.Series(close).rolling(window=10).mean().bfill(),
        }
    )
    return df.reset_index(drop=True)


def _run_random_episode(env) -> List[Trade]:
    """Run one episode with uniform-random actions; return trade history."""
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info.get("trade_history", [])


@pytest.mark.slow
def test_diagnostic_random_policy_trade_generation(capfd):
    """Ensure a random policy generates >1 trades and confirm obs scales."""
    data = _generate_dummy_data()

    env = environment_factory.create_environment(
        name="default", data=data, initial_balance=10_000, trade_fee=0.0
    )

    # 1. Observation scale ----
    obs, _ = env.reset()
    obs_min, obs_max, obs_mean = obs.min(), obs.max(), obs.mean()

    print("Observation vector stats – min / max / mean:", obs_min, obs_max, obs_mean)

    # Ensure observation elements are finite and within a reasonable range.
    assert np.isfinite(obs).all(), "Observation contains non-finite values"
    assert obs_max < 100.0, "Observation values look un-normalised (>100)"

    # 2. Random policy ----
    trade_history = _run_random_episode(env)
    print("Random policy total trades:", len(trade_history))

    # Basic expectation: with random actions over ~1000 steps and three discrete
    # choices, we expect multiple position changes.
    assert len(trade_history) > 1, "Random agent produced <=1 trades – env bug?"

    # Capture stdout for optional inspection when running the test directly.
    capfd.readouterr() 