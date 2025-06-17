#!/usr/bin/env python3
"""Tests for the trading environment (relocated)."""

import numpy as np
import pandas as pd
import pytest

from src.environments.trading_env import TradingEnv
from src.environments.types import Position


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10))
    data = {
        "open": [100, 102, 101, 103, 105, 107, 106, 108, 110, 109],
        "high": [101, 103, 102, 104, 106, 108, 107, 109, 111, 110],
        "low": [99, 101, 100, 102, 104, 106, 105, 107, 109, 108],
        "close": [101, 101, 102, 104, 105, 106, 106, 108, 110, 109],
        "volume": [1000] * 10,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def trading_env(sample_data: pd.DataFrame) -> TradingEnv:
    return TradingEnv(data=sample_data, initial_balance=10000.0, trade_fee=0.001)


@pytest.mark.unit
class TestTradingEnv:
    def test_initialization(self, trading_env: TradingEnv, sample_data: pd.DataFrame):
        assert trading_env.initial_balance == 10000.0
        assert trading_env.balance == 10000.0
        assert trading_env.current_step == 0
        assert trading_env._position == Position.FLAT
        pd.testing.assert_frame_equal(trading_env.data, sample_data)

    def test_reset(self, trading_env: TradingEnv):
        trading_env.current_step = 1
        trading_env.balance = 9000.0
        trading_env._position = Position.LONG
        trading_env.trade_history.append("dummy")

        observation, _ = trading_env.reset()
        assert trading_env.balance == trading_env.initial_balance
        assert trading_env.current_step == 0
        assert trading_env._position == Position.FLAT
        np.testing.assert_array_equal(observation, trading_env._get_observation())

    def test_get_observation(self, trading_env: TradingEnv):
        observation = trading_env._get_observation()
        expected_shape = (3 + 5,)
        assert observation.shape == expected_shape
        assert observation[0] == trading_env.data.iloc[0]["close"]
        assert observation[1] == trading_env.balance / trading_env.initial_balance
        assert observation[2] == Position.FLAT.value

    def test_step_full_flow(self, trading_env: TradingEnv):
        # Initial step – hold
        initial_balance = trading_env.balance
        _, reward, terminated, truncated, _ = trading_env.step(Position.FLAT.value)
        assert reward == 0
        assert not (terminated or truncated)
        assert trading_env.balance == initial_balance

        # Go long
        _, _, _, _, _ = trading_env.step(Position.LONG.value)
        assert trading_env._position == Position.LONG

        # Close
        _, reward_close, _, _, _ = trading_env.step(Position.FLAT.value)
        assert trading_env._position == Position.FLAT
        # Reward can be ± depending on price movement; ensure balance updated
        assert trading_env.balance != initial_balance

    def test_trade_history_tracking_and_reset(self, trading_env: TradingEnv):
        trading_env.step(Position.LONG.value)
        trading_env.step(Position.FLAT.value)
        assert len(trading_env.trade_history) == 1
        trading_env.reset()
        assert len(trading_env.trade_history) == 0

    def test_episode_end_conditions(self, trading_env: TradingEnv):
        # Exhaust data
        for _ in range(len(trading_env.data) - 2):
            _, _, terminated, truncated, _ = trading_env.step(Position.FLAT.value)
            assert not (terminated or truncated)
        _, _, terminated, truncated, _ = trading_env.step(Position.FLAT.value)
        assert terminated or truncated

    def test_balance_depletion_ends_episode(self, trading_env: TradingEnv):
        trading_env.balance = 1.0
        trading_env.initial_balance = 1.0
        trading_env.data.loc[trading_env.data.index[0], "close"] = 100
        trading_env.data.loc[trading_env.data.index[1], "close"] = 0
        _, _, terminated, truncated, _ = trading_env.step(Position.LONG.value)
        assert terminated or truncated
