#!/usr/bin/env python3
"""
Tests for the trading environment.
"""
import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from src.environments.trading_env import TradingEnv
from src.environments.types import Position


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    data = {
        "timestamp": pd.to_datetime(
            [
                "2023-01-01 12:00",
                "2023-01-01 13:00",
                "2023-01-01 14:00",
                "2023-01-01 15:00",
            ]
        ),
        "open": [100, 102, 105, 108],
        "high": [103, 106, 108, 110],
        "low": [99, 101, 104, 107],
        "close": [102, 105, 107, 109],
        "volume": [1000, 1200, 1100, 1300],
        "feature1": [0.1, 0.2, 0.3, 0.4],
    }
    df = pd.DataFrame(data).set_index("timestamp")
    return df


@pytest.fixture
def trading_env(sample_data: pd.DataFrame) -> TradingEnv:
    """Create a TradingEnv instance for testing."""
    return TradingEnv(data=sample_data, initial_balance=10000.0, trade_fee=0.001)


@pytest.mark.unit
class TestTradingEnv:
    """
    Test suite for the TradingEnv class.
    """

    def test_initialization(self, trading_env: TradingEnv, sample_data: pd.DataFrame):
        """
        Test if the environment is initialized correctly.
        """
        assert trading_env.initial_balance == 10000.0
        assert trading_env.balance == 10000.0
        assert trading_env.trade_fee == 0.001
        assert trading_env.current_step == 0
        assert trading_env._position == Position.FLAT
        assert trading_env._position_entry_price == 0.0
        assert isinstance(trading_env.action_space, spaces.Discrete)
        assert trading_env.action_space.n == len(Position)
        assert isinstance(trading_env.observation_space, spaces.Box)
        
        # Observation space shape = 3 (price, balance, pos) + num_features
        expected_shape = (3 + len(sample_data.columns),)
        assert trading_env.observation_space.shape == expected_shape

    def test_reset(self, trading_env: TradingEnv):
        """
        Test if the reset method correctly resets the environment's state.
        """
        # Change state
        trading_env.current_step = 1
        trading_env.balance = 5000.0
        trading_env._position = Position.LONG
        trading_env._position_entry_price = 102.0

        # Reset the environment
        observation, info = trading_env.reset()

        # Check if state is reset
        assert trading_env.current_step == 0
        assert trading_env.balance == trading_env.initial_balance
        assert trading_env._position == Position.FLAT
        assert trading_env._position_entry_price == 0.0
        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)

        # Verify observation content
        expected_price = trading_env.data.iloc[0]["close"]
        assert observation[0] == expected_price
        assert observation[1] == 1.0  # Normalized balance
        assert observation[2] == Position.FLAT.value

    @pytest.mark.parametrize(
        "action, expected_position, initial_position",
        [
            (Position.LONG, Position.LONG, Position.FLAT),
            (Position.SHORT, Position.SHORT, Position.FLAT),
            (Position.FLAT, Position.FLAT, Position.FLAT),
            (Position.LONG, Position.LONG, Position.LONG),  # No change
        ],
    )
    def test_take_action(
        self,
        trading_env: TradingEnv,
        action: Position,
        expected_position: Position,
        initial_position: Position,
    ):
        """Test the _take_action method."""
        trading_env._position = initial_position
        
        # Give a non-zero entry price if we start in a position
        initial_entry_price = 0.0
        if initial_position != Position.FLAT:
            initial_entry_price = 99.0  # An arbitrary price
            trading_env._position_entry_price = initial_entry_price

        trading_env._take_action(action.value)

        current_price = trading_env.data.iloc[0]["close"]

        assert trading_env._position == expected_position
        if expected_position != Position.FLAT:
            if initial_position == expected_position:
                # If position is unchanged, entry price should be unchanged
                assert trading_env._position_entry_price == initial_entry_price
            else:
                # If new position is opened, entry price should be current price
                assert trading_env._position_entry_price == current_price
        else:
            # If position is now flat, entry price should be 0
            assert trading_env._position_entry_price == 0.0
    
    def test_step(self, trading_env: TradingEnv):
        """
        Test a single step in the environment.
        """
        # Initial state: Flat
        initial_balance = trading_env.balance
        
        # Action: Go Long
        action = Position.LONG.value
        obs, reward, terminated, truncated, info = trading_env.step(action)

        # After step 0, current_step becomes 1
        assert trading_env.current_step == 1
        assert trading_env._position == Position.LONG
        assert info["position"] == "LONG"
        
        # PnL is calculated from step 1 data
        entry_price = trading_env.data.iloc[0]["close"] # 102
        current_price = trading_env.data.iloc[1]["close"] # 105
        
        # Check reward calculation
        unrealized_pnl = (current_price - entry_price) * (initial_balance / entry_price)
        expected_portfolio_value = initial_balance + unrealized_pnl
        expected_reward = expected_portfolio_value - initial_balance
        assert pytest.approx(reward) == expected_reward
        
        # Check info dict
        assert info["portfolio_value"] == pytest.approx(expected_portfolio_value)

    def test_long_short_trading_sequence(self, trading_env: TradingEnv):
        """
        Test a sequence of trades: going long, closing, and going short.
        """
        initial_balance = trading_env.initial_balance

        # 1. Go Long at step 0
        trading_env.step(Position.LONG.value)
        assert trading_env._position == Position.LONG
        entry_price_long = 102
        assert trading_env._position_entry_price == entry_price_long

        # 2. Close position (go Flat) at step 1
        trading_env.step(Position.FLAT.value)
        assert trading_env._position == Position.FLAT
        price_at_step_1 = 105
        profit_long = (price_at_step_1 - entry_price_long) * (initial_balance / entry_price_long)
        expected_balance_after_long = initial_balance + profit_long * (1 - trading_env.trade_fee)
        assert trading_env.balance == pytest.approx(expected_balance_after_long)

        # 3. Go Short at step 2
        trading_env.step(Position.SHORT.value)
        assert trading_env._position == Position.SHORT
        entry_price_short = 107
        assert trading_env._position_entry_price == entry_price_short

    def test_episode_end_on_data_depletion(self, trading_env: TradingEnv):
        """
        Test if the episode terminates when the data runs out.
        """
        # Step until the second to last data point
        for _ in range(len(trading_env.data) - 2):
            _, _, terminated, _, _ = trading_env.step(Position.FLAT.value)
            assert not terminated
        
        # The last step should trigger termination
        _, _, terminated, _, _ = trading_env.step(Position.FLAT.value)
        assert terminated

    def test_episode_end_on_balance_depletion(self, trading_env: TradingEnv):
        """
        Test if the episode terminates when the portfolio value is depleted.
        """
        trading_env.balance = 100.0  # Start with a balance of 100
        # Set up prices to cause a wipeout on the second step
        trading_env.data.loc[trading_env.data.index[0], "close"] = 100
        trading_env.data.loc[trading_env.data.index[1], "close"] = 201 # This will wipe out the portfolio

        # The first step opens the short position, which is immediately wiped out
        # by the price move to 201, so the episode should terminate.
        _, _, terminated, _, _ = trading_env.step(Position.SHORT.value)
        assert terminated, "Episode should terminate immediately on wipeout" 