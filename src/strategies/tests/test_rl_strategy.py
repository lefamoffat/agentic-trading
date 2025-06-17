#!/usr/bin/env python3
"""Tests for the RL-based trading strategy (relocated)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.environments.types import Position
from src.strategies.rl_strategy import RLStrategy


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.predict.return_value = (Position.LONG.value, None)
    return agent


@pytest.fixture
def mock_broker():
    broker = AsyncMock()
    broker.initial_balance = 100000
    broker.get_cash_balance.return_value = 100000
    broker.get_position.return_value = Position.FLAT
    broker.get_open_positions.return_value = {}
    broker.get_historical_data.return_value = pd.DataFrame(
        {
            "open": np.random.rand(200),
            "high": np.random.rand(200),
            "low": np.random.rand(200),
            "close": np.random.rand(200),
            "volume": np.random.rand(200),
        }
    )
    return broker


@pytest.fixture
def rl_strategy(mock_agent, mock_broker):
    return RLStrategy(
        agent=mock_agent,
        broker=mock_broker,
        trading_symbol="EUR/USD",
        timeframe="1h",
        window_size=200,
    )


@pytest.mark.asyncio
async def test_generate_signals_long(rl_strategy, mock_agent):
    latest_data = pd.DataFrame(
        {
            "open": np.random.rand(200),
            "high": np.random.rand(200),
            "low": np.random.rand(200),
            "close": np.random.rand(200),
            "volume": np.random.rand(200),
        }
    )
    mock_agent.predict.return_value = (Position.LONG.value, None)
    position = await rl_strategy.generate_signals(latest_data, rl_strategy.broker)
    assert position == Position.LONG
    mock_agent.predict.assert_called_once()


@pytest.mark.asyncio
async def test_execute_trade_opens_long_position(rl_strategy, mock_broker):
    await rl_strategy._execute_trade(Position.LONG)
    mock_broker.open_position.assert_called_once_with("BUY", "EUR/USD", 10000)


@pytest.mark.asyncio
async def test_execute_trade_opens_short_position(rl_strategy, mock_broker):
    await rl_strategy._execute_trade(Position.SHORT)
    mock_broker.open_position.assert_called_once_with("SELL", "EUR/USD", 10000)


@pytest.mark.asyncio
async def test_execute_trade_closes_long_position(rl_strategy, mock_broker):
    rl_strategy.current_position = {"dealId": "123", "direction": "BUY"}
    await rl_strategy._execute_trade(Position.FLAT)
    mock_broker.close_position.assert_called_once_with("123")


@pytest.mark.asyncio
async def test_execute_main_loop_single_pass(rl_strategy, mock_agent, mock_broker):
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = lambda _: setattr(rl_strategy, "_is_running", False)

        rl_strategy.start()
        await rl_strategy.execute()

        mock_broker.get_historical_data.assert_called_once()
        mock_agent.predict.assert_called_once()
        mock_broker.get_open_positions.assert_called_once()
        mock_broker.open_position.assert_called_once_with("BUY", "EUR/USD", 10000) 