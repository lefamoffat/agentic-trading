from unittest.mock import MagicMock

import numpy as np
import pytest

from src.environments.trading_env import Trade
from src.environments.types import Position
from src.environments.wrappers import EvaluationWrapper


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.reset.return_value = (
        np.array([0]),
        {"portfolio_value": 10000.0, "trade_history": []},
    )
    env.step.return_value = (
        np.array([0]),
        0,
        True,
        False,
        {
            "portfolio_value": 10100.0,
            "trade_history": [
                Trade(
                    entry_price=100,
                    exit_price=101,
                    position=Position.LONG,
                    profit=100.0,
                )
            ],
        },
    )
    return env


@pytest.mark.unit
class TestEvaluationWrapper:
    def test_wrapper_collects_data_over_episode(self, mock_env):
        wrapper = EvaluationWrapper(mock_env)
        wrapper.reset()
        wrapper.step(0)

        assert wrapper.portfolio_values == [10000.0, 10100.0]
        assert len(wrapper.trade_history) == 1
        assert wrapper.trade_history[0].profit == 100.0

    def test_wrapper_reset_clears_data(self, mock_env):
        wrapper = EvaluationWrapper(mock_env)
        wrapper.reset()
        wrapper.step(0)

        assert len(wrapper.portfolio_values) == 2
        wrapper.reset()
        assert len(wrapper.portfolio_values) == 1
        assert len(wrapper.trade_history) == 0
