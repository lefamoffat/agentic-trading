import pandas as pd
import pytest

from src.environments.trading_env import TradingEnv
from src.models.sb3.factory import agent_factory


def _dummy_env():
    df = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [1, 1], "close": [1.5, 2.5], "volume": [100, 100]})
    return TradingEnv(df)


@pytest.mark.unit
class TestAgentFactory:
    def test_create_registered_agent(self):
        env = _dummy_env()
        agent = agent_factory.create_agent("PPO", env)
        assert agent.__class__.__name__ == "PPOAgent"

    def test_unregistered_agent_raises(self):
        env = _dummy_env()
        with pytest.raises(ValueError):
            agent_factory.create_agent("NON_EXISTENT", env)
