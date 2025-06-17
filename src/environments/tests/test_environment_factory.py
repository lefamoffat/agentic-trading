import pandas as pd
import pytest

from src.environments.factory import environment_factory
from src.environments.trading_env import TradingEnv


def _df():
    return pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [10]})


@pytest.mark.unit
class TestEnvironmentFactory:
    def test_create_default_env(self):
        env = environment_factory.create_environment("default", _df())
        assert isinstance(env, TradingEnv)

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError):
            environment_factory.create_environment("unknown", _df()) 