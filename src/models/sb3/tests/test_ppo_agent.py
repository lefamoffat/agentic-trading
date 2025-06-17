import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.environments.trading_env import TradingEnv
from src.models.sb3.ppo import PPOAgent


class TestPPOAgent(unittest.TestCase):
    """Unit tests for the PPOAgent."""

    def setUp(self):
        """Set up the test environment."""
        dummy_data = pd.DataFrame({
            "open": [1.0, 1.1, 1.2], "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1], "close": [1.1, 1.2, 1.1],
            "volume": [100, 110, 120]
        })
        self.mock_env = TradingEnv(data=dummy_data)

        config_path = Path("config/agent_config.yaml")
        with open(config_path, "r") as f:
            self.expected_params = yaml.safe_load(f)["ppo"]

    def test_agent_creation(self):
        """Test if the PPOAgent can be created successfully."""
        agent = PPOAgent(env=self.mock_env)
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent, PPOAgent)

    def test_model_creation_with_default_params(self):
        """Test if the underlying Stable Baselines 3 model is created with the
        correct default parameters from the config file.
        """
        agent = PPOAgent(env=self.mock_env)

        with patch("src.models.sb3.ppo.PPO") as MockPPO:
            agent._create_model()

            MockPPO.assert_called_once()

            _, kwargs = MockPPO.call_args
            for key, value in self.expected_params.items():
                self.assertIn(key, kwargs)
                self.assertEqual(kwargs[key], value)


if __name__ == "__main__":
    unittest.main()
