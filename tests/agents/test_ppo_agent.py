import unittest
from unittest.mock import MagicMock
import pandas as pd
from pathlib import Path
import yaml

from src.agents.ppo_agent import PPOAgent
from src.environments.trading_env import TradingEnv

class TestPPOAgent(unittest.TestCase):
    """Unit tests for the PPOAgent."""

    def setUp(self):
        """Set up the test environment."""
        # Create a dummy environment for the agent
        dummy_data = pd.DataFrame({
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.1, 1.2, 1.1],
            "volume": [100, 110, 120]
        })
        self.mock_env = TradingEnv(data=dummy_data)
        
        # Load expected params from the config file to verify against
        config_path = Path("config/agent_config.yaml")
        with open(config_path, "r") as f:
            self.expected_params = yaml.safe_load(f)["ppo"]

    def test_agent_creation(self):
        """Test if the PPOAgent can be created successfully."""
        agent = PPOAgent(env=self.mock_env)
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent, PPOAgent)

    def test_load_model_params(self):
        """Test that the agent correctly loads parameters from the config file."""
        agent = PPOAgent(env=self.mock_env)
        loaded_params = agent._load_model_params("ppo")
        self.assertIsNotNone(loaded_params)
        self.assertEqual(loaded_params["learning_rate"], 0.0003)
        self.assertEqual(loaded_params["policy"], "MlpPolicy")

    def test_model_creation_with_default_params(self):
        """
        Test if the underlying Stable Baselines 3 model is created with the
        correct default parameters from the config file.
        """
        agent = PPOAgent(env=self.mock_env)
        
        # Since we mock PPO, we must also mock inspect.signature to return
        # the expected arguments, so the internal filtering logic works.
        with unittest.mock.patch("src.agents.ppo_agent.PPO") as MockPPO, \
             unittest.mock.patch("inspect.signature") as mock_signature:

            # Create a mock signature object that behaves as needed
            mock_sig_obj = unittest.mock.MagicMock()
            mock_sig_obj.parameters.keys.return_value = self.expected_params.keys()
            mock_signature.return_value = mock_sig_obj

            agent._create_model()
            
            # Assert that the PPO constructor was called once.
            MockPPO.assert_called_once()
            
            # Assert that the parameters passed to the constructor match the expected ones.
            _, kwargs = MockPPO.call_args
            for key, value in self.expected_params.items():
                self.assertIn(key, kwargs)
                self.assertEqual(kwargs[key], value)

if __name__ == "__main__":
    unittest.main() 