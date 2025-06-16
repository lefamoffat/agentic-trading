import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.callbacks.metrics_callback import MlflowMetricsCallback
from src.environments.wrappers import EvaluationWrapper

@pytest.fixture
def mock_base_env():
    """Fixture for a mock environment that the wrapper will use."""
    env = MagicMock()
    env.reset.return_value = (np.array([0]), {})
    env.step.return_value = (np.array([0]), 0, True, False, {}) # terminated
    return env

@patch('src.callbacks.metrics_callback.calculate_performance_metrics')
@patch('mlflow.log_metrics')
@pytest.mark.unit
class TestMlflowMetricsCallback:
    """Test suite for the MlflowMetricsCallback."""

    def test_callback_logs_metrics(self, mock_log_metrics, mock_calc_metrics, mock_base_env):
        """
        Test that the callback runs evaluation and logs all calculated metrics
        to both the SB3 logger and MLflow.
        """
        # 1. Setup mocks and test data
        mock_calc_metrics.return_value = {
            "sharpe_ratio": 1.5,
            "profit_pct": 10.0,
            "win_rate_pct": 60.0,
        }
        
        callback = MlflowMetricsCallback(
            eval_env=mock_base_env,
            eval_freq=10,
            n_eval_episodes=2,
            timeframe="1h"
        )
        
        # Mock the model and its logger, required by the BaseCallback
        mock_model = MagicMock()
        mock_model.logger = MagicMock()
        mock_model.logger.record = MagicMock()
        mock_model.predict.return_value = (0, None)  # Mock predict to return a tuple
        callback.model = mock_model
        
        # 2. Trigger the callback's evaluation logic
        callback.n_calls = 10
        callback._on_step()

        # 3. Assertions
        # It should run evaluation for each episode
        assert mock_base_env.reset.call_count == 2
        
        # It should calculate metrics for each episode
        assert mock_calc_metrics.call_count == 2
        
        # The first argument to the metric calculation should be the portfolio values
        # collected by the wrapper. The wrapper gets this from the 'info' dict.
        # Since we can't easily inspect the wrapper's internal state, we
        # trust the wrapper test and focus on the call signature here.
        args, kwargs = mock_calc_metrics.call_args
        assert isinstance(args[0], list)  # portfolio_values
        assert isinstance(args[1], list)  # trade_history
        assert args[2] == "1h"           # timeframe
        
        # It should log the averaged metrics to MLflow
        expected_logged_metrics = {"sharpe_ratio": 1.5, "profit_pct": 10.0, "win_rate_pct": 60.0}
        mock_log_metrics.assert_called_once_with(expected_logged_metrics, step=10)
        
        # It should also log each metric to the SB3 logger
        assert callback.model.logger.record.call_count == 3
        callback.model.logger.record.assert_any_call("eval/sharpe_ratio", 1.5)
        callback.model.logger.record.assert_any_call("eval/profit_pct", 10.0)
        callback.model.logger.record.assert_any_call("eval/win_rate_pct", 60.0) 