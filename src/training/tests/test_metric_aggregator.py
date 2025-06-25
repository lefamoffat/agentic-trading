"""Unit tests for MetricAggregator."""

from src.training.callbacks.metric_aggregator import MetricAggregator


class TestMetricAggregator:
    """Test the metric aggregation logic."""
    
    def test_initialization(self):
        """Test aggregator initializes correctly."""
        aggregator = MetricAggregator()
        summary = aggregator.summary()
        assert summary == {}  # Empty when no updates
    
    def test_single_metric_update(self):
        """Test adding single metrics works correctly."""
        aggregator = MetricAggregator()
        aggregator.update({"reward": 10.0, "loss": 0.5})
        
        summary = aggregator.summary()
        assert summary["avg_reward"] == 10.0
        assert summary["avg_loss"] == 0.5
    
    def test_multiple_updates_calculate_averages(self):
        """Test that multiple updates calculate proper averages."""
        aggregator = MetricAggregator()
        
        aggregator.update({"reward": 10.0, "loss": 1.0})
        aggregator.update({"reward": 20.0, "loss": 0.8})
        aggregator.update({"reward": 30.0, "loss": 0.6})
        
        summary = aggregator.summary()
        assert summary["avg_reward"] == 20.0  # (10+20+30)/3
        assert abs(summary["avg_loss"] - 0.8) < 0.01  # (1.0+0.8+0.6)/3 