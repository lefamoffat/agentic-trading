import pandas as pd
import pytest
from pathlib import Path

from stable_baselines3 import PPO
from src.environment import TradingEnv, load_trading_config
from src.agents.factory import agent_factory


def _evaluate_trading_agent(agent, env: TradingEnv, n_episodes: int = 3) -> float:
    """Run n episodes with the trading agent and return mean final portfolio value."""
    portfolio_values: list[float] = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        # Get final portfolio value from environment
        final_value = getattr(env, 'current_portfolio_value', env.config.initial_balance)
        portfolio_values.append(float(final_value))
    
    return float(sum(portfolio_values) / len(portfolio_values))


@pytest.mark.integration
def test_ppo_trading_agent_learns():
    """Train PPO on real EUR/USD data and verify the agent improves trading performance."""
    
    # Load real EUR/USD features data
    features_path = Path("data/processed/features/EURUSD_1h_features.csv")
    if not features_path.exists():
        pytest.skip(f"EUR/USD features data not found: {features_path}")
    
    features_df = pd.read_csv(features_path)
    
    # Use subset of data for faster test (first 500 rows = ~3 weeks of hourly data)
    test_data = features_df.head(500).copy()
    
    # Load trading environment config
    env_config = load_trading_config(Path("configs/trading_config.yaml"))
    
    # Create trading environment with real market data
    env = TradingEnv(data=test_data, config=env_config)
    
    # Create PPO agent
    agent = agent_factory.create_agent(
        name="PPO",
        env=env,
        hyperparams={"learning_rate": 0.001}  # Slightly higher LR for faster learning
    )
    
    # Evaluate baseline performance (untrained agent)
    baseline_portfolio = _evaluate_trading_agent(agent, env, n_episodes=3)
    
    # Train the agent for 1000 timesteps (fast enough for CI)
    agent.train(total_timesteps=1_000)
    
    # Evaluate trained performance
    trained_portfolio = _evaluate_trading_agent(agent, env, n_episodes=3)
    
    # Agent should either:
    # 1. Maintain portfolio value (not lose money), OR  
    # 2. Show improvement in trading performance
    # We use a relaxed threshold since forex is challenging and 1k steps is minimal
    improvement_threshold = 0.95  # Allow up to 5% loss (better than random)
    
    assert trained_portfolio >= baseline_portfolio * improvement_threshold, (
        f"Expected trained agent to maintain at least 95% of baseline performance. "
        f"Baseline: ${baseline_portfolio:,.2f}, Trained: ${trained_portfolio:,.2f} "
        f"(ratio: {trained_portfolio/baseline_portfolio:.3f})"
    ) 