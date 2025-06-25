"""Configuration validation for training experiments."""

from typing import Dict, Any


def validate_config(config: Dict[str, Any]) -> None:
    """Validate training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Required fields
    required_fields = ["agent_type", "symbol", "timeframe", "timesteps"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate agent type
    valid_agent_types = ["ppo", "a2c", "sac", "dqn"]
    if config["agent_type"] not in valid_agent_types:
        raise ValueError(f"Invalid agent type: {config['agent_type']}. Must be one of: {valid_agent_types}")
    
    # Validate symbol format (basic check)
    symbol = config["symbol"]
    if len(symbol) < 3 or "/" not in symbol:
        raise ValueError(f"Invalid symbol: {symbol}. Expected format: 'EUR/USD'")
    
    # Validate timeframe
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    if config["timeframe"] not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {config['timeframe']}. Must be one of: {valid_timeframes}")
    
    # Validate timesteps
    if not isinstance(config["timesteps"], int) or config["timesteps"] <= 0:
        raise ValueError(f"Invalid timesteps: {config['timesteps']}. Must be a positive integer") 