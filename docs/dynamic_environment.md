# Dynamic Environment System

The Agentic Trading system features a fully dynamic, model-agnostic environment that can adapt to any ML framework and configuration.

## 🎯 Key Features

### Model-Agnostic Actions

The environment accepts actions from **any** ML framework or input source:

```python
# Stable-Baselines3 (numpy arrays)
action = np.array([1])  # Long position
env.step(action)

# PyTorch tensors
action = torch.tensor([0])  # Close position
env.step(action)

# LLM string commands
action = "buy"  # or "sell", "close", "open_long", "open_short"
env.step(action)

# Human input
action = 2  # Short position (integer)
env.step(action)

# TradingAction enums
action = TradingAction.OPEN_LONG
env.step(action)
```

### Dynamic Observations

Observations are fully configurable via the trading configuration:

```python
# Configure any market features from your data
config = TradingEnvironmentConfig(
    observation_features=['close', 'volume', 'rsi', 'macd', 'custom_indicator'],
    include_time_features=True,    # Market hours, day of week
    include_portfolio_state=True,  # Balance, positions, PnL
)

# Resulting observation includes:
# - Market features: 5 (configurable)
# - Portfolio features: 4 (balance, position, entry_price, unrealized_pnl)
# - Time features: 9 (market_open, time_of_day, day_monday...day_sunday)
# Total: 18 features
```

### Time-Aware Trading

The environment understands trading hours and market sessions:

```yaml
# trading_config.yaml
trading_hours:
    start: "07:00"
    end: "17:00"
    timezone: "UTC"
    exclude_weekends: true
    exclude_holidays: true
```

The time observation component provides:

-   `market_open`: Binary flag (0=closed, 1=open)
-   `time_of_day_normalized`: Normalized hour (0.0-1.0)
-   `day_monday` through `day_sunday`: One-hot encoding of weekdays

## 🔧 Configuration System

### Complete YAML-to-Code Mapping

Every parameter in `trading_config.yaml` maps to `TradingEnvironmentConfig`:

```yaml
# trading_config.yaml
instrument:
    pip_value: 0.0001
    min_trade_size: 1000
    max_trade_size: 100000

risk:
    max_drawdown: 0.10
    stop_loss: 0.02
    take_profit: 0.04
    max_daily_loss: 0.05

trading_hours:
    start: "07:00"
    end: "17:00"
    exclude_weekends: true
```

```python
# Automatically mapped to Python config
config = load_trading_config("configs/trading_config.yaml")
print(config.pip_value)        # 0.0001
print(config.min_trade_size)   # 1000.0
print(config.max_drawdown)     # 0.10
print(config.trading_start_hour)  # 7
```

### Dynamic Feature Selection

Choose any combination of market features from your data:

```python
# Basic OHLCV
observation_features=['open', 'high', 'low', 'close', 'volume']

# With technical indicators (if present in data)
observation_features=['close', 'volume', 'rsi', 'macd', 'bollinger_upper']

# Custom features
observation_features=['price_momentum', 'volatility_regime', 'sentiment_score']
```

## 🏗️ Architecture

### Modular Components

```python
# Environment is composed of pluggable components
class TradingEnv:
    def __init__(self, data, config):
        # Model-agnostic action handler
        self.action_handler = DiscreteActionSpace()

        # Dynamic observation system
        self.observation_handler = CompositeObservation(
            market_features=config.observation_features,
            include_time_features=config.include_time_features
        )

        # Configurable reward system
        self.reward_calculator = CompositeReward(config.reward_system)
```

### Action Processing Pipeline

1. **Input Normalization**: Any input type → standard format
2. **Validation**: Ensure action is valid for current state
3. **Execution**: Convert to trading signal and execute
4. **Feedback**: Return observation, reward, termination status

### Observation Processing Pipeline

1. **Market Data**: Extract configured features from DataFrame
2. **Portfolio State**: Current balance, position, PnL
3. **Time Features**: Market hours, session information
4. **Normalization**: Scale all features appropriately
5. **Composition**: Combine into single observation vector

## 🎮 Usage Examples

### Basic Environment Creation

```python
from src.environment import TradingEnv, TradingEnvironmentConfig, FeeStructure

# Load configuration from YAML
config = TradingEnvironmentConfig(
    initial_balance=100000.0,
    fee_structure=FeeStructure.SPREAD_BASED,
    observation_features=['close', 'volume', 'high', 'low'],
    include_time_features=True
)

# Create environment with market data
env = TradingEnv(data=market_data, config=config)
```

### Training with Different ML Frameworks

```python
# Stable-Baselines3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)

# Manual control
obs, _ = env.reset()
action = "buy"  # String action
obs, reward, terminated, truncated, info = env.step(action)
```

### Custom Observations

```python
# Create environment with custom feature set
config = TradingEnvironmentConfig(
    observation_features=['close', 'custom_indicator', 'market_regime'],
    include_time_features=False,  # Disable time features
    include_portfolio_state=True   # Keep portfolio state
)

env = TradingEnv(data=data_with_custom_features, config=config)
print(f"Observation size: {env.observation_space.shape[0]}")
# Output: Observation size: 7 (3 market + 4 portfolio)
```

## 🧪 Testing & Validation

The dynamic environment system is thoroughly tested:

-   **35 observation tests**: Cover all feature combinations
-   **16 action tests**: Test all input frameworks
-   **21 environment tests**: End-to-end functionality
-   **Integration tests**: Real training workflows

All tests verify that:

-   Configuration changes flow through to behavior
-   Actions work across all frameworks
-   Observations adapt to any feature configuration
-   Time features respect trading hours

## 🚀 Benefits

1. **Future-Proof**: Works with any ML framework (current or future)
2. **Flexible**: Easy to add new features, actions, or reward functions
3. **Configurable**: Change behavior without code changes
4. **Testable**: Isolated components with comprehensive tests
5. **Production-Ready**: Robust error handling and validation
