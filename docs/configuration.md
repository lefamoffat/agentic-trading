# Configuration

The system uses a combination of YAML files and environment variables for configuration.

## YAML Configuration Files

All YAML configuration files are located in the `config/` directory.

-   **`agent_config.yaml`**: Contains settings for the reinforcement learning agent, including the algorithm's hyperparameters (e.g., learning rate, batch size for PPO) and training-specific parameters like evaluation frequency.

-   **`data_config.yaml`**: Defines data sources, API credentials (if not using environment variables), and other data-related settings.

-   **`trading_config.yaml`**: Holds parameters related to the trading logic, such as initial balance, trade fees, and risk management settings like stop-loss or take-profit levels.

-   **`qlib_config.yaml`**: Used for initializing Qlib. It specifies the data provider, feature set, and other settings for the backtesting and data analysis engine.

## Environment Variables (.env)

Sensitive information and environment-specific settings are managed via a `.env` file in the project root. A `.env.example` file is provided as a template.

```bash
# API Keys
FOREX_COM_USERNAME=your_username_here
FOREX_COM_PASSWORD=your_password_here
FOREX_COM_APP_KEY=your_app_key_here
FOREX_COM_SANDBOX=true # Use 'true' for paper trading, 'false' for live

# Environment
ENVIRONMENT=development # Can be 'development', 'production', or 'testing'
LOG_LEVEL=INFO # e.g., DEBUG, INFO, WARNING, ERROR

# Trading
POSITION_SIZE=10000
MAX_DRAWDOWN=0.10
STOP_LOSS=0.02
```
