# Environment Variables Setup

Create a `.env` file in your project root with the following variables:

## Required API Keys

```bash
# Forex.com API credentials (GainCapital/StoneX API)
FOREX_COM_USERNAME=your_forex_com_username_here
FOREX_COM_PASSWORD=your_forex_com_password_here
```

## Environment Configuration

```bash
# Application environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Trading Configuration

```bash
# Trading parameters
POSITION_SIZE=10000
MAX_DRAWDOWN=0.10
STOP_LOSS=0.02
```

## Data Storage Paths

```bash
# Data directories
DATA_PATH=./data
QLIB_DATA_PATH=./data/qlib_format
```

## Model Storage

```bash
# Model directories
MODEL_PATH=./data/models
CHECKPOINT_PATH=./data/models/checkpoints
```

## Getting Forex.com API Credentials

1. **Contact Forex.com support** to request API access
2. **Forex.com uses GainCapital/StoneX API**
3. **API Documentation**: https://docs.labs.gaincapital.com/
4. **Obtain credentials:**
    - Username: Your trading login username
    - Password: Your trading login password
    - AppKey: Provided by GainCapital for API access

### For Testing/Development:

-   Use demo/sandbox environment for testing
-   Production trading requires live account credentials
-   API access typically requires professional/institutional account

## Symbol Mapping

The system uses a mapping layer between common symbols and broker-specific formats:

-   **Common format**: `EUR/USD`, `GBP/USD`, etc.
-   **Forex.com format**: `EUR_USD`, `GBP_USD`, etc.

This mapping is handled automatically by the `SymbolMapper` class.
