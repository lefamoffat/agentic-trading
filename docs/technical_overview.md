# Technical Overview

This document provides a high-level overview of the key technical components of the Agentic Trading system.

## Project Structure

The codebase is organized into several key directories:

```
agentic-trading/
├── configs/                 # YAML configuration files
├── src/                    # Main source code
│   ├── agents/            # RL agent implementations (Stable-Baselines3)
│   ├── brokers/           # Broker integrations (forex.com working)
│   ├── market_data/       # Centralized market data handling
│   ├── data/              # Data utilities & market calendars
│   ├── environment/       # Dynamic, model-agnostic RL environment
│   ├── tracking/          # ML tracking
│   ├── intelligence/      # LLM-based configuration intelligence
│   ├── callbacks/         # Training callbacks
│   ├── utils/             # Core utilities
│   └── types/             # Centralized type definitions
├── apps/                  # Web applications
│   ├── dashboard/         # Multi-page Dash dashboard
│   └── cli/               # Typer-powered CLI interface
├── scripts/               # Executable scripts
│   ├── training/          # RL agent training scripts
│   ├── dashboard/         # Dashboard launch scripts
│   └── setup/             # Project initialization
├── data/                  # Data storage (raw, processed, qlib, models)
├── integration_tests/     # Integration tests
└── docs/                  # Documentation
```

For a more detailed breakdown, refer to the [Architecture Guide](architecture.md).

## Market Data & Feature Engineering

### Centralized Market Data Module

The system uses a centralized `src/market_data/` module that provides:

-   **Source-agnostic data fetching** from brokers, APIs, and other providers
-   **Intelligent caching** to avoid re-downloading data
-   **Qlib binary format integration** for efficient feature engineering
-   **Type-safe data contracts** using Pydantic models
-   **Date range based data retrieval** inspired by Alpaca's API

### Feature Engineering with Qlib

Feature engineering is handled exclusively by **Microsoft Qlib**, providing access to a vast collection of technical indicators and alpha factors, including the renowned `Alpha158` and `Alpha360` collections.

The system integrates Qlib through the market_data module, which automatically handles:

-   Data format conversion to Qlib's binary format
-   Feature calculation using Qlib's alpha libraries
-   Seamless integration with the training pipeline

### Simplified Data Preparation

The old multi-step process has been replaced with intelligent CLI commands:

```bash
# Previous multi-step approach
scripts/data/download_historical.py → scripts/data/dump_bin.py → scripts/features/build_features.py

# Current streamlined approach
uv run agentic download-data    # Downloads with intelligent caching
uv run agentic prepare-data     # Complete pipeline: download → qlib → features
```

The new system includes:

-   **Intelligent Caching**: Automatic cache hits avoid redundant downloads
-   **Clean Directory Structure**: Organized data storage in `data/cache/`, `data/qlib_data/`, etc.
-   **Type-Safe Contracts**: Pydantic models ensure data consistency

## ML Tracking Architecture

### Current Implementation: Aim-Only with Generic Design

The system implements a **generic ML tracking architecture** designed for easy backend switching, but **currently only supports Aim backend**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│         (NEVER references specific backends)                │
├─────────────────────────────────────────────────────────────┤
│  Training Service  │  Dashboard Service  │  Callbacks       │
├─────────────────────────────────────────────────────────────┤
│                 Generic Tracking Interface                  │
│  MLTracker  │  ExperimentRepository  │  TrainingMetrics     │
├─────────────────────────────────────────────────────────────┤
│                    src/tracking/ Module                     │
│              (Backend-specific code allowed)                │
├─────────────────────────────────────────────────────────────┤
│                      Aim Backend ONLY                       │
│              (WandB, MLflow ready for implementation)       │
└─────────────────────────────────────────────────────────────┘
```

**Current State**: Only `src/tracking/backends/aim_backend.py` is implemented. The architecture supports adding WandB, MLflow, TensorBoard, etc. in the `backends/` directory, but these are not yet implemented.

### Backend Isolation Rules

**✅ ALLOWED:**

-   Backend-specific imports **within** `src/tracking/` module
-   Backend-specific configuration functions in `src/tracking/factory.py`
-   Backend-specific tests in `integration_tests/` (testing outside module)

**❌ NOT ALLOWED:**

-   Aim/MLflow/WandB references **outside** `src/tracking/` module
-   Application code using backend-specific APIs

### Configuration (Aim Backend Only)

```bash
# Production - Aim backend (only option currently)
export ML_STORAGE_PATH=./aim_logs
export ML_EXPERIMENT_NAME="AgenticTrading"
```

### Generic Usage Pattern

```python
# ✅ CORRECT: Generic usage (anywhere in codebase)
from src.tracking import get_ml_tracker, get_experiment_repository
tracker = await get_ml_tracker()  # Backend determined by config

# ❌ WRONG: Backend-specific usage (outside src/tracking/)
from src.tracking.factory import configure_aim_backend  # DON'T DO THIS
```

## Dashboard & Monitoring

### Multi-Page Dashboard

The system includes a comprehensive web dashboard built with Dash and Plotly:

-   **Overview Page**: System status, recent experiments, performance metrics, and top model leaderboard
-   **Experiments Page**: Browse and filter all training runs with comparison charts
-   **Single Experiment**: Detailed analysis of individual training runs with metrics and parameters
-   **Single Model**: Model management with live trading controls and backtesting interface
-   **Data Pipeline**: Real-time monitoring of data sources, quality metrics, and cache status

### Real-time Features

-   Auto-refresh intervals (15-30 seconds) for live data updates
-   Interactive Plotly charts with filtering and sorting
-   Bootstrap-based responsive design
-   Generic ML tracking integration with graceful backend fallbacks
-   Error handling and graceful degradation

### Backend Integration

The dashboard integrates both messaging and tracking systems:

-   **Real-time Updates**: Live experiment monitoring through messaging system
-   **Rich Experiment Data**: Full access to Aim's experiment tracking capabilities
-   **Unified Interface**: Combines live data (messaging) with historical analysis (tracking)
-   **Graceful Fallbacks**: UI remains functional even if backends are unavailable

**See Also**: [Experiment Data Management Guide](experiment_data_management.md) for complete integration details

### Launch Dashboard

```bash
# Launch dashboard (works with Aim backend)
uv run python scripts/dashboard/launch_dashboard.py

# Optional: Start Aim UI for detailed experiment analysis
uv run aim up

# Access dashboard at http://localhost:8050
# Aim UI available at http://localhost:43800 (if started)
```

## CLI Interface

### Typer-Powered CLI

The system includes a comprehensive CLI built with Typer:

```bash
# Complete workflow
uv run agentic quickstart --symbol "EUR/USD" --timeframe 1h --timesteps 20000

# Individual commands
uv run agentic download-data --symbol "EUR/USD" --timeframe 1h --bars 365
uv run agentic prepare-data --symbol "EUR/USD" --timeframe 1h --bars 365
uv run agentic train --symbol "EUR/USD" --timeframe 1h --timesteps 20000
uv run agentic optimize --symbol "EUR/USD" --trials 10
```

### Benefits

-   **Zero Boilerplate**: Complete workflows in single commands
-   **Intelligent Caching**: Automatic cache management
-   **Type Safety**: Full validation and error handling
-   **Progress Tracking**: Real-time progress bars and status updates

## Testing Architecture

Our comprehensive test suite follows strict testing patterns and automatically detects system issues:

### Test Structure

Following `@testing.mdc` patterns:

```
src/*/tests/              # Unit/component tests (module-only, no external deps)
integration_tests/        # Integration tests (cross-module, external deps, Aim backend)
```

### Test Commands

```bash
# Run by test type (following @testing.mdc markers)
uv run pytest -m unit                    # Pure logic tests, <50ms
uv run pytest -m component               # Component tests with mocks, <1s
uv run pytest -m integration             # Integration tests with real backends

# Health monitoring validation (catches schema issues)
uv run pytest integration_tests/test_backend_schema_validation.py

# CLI integration testing
uv run pytest integration_tests/test_cli_health_monitoring.py
```

### Key Testing Features

-   ✅ **Automated Issue Detection**: Would catch SQLite schema issues like "no such table: run" automatically
-   ✅ **Health Monitoring**: Comprehensive backend health validation with error recovery testing
-   ✅ **Module Isolation**: Unit tests stay within modules, integration tests for cross-module behavior
-   ✅ **CLI Integration**: Complete workflow testing with health checks and error detection

## Key Technical Benefits

### ML Tracking with Aim Backend

-   **Rich Experiment Tracking**: Comprehensive experiment logging and visualization with Aim
-   **Trading-Specific Metrics**: Specialized metrics for financial applications
-   **Clean Interface**: Generic protocols that keep application code backend-agnostic
-   **Production Ready**: Aim backend with rich visualizations and analysis
-   **Extensible Design**: Ready for additional backends (WandB, MLflow, TensorBoard)

### Dynamic Environment System

-   **Model Agnostic**: Works with any ML framework (SB3, PyTorch, LLMs, human input)
-   **Configurable Observations**: Any combination of market features and technical indicators
-   **Time-Aware Trading**: Respects market hours and trading sessions
-   **Flexible Actions**: Support for discrete, continuous, and string-based actions

### Event-Driven Architecture

-   **Real-time Messaging**: Event-driven system with <100ms latency
-   **Scalable Design**: Redis pub/sub for production scalability
-   **Memory Broker**: In-memory messaging for development
-   **Health Monitoring**: System health checks and diagnostics
