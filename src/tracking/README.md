# ML Tracking System

The ML tracking system provides comprehensive experiment logging, analysis, and model management for machine learning experiments. It offers a backend-agnostic design with rich functionality for tracking training metrics, hyperparameters, and model artifacts.

## 🎯 What Does This Do?

The tracking system is your **experiment lab notebook** that automatically records everything about your ML experiments:

-   **Training Metrics**: Loss, accuracy, rewards, and custom metrics over time
-   **Hyperparameters**: Learning rates, model architectures, training configurations
-   **Model Artifacts**: Trained models, checkpoints, and performance evaluations
-   **Experiment Comparison**: Compare different runs to find the best performing models
-   **Rich Visualizations**: Charts, graphs, and analysis tools for deep insights

Think of it as a **scientist's laboratory journal** that never forgets anything and can instantly show you patterns across hundreds of experiments.

## 🏗️ Architecture Overview

The tracking system uses a **generic, backend-agnostic design** that currently supports Aim but can easily add other backends:

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
│                      Aim Backend                            │
│              (WandB, MLflow ready for future)               │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

-   **Backend Independence**: Switch between tracking systems without changing application code
-   **Rich Experiment Data**: Comprehensive logging of all experiment aspects
-   **Persistent Storage**: Data survives system restarts and crashes
-   **Advanced Analysis**: Rich querying and comparison capabilities
-   **Production Ready**: Designed for high-throughput training workloads

## 🧩 Components

### Core Interfaces (Protocols)

**`MLTracker`** - Active Experiment Tracking

-   Start and manage individual experiment runs
-   Log metrics, parameters, and artifacts in real-time
-   Handle training callbacks and checkpoints

**`ExperimentRepository`** - Historical Data Access

-   Query and retrieve past experiments
-   Compare multiple experiments
-   Export and analyze experiment data

**`TrackingBackend`** - Backend Implementation

-   Interface for specific tracking systems (Aim, WandB, etc.)
-   Health monitoring and configuration management

### Data Models

**`TrainingMetrics`** - Real-time Training Data

```python
TrainingMetrics(
    step=1000,
    loss=0.045,
    reward=1250.5,
    episode_length=100,
    accuracy=0.892
)
```

**`ExperimentSummary`** - Experiment Overview

```python
ExperimentSummary(
    experiment_id="exp_20241201_001",
    name="PPO EUR/USD Training",
    status=ExperimentStatus.COMPLETED,
    start_time=datetime.now(),
    duration=timedelta(hours=2),
    final_metrics={"reward": 2150.0, "win_rate": 0.68}
)
```

**`ModelArtifact`** - Saved Models and Files

```python
ModelArtifact(
    name="best_model.zip",
    artifact_type=ArtifactType.MODEL,
    file_path="models/exp_123/best_model.zip",
    metadata={"validation_score": 0.95}
)
```

### Current Backend: Aim

**Aim Integration** provides:

-   Rich web UI for experiment visualization
-   Time-series metric tracking
-   Hyperparameter optimization support
-   Model comparison tools
-   Export capabilities

## 🚀 Quick Start

### Basic Experiment Tracking

```python
from src.tracking import get_ml_tracker

# Get a tracker instance
tracker = await get_ml_tracker()

# Start a new experiment run
run = await tracker.create_run(
    experiment_name="PPO_EUR_USD_Training",
    config={
        "agent_type": "PPO",
        "learning_rate": 0.0003,
        "symbol": "EUR/USD",
        "timesteps": 50000
    }
)

# Log metrics during training
await tracker.log_metrics(run, TrainingMetrics(
    step=1000,
    loss=0.045,
    reward=1250.5,
    episode_length=100
))

# Log custom metrics
await tracker.log_metric(run, "win_rate", 0.68, step=1000)

# Save model artifacts
await tracker.log_artifact(run, "best_model.zip", ArtifactType.MODEL)

# Finish the experiment
await tracker.finish_run(run, status=ExperimentStatus.COMPLETED)
```

### Experiment Repository (Analysis)

```python
from src.tracking import get_experiment_repository

# Get repository for querying experiments
repo = await get_experiment_repository()

# Get all experiments
experiments = await repo.list_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.status} - Final reward: {exp.final_metrics.get('reward', 'N/A')}")

# Get specific experiment
experiment = await repo.get_experiment("exp_20241201_001")
print(f"Duration: {experiment.duration}")
print(f"Final metrics: {experiment.final_metrics}")

# Get experiment metrics over time
metrics = await repo.get_experiment_metrics("exp_20241201_001")
for metric in metrics:
    print(f"Step {metric.step}: Loss={metric.value}")

# Compare experiments
experiments = await repo.list_experiments(status=ExperimentStatus.COMPLETED)
best_experiment = max(experiments, key=lambda x: x.final_metrics.get('reward', 0))
print(f"Best experiment: {best_experiment.name} with reward {best_experiment.final_metrics['reward']}")
```

### Stable-Baselines3 Integration

```python
from src.tracking.callbacks import TrackingCallback
from stable_baselines3 import PPO

# Create tracking callback
tracking_callback = TrackingCallback(
    experiment_name="PPO_Training",
    config={"learning_rate": 0.0003, "symbol": "EUR/USD"},
    eval_freq=1000  # Evaluate every 1000 steps
)

# Train with automatic tracking
model = PPO("MlpPolicy", env)
model.learn(
    total_timesteps=50000,
    callback=tracking_callback
)

# Best model is automatically saved
print(f"Best model saved at: {tracking_callback.best_model_path}")
```

## 🔧 Configuration

### Environment Variables

```bash
# Backend selection (currently only 'aim' supported)
ML_TRACKING_BACKEND=aim

# Aim configuration
ML_STORAGE_PATH=./aim_logs
ML_EXPERIMENT_NAME=AgenticTrading

# Optional: Custom repository location
AIM_REPO_PATH=/custom/path/to/aim/repo
```

### Programmatic Configuration

```python
from src.tracking.factory import configure_aim_backend

# Configure Aim backend
configure_aim_backend(
    repo_path="./my_experiments",
    experiment_name="My_Trading_Bot"
)

# Get configured tracker
tracker = await get_ml_tracker()
```

### Health Monitoring

```python
from src.tracking.factory import health_check

# Check tracking system health
health = await health_check()
print(f"Tracking system healthy: {health.is_healthy}")
if not health.is_healthy:
    print(f"Issues: {health.issues}")
```

## 📊 Real-World Example: Complete Training Workflow

Here's how the tracking system captures a complete training session:

### Step 1: Configure and Start Tracking

```python
# scripts/training/train_agent.py
from src.tracking import get_ml_tracker, TrainingMetrics, ExperimentStatus
from src.tracking.callbacks import TrackingCallback

# Start experiment tracking
tracker = await get_ml_tracker()
run = await tracker.create_run(
    experiment_name="PPO_EUR_USD_Optimization",
    config={
        "agent_type": "PPO",
        "learning_rate": 0.0003,
        "batch_size": 2048,
        "symbol": "EUR/USD",
        "timeframe": "1h",
        "timesteps": 100000,
        "environment": "TradingEnv-v1"
    }
)
```

### Step 2: Training with Automatic Logging

```python
from stable_baselines3 import PPO

# Create callback for automatic tracking
callback = TrackingCallback(
    experiment_name="PPO_EUR_USD_Optimization",
    config=config,
    eval_freq=2000,  # Evaluate every 2000 steps
    save_freq=10000  # Save model every 10000 steps
)

# Train with automatic tracking
model = PPO("MlpPolicy", env, learning_rate=0.0003)
model.learn(
    total_timesteps=100000,
    callback=callback
)

# Training automatically logs:
# - Training metrics every step
# - Evaluation metrics every 2000 steps
# - Model checkpoints every 10000 steps
# - Best model when training completes
```

### Step 3: Manual Metrics (Custom Logic)

```python
# Custom metrics during training
for episode in range(num_episodes):
    episode_reward = 0
    trades_made = 0

    # ... trading logic ...

    # Log custom business metrics
    await tracker.log_metrics(run, TrainingMetrics(
        step=episode,
        reward=episode_reward,
        trades_count=trades_made,
        win_rate=wins / total_trades if total_trades > 0 else 0,
        sharpe_ratio=calculate_sharpe_ratio(returns),
        max_drawdown=calculate_max_drawdown(portfolio_values)
    ))
```

### Step 4: Complete and Analyze

```python
# Mark experiment as completed
await tracker.finish_run(run,
    status=ExperimentStatus.COMPLETED,
    final_metrics={
        "total_reward": 5250.75,
        "win_rate": 0.684,
        "sharpe_ratio": 1.42,
        "max_drawdown": 0.088
    }
)

# Analysis and comparison
repo = await get_experiment_repository()
all_experiments = await repo.list_experiments()

# Find best performing experiment
best_exp = max(all_experiments, key=lambda x: x.final_metrics.get('sharpe_ratio', 0))
print(f"Best Sharpe ratio: {best_exp.final_metrics['sharpe_ratio']:.3f}")

# Get detailed metrics for analysis
metrics = await repo.get_experiment_metrics(best_exp.experiment_id)
rewards = [m.value for m in metrics if m.name == 'reward']
print(f"Reward progression: {rewards[:5]} ... {rewards[-5:]}")
```

## 🎨 Aim Web UI

When using the Aim backend, you get a rich web interface:

### Launch Aim UI

```bash
# Start Aim web interface
uv run aim up

# Open browser to http://localhost:43800
# View all experiments, metrics, and comparisons
```

### Features Available in Aim UI:

-   **Experiment Dashboard**: Overview of all runs
-   **Metrics Visualization**: Interactive charts and graphs
-   **Hyperparameter Analysis**: Compare parameter effects
-   **Model Comparison**: Side-by-side experiment comparison
-   **Export Tools**: Export data and visualizations

## 🧪 Testing

The tracking system includes comprehensive tests:

```bash
# Run tracking tests
uv run pytest src/tracking/tests/ -v

# Test specific components
uv run pytest src/tracking/tests/test_factory.py      # Configuration
uv run pytest src/tracking/tests/test_callbacks.py   # SB3 integration
uv run pytest src/tracking/tests/test_utils.py       # Utility functions
uv run pytest src/tracking/tests/test_protocols.py   # Interface compliance

# Integration tests with real Aim backend
uv run pytest integration_tests/test_aim_integration.py
```

## 🔍 Troubleshooting

### Common Issues

**Problem:** "No tracking backend configured"

```python
# Solution: Configure the backend
from src.tracking.factory import configure_aim_backend
configure_aim_backend()

# Or set environment variable
export ML_TRACKING_BACKEND=aim
```

**Problem:** Aim UI not showing experiments

```bash
# Check Aim repository location
ls -la ./aim_logs  # Should contain .aim directory

# Verify experiments are being logged
python -c "
import asyncio
from src.tracking import get_experiment_repository

async def check():
    repo = await get_experiment_repository()
    experiments = await repo.list_experiments()
    print(f'Found {len(experiments)} experiments')

asyncio.run(check())
"
```

**Problem:** High memory usage during training

```python
# Solution: Reduce logging frequency
callback = TrackingCallback(
    eval_freq=5000,  # Increase interval
    log_interval=100  # Log less frequently
)
```

### Performance Tips

1. **Batch metric logging** - Use `log_metrics()` instead of multiple `log_metric()` calls
2. **Reasonable logging frequency** - Don't log every single step for long training runs
3. **Clean up old experiments** - Archive or delete unused experiment data
4. **Monitor storage** - Aim repositories can grow large with many experiments

## 🔗 Integration with Messaging

The tracking system works together with the messaging system to provide complete experiment data management:

-   **Tracking**: Persistent experiment data and analysis (this system)
-   **Messaging**: Real-time communication and temporary state ([Messaging README](../messaging/README.md))

**Key Integration Points:**

-   Detailed metrics logging (tracking) + Real-time progress updates (messaging)
-   Historical analysis (tracking) + Live experiment monitoring (messaging)
-   Permanent experiment records (tracking) + Temporary session state (messaging)

**See Also:**

-   **[Experiment Data Management Guide](../docs/experiment_data_management.md)** - Complete guide on how both systems work together
-   **[Messaging System README](../messaging/README.md)** - Detailed messaging system documentation
-   **[Development Guide](../docs/development.md)** - Architecture guidelines for both systems

## 🚀 Future Enhancements

The generic design enables easy addition of new backends:

### Planned Backends

-   **Weights & Biases** - Cloud-based experiment tracking
-   **MLflow** - Open-source ML lifecycle management
-   **TensorBoard** - Google's visualization toolkit

### Adding New Backends

```python
# Example: Adding WandB backend
class WandBBackend(TrackingBackend):
    async def create_run(self, experiment_name: str, config: dict) -> ExperimentRun:
        # Implementation for WandB
        pass

    # ... implement other TrackingBackend methods
```

The application code remains unchanged when new backends are added!
