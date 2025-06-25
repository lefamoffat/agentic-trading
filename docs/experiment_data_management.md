# Experiment Data Management

This guide explains how the Agentic Trading platform manages experiment data using two complementary systems: **Messaging** for real-time communication and **Tracking** for persistent experiment storage. Together, they provide a complete solution for monitoring, analyzing, and managing your trading experiments.

## 🎯 The Big Picture

Imagine you're running a machine learning experiment to train a trading bot. You want to:

1. **See progress in real-time** while training is happening
2. **Store all experiment data** for later analysis
3. **Compare different experiments** to find the best model
4. **Monitor from multiple places** (dashboard, CLI, scripts)

The Agentic Trading platform uses **two specialized systems** working together to achieve this:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENT DATA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Training Process                                                               │
│       │                                                                         │
│       ├─── Real-time Updates ──►  MESSAGING SYSTEM   ──► Dashboard (Live)      │
│       │     (progress, status)      (Redis/Memory)        CLI (Monitoring)     │
│       │                                                                         │
│       └─── Experiment Data ────►  TRACKING SYSTEM   ──► Analysis Tools        │
│             (metrics, models)       (Aim Backend)        Model Comparison      │
│                                                           Historical Review     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 How They Work Together

### Messaging System: Real-Time Communication

-   **Purpose**: Live updates and immediate feedback
-   **Data Type**: Temporary, fast-changing information
-   **Storage**: Redis (production) or Memory (development)
-   **Lifespan**: Session-based (data may not persist between restarts)

### Tracking System: Persistent Experiment Data

-   **Purpose**: Long-term experiment analysis and model management
-   **Data Type**: Comprehensive experiment records
-   **Storage**: Aim backend (persistent files)
-   **Lifespan**: Permanent (survives restarts, can be analyzed weeks/months later)

## 📊 Data Flow Example

Let's walk through what happens when you train a PPO agent:

### Step 1: Training Starts

**Messaging System** broadcasts immediately:

```python
# Real-time notification
await channel.publish_status("exp_123", "running", "PPO training started")
```

**Tracking System** creates experiment record:

```python
# Persistent experiment record
run = await tracker.create_run(
    experiment_name="PPO_EUR_USD_Training",
    config={"agent_type": "PPO", "learning_rate": 0.0003}
)
```

### Step 2: Training Loop (Every Few Steps)

**Messaging System** sends live updates:

```python
# Dashboard sees this immediately
await channel.publish_progress("exp_123",
    current_step=1000,
    total_steps=50000,
    progress=2.0
)
```

**Tracking System** logs detailed metrics:

```python
# Stored permanently for analysis
await tracker.log_metrics(run, TrainingMetrics(
    step=1000,
    loss=0.045,
    reward=1250.5,
    episode_length=100
))
```

### Step 3: Training Completes

**Messaging System** notifies completion:

```python
# Everyone knows it's done
await channel.publish_status("exp_123", "completed", "Training finished successfully")
```

**Tracking System** stores final results:

```python
# Permanent record for comparison
await tracker.finish_run(run,
    status=ExperimentStatus.COMPLETED,
    final_metrics={"total_reward": 5250.75, "win_rate": 0.684}
)
```

## 🏗️ Practical Usage Patterns

### Pattern 1: Development Workflow

```python
from src.messaging.channels.training import TrainingChannel
from src.tracking import get_ml_tracker

# Set up both systems
channel = TrainingChannel(namespace="development")
tracker = await get_ml_tracker()

# Start experiment
experiment_id = "dev_exp_001"
await channel.create_experiment(experiment_id, config)
run = await tracker.create_run("Development_Test", config)

# During training loop
for step in range(timesteps):
    # ... training logic ...

    # Real-time updates (every 100 steps)
    if step % 100 == 0:
        await channel.publish_progress(experiment_id, step, timesteps)

    # Detailed logging (every 1000 steps)
    if step % 1000 == 0:
        await tracker.log_metrics(run, TrainingMetrics(
            step=step, loss=current_loss, reward=episode_reward
        ))

# Cleanup
await channel.publish_status(experiment_id, "completed")
await tracker.finish_run(run, ExperimentStatus.COMPLETED)
```

### Pattern 2: Dashboard Monitoring

```python
# Real-time dashboard updates (using messaging)
async def get_live_experiments():
    channel = TrainingChannel()
    experiments = await channel.list_experiments(status="running")
    return experiments

# Historical analysis (using tracking)
async def get_experiment_history():
    repo = await get_experiment_repository()
    experiments = await repo.list_experiments()
    return experiments

# Combined view: Live + Historical
async def get_dashboard_data():
    live_experiments = await get_live_experiments()      # From messaging
    historical_data = await get_experiment_history()     # From tracking

    return {
        "live": live_experiments,
        "history": historical_data,
        "comparison": analyze_performance(historical_data)
    }
```

## 🔧 Configuration for Both Systems

### Environment Setup

```bash
# Messaging Configuration
MESSAGE_BROKER_TYPE=redis          # or 'memory' for development
REDIS_HOST=localhost
REDIS_PORT=6379

# Tracking Configuration
ML_TRACKING_BACKEND=aim
ML_STORAGE_PATH=./aim_logs
ML_EXPERIMENT_NAME=AgenticTrading

# Start Aim UI for experiment visualization
uv run aim up  # Access at http://localhost:43800
```

## 📱 Accessing Your Data

### Dashboard (Real-time + Historical)

The dashboard combines both systems for a complete view:

```bash
# Launch dashboard
uv run python scripts/dashboard/launch_dashboard.py

# Access at http://localhost:8050
# Shows: Live experiments (messaging) + Historical analysis (tracking)
```

### Aim UI (Detailed Analysis)

For deep experiment analysis:

```bash
# Launch Aim UI
uv run aim up

# Access at http://localhost:43800
# Shows: Detailed metrics, comparisons, visualizations
```

This dual-system approach gives you the best of both worlds: **immediate feedback during training** and **comprehensive analysis capabilities** for optimizing your trading algorithms!
