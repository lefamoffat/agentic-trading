# Messaging System

The messaging system provides real-time communication between different components of the Agentic Trading platform. It enables live updates during training, real-time dashboard monitoring, and efficient data exchange between services.

## 🎯 What Does This Do?

The messaging system acts as a **real-time communication hub** that allows different parts of the application to send and receive messages instantly. Think of it like a WhatsApp group chat, but for your trading algorithms:

-   **Training scripts** can broadcast their progress to anyone listening
-   **Dashboard** can show live updates without constantly refreshing
-   **CLI tools** can monitor what's happening in real-time
-   **Multiple experiments** can run simultaneously without interfering with each other

## 🏗️ Architecture Overview

The messaging system has two main purposes:

### 1. **Pub/Sub Messaging** (Real-time Communication)

```
Training Process  ──publish──►  Message Broker  ──subscribe──►  Dashboard
       │                              │                              │
       │                              ▼                              │
       └──── "Training started" ──►  Redis/Memory  ◄──── "Show progress bar"
```

### 2. **Data Storage** (Experiment State)

```
Training Process  ──store──►  Message Broker  ──retrieve──►  Dashboard
       │                          │                             │
       │                          ▼                             │
       └─ experiment_123 ──►  Redis/Memory  ◄─── "Get experiment data"
```

## 🧩 Components

### Message Brokers

**`MemoryBroker`** - Development & Testing

-   Stores everything in RAM
-   Perfect for development and testing
-   No external dependencies
-   Data lost when application stops

**`RedisBroker`** - Production

-   Uses Redis for persistent storage
-   Handles high-throughput messaging
-   Data survives application restarts
-   Supports distributed systems

### Channels (Business Logic Wrappers)

**`TrainingChannel`** - Training-Specific Operations

-   Manages training experiment lifecycle
-   Publishes training events (start, progress, completion, errors)
-   Stores experiment metadata
-   Provides training-specific subscriptions

**`BaseChannel`** - Generic Channel Operations

-   Topic namespacing (`namespace.subtopic`)
-   Storage key generation (`namespace:resource:id:field`)
-   Common publishing and subscription patterns

### Events (Structured Data)

**Training Events:**

-   `StatusUpdateEvent` - Training started/completed/failed
-   `ProgressUpdateEvent` - Current step, epoch, progress percentage
-   `MetricsUpdateEvent` - Loss, accuracy, performance metrics
-   `LogUpdateEvent` - Training logs and debug information
-   `ErrorEvent` - Error messages and stack traces

## 🚀 Quick Start

### Basic Usage

```python
from src.messaging.factory import get_message_broker

# Get a message broker (Memory for dev, Redis for production)
broker = await get_message_broker()

# Publish a message
await broker.publish("training.status", {
    "experiment_id": "exp_123",
    "status": "running",
    "progress": 0.45
})

# Subscribe to messages
async for message in broker.subscribe("training.*"):
    print(f"Received: {message.topic} - {message.data}")
```

### Training Integration

```python
from src.messaging.channels.training import TrainingChannel

# Create a training channel
channel = TrainingChannel(namespace="experiment_123")

# Start an experiment
await channel.create_experiment("exp_123", {
    "agent_type": "PPO",
    "symbol": "EUR/USD",
    "timesteps": 10000
})

# Update progress during training
await channel.publish_progress("exp_123",
    current_step=500,
    total_steps=10000,
    epoch=5
)

# Update metrics
await channel.publish_metrics("exp_123", {
    "loss": 0.045,
    "reward": 1250.50,
    "episode_length": 100
})

# Mark as completed
await channel.publish_status("exp_123", "completed", "Training finished successfully")
```

### Dashboard Monitoring

```python
from src.messaging.channels.training import TrainingChannel

# Monitor all training experiments
channel = TrainingChannel()

# Get experiment data
experiment = await channel.get_experiment("exp_123")
print(f"Status: {experiment['status']}")
print(f"Progress: {experiment['progress']}%")

# Subscribe to live updates
async for message in channel.subscribe_to_status_updates():
    if message.data['experiment_id'] == 'exp_123':
        print(f"Update: {message.data['status']}")
```

## 🔧 Configuration

### Environment Variables

```bash
# Message broker type
MESSAGE_BROKER_TYPE=memory        # or 'redis'

# Redis configuration (when using RedisBroker)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password      # optional
```

### Programmatic Configuration

```python
from src.messaging.factory import get_message_broker
from src.messaging.config import MessageBrokerConfig

# Memory broker with custom settings
config = MessageBrokerConfig(
    broker_type="memory",
    max_queue_size=10000,
    max_subscribers=100
)
broker = await get_message_broker(config)

# Redis broker with custom settings
config = MessageBrokerConfig(
    broker_type="redis",
    redis_host="redis.example.com",
    redis_port=6379,
    redis_db=1
)
broker = await get_message_broker(config)
```

## 📊 Real-World Example: Training Monitoring

Here's how the messaging system works in practice during a training session:

### Step 1: Training Process Starts

```python
# scripts/training/train_agent.py
from src.messaging.channels.training import TrainingChannel

channel = TrainingChannel(namespace="ppo_training")

# Create experiment record
await channel.create_experiment("exp_20241201_001", {
    "agent_type": "PPO",
    "symbol": "EUR/USD",
    "timesteps": 50000,
    "learning_rate": 0.0003
})

# Notify that training started
await channel.publish_status("exp_20241201_001", "running", "PPO training started")
```

### Step 2: Training Loop Updates

```python
# During training loop
for step in range(50000):
    # ... training logic ...

    if step % 100 == 0:  # Update every 100 steps
        await channel.publish_progress("exp_20241201_001",
            current_step=step,
            total_steps=50000,
            epoch=step // 2048
        )

    if step % 500 == 0:  # Metrics every 500 steps
        await channel.publish_metrics("exp_20241201_001", {
            "loss": current_loss,
            "reward": episode_reward,
            "epsilon": current_epsilon
        })
```

### Step 3: Dashboard Shows Live Updates

```python
# apps/dashboard/callbacks.py
@app.callback(Output('progress-bar', 'value'), Input('interval', 'n_intervals'))
def update_progress(n):
    channel = TrainingChannel()
    experiment = await channel.get_experiment("exp_20241201_001")
    return experiment.get('progress', 0)
```

### Step 4: CLI Monitoring

```bash
# Real-time monitoring from CLI
uv run python -c "
import asyncio
from src.messaging.channels.training import TrainingChannel

async def monitor():
    channel = TrainingChannel()
    async for event in channel.subscribe_to_progress_updates():
        print(f'Step {event.data["current_step"]}/{event.data["total_steps"]} - {event.data["progress"]:.1f}%')

asyncio.run(monitor())
"
```

## 🧪 Testing

The messaging system includes comprehensive tests covering all scenarios:

```bash
# Run messaging tests
uv run pytest src/messaging/tests/ -v

# Test specific components
uv run pytest src/messaging/tests/test_memory_broker.py    # Memory broker
uv run pytest src/messaging/tests/test_redis_broker.py     # Redis broker
uv run pytest src/messaging/tests/test_training_channel.py # Training logic
uv run pytest src/messaging/tests/test_factory.py          # Configuration
```

## 🔍 Troubleshooting

### Common Issues

**Problem:** Messages not being received

```bash
# Check broker health
python -c "
import asyncio
from src.messaging.factory import get_message_broker

async def check():
    broker = await get_message_broker()
    health = await broker.health_check()
    print(f'Broker healthy: {health}')

asyncio.run(check())
"
```

**Problem:** Redis connection errors

```bash
# Verify Redis is running
redis-cli ping
# Should return: PONG

# Check Redis configuration
redis-cli config get "*"
```

**Problem:** Memory broker losing data

-   Switch to Redis broker for persistence
-   Memory broker is designed for development only

### Performance Tips

1. **Use pattern subscriptions wisely** - `training.*` is more efficient than multiple specific subscriptions
2. **Batch metrics updates** - Don't send metrics every single step, use intervals
3. **Clean up subscriptions** - Always close subscriptions when done
4. **Monitor memory usage** - Memory broker can consume RAM with high message volumes

## 🔗 Integration with Tracking

The messaging system works together with the tracking system to provide complete experiment data management:

-   **Messaging**: Real-time communication and temporary state (this system)
-   **Tracking**: Persistent experiment data and analysis ([Tracking README](../tracking/README.md))

**Key Integration Points:**

-   Real-time progress updates (messaging) + Detailed metrics logging (tracking)
-   Live experiment monitoring (messaging) + Historical analysis (tracking)
-   Temporary session state (messaging) + Permanent experiment records (tracking)

**See Also:**

-   **[Experiment Data Management Guide](../docs/experiment_data_management.md)** - Complete guide on how both systems work together
-   **[Tracking System README](../tracking/README.md)** - Detailed tracking system documentation
-   **[Development Guide](../docs/development.md)** - Architecture guidelines for both systems
