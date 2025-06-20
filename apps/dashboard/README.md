# Agentic Trading Dashboard

A modern Dash-based multi-page dashboard for monitoring and managing the agentic trading system.

## Features

### 📊 Overview Page (`/`)

-   **System Status Cards**: Real-time monitoring of system components
-   **Recent Experiments**: Quick view of latest training runs
-   **Performance Trends**: Time series charts of model performance
-   **Top Models**: Leaderboard of best performing models

### 🧪 Experiments Page (`/experiments`)

-   **Experiment Browser**: Searchable table of all training runs
-   **Advanced Filtering**: Filter by status, agent type, symbol
-   **Performance Comparison**: Interactive charts comparing experiment results
-   **MLflow Integration**: Direct links to detailed experiment data

### 🔬 Single Experiment (`/experiment/{id}`)

-   **Detailed Metrics**: Training progress, hyperparameters, results
-   **Intelligence Decisions**: LLM/Agent decision tracking
-   **Training Visualizations**: Reward curves, loss plots
-   **Export Capabilities**: Download results and artifacts

### 🤖 Model Management (`/model/{id}`)

-   **Live Trading Controls**: Start/stop live trading sessions
-   **Backtesting Interface**: Configure and run backtests
-   **Demo Mode**: Safe simulation environment
-   **Model Performance**: Real-time metrics and analytics

### 🔄 Data Pipeline (`/data-pipeline`)

-   **Data Source Status**: Monitor all data connections
-   **Quality Metrics**: Data validation and quality scores
-   **Cache Management**: Hit rates, refresh controls
-   **Operations Log**: Recent data processing activities

## Quick Start

### Launch Dashboard

```bash
# Basic launch (port 8050)
uv run python scripts/dashboard/launch_dashboard.py

# Debug mode with custom port
uv run python scripts/dashboard/launch_dashboard.py --debug --port 8051

# Bind to specific host
uv run python scripts/dashboard/launch_dashboard.py --host 0.0.0.0 --port 8050
```

### Access Dashboard

-   Open your browser to `http://localhost:8050`
-   Navigate between pages using the top navigation bar
-   All pages auto-refresh to show real-time data

## Architecture

### Multi-Page Structure

```
apps/dashboard/
├── app.py              # Main application factory
├── components/         # Reusable UI components
│   └── navigation.py   # Navigation bar
└── pages/              # Individual page modules
    ├── overview.py     # System overview
    ├── experiments.py  # All experiments
    ├── single_experiment.py
    ├── single_model.py
    └── data_pipeline.py
```

### Technology Stack

-   **Dash**: Web application framework
-   **Plotly**: Interactive visualizations
-   **Bootstrap**: UI styling via dash-bootstrap-components
-   **MLflow**: Experiment tracking integration
-   **Pandas**: Data manipulation and display

## Features

### Real-Time Updates

-   Auto-refresh intervals for live data
-   WebSocket support for instant updates
-   Configurable refresh rates per page

### MLflow Integration

-   Direct connection to MLflow tracking server
-   Experiment metadata and metrics
-   Model registry integration
-   Artifact download links

### Responsive Design

-   Mobile-friendly interface
-   Bootstrap-based responsive layouts
-   Optimized for desktop and tablet viewing

## Development

### Adding New Pages

1. Create page module in `apps/dashboard/pages/`
2. Define `layout()` function returning Dash components
3. Add route in `app.py` routing callback
4. Update navigation in `components/navigation.py`

### Custom Components

-   Place reusable components in `apps/dashboard/components/`
-   Follow Bootstrap styling conventions
-   Use Dash callback decorators for interactivity

### Testing

```bash
# Test dashboard imports
uv run python -c "from apps.dashboard import create_app; print('✅ Dashboard OK')"

# Run in debug mode
uv run python scripts/dashboard/launch_dashboard.py --debug
```

## Configuration

### Environment Variables

-   `MLFLOW_TRACKING_URI`: MLflow server URL
-   `MLFLOW_EXPERIMENT_NAME`: Default experiment name
-   `LOG_LEVEL`: Logging verbosity

### Dashboard Settings

-   Modify refresh intervals in page modules
-   Customize styling in component files
-   Update data sources in data pipeline page

---

**Ready to monitor your trading system!** 🚀
