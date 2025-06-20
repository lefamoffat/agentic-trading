"""Single Model Page.

Detailed view of a trained model with deployment and simulation capabilities.
"""

from typing import Optional

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go

from src.utils.logger import get_logger
from apps.dashboard.services.mlflow_service import mlflow_service

logger = get_logger(__name__)


def layout(model_id: Optional[str] = None) -> html.Div:
    """
    Create the single model page layout.
    
    Args:
        model_id: ID of the model/experiment to display
        
    Returns:
        Single model page layout
    """
    if not model_id:
        return html.Div([
            html.H3("No Model Selected", className="text-center mt-5"),
            html.P("Please select a model to view details.", className="text-center"),
            dbc.Button("Go to Experiments", href="/experiments", color="primary", className="d-block mx-auto")
        ])
    
    # Get model data from MLflow (same as experiment data)
    model = mlflow_service.get_experiment_by_id(model_id)
    
    if not model:
        return html.Div([
            html.H3("Model Not Found", className="text-center mt-5"),
            html.P(f"Model {model_id} could not be found.", className="text-center"),
            dbc.Button("Go to Experiments", href="/experiments", color="primary", className="d-block mx-auto")
        ])
    
    return html.Div([
        # Model header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-robot me-3"),
                    f"Model: {model['name']}"
                ], className="text-primary mb-4"),
                dbc.Breadcrumb(items=[
                    {"label": "Dashboard", "href": "/"},
                    {"label": "Experiments", "href": "/experiments"},
                    {"label": "Model Management", "active": True}
                ])
            ])
        ], className="mb-4"),
        
        # Model status and performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"Status: {get_model_status(model)}", 
                               className=f"text-{get_status_color(model)}"),
                        html.P(f"Agent: {model['agent']} | Symbol: {model['symbol']}"),
                        html.Small(f"Trained: {model['start_time'].strftime('%Y-%m-%d') if model['start_time'] else 'Unknown'}", 
                                 className="text-muted")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{model['sharpe_ratio']:.3f}" if model['sharpe_ratio'] else "N/A", 
                               className="text-primary mb-0"),
                        html.P("Sharpe Ratio", className="text-muted mb-0")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{model['total_return']:.1f}%" if model['total_return'] else "N/A", 
                               className="text-success mb-0"),
                        html.P("Total Return", className="text-muted mb-0")
                    ])
                ])
            ], width=4),
        ], className="mb-4"),
        
        # Model management actions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-cogs me-2"),
                            "Model Management"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Deployment Status"),
                                dbc.Badge("Not Deployed", color="secondary", pill=True),
                                html.Hr(),
                                dbc.ButtonGroup([
                                    dbc.Button([
                                        html.I(className="fas fa-rocket me-1"),
                                        "Deploy Model"
                                    ], color="success", disabled=True),
                                    dbc.Button([
                                        html.I(className="fas fa-stop me-1"),
                                        "Stop Model"
                                    ], color="danger", disabled=True),
                                ], className="d-grid gap-2")
                            ], width=6),
                            dbc.Col([
                                html.H6("Model Actions"),
                                dbc.ButtonGroup([
                                    dbc.Button([
                                        html.I(className="fas fa-play me-1"),
                                        "Run Simulation"
                                    ], color="primary", id="run-simulation-btn"),
                                    dbc.Button([
                                        html.I(className="fas fa-download me-1"),
                                        "Download Model"
                                    ], color="outline-info"),
                                    dbc.Button([
                                        html.I(className="fas fa-copy me-1"),
                                        "Clone Model"
                                    ], color="outline-secondary"),
                                ], className="d-grid gap-2", vertical=True)
                            ], width=6)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Simulation controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-chart-line me-2"),
                            "Backtesting & Simulation"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Symbol", className="form-label"),
                                dcc.Dropdown(
                                    id="simulation-symbol",
                                    options=[
                                        {"label": model['symbol'], "value": model['symbol']},
                                        {"label": "EUR/USD", "value": "EUR/USD"},
                                        {"label": "GBP/USD", "value": "GBP/USD"},
                                        {"label": "USD/JPY", "value": "USD/JPY"}
                                    ],
                                    value=model['symbol'],
                                    clearable=False
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Start Date", className="form-label"),
                                dcc.DatePickerSingle(
                                    id="simulation-start-date",
                                    date="2024-01-01",
                                    display_format="YYYY-MM-DD"
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("End Date", className="form-label"),
                                dcc.DatePickerSingle(
                                    id="simulation-end-date",
                                    date="2024-12-31",
                                    display_format="YYYY-MM-DD"
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Actions", className="form-label"),
                                html.Br(),
                                dbc.Button([
                                    html.I(className="fas fa-play me-1"),
                                    "Start Simulation"
                                ], color="primary", id="start-simulation", disabled=True)
                            ], width=3)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Performance analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Metrics"),
                    dbc.CardBody([
                        create_performance_metrics_table(model)
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Configuration"),
                    dbc.CardBody([
                        create_model_config_table(model)
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Simulation results placeholder
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Simulation Results"),
                    dbc.CardBody([
                        html.Div(id="simulation-results", children=[
                            html.P("Run a simulation to see results here.", className="text-muted text-center"),
                            html.I(className="fas fa-chart-area fa-3x text-muted d-block text-center mt-3")
                        ])
                    ])
                ])
            ], width=12)
        ])
    ])


def get_model_status(model: dict) -> str:
    """Get model deployment status."""
    if model['status'] == 'FINISHED':
        return "Ready for Deployment"
    elif model['status'] == 'RUNNING':
        return "Training in Progress"
    else:
        return "Training Failed"


def get_status_color(model: dict) -> str:
    """Get status color for model."""
    if model['status'] == 'FINISHED':
        return "success"
    elif model['status'] == 'RUNNING':
        return "warning"
    else:
        return "danger"


def create_performance_metrics_table(model: dict) -> html.Div:
    """Create performance metrics table."""
    metrics = [
        ("Sharpe Ratio", f"{model['sharpe_ratio']:.3f}" if model['sharpe_ratio'] else "N/A"),
        ("Total Return", f"{model['total_return']:.2f}%" if model['total_return'] else "N/A"),
        ("Max Drawdown", f"{model['max_drawdown']:.2f}%" if model['max_drawdown'] else "N/A"),
        ("Win Rate", f"{model['win_rate']:.1f}%" if model['win_rate'] else "N/A"),
        ("Avg Trade Return", f"{model['avg_trade_return']:.2f}%" if model['avg_trade_return'] else "N/A"),
    ]
    
    rows = []
    for metric, value in metrics:
        rows.append(
            html.Tr([
                html.Td(metric, className="fw-bold"),
                html.Td(value)
            ])
        )
    
    return html.Table([
        html.Tbody(rows)
    ], className="table table-sm")


def create_model_config_table(model: dict) -> html.Div:
    """Create model configuration table."""
    config = [
        ("Agent Type", model['agent']),
        ("Symbol", model['symbol']),
        ("Timeframe", model['timeframe']),
        ("Training Steps", model['timesteps']),
        ("Training Duration", calculate_training_duration(model)),
    ]
    
    rows = []
    for key, value in config:
        rows.append(
            html.Tr([
                html.Td(key, className="fw-bold"),
                html.Td(str(value))
            ])
        )
    
    return html.Table([
        html.Tbody(rows)
    ], className="table table-sm")


def calculate_training_duration(model: dict) -> str:
    """Calculate training duration."""
    if not model['start_time'] or not model['end_time']:
        return "Unknown"
    
    duration = model['end_time'] - model['start_time']
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, _ = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m"
    else:
        return f"{int(minutes)}m" 