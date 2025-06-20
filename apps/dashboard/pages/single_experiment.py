"""Single Experiment Page.

Detailed view of a single experiment with metrics, parameters, and intelligence decisions.
"""

from typing import Optional

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.utils.logger import get_logger
from apps.dashboard.services.mlflow_service import mlflow_service

logger = get_logger(__name__)


def layout(experiment_id: Optional[str] = None) -> html.Div:
    """
    Create the single experiment page layout.
    
    Args:
        experiment_id: ID of the experiment to display
        
    Returns:
        Single experiment page layout
    """
    if not experiment_id:
        return html.Div([
            html.H3("No Experiment Selected", className="text-center mt-5"),
            html.P("Please select an experiment to view details.", className="text-center"),
            dbc.Button("Go to Experiments", href="/experiments", color="primary", className="d-block mx-auto")
        ])
    
    # Get experiment data from MLflow
    experiment = mlflow_service.get_experiment_by_id(experiment_id)
    
    if not experiment:
        return html.Div([
            html.H3("Experiment Not Found", className="text-center mt-5"),
            html.P(f"Experiment {experiment_id} could not be found.", className="text-center"),
            dbc.Button("Go to Experiments", href="/experiments", color="primary", className="d-block mx-auto")
        ])
    
    return html.Div([
        # Experiment header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-microscope me-3"),
                    f"Experiment: {experiment['name']}"
                ], className="text-primary mb-4"),
                dbc.Breadcrumb(items=[
                    {"label": "Dashboard", "href": "/"},
                    {"label": "Experiments", "href": "/experiments"},
                    {"label": experiment['name'][:20] + "..." if len(experiment['name']) > 20 else experiment['name'], "active": True}
                ])
            ])
        ], className="mb-4"),
        
        # Experiment status and key metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"Status: {experiment['status']}", 
                               className=f"text-{'success' if experiment['status'] == 'FINISHED' else 'warning' if experiment['status'] == 'RUNNING' else 'danger'}"),
                        html.P(f"Agent: {experiment['agent']} | Symbol: {experiment['symbol']}"),
                        html.Small(f"Started: {experiment['start_time'].strftime('%Y-%m-%d %H:%M:%S') if experiment['start_time'] else 'Unknown'}", 
                                 className="text-muted")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{experiment['sharpe_ratio']:.3f}" if experiment['sharpe_ratio'] else "N/A", 
                               className="text-primary mb-0"),
                        html.P("Sharpe Ratio", className="text-muted mb-0")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{experiment['total_return']:.1f}%" if experiment['total_return'] else "N/A", 
                               className="text-success mb-0"),
                        html.P("Total Return", className="text-muted mb-0")
                    ])
                ])
            ], width=4),
        ], className="mb-4"),
        
        # Additional metrics row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Additional Metrics", className="card-title"),
                        html.P(f"Max Drawdown: {experiment['max_drawdown']:.2f}%" if experiment['max_drawdown'] else "N/A"),
                        html.P(f"Win Rate: {experiment['win_rate']:.1f}%" if experiment['win_rate'] else "N/A"),
                        html.P(f"Avg Trade Return: {experiment['avg_trade_return']:.2f}%" if experiment['avg_trade_return'] else "N/A"),
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Training Info", className="card-title"),
                        html.P(f"Timeframe: {experiment['timeframe']}"),
                        html.P(f"Training Steps: {experiment['timesteps']}"),
                        html.P(f"Duration: {calculate_duration(experiment)}"),
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Parameters and configuration
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Parameters"),
                    dbc.CardBody([
                        create_parameters_table(experiment['all_params'])
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("All Metrics"),
                    dbc.CardBody([
                        create_metrics_table(experiment['all_metrics'])
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Actions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Actions"),
                    dbc.CardBody([
                        dbc.ButtonGroup([
                            dbc.Button([
                                html.I(className="fas fa-robot me-1"),
                                "Manage Model"
                            ], href=f"/model/{experiment_id}", color="primary"),
                            dbc.Button([
                                html.I(className="fas fa-external-link-alt me-1"),
                                "Open in MLflow"
                            ], color="outline-secondary", external_link=True),
                            dbc.Button([
                                html.I(className="fas fa-download me-1"),
                                "Download Artifacts"
                            ], color="outline-info"),
                        ])
                    ])
                ])
            ], width=12)
        ])
    ])


def calculate_duration(experiment: dict) -> str:
    """Calculate experiment duration."""
    if not experiment['start_time']:
        return "Unknown"
    
    if not experiment['end_time']:
        if experiment['status'] == 'RUNNING':
            return "Running..."
        else:
            return "Unknown"
    
    duration = experiment['end_time'] - experiment['start_time']
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


def create_parameters_table(params: dict) -> html.Div:
    """Create a table of experiment parameters."""
    if not params:
        return html.P("No parameters recorded.", className="text-muted")
    
    table_data = []
    for key, value in params.items():
        table_data.append([key, str(value)])
    
    df = pd.DataFrame(table_data, columns=["Parameter", "Value"])
    
    return dbc.Table.from_dataframe(
        df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size="sm"
    )


def create_metrics_table(metrics: dict) -> html.Div:
    """Create a table of experiment metrics."""
    if not metrics:
        return html.P("No metrics recorded.", className="text-muted")
    
    table_data = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        table_data.append([key, formatted_value])
    
    df = pd.DataFrame(table_data, columns=["Metric", "Value"])
    
    return dbc.Table.from_dataframe(
        df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size="sm"
    ) 