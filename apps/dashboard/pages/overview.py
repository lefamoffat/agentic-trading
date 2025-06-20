"""Overview Page.

Main dashboard page showing system status, recent experiments, and key performance metrics.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.utils.logger import get_logger
from apps.dashboard.services.mlflow_service import mlflow_service

logger = get_logger(__name__)


def layout() -> html.Div:
    """
    Create the overview page layout.
    
    Returns:
        Overview page layout with system status and metrics
    """
    # Get real data from MLflow
    summary = mlflow_service.get_experiments_summary()
    
    return html.Div([
        # Page header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-tachometer-alt me-3"),
                    "System Overview"
                ], className="text-primary mb-4"),
                html.P("Monitor your trading system's performance, experiments, and data pipeline status.",
                       className="lead text-muted")
            ])
        ]),
        
        # System status cards
        dbc.Row([
            dbc.Col([
                create_status_card(
                    "MLflow Status", 
                    "connected" if mlflow_service.connected else "disconnected",
                    "success" if mlflow_service.connected else "danger", 
                    "fas fa-server"
                )
            ], width=3),
            dbc.Col([
                create_status_card(
                    "Total Experiments", 
                    str(summary["total_runs"]), 
                    "info", 
                    "fas fa-flask"
                )
            ], width=3),
            dbc.Col([
                create_status_card(
                    "Active Runs", 
                    str(summary["active_runs"]), 
                    "warning" if summary["active_runs"] > 0 else "secondary", 
                    "fas fa-play-circle"
                )
            ], width=3),
            dbc.Col([
                create_status_card(
                    "Symbols Traded", 
                    str(len(summary["unique_symbols"])), 
                    "success", 
                    "fas fa-chart-line"
                )
            ], width=3),
        ], className="mb-4"),
        
        # Show available symbols and agents if connected
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("📊 Your Trading Data", className="alert-heading"),
                    html.P(f"Symbols: {', '.join(summary['unique_symbols']) if summary['unique_symbols'] else 'None found'}"),
                    html.P(f"Agents: {', '.join(summary['unique_agents']) if summary['unique_agents'] else 'None found'}"),
                ], color="info" if mlflow_service.connected else "warning")
            ], width=12)
        ], className="mb-4"),
        
        # Recent experiments and performance charts
        dbc.Row([
            # Recent experiments
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-flask me-2"),
                            "Recent Experiments"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="recent-experiments-table", children=[
                            create_recent_experiments_table()
                        ])
                    ])
                ])
            ], width=6),
            
            # Performance chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-chart-line me-2"),
                            "Performance Trends"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="performance-chart",
                            figure=create_performance_chart()
                        )
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Model performance comparison
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-trophy me-2"),
                            "Top Performing Models"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="model-performance-table", children=[
                            create_top_models_table()
                        ])
                    ])
                ])
            ], width=12),
        ]),
    ])


def create_status_card(title: str, value: str, color: str, icon: str) -> dbc.Card:
    """
    Create a status card component.
    
    Args:
        title: Card title
        value: Status value to display
        color: Bootstrap color theme
        icon: FontAwesome icon class
        
    Returns:
        Status card component
    """
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className=f"{icon} fa-2x"),
                ], className=f"text-{color}"),
                html.Div([
                    html.H4(value, className="fw-bold mb-0"),
                    html.P(title, className="text-muted mb-0")
                ])
            ], className="d-flex justify-content-between align-items-center")
        ])
    ], className=f"border-left-{color} shadow-sm")


def create_recent_experiments_table() -> html.Div:
    """Create table of recent experiments from MLflow."""
    try:
        experiments = mlflow_service.get_recent_experiments(limit=5)
        
        if not experiments:
            return html.P("No experiments found. Run some training to see data here!", className="text-muted")
        
        # Create table rows
        table_rows = []
        for exp in experiments:
            status_color = {
                "FINISHED": "success",
                "FAILED": "danger", 
                "RUNNING": "warning"
            }.get(exp['status'], "secondary")
            
            row = dbc.Row([
                dbc.Col([
                    html.Strong(exp['name'][:20] + "..." if len(exp['name']) > 20 else exp['name']),
                    html.Br(),
                    html.Small(f"{exp['agent']} • {exp['symbol']}", className="text-muted")
                ], width=4),
                dbc.Col([
                    dbc.Badge(exp['status'], color=status_color, pill=True)
                ], width=2),
                dbc.Col([
                    html.Strong(f"{exp['sharpe_ratio']:.3f}" if exp['sharpe_ratio'] else "N/A")
                ], width=2),
                dbc.Col([
                    html.Small(exp['start_time'].strftime("%m/%d %H:%M") if exp['start_time'] else "Unknown")
                ], width=2),
                dbc.Col([
                    dbc.Button("View", size="sm", color="outline-primary", 
                              href=f"/experiment/{exp['id']}")
                ], width=2)
            ], className="py-2 border-bottom align-items-center")
            table_rows.append(row)
        
        return html.Div([
            # Table header
            dbc.Row([
                dbc.Col(html.Strong("Experiment"), width=4),
                dbc.Col(html.Strong("Status"), width=2),
                dbc.Col(html.Strong("Sharpe"), width=2),
                dbc.Col(html.Strong("Started"), width=2),
                dbc.Col(html.Strong("Actions"), width=2),
            ], className="py-2 border-bottom fw-bold"),
            
            # Table rows
            html.Div(table_rows)
        ])
        
    except Exception as e:
        logger.error(f"Failed to create recent experiments table: {e}")
        return html.P("Error loading experiments.", className="text-danger")


def create_performance_chart() -> go.Figure:
    """Create performance chart from real MLflow data."""
    try:
        experiments = mlflow_service.get_recent_experiments(limit=20)
        
        if not experiments:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No experiment data available.<br>Run some training to see performance trends!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(
                title="Performance Trends",
                template="plotly_white",
                height=300,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Filter experiments with sharpe ratio and sort by date
        valid_experiments = [exp for exp in experiments if exp['sharpe_ratio'] is not None and exp['start_time']]
        valid_experiments.sort(key=lambda x: x['start_time'])
        
        if not valid_experiments:
            fig = go.Figure()
            fig.add_annotation(
                text="No performance metrics found.<br>Check that your experiments log 'eval/sharpe_ratio'",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(
                title="Performance Trends",
                template="plotly_white",
                height=300,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Create DataFrame for plotting
        df = pd.DataFrame([{
            'date': exp['start_time'],
            'sharpe_ratio': exp['sharpe_ratio'],
            'symbol': exp['symbol'],
            'agent': exp['agent']
        } for exp in valid_experiments])
        
        fig = px.line(
            df, 
            x='date', 
            y='sharpe_ratio',
            color='symbol',
            title="Sharpe Ratio Over Time",
            hover_data=['agent']
        )
        
        fig.update_layout(
            xaxis_title="Experiment Date",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            height=300
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create performance chart: {e}")
        return go.Figure()


def create_top_models_table() -> html.Div:
    """Create table of top performing models from MLflow."""
    try:
        models = mlflow_service.get_top_models(limit=5)
        
        if not models:
            return html.P("No models with performance metrics found.", className="text-muted")
        
        # Create table data
        table_data = []
        for model in models:
            table_data.append([
                model['name'][:30] + "..." if len(model['name']) > 30 else model['name'],
                f"{model['sharpe_ratio']:.3f}" if model['sharpe_ratio'] else "N/A",
                f"{model['total_return']:.2f}%" if model['total_return'] else "N/A",
                f"{model['max_drawdown']:.2f}%" if model['max_drawdown'] else "N/A",
                model['symbol'],
                model['start_time'].strftime("%m/%d %H:%M") if model['start_time'] else "Unknown"
            ])
        
        return dbc.Table.from_dataframe(
            pd.DataFrame(table_data, columns=[
                "Model Name", "Sharpe Ratio", "Total Return", "Max Drawdown", "Symbol", "Date"
            ]),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm"
        )
        
    except Exception as e:
        logger.error(f"Failed to create top models table: {e}")
        return html.P("Error loading model performance.", className="text-danger") 