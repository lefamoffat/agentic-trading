"""Experiments Page.

Browse and compare all training runs with filtering and search capabilities.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd

from src.utils.logger import get_logger
from apps.dashboard.services.mlflow_service import mlflow_service

logger = get_logger(__name__)


def layout() -> html.Div:
    """
    Create the experiments page layout.
    
    Returns:
        Experiments page layout with filtering and comparison tools
    """
    # Get summary for filter options
    summary = mlflow_service.get_experiments_summary()
    
    return html.Div([
        # Page header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-flask me-3"),
                    "Experiments"
                ], className="text-primary mb-4"),
                html.P("Browse, filter, and compare all training experiments.",
                       className="lead text-muted")
            ])
        ]),
        
        # Connection status
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("📊 MLflow Connection", className="alert-heading"),
                    html.P(f"Status: {'Connected' if mlflow_service.connected else 'Disconnected'}"),
                    html.P(f"Total Runs: {summary['total_runs']} | Active: {summary['active_runs']}"),
                ], color="success" if mlflow_service.connected else "warning")
            ], width=12)
        ], className="mb-4"),
        
        # Filters and controls
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Status Filter", className="form-label"),
                        dcc.Dropdown(
                            id="status-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Finished", "value": "FINISHED"},
                                {"label": "Running", "value": "RUNNING"},
                                {"label": "Failed", "value": "FAILED"}
                            ],
                            value="all",
                            clearable=False
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("Agent Type", className="form-label"),
                        dcc.Dropdown(
                            id="agent-filter",
                            options=[{"label": "All", "value": "all"}] + 
                                   [{"label": agent, "value": agent} for agent in summary["unique_agents"]],
                            value="all",
                            clearable=False
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("Symbol", className="form-label"),
                        dcc.Dropdown(
                            id="symbol-filter",
                            options=[{"label": "All", "value": "all"}] + 
                                   [{"label": symbol, "value": symbol} for symbol in summary["unique_symbols"]],
                            value="all",
                            clearable=False
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("Actions", className="form-label"),
                        html.Br(),
                        dbc.ButtonGroup([
                            dbc.Button([
                                html.I(className="fas fa-sync me-1"),
                                "Refresh"
                            ], id="refresh-experiments", color="outline-primary", size="sm"),
                        ], size="sm")
                    ], width=3)
                ])
            ])
        ], className="mb-4"),
        
        # Experiments table
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-table me-2"),
                    "All Experiments"
                ], className="card-title mb-0")
            ]),
            dbc.CardBody([
                html.Div(id="experiments-table-container", children=[
                    create_experiments_table()
                ])
            ])
        ], className="mb-4"),
        
        # Comparison chart
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Performance Comparison"
                ], className="card-title mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="experiments-comparison-chart",
                    figure=create_comparison_chart()
                )
            ])
        ])
    ])


def create_experiments_table() -> html.Div:
    """Create table of all experiments from MLflow."""
    try:
        experiments = mlflow_service.get_recent_experiments(limit=50)  # Get more experiments
        
        if not experiments:
            return html.Div([
                html.P("No experiments found.", className="text-muted text-center"),
                html.P("Run some training experiments to see data here!", className="text-muted text-center"),
                dbc.Button("View Training Scripts", href="/docs", color="primary", outline=True, className="d-block mx-auto mt-3")
            ])
        
        # Create enhanced table with more details
        table_data = []
        for exp in experiments:
            # Format duration
            duration = "N/A"
            if exp['start_time'] and exp['end_time']:
                duration_delta = exp['end_time'] - exp['start_time']
                hours, remainder = divmod(duration_delta.total_seconds(), 3600)
                minutes, _ = divmod(remainder, 60)
                duration = f"{int(hours)}h {int(minutes)}m"
            elif exp['status'] == 'RUNNING':
                duration = "Running..."
            
            table_data.append([
                exp['name'][:25] + "..." if len(exp['name']) > 25 else exp['name'],
                exp['status'],
                exp['agent'],
                exp['symbol'],
                f"{exp['sharpe_ratio']:.3f}" if exp['sharpe_ratio'] else "N/A",
                f"{exp['total_return']:.1f}%" if exp['total_return'] else "N/A",
                exp['start_time'].strftime("%Y-%m-%d %H:%M") if exp['start_time'] else "Unknown",
                duration
            ])
        
        df = pd.DataFrame(table_data, columns=[
            "Experiment", "Status", "Agent", "Symbol", "Sharpe", "Return", "Started", "Duration"
        ])
        
        return html.Div([
            dbc.Table.from_dataframe(
                df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm"
            ),
            html.P(f"Showing {len(experiments)} experiments", className="text-muted mt-2")
        ])
        
    except Exception as e:
        logger.error(f"Failed to create experiments table: {e}")
        return html.P("Error loading experiments.", className="text-danger")


def create_comparison_chart() -> px.bar:
    """Create performance comparison chart from real MLflow data."""
    try:
        experiments = mlflow_service.get_recent_experiments(limit=20)
        
        if not experiments:
            fig = px.bar(title="No experiment data available")
            fig.add_annotation(
                text="Run some experiments to see performance comparison!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(
                template="plotly_white",
                height=400,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Filter experiments with sharpe ratio
        valid_experiments = [exp for exp in experiments if exp['sharpe_ratio'] is not None]
        
        if not valid_experiments:
            fig = px.bar(title="No performance metrics available")
            fig.add_annotation(
                text="No experiments with Sharpe ratio metrics found.<br>Check that experiments log 'eval/sharpe_ratio'",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(
                template="plotly_white",
                height=400,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Sort by sharpe ratio and take top performers
        valid_experiments.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        top_experiments = valid_experiments[:15]  # Show top 15
        
        df = pd.DataFrame([{
            'name': exp['name'][:15] + "..." if len(exp['name']) > 15 else exp['name'],
            'sharpe_ratio': exp['sharpe_ratio'],
            'agent': exp['agent'],
            'symbol': exp['symbol'],
            'full_name': exp['name']
        } for exp in top_experiments])
        
        fig = px.bar(
            df,
            x='name',
            y='sharpe_ratio',
            color='agent',
            title="Top Performing Experiments (by Sharpe Ratio)",
            hover_data=['symbol', 'full_name']
        )
        
        fig.update_layout(
            xaxis_title="Experiment",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            height=400,
            xaxis={'tickangle': -45}
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to generate comparison chart: {e}")
        return px.bar(title="Error generating chart") 