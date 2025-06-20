"""Data Pipeline Page.

Monitor data quality, source status, and cache management.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def layout() -> html.Div:
    """
    Create the data pipeline page layout.
    
    Returns:
        Data pipeline monitoring page layout
    """
    return html.Div([
        # Page header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-database me-3"),
                    "Data Pipeline"
                ], className="text-primary mb-4"),
                html.P("Monitor data quality, source status, and cache management.",
                       className="lead text-muted")
            ])
        ]),
        
        # Data source status cards
        dbc.Row([
            dbc.Col([
                create_data_source_card("Forex.com", "connected", "success", "2 min ago")
            ], width=3),
            dbc.Col([
                create_data_source_card("Yahoo Finance", "connected", "success", "5 min ago")
            ], width=3),
            dbc.Col([
                create_data_source_card("Qlib Cache", "healthy", "info", "Fresh")
            ], width=3),
            dbc.Col([
                create_data_source_card("Market Data", "processing", "warning", "Running")
            ], width=3),
        ], className="mb-4"),
        
        # Data quality and cache statistics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-chart-pie me-2"),
                            "Data Quality Score"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="data-quality-chart")
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-history me-2"),
                            "Cache Hit Rate"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="cache-hit-rate-chart")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Recent data operations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-list me-2"),
                            "Recent Data Operations"
                        ], className="card-title mb-0")
                    ]),
                    dbc.CardBody(id="recent-operations-table")
                ])
            ], width=12),
        ], className="mb-4"),
        
        # Data management controls
        dbc.Card([
            dbc.CardHeader("Data Management"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-sync me-1"),
                            "Refresh All Data"
                        ], color="primary", id="refresh-all-data")
                    ], width=3),
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-trash me-1"),
                            "Clear Cache"
                        ], color="warning", id="clear-cache")
                    ], width=3),
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-download me-1"),
                            "Export Data"
                        ], color="info", id="export-data")
                    ], width=3),
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-check me-1"),
                            "Validate Data"
                        ], color="success", id="validate-data")
                    ], width=3),
                ])
            ])
        ]),
        
        # Auto-refresh interval
        dcc.Interval(
            id="data-pipeline-refresh-interval",
            interval=15*1000,  # Refresh every 15 seconds
            n_intervals=0
        )
    ])


def create_data_source_card(name: str, status: str, color: str, last_update: str) -> dbc.Card:
    """
    Create a data source status card.
    
    Args:
        name: Data source name
        status: Current status
        color: Bootstrap color theme
        last_update: Last update time
        
    Returns:
        Data source status card
    """
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H5(name, className="card-title"),
                html.H6(status.title(), className=f"text-{color} mb-2"),
                html.Small(f"Last update: {last_update}", className="text-muted")
            ])
        ])
    ], className=f"border-left-{color} shadow-sm")


@callback(
    Output("data-quality-chart", "figure"),
    Input("data-pipeline-refresh-interval", "n_intervals")
)
def update_data_quality_chart(n_intervals: int) -> go.Figure:
    """
    Update the data quality chart.
    
    Args:
        n_intervals: Number of refresh intervals
        
    Returns:
        Data quality gauge chart
    """
    try:
        # Mock data quality score
        quality_score = 87  # This would come from actual data validation
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score"},
            delta={'reference': 85},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, showlegend=False)
        return fig
        
    except Exception as e:
        logger.error(f"Failed to generate data quality chart: {e}")
        return go.Figure()


@callback(
    Output("cache-hit-rate-chart", "figure"),
    Input("data-pipeline-refresh-interval", "n_intervals")
)
def update_cache_hit_rate_chart(n_intervals: int) -> px.line:
    """
    Update the cache hit rate chart.
    
    Args:
        n_intervals: Number of refresh intervals
        
    Returns:
        Cache hit rate time series chart
    """
    try:
        # Generate mock cache hit rate data
        import random
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='h'
        )
        
        hit_rates = []
        base_rate = 75
        for _ in dates:
            base_rate += random.uniform(-5, 5)
            base_rate = max(50, min(95, base_rate))  # Keep in bounds
            hit_rates.append(base_rate)
        
        df = pd.DataFrame({
            'time': dates,
            'hit_rate': hit_rates
        })
        
        fig = px.line(
            df,
            x='time',
            y='hit_rate',
            title="Cache Hit Rate (24h)",
            color_discrete_sequence=['#28a745']
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Hit Rate (%)",
            template="plotly_white",
            height=300
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to generate cache hit rate chart: {e}")
        return px.line(title="Error generating chart")


@callback(
    Output("recent-operations-table", "children"),
    Input("data-pipeline-refresh-interval", "n_intervals")
)
def update_recent_operations(n_intervals: int) -> html.Div:
    """
    Update the recent operations table.
    
    Args:
        n_intervals: Number of refresh intervals
        
    Returns:
        Recent operations table component
    """
    try:
        # Generate mock operations data
        operations = generate_mock_operations(limit=10)
        
        if not operations:
            return html.P("No recent operations.", className="text-muted")
        
        # Create table rows
        table_rows = []
        for op in operations:
            status_color = {
                "SUCCESS": "success",
                "FAILED": "danger",
                "RUNNING": "warning"
            }.get(op['status'], "secondary")
            
            row = dbc.Row([
                dbc.Col(op['operation'], width=3),
                dbc.Col(op['symbol'], width=2),
                dbc.Col([
                    dbc.Badge(op['status'], color=status_color, pill=True)
                ], width=2),
                dbc.Col(f"{op['records']:,}" if op['records'] else "N/A", width=2),
                dbc.Col(op['timestamp'].strftime("%H:%M:%S"), width=2),
                dbc.Col(op['duration'], width=1)
            ], className="py-2 border-bottom align-items-center")
            table_rows.append(row)
        
        return html.Div([
            # Table header
            dbc.Row([
                dbc.Col(html.Strong("Operation"), width=3),
                dbc.Col(html.Strong("Symbol"), width=2),
                dbc.Col(html.Strong("Status"), width=2),
                dbc.Col(html.Strong("Records"), width=2),
                dbc.Col(html.Strong("Time"), width=2),
                dbc.Col(html.Strong("Duration"), width=1),
            ], className="py-2 border-bottom fw-bold"),
            
            # Table rows
            html.Div(table_rows)
        ])
        
    except Exception as e:
        logger.error(f"Failed to load recent operations: {e}")
        return html.P("Failed to load operations.", className="text-danger")


def generate_mock_operations(limit: int = 10) -> List[Dict[str, Any]]:
    """Generate mock data operations for display."""
    import random
    
    operations = []
    operation_types = ["Data Fetch", "Cache Update", "Feature Engineering", "Data Validation"]
    symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
    statuses = ["SUCCESS", "FAILED", "RUNNING"]
    
    for i in range(limit):
        operations.append({
            'operation': random.choice(operation_types),
            'symbol': random.choice(symbols),
            'status': random.choice(statuses),
            'records': random.randint(1000, 50000) if random.random() > 0.3 else None,
            'timestamp': datetime.now() - timedelta(minutes=random.randint(1, 120)),
            'duration': f"{random.randint(1, 30)}s"
        })
    
    # Sort by timestamp (most recent first)
    operations.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return operations 