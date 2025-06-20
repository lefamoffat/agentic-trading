"""Navigation Component.

Creates the main navigation bar for the trading dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html


def create_navbar() -> dbc.Navbar:
    """
    Create the main navigation bar for the dashboard.
    
    Returns:
        Bootstrap navbar component with all page links
    """
    return dbc.Navbar(
        dbc.Container([
            # Brand logo/title
            dbc.NavbarBrand([
                html.I(className="fas fa-chart-line me-2"),
                "Agentic Trading Dashboard"
            ], href="/", className="text-white fw-bold"),
            
            # Navigation links
            dbc.Nav([
                dbc.NavItem(dbc.NavLink([
                    html.I(className="fas fa-home me-1"),
                    "Overview"
                ], href="/", className="text-white")),
                
                dbc.NavItem(dbc.NavLink([
                    html.I(className="fas fa-flask me-1"),
                    "Experiments"
                ], href="/experiments", className="text-white")),
                
                dbc.NavItem(dbc.NavLink([
                    html.I(className="fas fa-database me-1"),
                    "Data Pipeline"
                ], href="/data-pipeline", className="text-white")),
                
                # Dropdown for advanced features
                dbc.DropdownMenu([
                    dbc.DropdownMenuItem("Model Management", href="/models"),
                    dbc.DropdownMenuItem("Live Trading", href="/live-trading"),
                    dbc.DropdownMenuItem("Backtesting", href="/backtesting"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Settings", href="/settings"),
                ], label=[
                    html.I(className="fas fa-cog me-1"),
                    "Tools"
                ], nav=True, className="text-white"),
                
            ], className="ms-auto", navbar=True),
            
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-3"
    ) 