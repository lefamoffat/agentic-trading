"""Main Dashboard Application Factory.

Creates and configures the Dash application with multi-page routing,
MLflow integration, and all dashboard pages.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from apps.dashboard.pages import (
    overview,
    experiments,
    single_experiment,
    single_model,
    data_pipeline
)
from apps.dashboard.components.navigation import create_navbar
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> dash.Dash:
    """
    Create and configure the Dash application.
    
    Returns:
        Configured Dash application instance
    """
    # Initialize Dash app with Bootstrap theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        title="Agentic Trading Dashboard"
    )
    
    # Define app layout with navigation and page content
    app.layout = dbc.Container([
        dcc.Location(id="url", refresh=False),
        create_navbar(),
        html.Hr(),
        html.Div(id="page-content", className="mt-4")
    ], fluid=True)
    
    # Register page routing callback
    register_callbacks(app)
    
    logger.info("Dashboard application created successfully")
    return app


def register_callbacks(app: dash.Dash) -> None:
    """Register application callbacks for routing and interactivity."""
    
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def display_page(pathname: str) -> html.Div:
        """Route to appropriate page based on URL pathname."""
        
        logger.debug(f"Routing to page: {pathname}")
        
        if pathname == "/" or pathname == "/overview":
            return overview.layout()
        elif pathname == "/experiments":
            return experiments.layout()
        elif pathname.startswith("/experiment/"):
            experiment_id = pathname.split("/")[-1]
            return single_experiment.layout(experiment_id)
        elif pathname.startswith("/model/"):
            model_id = pathname.split("/")[-1]
            return single_model.layout(model_id)
        elif pathname == "/models":
            # Models list page - show top models
            return html.Div([
                html.H1([
                    html.I(className="fas fa-robot me-3"),
                    "Model Management"
                ], className="text-primary mb-4"),
                html.P("Manage and deploy your trained models.", className="lead text-muted"),
                
                dbc.Card([
                    dbc.CardHeader("Available Models"),
                    dbc.CardBody([
                        html.P("Select a model from the experiments page to manage it.", className="text-muted"),
                        dbc.Button("View Experiments", href="/experiments", color="primary")
                    ])
                ])
            ])
        elif pathname == "/data-pipeline":
            return data_pipeline.layout()
        else:
            return html.Div([
                html.H1("404 - Page Not Found", className="text-center mt-5"),
                html.P("The requested page could not be found.", className="text-center"),
                dbc.Button("Go to Overview", href="/", color="primary", className="d-block mx-auto")
            ])


if __name__ == "__main__":
    # Create and run the app in development mode
    app = create_app()
    app.run_server(debug=True, host="0.0.0.0", port=8050) 