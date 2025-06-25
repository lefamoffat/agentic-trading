from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.core.dependencies import get_experiments_service, get_stream_manager  # noqa: F401
from apps.api.routers import experiments_http, experiments_ws


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    app = FastAPI(title="Agentic Trading API", version="0.1.0")

    # CORS for local dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(experiments_http.router, prefix="/experiments", tags=["experiments"])
    app.include_router(experiments_ws.router, tags=["experiments-ws"])

    @app.get("/", include_in_schema=False)
    async def _root():  # noqa: WPS430
        """Health-check root endpoint."""
        return {"status": "ok", "app": "Agentic Trading API"}

    return app 