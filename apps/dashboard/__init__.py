"""Trading Dashboard Application.

A multi-page Dash application for monitoring and managing the agentic trading system.
Provides interfaces for experiment tracking, model management, and data pipeline monitoring.
"""

from apps.dashboard.app import create_app

__all__ = ["create_app"] 