#!/usr/bin/env python3
"""Launch the Trading Dashboard.

Simple script to start the Dash-based trading dashboard application.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.dashboard.app import create_app
# Use lightweight logging instead of heavy config-dependent logger
import logging
logger = logging.getLogger(__name__)

def main():
    """Main entry point for launching the dashboard."""
    parser = argparse.ArgumentParser(description="Launch the Trading Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("🚀 Starting Agentic Trading Dashboard")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("=" * 50)
    
    try:
        # Create the Dash app
        app = create_app()
        
        logger.info(f"Dashboard starting on http://{args.host}:{args.port}")
        
        # Run the server
        app.run(
            debug=args.debug,
            host=args.host,
            port=args.port
        )
        
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 