#!/usr/bin/env python3
"""
Project initialization script for agentic-trading
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.utils.settings import settings
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


def init_project():
    """Initialize the agentic-trading project"""
    logger.info("Initializing agentic-trading project...")
    
    # Create necessary directories
    logger.info("Creating project directories...")
    settings.create_directories()
    
    # Create additional data subdirectories
    timeframes = ["5m", "15m", "1h", "4h", "1d"]
    for timeframe in timeframes:
        (settings.data_path / "raw" / "historical" / timeframe).mkdir(parents=True, exist_ok=True)
    
    logger.info("Project directories created successfully")
    
    # Test configuration loading
    try:
        logger.info("Testing configuration loading...")
        config_loader = ConfigLoader()
        
        # Test loading each config file
        agent_config = config_loader.get_agent_config()
        data_config = config_loader.get_data_config()
        trading_config = config_loader.get_trading_config()
        qlib_config = config_loader.get_qlib_config()
        
        logger.info("✅ Agent config loaded successfully")
        logger.info("✅ Data config loaded successfully")
        logger.info("✅ Trading config loaded successfully")
        logger.info("✅ Qlib config loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False
    
    # Check environment variables
    logger.info("Checking environment variables...")
    if settings.validate_api_keys():
        logger.info("✅ All API keys are configured")
    else:
        logger.warning("⚠️  Some API keys are missing. Create a .env file with your API keys.")
        logger.warning("   Copy .env.example to .env and fill in your API keys")
    
    logger.info("🎉 Project initialization completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Set up your API keys in .env file")
    logger.info("2. Install dependencies: uv sync")
    logger.info("3. Download historical data: python scripts/data/download_historical.py")
    
    return True


if __name__ == "__main__":
    success = init_project()
    sys.exit(0 if success else 1) 