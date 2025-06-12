"""
Settings and environment variables management
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Settings:
    """Application settings and environment variables"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize settings
        
        Args:
            env_file: Path to .env file (optional)
        """
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        else:
            # Try to load .env from current directory
            if Path(".env").exists():
                load_dotenv(".env")
    
    # Environment
    @property
    def environment(self) -> str:
        """Get environment (development, staging, production)"""
        return os.getenv("ENVIRONMENT", "development")
    
    @property
    def log_level(self) -> str:
        """Get log level"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    # Forex.com (GainCapital) credentials
    @property
    def forex_com_username(self) -> Optional[str]:
        """Get forex.com username"""
        return os.getenv("FOREX_COM_USERNAME")
    
    @property
    def forex_com_password(self) -> Optional[str]:
        """Get forex.com password"""
        return os.getenv("FOREX_COM_PASSWORD")
    
    @property
    def forex_com_app_key(self) -> Optional[str]:
        """Get forex.com app key"""
        return os.getenv("FOREX_COM_APP_KEY")
    
    @property
    def forex_com_sandbox(self) -> bool:
        """Get forex.com sandbox mode"""
        return os.getenv("FOREX_COM_SANDBOX", "true").lower() == "true"
    
    @property
    def alpha_vantage_api_key(self) -> Optional[str]:
        """Get Alpha Vantage API key"""
        return os.getenv("ALPHA_VANTAGE_API_KEY")
    
    # Trading Configuration
    @property
    def position_size(self) -> float:
        """Get default position size"""
        return float(os.getenv("POSITION_SIZE", "10000"))
    
    @property
    def max_drawdown(self) -> float:
        """Get maximum drawdown limit"""
        return float(os.getenv("MAX_DRAWDOWN", "0.10"))
    
    @property
    def stop_loss(self) -> float:
        """Get default stop loss"""
        return float(os.getenv("STOP_LOSS", "0.02"))
    
    # Paths
    @property
    def data_path(self) -> Path:
        """Get data storage path"""
        return Path(os.getenv("DATA_PATH", "./data"))
    
    @property
    def qlib_data_path(self) -> Path:
        """Get Qlib data path"""
        return Path(os.getenv("QLIB_DATA_PATH", "./data/qlib_format"))
    
    @property
    def model_path(self) -> Path:
        """Get model storage path"""
        return Path(os.getenv("MODEL_PATH", "./data/models"))
    
    @property
    def checkpoint_path(self) -> Path:
        """Get checkpoint storage path"""
        return Path(os.getenv("CHECKPOINT_PATH", "./data/models/checkpoints"))
    
    @property
    def logs_path(self) -> Path:
        """Get logs path"""
        return Path(os.getenv("LOGS_PATH", "./logs"))
    
    @property
    def results_path(self) -> Path:
        """Get results path"""
        return Path(os.getenv("RESULTS_PATH", "./results"))
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_path,
            self.qlib_data_path,
            self.model_path,
            self.checkpoint_path,
            self.logs_path,
            self.results_path,
            self.data_path / "raw" / "historical",
            self.data_path / "processed" / "features",
            self.logs_path / "training",
            self.logs_path / "backtesting",
            self.logs_path / "trading",
            self.logs_path / "system"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_api_keys(self) -> bool:
        """
        Validate that required API keys are present
        
        Returns:
            True if all required keys are present
        """
        required_keys = [
            self.forex_com_username,
            self.forex_com_password,
            self.forex_com_app_key,
            self.alpha_vantage_api_key
        ]
        
        return all(key is not None for key in required_keys)


# Global settings instance
settings = Settings() 