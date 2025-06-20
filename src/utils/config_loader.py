"""Configuration loader for YAML files with environment variable substitution
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Load and manage YAML configuration files with environment variable substitution"""

    def __init__(self, config_dir: str = "configs"):
        """Initialize the config loader

        Args:
            config_dir: Directory containing configuration files

        """
        self.config_dir = Path(config_dir)
        self._configs = {}

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file

        Args:
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Dict containing the configuration

        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as file:
            content = file.read()

        # Substitute environment variables
        content = self._substitute_env_vars(content)

        # Parse YAML
        config = yaml.safe_load(content)

        # Cache the config
        self._configs[config_name] = config

        return config

    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in the format ${VAR_NAME}

        Args:
            content: YAML content as string

        Returns:
            Content with environment variables substituted

        """
        pattern = r'\$\{([^}]+)\}'

        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, f"${{{var_name}}}")  # Keep original if not found

        return re.sub(pattern, replace_env_var, content)

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.load_config("agent_config")

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.load_config("trading_config")

    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """Reload a configuration file (clears cache)

        Args:
            config_name: Name of the config to reload

        Returns:
            Reloaded configuration

        """
        if config_name in self._configs:
            del self._configs[config_name]
        return self.load_config(config_name)

    def clear_cache(self):
        """Clear all cached configurations"""
        self._configs.clear()
