#!/usr/bin/env python3
"""Configuration generator for converting LLM recommendations to valid configs."""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.environment import TradingEnvironmentConfig, ActionType, FeeStructure, RewardSystem, load_trading_config
from src.utils.logger import get_logger


class ConfigGenerator:
    """Converts LLM recommendations into valid trading configurations."""
    
    def __init__(self, base_config_path: Optional[Path] = None):
        """Initialize config generator."""
        self.logger = get_logger(self.__class__.__name__)
        self.base_config_path = base_config_path or Path("configs/trading_config.yaml")
        
        # Load base configuration
        try:
            self.base_config = load_trading_config(self.base_config_path)
            self.logger.info(f"Loaded base config from {self.base_config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load base config: {e}. Using defaults.")
            self.base_config = self._get_default_config()
    
    def _get_default_config(self) -> TradingEnvironmentConfig:
        """Get default trading environment configuration."""
        return TradingEnvironmentConfig(
            initial_balance=100000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            spread=0.0001,
            commission_rate=0.0,
            slippage_pips=0.5,
            action_type=ActionType.DISCRETE_THREE,
            reward_system=RewardSystem.REALIZED_PNL,
            position_sizing_method="fixed",
            max_positions=1,
        )
    
    def generate_config(
        self, 
        recommendations: Dict[str, Any],
        action: str = "start"
    ) -> TradingEnvironmentConfig:
        """Generate trading configuration from LLM recommendations."""
        # Start with base configuration
        config = deepcopy(self.base_config)
        
        # Extract recommended changes
        recommended_changes = recommendations.get("recommended_changes", {})
        
        if not recommended_changes:
            self.logger.info("No recommended changes, returning base config")
            return config
        
        # Apply changes
        for param_name, change_info in recommended_changes.items():
            if not isinstance(change_info, dict) or "value" not in change_info:
                continue
            
            new_value = change_info["value"]
            if self._apply_parameter_change(config, param_name, new_value):
                self.logger.info(f"Applied {param_name} = {new_value}")
        
        return config
    
    def _apply_parameter_change(
        self, 
        config: TradingEnvironmentConfig, 
        param_name: str, 
        new_value: Any
    ) -> bool:
        """Apply a single parameter change to the configuration."""
        # Mapping of parameter names to config attributes
        param_mapping = {
            "initial_balance": "initial_balance",
            "spread": "spread",
            "commission_rate": "commission_rate",
            "slippage_pips": "slippage_pips",
            "position_sizing_method": "position_sizing_method",
            "max_positions": "max_positions",
        }
        
        config_attr = param_mapping.get(param_name)
        
        if config_attr is None:
            self.logger.debug(f"Parameter {param_name} not mapped to environment config")
            return False
        
        if not hasattr(config, config_attr):
            self.logger.warning(f"Configuration does not have attribute: {config_attr}")
            return False
        
        try:
            # Validate and convert value if necessary
            if param_name in ["initial_balance", "spread", "commission_rate", "slippage_pips"]:
                new_value = float(new_value)
            elif param_name in ["max_positions"]:
                new_value = int(new_value)
            
            setattr(config, config_attr, new_value)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set {config_attr} = {new_value}: {e}")
            return False
    
    def generate_agent_config_changes(
        self, 
        recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract agent configuration changes from recommendations."""
        agent_changes = {}
        recommended_changes = recommendations.get("recommended_changes", {})
        
        # Parameters that belong to agent config, not environment config
        agent_params = ["learning_rate", "batch_size", "n_epochs", "gamma", "clip_range"]
        
        for param_name, change_info in recommended_changes.items():
            if param_name in agent_params and isinstance(change_info, dict):
                new_value = change_info.get("new") or change_info.get("value")
                if new_value is not None:
                    agent_changes[param_name] = {
                        "value": new_value,
                        "reason": change_info.get("reason", "LLM recommendation")
                    }
        
        return agent_changes 