#!/usr/bin/env python3
"""Core trading environment implementation.

This module provides the main TradingEnv class,
replacing all old code with clean, well-tested implementations.
"""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import logging

from src.environment.actions.base import BaseActionHandler
from src.environment.actions.discrete import DiscreteActionSpace, TradingAction
from src.environment.config import TradingEnvironmentConfig
from src.environment.observations.composite import CompositeObservation
from src.environment.reward_system.composite import CompositeReward
from src.environment.state.position import PositionManager
from src.environment.state.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """Clean trading environment implementation using modular components.
    
    This environment is model-agnostic and can accept actions from:
    - Stable-Baselines3 (numpy arrays)
    - PyTorch (tensors)
    - TensorFlow (tensors)
    - LLMs (strings)
    - Human input (integers/strings)
    - Any future ML framework
    
    The environment uses dynamic configuration to support different:
    - Action spaces (discrete_3, discrete_5, continuous)
    - Observation systems
    - Reward functions
    - Fee structures
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, data: pd.DataFrame, config: TradingEnvironmentConfig):
        """Initialize the trading environment.
        
        Args:
            data: Market data DataFrame with OHLCV data
            config: Environment configuration object
            
        Raises:
            ValueError: If data is invalid or config is malformed
        """
        super().__init__()
        
        # Validate inputs
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Data must be a non-empty pandas DataFrame")
        
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        self.data = data.reset_index(drop=True)  # Ensure clean index
        self.config = config
        
        # Initialize components
        self._init_action_space()
        self._init_state_managers()
        self._init_observation_system()
        self._init_reward_system()
        self._init_observation_space()
        
        # Environment state
        self.current_step = 0
        
    def _init_action_space(self) -> None:
        """Initialize action space based on configuration."""
        if self.config.action_type.value == "discrete_3":
            self.action_handler: BaseActionHandler = DiscreteActionSpace()
            self.action_space = self.action_handler.action_space
        else:
            raise NotImplementedError(f"Action type {self.config.action_type} not implemented")
    
    def _init_state_managers(self) -> None:
        """Initialize position and portfolio managers."""
        self.position_manager = PositionManager()
        self.portfolio_tracker = PortfolioTracker(
            initial_balance=self.config.initial_balance,
            fee_structure=self.config.fee_structure,
            spread=self.config.spread,
            commission_rate=self.config.commission_rate
        )
    
    def _init_observation_system(self) -> None:
        """Initialize observation system."""
        self.observation_handler = CompositeObservation(config=self.config)
    
    def _init_reward_system(self) -> None:
        """Initialize reward system based on configuration."""
        if self.config.reward_system.value == "realized_pnl":
            reward_components = {"pnl": 1.0}
        else:
            raise NotImplementedError(f"Reward system {self.config.reward_system} not implemented")
        
        self.reward_calculator = CompositeReward(reward_components)
    
    def _init_observation_space(self) -> None:
        """Initialize observation space."""
        obs_size = self.observation_handler.observation_size
        low_bounds, high_bounds = self.observation_handler.get_observation_space_bounds()
        
        self.observation_space = spaces.Box(
            low=low_bounds,
            high=high_bounds,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self, *, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        logger.info("Resetting environment")
        super().reset(seed=seed)
        
        # Reset all components
        self.current_step = 0
        self.position_manager.reset()
        self.portfolio_tracker.reset()
        self.observation_handler.reset()
        self.reward_calculator.reset()
        
        # Get initial observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        logger.info(f"Environment reset complete. Portfolio value: {self.portfolio_value}")
        return observation, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment with model-agnostic action input.
        
        This method accepts actions from any ML framework or source:
        - SB3: numpy.ndarray containing action index
        - PyTorch: torch.Tensor containing action index  
        - TensorFlow: tf.Tensor containing action index
        - LLMs: string action names like "open_long", "close"
        - Human: integer action indices or string names
        
        Args:
            action: Action from any source (numpy array, tensor, string, int, etc.)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            
        Raises:
            ValueError: If action cannot be validated or converted
        """
        try:
            # Use the model-agnostic action handler to validate and convert
            trading_action = self.action_handler.validate_action(action)
            completed_trade = self._execute_action(trading_action)
            
            # Advance time
            self.current_step += 1
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(
                trade=completed_trade,
                portfolio_balance=self.portfolio_tracker.balance
            )
            
            # Check termination conditions
            terminated = self._check_termination()
            truncated = False  # No time limits in this implementation
            
            # Get observation and info
            observation = self._get_observation()
            info = self._get_info()
            
            # Add action debugging info to info dict
            info["action_debug"] = self.action_handler.get_action_info(action)
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            # Provide helpful debugging information
            action_info = {
                "original_action": str(action),
                "action_type": str(type(action).__name__),
                "error": str(e)
            }
            raise ValueError(
                f"Failed to process action in environment step. "
                f"Action info: {action_info}"
            ) from e
    
    def _execute_action(self, action: TradingAction) -> Optional[Any]:
        """Execute a trading action and return any completed trade.
        
        Args:
            action: Trading action to execute
            
        Returns:
            Completed trade if position was closed, None otherwise
        """
        current_price = self.data.iloc[self.current_step]["close"]
        completed_trade = None
        
        if action == TradingAction.OPEN_LONG:
            if self.position_manager.is_flat:
                self.position_manager.open_long(current_price, self.current_step)
        
        elif action == TradingAction.OPEN_SHORT:
            if self.position_manager.is_flat:
                self.position_manager.open_short(current_price, self.current_step)
        
        elif action == TradingAction.CLOSE:
            if not self.position_manager.is_flat:
                # Calculate position size and close
                position_size = self.portfolio_tracker.calculate_position_size(
                    self.position_manager.entry_price, 
                    self.config.position_sizing_method
                )
                
                # Close position and get trade record
                completed_trade = self.position_manager.close_position(
                    current_price, self.current_step, position_size
                )
                
                # Apply trade result to portfolio
                trade_value = position_size * current_price
                self.portfolio_tracker.apply_trade_result(
                    completed_trade.profit, trade_value
                )
        
        return completed_trade
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate.
        
        Returns:
            True if episode should end
        """
        # Terminate if we've run out of data
        if self.current_step >= len(self.data) - 1:
            logger.info(f"Episode terminated: Reached end of data (step {self.current_step} >= {len(self.data)-1})")
            return True
        
        # Terminate if portfolio value is too low
        if self.portfolio_value <= 0:
            logger.info(f"Episode terminated: Portfolio value too low ({self.portfolio_value})")
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Current observation array
        """
        return self.observation_handler.get_observation(
            self.data, self.current_step, 
            self.position_manager, self.portfolio_tracker
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information.
        
        Returns:
            Dictionary with diagnostic information
        """
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "balance": self.portfolio_tracker.balance,
            "position": self.position_manager.position.name,
            "entry_price": self.position_manager.entry_price,
            "total_trades": len(self.position_manager.completed_trades),
            "trade_history": self.position_manager.completed_trades,
            "total_fees_paid": self.portfolio_tracker.total_fees_paid,
        }
    
    @property
    def portfolio_value(self) -> float:
        """Calculate current portfolio value including unrealized P&L.
        
        Returns:
            Total portfolio value
        """
        if self.position_manager.is_flat:
            return self.portfolio_tracker.balance
        
        # Calculate unrealized P&L
        current_price = self.data.iloc[self.current_step]["close"]
        position_size = self.portfolio_tracker.calculate_position_size(
            self.position_manager.entry_price,
            self.config.position_sizing_method
        )
        unrealized_pnl = self.position_manager.calculate_unrealized_pnl(
            current_price, position_size
        )
        
        return self.portfolio_tracker.balance + unrealized_pnl
    
    def render(self, mode: str = "human") -> None:
        """Render the environment state.
        
        Args:
            mode: Render mode (only "human" supported)
        """
        if mode == "human":
            logger.info(f"Step: {self.current_step}")
            logger.info(f"Portfolio Value: ${self.portfolio_value:.2f}")
            logger.info(f"Position: {self.position_manager.position.name}")
            if not self.position_manager.is_flat:
                logger.info(f"Entry Price: {self.position_manager.entry_price}")
            logger.info(f"Total Trades: {len(self.position_manager.completed_trades)}")
            logger.info("-" * 40) 