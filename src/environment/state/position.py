#!/usr/bin/env python3
"""Position state management for trading environment.

This module handles position state with proper validation.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Position(Enum):
    """Trading position types."""
    SHORT = 0
    FLAT = 1  
    LONG = 2

@dataclass
class Trade:
    """Represents a completed trade."""
    entry_price: float
    exit_price: float
    position: Position
    profit: float
    entry_step: int
    exit_step: int

class PositionManager:
    """Manages trading positions with proper validation.
    
    This replaces the buggy position logic from legacy code that allowed
    invalid states like position without entry price.
    """
    
    def __init__(self):
        """Initialize position manager."""
        self.reset()
    
    def reset(self) -> None:
        """Reset to initial flat position."""
        self._position = Position.FLAT
        self._entry_price: Optional[float] = None
        self._entry_step: Optional[int] = None
        self._completed_trades: list[Trade] = []
    
    @property
    def position(self) -> Position:
        """Current position type."""
        return self._position
    
    @property
    def entry_price(self) -> Optional[float]:
        """Entry price of current position (None if flat)."""
        return self._entry_price
    
    @property
    def entry_step(self) -> Optional[int]:
        """Step when current position was opened (None if flat)."""
        return self._entry_step
    
    @property
    def is_flat(self) -> bool:
        """True if no position is open."""
        return self._position == Position.FLAT
    
    @property
    def is_long(self) -> bool:
        """True if long position is open."""
        return self._position == Position.LONG
    
    @property
    def is_short(self) -> bool:
        """True if short position is open."""
        return self._position == Position.SHORT
    
    @property
    def completed_trades(self) -> list[Trade]:
        """List of completed trades."""
        return self._completed_trades.copy()
    
    def open_long(self, price: float, step: int) -> None:
        """Open a long position.
        
        Args:
            price: Entry price (must be positive)
            step: Current step number
            
        Raises:
            ValueError: If price is invalid or position already open
        """
        if price <= 0:
            raise ValueError(f"Entry price must be positive, got {price}")
        
        if not self.is_flat:
            raise ValueError(f"Cannot open long: already have {self._position.name} position")
        
        self._position = Position.LONG
        self._entry_price = price
        self._entry_step = step
    
    def open_short(self, price: float, step: int) -> None:
        """Open a short position.
        
        Args:
            price: Entry price (must be positive)
            step: Current step number
            
        Raises:
            ValueError: If price is invalid or position already open
        """
        if price <= 0:
            raise ValueError(f"Entry price must be positive, got {price}")
        
        if not self.is_flat:
            raise ValueError(f"Cannot open short: already have {self._position.name} position")
        
        self._position = Position.SHORT
        self._entry_price = price
        self._entry_step = step
    
    def close_position(self, exit_price: float, exit_step: int, position_size: float) -> Trade:
        """Close current position and record trade.
        
        Args:
            exit_price: Exit price (must be positive)
            exit_step: Current step number
            position_size: Size of position in base currency
            
        Returns:
            Trade object representing the completed trade
            
        Raises:
            ValueError: If no position to close or invalid parameters
        """
        if self.is_flat:
            raise ValueError("Cannot close position: no position is open")
        
        if exit_price <= 0:
            raise ValueError(f"Exit price must be positive, got {exit_price}")
        
        if self._entry_price is None or self._entry_step is None:
            raise ValueError("Invalid position state: missing entry data")
        
        # Calculate profit
        if self.is_long:
            profit = (exit_price - self._entry_price) * position_size
        else:  # short
            profit = (self._entry_price - exit_price) * position_size
        
        # Create trade record
        trade = Trade(
            entry_price=self._entry_price,
            exit_price=exit_price,
            position=self._position,
            profit=profit,
            entry_step=self._entry_step,
            exit_step=exit_step
        )
        
        # Reset to flat
        self._position = Position.FLAT
        self._entry_price = None
        self._entry_step = None
        self._completed_trades.append(trade)
        
        return trade
    
    def calculate_unrealized_pnl(self, current_price: float, position_size: float) -> float:
        """Calculate unrealized P&L for current position.
        
        Args:
            current_price: Current market price
            position_size: Size of position in base currency
            
        Returns:
            Unrealized P&L (0.0 if flat)
            
        Raises:
            ValueError: If position is open but has invalid entry price
        """
        if self.is_flat:
            return 0.0
        
        if self._entry_price is None:
            raise ValueError("Invalid position state: open position without entry price")
        
        if current_price <= 0:
            raise ValueError(f"Current price must be positive, got {current_price}")
        
        if self.is_long:
            return (current_price - self._entry_price) * position_size
        else:  # short
            return (self._entry_price - current_price) * position_size 