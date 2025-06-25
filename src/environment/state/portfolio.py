#!/usr/bin/env python3
"""Portfolio management for trading environment.

This module handles portfolio state tracking including balance, trades, and PnL.
"""
from typing import List

from src.environment.config import FeeStructure

class PortfolioTracker:
    """Tracks portfolio state including balance, fees, and trade statistics."""
    
    def __init__(self, initial_balance: float, fee_structure: FeeStructure, 
                 spread: float = 0.0001, commission_rate: float = 0.0):
        """Initialize portfolio tracker.
        
        Args:
            initial_balance: Starting balance
            fee_structure: How fees are calculated
            spread: Bid/ask spread for forex
            commission_rate: Commission rate for stocks
            
        Raises:
            ValueError: If initial_balance is not positive
        """
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        self.initial_balance = initial_balance
        self.fee_structure = fee_structure
        self.spread = spread
        self.commission_rate = commission_rate
        
        self.reset()
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.balance = self.initial_balance
        self.total_fees_paid = 0.0
        self.total_trades = 0
    
    def calculate_transaction_cost(self, trade_value: float) -> float:
        """Calculate transaction cost based on fee structure.
        
        Args:
            trade_value: Value of the trade
            
        Returns:
            Transaction cost amount
        """
        if self.fee_structure == FeeStructure.SPREAD_BASED:
            # For forex: cost is half the spread on the trade value
            return abs(trade_value) * self.spread / 2
        elif self.fee_structure == FeeStructure.COMMISSION:
            # For stocks: percentage commission
            return abs(trade_value) * self.commission_rate
        elif self.fee_structure == FeeStructure.COMBINED:
            # Both spread and commission
            spread_cost = abs(trade_value) * self.spread / 2
            commission_cost = abs(trade_value) * self.commission_rate
            return spread_cost + commission_cost
        else:
            return 0.0
    
    def apply_trade_result(self, profit: float, trade_value: float) -> float:
        """Apply trade profit/loss and deduct transaction costs.
        
        Args:
            profit: Profit/loss from the trade
            trade_value: Total value of the trade
            
        Returns:
            Net profit after fees
        """
        transaction_cost = self.calculate_transaction_cost(trade_value)
        net_profit = profit - transaction_cost
        
        self.balance += net_profit
        self.total_fees_paid += transaction_cost
        self.total_trades += 1
        
        return net_profit
    
    def calculate_position_size(self, price: float, sizing_method: str = "fixed") -> float:
        """Calculate position size based on current balance and sizing method.
        
        Args:
            price: Current price
            sizing_method: How to size positions ("fixed" for now)
            
        Returns:
            Position size in base currency units
        """
        if sizing_method == "fixed":
            # Use full balance for position sizing
            return self.balance / price
        else:
            # Future: implement other sizing methods
            raise NotImplementedError(f"Sizing method '{sizing_method}' not implemented")
    
    def calculate_portfolio_value(self, current_price: float, position_size: float, unrealized_pnl: float) -> float:
        """Calculate total portfolio value.
        
        Args:
            current_price: Current market price
            position_size: Current position size
            unrealized_pnl: Unrealized P&L from open position
            
        Returns:
            Total portfolio value
        """
        return self.balance + unrealized_pnl
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage.
        
        Returns:
            Total return as percentage
        """
        return ((self.balance - self.initial_balance) / self.initial_balance) * 100
    
    @property
    def net_balance(self) -> float:
        """Get current balance.
        
        Returns:
            Current balance
        """
        return self.balance 