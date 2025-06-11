"""
Data handling modules for the trading system.

This package contains:
- Market calendars for different asset classes
- Data processing pipeline
- Feature engineering framework
- Data validation and quality checks
"""

from .processor import DataProcessor
from .calendars import BaseCalendar, ForexCalendar
from .calendars.factory import CalendarFactory, calendar_factory

__all__ = [
    'DataProcessor',
    'BaseCalendar', 
    'ForexCalendar',
    'CalendarFactory',
    'calendar_factory'
] 