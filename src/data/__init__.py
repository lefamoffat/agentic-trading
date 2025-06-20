"""Data handling modules for the trading system.

This package contains:
- Market calendars for different asset classes
- Data processing pipeline
- Feature engineering framework
- Data validation and quality checks
"""

from src.data.calendars import BaseCalendar, ForexCalendar
from src.data.calendars.factory import CalendarFactory, calendar_factory
from src.data.processor import DataProcessor

__all__ = [
    'BaseCalendar',
    'CalendarFactory',
    'DataProcessor',
    'ForexCalendar',
    'calendar_factory'
]
