"""Specific type definitions for the Forex.com broker module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForexComApiResponseKeys:
    """Stores constant keys found in Forex.com API responses.
    This provides a single source of truth and avoids hardcoded strings.
    """

    PRICE_BARS = "PriceBars"
    BAR_HISTORY = "BarHistory"
    MARKET_ID = "MarketId"
    MARKETS = "Markets"
    ACTIVE_ORDERS = "ActiveOrders"
    OPEN_POSITIONS = "OpenPositions"
    TRADING_ACCOUNTS = "TradingAccounts"


@dataclass(frozen=True)
class ForexComApiParams:
    """Stores constant keys for parameters sent in Forex.com API requests.
    """

    PRICE_BARS = "PriceBars"
    INTERVAL = "interval"
    SPAN = "span"
    MAX_BARS = "maxbars"
