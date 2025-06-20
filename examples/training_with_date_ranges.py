"""Examples of using the new market_data module with date ranges."""
from datetime import datetime, timezone
from src.types import Timeframe, DataSource
from src.market_data.coordinator import (
    prepare_for_training,
    prepare_for_training_bars_after,
    prepare_for_training_bars_before
)


async def example_2008_crisis_training():
    """Train agent on 2008 financial crisis data."""
    
    # PREFERRED: Exact date range
    features_path = await prepare_for_training(
        symbol="EUR/USD",
        timeframe=Timeframe.D1,
        source=DataSource.FOREX_COM,  # Only available source currently
        start_date=datetime(2008, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2009, 6, 1, tzinfo=timezone.utc)  # Include recovery period
    )
    
    print(f"2008 crisis training data ready: {features_path}")


async def example_bars_after_date():
    """Get 1000 bars starting from a specific date."""
    
    features_path = await prepare_for_training_bars_after(
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        source=DataSource.FOREX_COM,
        start_date=datetime(2020, 3, 1, tzinfo=timezone.utc),  # COVID crash start
        bars=1000
    )
    
    print(f"COVID period training data ready: {features_path}")


async def example_bars_before_date():
    """Get 500 bars ending at a specific date."""
    
    features_path = await prepare_for_training_bars_before(
        symbol="EUR/USD", 
        timeframe=Timeframe.H4,
        source=DataSource.FOREX_COM,
        end_date=datetime(2016, 6, 24, tzinfo=timezone.utc),  # Brexit vote
        bars=500
    )
    
    print(f"Pre-Brexit training data ready: {features_path}")


async def example_recent_data():
    """Get recent data for live trading preparation."""
    
    # Get last 30 days for current market conditions
    end_date = datetime.now(timezone.utc)
    start_date = end_date.replace(day=end_date.day-30)
    
    features_path = await prepare_for_training(
        symbol="EUR/USD",
        timeframe=Timeframe.M15,
        source=DataSource.FOREX_COM,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Recent market data ready: {features_path}")


# Training script integration example
async def train_agent_with_historical_data():
    """How training scripts would use the new API."""
    
    # Replace the old run_data_preparation_pipeline call with:
    features_path = await prepare_for_training(
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        source=DataSource.FOREX_COM,
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
    )
    
    # Rest of training logic remains the same
    # train_df = pd.read_csv(features_path)
    # ... create environments, train agent, etc.
    
    return features_path 