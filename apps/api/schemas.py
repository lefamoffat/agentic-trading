from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ExperimentLaunchRequest(BaseModel):
    agent_type: str = Field("ppo", description="RL agent type")
    symbol: str = Field("EUR/USD")
    timeframe: str = Field("1h")
    timesteps: int = Field(50000, ge=1000)
    learning_rate: float = Field(0.0003, gt=0)
    initial_balance: float = Field(10000.0, gt=0)

class ExperimentLaunchResponse(BaseModel):
    experiment_id: str
    status: str 