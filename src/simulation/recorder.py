from __future__ import annotations

"""Session recording helper for live/back-test data.

Records tuples of (step, observation dict, action, reward, equity).
Writes them to Parquet for later offline RL / behavioural cloning.
"""

from pathlib import Path
from typing import Any, List

import pandas as pd

__all__ = ["SessionRecorder"]


class SessionRecorder:
    """Accumulate step data and persist to Parquet."""

    def __init__(self) -> None:
        self._rows: List[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        step: int,
        observation: Any,
        action: int,
        reward: float,
        equity: float,
    ) -> None:
        self._rows.append(
            {
                "step": step,
                "observation": observation if isinstance(observation, (float, int)) else str(observation),
                "action": action,
                "reward": reward,
                "equity": equity,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._rows)

    def save(self, path: str | Path) -> Path:
        df = self.to_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path 