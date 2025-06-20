#!/usr/bin/env python3
"""SB3 → MLflow PyFunc wrapper for agents."""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import stable_baselines3

from src.agents.helpers import build_observation

__all__ = ["Sb3ModelWrapper"]


class Sb3ModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for SB3 agents."""

    def __init__(self, policy_name: str):
        super().__init__()
        self._policy_name = policy_name
        self._model = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load SB3 model from MLflow context."""
        try:
            policy_cls = getattr(stable_baselines3, self._policy_name.upper())
        except AttributeError as exc:
            raise AttributeError(
                f"SB3 algorithm '{self._policy_name}' not found in stable_baselines3."
            ) from exc
        self._model = policy_cls.load(context.artifacts["model_path"])

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded SB3 model."""
        if self._model is None:
            raise RuntimeError("Model not loaded - call load_context() first.")

        obs = build_observation(model_input)
        actions, _ = self._model.predict(obs, deterministic=True)
        return actions 