#!/usr/bin/env python3
"""SB3 → MLflow PyFunc wrapper class."""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import stable_baselines3

from .helpers import build_observation


class Sb3ModelWrapper(mlflow.pyfunc.PythonModel):
    """A thin PyFunc wrapper around any SB3 policy class."""

    def __init__(self, policy_name: str):
        super().__init__()
        self._policy_name = policy_name
        self._model = None  # type: stable_baselines3.common.base_class.BaseAlgorithm | None

    # ------------------------------------------------------------------
    # MLflow lifecycle hooks
    # ------------------------------------------------------------------

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        # `stable_baselines3` exposes algorithm classes at the top level using
        # uppercase names (``PPO``, ``A2C``, etc.).  The run metadata may pass
        # them in lowercase, so we normalise here.

        try:
            policy_cls = getattr(stable_baselines3, self._policy_name.upper())
        except AttributeError as exc:
            raise AttributeError(
                f"SB3 algorithm '{self._policy_name}' not found in stable_baselines3 package."
            ) from exc
        self._model = policy_cls.load(context.artifacts["model_path"])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        if self._model is None:
            raise RuntimeError("Model not loaded - call load_context() first.")

        obs = build_observation(model_input)
        actions, _ = self._model.predict(obs, deterministic=True)
        return actions
