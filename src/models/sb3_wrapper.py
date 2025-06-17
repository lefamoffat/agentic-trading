#!/usr/bin/env python3
"""Stable-Baselines3 → MLflow PyFunc wrapper.

This lives in a dedicated module so that training scripts remain short
and to make the wrapper re-usable for inference / serving contexts.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import stable_baselines3
import mlflow

# NOTE: we intentionally avoid importing Stable-Baselines3 BaseAlgorithm at
# import time to keep this module import-light for API servers that only
# need the predict path.


class Sb3ModelWrapper(mlflow.pyfunc.PythonModel):
    """A thin PyFunc wrapper around any SB3 policy class.

    The input signature is inferred from the *type hint* on
    ``predict(..., model_input: pd.DataFrame)`` by MLflow 3, so we don't
    have to pass an `input_example` purely for schema inference – though
    the training script still provides one for model validation.
    """

    def __init__(self, policy_name: str):
        super().__init__()
        self._policy_name = policy_name
        self._model = None  # Will be loaded lazily in `load_context`.

    # ------------------------------------------------------------------
    # Required MLflow hooks
    # ------------------------------------------------------------------

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:  # noqa: D401
        """Load the concrete SB3 model from the artifact directory."""
        policy_cls = getattr(stable_baselines3, self._policy_name)
        self._model = policy_cls.load(context.artifacts["model_path"])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        """Return *actions* for a batch of inputs.

        Args:
            model_input: A DataFrame with the *same columns* used during
                training.  We convert it to the environment's expected
                observation tensor.

        Returns:
            A NumPy array of actions predicted by the SB3 policy.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — `load_context` was not called.")

        # ------------------------------------------------------------------
        # Build observation.  This mirrors the training env's observation
        #   [close, balance_norm, position, feature1, feature2, ...]
        # ------------------------------------------------------------------
        features = model_input.to_numpy(dtype=np.float32)

        close_vals = (
            model_input["close"].to_numpy(dtype=np.float32).reshape(-1, 1)
            if "close" in model_input.columns
            else features[:, [0]]
        )
        balance_vals = np.ones((features.shape[0], 1), dtype=np.float32)
        position_vals = np.ones((features.shape[0], 1), dtype=np.float32)

        obs = np.concatenate([close_vals, balance_vals, position_vals, features], axis=1)

        actions, _ = self._model.predict(obs, deterministic=True)
        return actions 