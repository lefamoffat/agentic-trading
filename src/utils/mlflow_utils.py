from __future__ import annotations

"""MLflow utility helpers."""

from mlflow.tracking import MlflowClient

__all__ = ["latest_version"]


def latest_version(model_name: str, stage: str | None = None) -> str:
    """Return latest version string for *model_name*.

    Args:
        model_name: Name in MLflow Model Registry.
        stage: Optional stage filter (``None`` → any stage).

    Returns:
        Version identifier (str).

    Raises:
        ValueError: If no versions found.
    """
    client = MlflowClient()
    filter_str = f"name = '{model_name}'"
    if stage:
        filter_str += f" and current_stage = '{stage.upper()}'"
    versions = client.search_model_versions(filter_str)
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}' (stage={stage})")
    # pick highest numerical version
    return max(versions, key=lambda v: int(v.version)).version 