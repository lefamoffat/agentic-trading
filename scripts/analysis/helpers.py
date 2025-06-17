from concurrent.futures import ThreadPoolExecutor, TimeoutError
from mlflow.tracking import MlflowClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _safe_list_models(client: MlflowClient):
    """Return list of registered models handling MLflow API differences."""
    try:
        return client.list_registered_models()
    except AttributeError:
        return list(client.search_registered_models())

def fetch_models_with_timeout(client: MlflowClient, timeout: int = 3):
    """Return registered models or empty list if server unavailable within timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_safe_list_models, client)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.warning("Timed-out connecting to MLflow server (>%ss)", timeout)
            return []
