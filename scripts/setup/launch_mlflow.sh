#!/bin/bash
#
# Launch a local MLflow tracking server in Docker.
#
# Usage:
#   ./scripts/setup/launch_mlflow.sh [HOST_PORT]
#
# If HOST_PORT is omitted it defaults to 5001.
# -----------------------------------------------------------------------------
set -e

# ------------------------ Config -------------------------------------------
HOST_PORT=${1:-5001}     # First CLI arg or 5001
CONTAINER_NAME="mlflow_server"

# Get the project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
MLFLOW_DATA_DIR="$PROJECT_ROOT/mlflow_data"

# Create the data directory if it doesn't exist
for sub in "" "/models"; do
  if [ ! -d "$MLFLOW_DATA_DIR$sub" ]; then
    echo "Creating directory: $MLFLOW_DATA_DIR$sub"
    mkdir -p "$MLFLOW_DATA_DIR$sub"
  fi
done

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running."
    echo "Please start Docker and try again."
    exit 1
fi

# Check for and forcibly remove any existing container with the same name
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Found existing $CONTAINER_NAME container. Forcibly removing it..."
    docker rm -f $CONTAINER_NAME
fi

# ------------------------ Launch -------------------------------------------
echo "Launching MLflow Tracking Server on port $HOST_PORT..."

# MLflow Docker image (official).  Override via MLFLOW_IMAGE env var or 2nd CLI arg.
IMAGE_TAG=${MLFLOW_IMAGE:-${2:-ghcr.io/mlflow/mlflow:v3.1.0}}

# We mount the project-local `mlflow_data` directory at /mlflow inside the
# container.  That path is used both for the backend store (runs,
# experiments, metrics) *and* as the artifact root and model-registry
# location, so everything is persisted on the host.

docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$HOST_PORT":5000 \
  -v "$MLFLOW_DATA_DIR":/mlflow \
  --user $(id -u):$(id -g) \
  "$IMAGE_TAG" mlflow server --host 0.0.0.0 --port 5000 \
      --backend-store-uri /mlflow \
      --artifacts-destination /mlflow \
      --serve-artifacts

echo "MLflow server started successfully."
echo "Access the UI at: http://localhost:$HOST_PORT"
echo "Stop with: docker stop $CONTAINER_NAME" 