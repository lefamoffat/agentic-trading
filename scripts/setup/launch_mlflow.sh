#!/bin/bash
# This script launches a local MLflow tracking server using Docker.

set -e

HOST_PORT=5000
CONTAINER_PORT=5000
SERVER_NAME="agentic_trading_mlflow_server"
DATA_DIR_NAME="mlflow_data"

# Get the absolute path of the project root
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../")
DATA_PATH="$PROJECT_ROOT/$DATA_DIR_NAME"
ARTIFACT_ROOT="/mlflow_artifacts_store"

# Create the data directory if it doesn't exist
mkdir -p "$DATA_PATH"

echo "Project Root: $PROJECT_ROOT"
echo "MLflow Data Path (Host): $DATA_PATH"

# Check if the container is already running
if [ "$(docker ps -q -f name=^/${SERVER_NAME}$)" ]; then
    echo "MLflow server is already running."
    echo "Access it at http://localhost:$HOST_PORT"
    echo "To stop it, run: docker stop $SERVER_NAME"
    exit 0
fi

# Check if the container exists but is stopped
if [ "$(docker ps -aq -f status=exited -f name=^/${SERVER_NAME}$)" ]; then
    echo "Found a stopped MLflow server container. Removing it..."
    docker rm "$SERVER_NAME"
fi

echo "Starting new MLflow server..."
docker run -d \
    --name "$SERVER_NAME" \
    -p "$HOST_PORT":"$CONTAINER_PORT" \
    -v "$DATA_PATH":"$ARTIFACT_ROOT" \
    evk02/mlflow:latest \
    mlflow server \
    --host 0.0.0.0 \
    --port "$CONTAINER_PORT" \
    --backend-store-uri "$ARTIFACT_ROOT" \
    --default-artifact-root "$ARTIFACT_ROOT"

echo ""
echo "✅ MLflow server started successfully in the background."
echo "   UI available at: http://localhost:$HOST_PORT"
echo ""
echo "To view logs, run: docker logs -f $SERVER_NAME"
echo "To stop the server, run: docker stop $SERVER_NAME" 