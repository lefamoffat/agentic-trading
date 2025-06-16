#!/bin/bash
#
# This script launches a local MLflow tracking server using Docker.
#
# The server will be accessible at http://localhost:5000.
# Artifacts and data will be stored in the `mlflow_data` directory
# in the project root.
#
set -e

# Get the project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
MLFLOW_DATA_DIR="$PROJECT_ROOT/mlflow_data"

# Create the data directory if it doesn't exist
if [ ! -d "$MLFLOW_DATA_DIR" ]; then
    echo "Creating MLflow data directory at: $MLFLOW_DATA_DIR"
    mkdir -p "$MLFLOW_DATA_DIR"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running."
    echo "Please start Docker and try again."
    exit 1
fi

# Check for and forcibly remove any existing container with the same name
if [ "$(docker ps -aq -f name=mlflow_server)" ]; then
    echo "Found existing mlflow_server container. Forcibly removing it..."
    docker rm -f mlflow_server
fi

echo "Launching MLflow Tracking Server on port 5001..."
docker run -d \
    --name mlflow_server \
    -p 5001:5000 \
    -v "$MLFLOW_DATA_DIR:/app/mlruns" \
    bitnami/mlflow:latest

echo "MLflow server started successfully."
echo "Access it at: http://localhost:5001"
echo "To stop the server, run: docker stop mlflow_server" 