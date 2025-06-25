#!/usr/bin/env bash
# Stop and remove the Redis podman container started by start_redis.sh
set -euo pipefail
CONTAINER_NAME="agentic_redis"

if podman ps --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  echo "Stopping Redis container '${CONTAINER_NAME}'..."
  podman stop "${CONTAINER_NAME}" >/dev/null
fi

if podman ps -a --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  echo "Removing Redis container '${CONTAINER_NAME}'..."
  podman rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Redis container cleaned up." 