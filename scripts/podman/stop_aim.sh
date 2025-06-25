#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="agentic_aim"

if podman ps --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  echo "Stopping Aim server '${CONTAINER_NAME}'..."
  podman stop "${CONTAINER_NAME}" >/dev/null
fi

if podman ps -a --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  echo "Removing Aim server '${CONTAINER_NAME}' container..."
  podman rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Aim server container cleaned up." 