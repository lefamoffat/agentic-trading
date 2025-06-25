#!/usr/bin/env bash
# Start a disposable Redis container with Podman.
# Usage: ./scripts/podman/start_redis.sh [PORT]
# Defaults: PORT=6379  CONTAINER_NAME=agentic_redis
set -euo pipefail

PORT="${1:-6379}"
CONTAINER_NAME="agentic_redis"
IMAGE="docker.io/library/redis:7-alpine"

# If container already running, just print info
if podman ps --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  echo "Redis container '${CONTAINER_NAME}' already running on port ${PORT}."
  exit 0
fi

# Remove any stopped container with same name
if podman ps -a --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  podman rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting Redis (${IMAGE}) on port ${PORT}..."
podman run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:6379" \
  --health-cmd "redis-cli ping" \
  --health-interval 5s \
  --health-timeout 2s \
  --health-retries 5 \
  "${IMAGE}" > /dev/null

# Wait for the TCP port to open (max 30 s)
printf "Waiting for Redis to accept connections on localhost:%s ..." "${PORT}"
for _ in {1..30}; do
  if nc -z localhost "${PORT}" 2>/dev/null; then
    echo " done."
    echo "Redis server running at redis://localhost:${PORT} (container: ${CONTAINER_NAME})"
    exit 0
  fi
  sleep 1
  printf '.'
done

echo "\n⚠️  Redis container started, but port localhost:${PORT} is not reachable." >&2
echo "   This typically indicates the Podman VM's port-forwarding proxy (gvproxy) is not active." >&2
echo "   Troubleshooting checklist:" >&2
echo "     • podman machine list            # confirm the VM is running" >&2
echo "     • podman machine inspect         # verify \"PortForwarding\": true" >&2
echo "     • podman machine stop && podman machine start" >&2
echo "       (restarts the proxy)" >&2
exit 1 