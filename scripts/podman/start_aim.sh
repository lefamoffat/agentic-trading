#!/usr/bin/env bash
# Start an Aim server via Podman for experiment tracking.
# Usage: ./scripts/podman/start_aim.sh [PORT]
set -euo pipefail
PORT="${1:-43800}"
CONTAINER_NAME="agentic_aim"
# Official Aim image hosts both the UI (cmd `aim up`) and the remote
# tracking server (cmd `aim server`). We override the default command to bind
# the UI to 0.0.0.0 (inside the container) so that Podman/Docker port-forwarding
# can expose it to the host.
IMAGE="aimstack/aim:3.29.1"

if podman ps --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  echo "Aim server '${CONTAINER_NAME}' already running on port ${PORT}."
  exit 0
fi

if podman ps -a --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  podman rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting Aim server on port ${PORT}..."
podman run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:43800" \
  -v "$(pwd)/.aim:/root/.aim" \
  "${IMAGE}" \
  aim up --host 0.0.0.0 --port 43800 > /dev/null

# Wait for the UI to respond (max 30 s). This ensures the caller only returns
# when the port is actually reachable from the host. If port-forwarding is
# disabled in `podman machine`, the check will time-out and emit a hint.
printf "Waiting for Aim UI to become reachable on http://localhost:%s ..." "${PORT}"
for _ in {1..30}; do
  if curl -sf "http://localhost:${PORT}" >/dev/null; then
    echo " done."
    echo "Aim server running at http://localhost:${PORT}"
    exit 0
  fi
  sleep 1
  printf '.'
done

echo "\n⚠️  Aim container started, but the port is not reachable on localhost:${PORT}." >&2
echo "   This usually means the Podman VM's automatic port forwarding is disabled or broken." >&2
echo "   Troubleshooting checklist:" >&2
echo "     • podman machine list            # confirm your default VM is running" >&2
echo "     • podman machine inspect         # look for \"PortForwarding\": true" >&2
echo "     • podman machine stop && podman machine start" >&2
echo "       (re-starts gVProxy which handles the forwarding)" >&2
echo "" >&2
echo "If the forwarding flag is already true and the restart did not help, please open" >&2
echo "an issue with \`podman machine inspect --format '{{json .}}'\` so we can reproduce." >&2

exit 1 