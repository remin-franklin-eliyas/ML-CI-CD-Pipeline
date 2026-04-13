#!/usr/bin/env bash
# =============================================================================
# blue_green_deploy.sh – Zero-downtime blue/green deployment simulation.
#
# Strategy
# --------
# Two container "slots" (blue / green) run in parallel on different ports.
# Traffic is switched from the active slot to the idle slot only after the
# new slot passes a health check.  The old slot is then stopped.
#
# Usage
# -----
#   ./deploy/blue_green_deploy.sh [IMAGE_NAME] [IMAGE_TAG]
#
# Prerequisites: Docker must be running.  No external load balancer required
# for the simulation – the "switch" is printed as the port that is now active.
# =============================================================================
set -euo pipefail

IMAGE_NAME="${1:-ml-iris-classifier}"
IMAGE_TAG="${2:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

BLUE_CONTAINER="ml-api-blue"
GREEN_CONTAINER="ml-api-green"
BLUE_PORT=8001
GREEN_PORT=8002

# Helper ─────────────────────────────────────────────────────────────────────
is_running() { docker ps -q --filter "name=^/${1}$" | grep -q .; }
is_present() { docker ps -aq --filter "name=^/${1}$" | grep -q .; }

wait_healthy() {
    local port="$1"
    echo "  Waiting for health-check on port ${port} …"
    for _ in $(seq 1 30); do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  Service on port ${port} is healthy ✓"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: Service on port ${port} did not become healthy."
    return 1
}

# ── Determine active / idle slot ─────────────────────────────────────────────
if is_running "${BLUE_CONTAINER}"; then
    ACTIVE_COLOR="blue"
    ACTIVE_CONTAINER="${BLUE_CONTAINER}"
    ACTIVE_PORT="${BLUE_PORT}"
    IDLE_COLOR="green"
    IDLE_CONTAINER="${GREEN_CONTAINER}"
    IDLE_PORT="${GREEN_PORT}"
else
    ACTIVE_COLOR="green"
    ACTIVE_CONTAINER="${GREEN_CONTAINER}"
    ACTIVE_PORT="${GREEN_PORT}"
    IDLE_COLOR="blue"
    IDLE_CONTAINER="${BLUE_CONTAINER}"
    IDLE_PORT="${BLUE_PORT}"
fi

echo "============================================================"
echo "  Blue-Green Deployment"
echo "============================================================"
echo "  Image    : ${FULL_IMAGE}"
echo "  Active   : ${ACTIVE_COLOR} (port ${ACTIVE_PORT})"
echo "  Deploying: ${IDLE_COLOR}  (port ${IDLE_PORT})"
echo "------------------------------------------------------------"

# ── Pull new image ────────────────────────────────────────────────────────────
echo "Pulling ${FULL_IMAGE} …"
docker pull "${FULL_IMAGE}"

# ── Remove stale idle container ───────────────────────────────────────────────
if is_present "${IDLE_CONTAINER}"; then
    echo "Removing stale '${IDLE_CONTAINER}' …"
    docker stop "${IDLE_CONTAINER}" 2>/dev/null || true
    docker rm   "${IDLE_CONTAINER}"
fi

# ── Start idle slot with new image ────────────────────────────────────────────
echo "Starting '${IDLE_CONTAINER}' on port ${IDLE_PORT} …"
docker run -d \
    --name "${IDLE_CONTAINER}" \
    -p "${IDLE_PORT}:8000" \
    --restart unless-stopped \
    -e MODEL_VERSION="${IMAGE_TAG}" \
    "${FULL_IMAGE}"

# ── Health-check the new slot ─────────────────────────────────────────────────
if ! wait_healthy "${IDLE_PORT}"; then
    echo "New slot unhealthy – aborting deployment, keeping ${ACTIVE_COLOR} active."
    docker stop "${IDLE_CONTAINER}" || true
    docker rm   "${IDLE_CONTAINER}" || true
    exit 1
fi

# ── Switch traffic (simulation: print the active port) ───────────────────────
echo ""
echo "  ✓  Traffic switched from ${ACTIVE_COLOR}:${ACTIVE_PORT} → ${IDLE_COLOR}:${IDLE_PORT}"
echo "  Update your load balancer / nginx upstream to port ${IDLE_PORT}."
echo ""

# ── Stop old slot ─────────────────────────────────────────────────────────────
if is_running "${ACTIVE_CONTAINER}"; then
    echo "Stopping old '${ACTIVE_CONTAINER}' (port ${ACTIVE_PORT}) …"
    docker stop "${ACTIVE_CONTAINER}"
    docker rm   "${ACTIVE_CONTAINER}"
fi

echo "============================================================"
echo "  Blue-Green deployment complete ✓"
echo "  Active slot : ${IDLE_COLOR} on port ${IDLE_PORT}"
echo "============================================================"
