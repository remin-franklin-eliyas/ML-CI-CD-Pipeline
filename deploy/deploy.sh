#!/usr/bin/env bash
# =============================================================================
# deploy.sh – Pull the latest Docker image and (re)start the API container.
#
# Usage
# -----
#   ./deploy/deploy.sh [IMAGE_NAME] [IMAGE_TAG] [--dry-run]
#
# Examples
#   ./deploy/deploy.sh myorg/ml-iris-classifier latest
#   ./deploy/deploy.sh myorg/ml-iris-classifier abc1234 --dry-run
#
# Environment Variables
# ---------------------
#   CONTAINER_NAME   Name for the running container.  Default: ml-api
#   HOST_PORT        Host port to bind.               Default: 8000
#   CONTAINER_PORT   Container port.                  Default: 8000
# =============================================================================
set -euo pipefail

IMAGE_NAME="${1:-ml-iris-classifier}"
IMAGE_TAG="${2:-latest}"
DRY_RUN="${3:-}"

CONTAINER_NAME="${CONTAINER_NAME:-ml-api}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "======================================================"
echo "  ML Model Deployment Script"
echo "======================================================"
echo "  Image     : ${FULL_IMAGE}"
echo "  Container : ${CONTAINER_NAME}"
echo "  Port      : ${HOST_PORT} → ${CONTAINER_PORT}"
echo "------------------------------------------------------"

# ── Dry-run mode ──────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "[DRY-RUN] Steps that would be executed:"
    echo "  1.  docker pull ${FULL_IMAGE}"
    echo "  2.  docker stop  ${CONTAINER_NAME}  (if running)"
    echo "  3.  docker rm    ${CONTAINER_NAME}  (if exists)"
    echo "  4.  docker run -d --name ${CONTAINER_NAME} \\"
    echo "        -p ${HOST_PORT}:${CONTAINER_PORT} \\"
    echo "        --restart unless-stopped \\"
    echo "        -e MODEL_VERSION=${IMAGE_TAG} \\"
    echo "        ${FULL_IMAGE}"
    echo "  5.  Wait for /health to return HTTP 200"
    echo "[DRY-RUN] Deployment simulation complete ✓"
    exit 0
fi

# ── Pull latest image ────────────────────────────────────────────────────
echo "Pulling image …"
docker pull "${FULL_IMAGE}"

# ── Stop existing container ──────────────────────────────────────────────
if docker ps -q --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
    echo "Stopping running container '${CONTAINER_NAME}' …"
    docker stop "${CONTAINER_NAME}"
fi

if docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
    echo "Removing existing container '${CONTAINER_NAME}' …"
    docker rm "${CONTAINER_NAME}"
fi

# ── Start new container ──────────────────────────────────────────────────
echo "Starting new container …"
docker run -d \
    --name "${CONTAINER_NAME}" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    --restart unless-stopped \
    -e MODEL_VERSION="${IMAGE_TAG}" \
    "${FULL_IMAGE}"

# ── Health-check loop ────────────────────────────────────────────────────
echo "Waiting for the service to become healthy …"
MAX_ATTEMPTS=30
ATTEMPT=0
until curl -sf "http://localhost:${HOST_PORT}/health" > /dev/null 2>&1; do
    ATTEMPT=$(( ATTEMPT + 1 ))
    if [[ ${ATTEMPT} -ge ${MAX_ATTEMPTS} ]]; then
        echo "ERROR: Service did not become healthy after ${MAX_ATTEMPTS} attempts."
        echo "Container logs:"
        docker logs "${CONTAINER_NAME}" --tail 50
        exit 1
    fi
    sleep 2
done

echo "======================================================"
echo "  Deployment complete ✓"
echo "  API docs : http://localhost:${HOST_PORT}/docs"
echo "  Health   : http://localhost:${HOST_PORT}/health"
echo "======================================================"
