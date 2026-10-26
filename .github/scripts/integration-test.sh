#!/usr/bin/env bash
set -Eeuo pipefail

: "${IMAGE:?IMAGE must contain the built API image tag}"

suffix="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
prefix="rag-integration-${suffix}"
network="${prefix}-net"
minio_volume="${prefix}-minio-data"

mongo_name="${prefix}-mongo"
qdrant_name="${prefix}-qdrant"
minio_name="${prefix}-minio"
ollama_name="${prefix}-ollama"
api_name="${prefix}-api"

work_dir="$(mktemp -d)"
env_file="${work_dir}/integration.env"
health_file="${work_dir}/health.json"

cleanup() {
    local exit_code=$?
    trap - EXIT

    if (( exit_code != 0 )); then
        echo "::error::Integration test failed. Printing service logs."
        for container_name in \
            "$api_name" \
            "$ollama_name" \
            "$minio_name" \
            "$qdrant_name" \
            "$mongo_name"; do
            if docker container inspect "$container_name" >/dev/null 2>&1; then
                echo "========== LOGS: ${container_name} =========="
                docker logs --tail=120 "$container_name" 2>&1 || true
            fi
        done
    fi

    for container_name in \
        "$api_name" \
        "$ollama_name" \
        "$minio_name" \
        "$qdrant_name" \
        "$mongo_name"; do
        docker rm -f "$container_name" >/dev/null 2>&1 || true
    done

    docker network rm "$network" >/dev/null 2>&1 || true
    docker volume rm "$minio_volume" >/dev/null 2>&1 || true
    rm -rf "$work_dir" || true
    exit "$exit_code"
}
trap cleanup EXIT

wait_for_http() {
    local label="$1"
    local url="$2"
    local max_attempts="${3:-60}"
    local attempt

    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        if curl --fail --silent --max-time 3 "$url" >/dev/null 2>&1; then
            echo "Ready: ${label}"
            return 0
        fi
        sleep 2
    done

    echo "::error::Timed out waiting for ${label}: ${url}" >&2
    return 1
}

wait_for_mongo() {
    local attempt
    for ((attempt = 1; attempt <= 60; attempt++)); do
        if docker exec "$mongo_name" mongosh --quiet \
            --eval 'db.adminCommand({ ping: 1 }).ok' >/dev/null 2>&1; then
            echo "Ready: MongoDB"
            return 0
        fi
        sleep 2
    done

    echo "::error::MongoDB did not become ready." >&2
    return 1
}

echo "Starting an isolated, disposable integration environment."
echo "API image: ${IMAGE}"

docker network create "$network" >/dev/null
docker volume create "$minio_volume" >/dev/null

# MongoDB: real database connectivity and index/collection initialization.
docker run --detach \
    --name "$mongo_name" \
    --network "$network" \
    --network-alias mongo \
    mongo:8 >/dev/null

# Qdrant: real collection initialization.
docker run --detach \
    --name "$qdrant_name" \
    --network "$network" \
    --network-alias qdrant \
    --publish 127.0.0.1:16333:6333 \
    qdrant/qdrant:v1.13.6 >/dev/null

# MinIO: use the same image family as docker-compose, but isolated test credentials/data.
docker run --detach \
    --name "$minio_name" \
    --network "$network" \
    --network-alias minio \
    --volume "${minio_volume}:/data" \
    --publish 127.0.0.1:19000:9000 \
    --env MINIO_ROOT_USER=ci-minio-user \
    --env MINIO_ROOT_PASSWORD=ci-minio-password-123 \
    --entrypoint sh \
    cgr.dev/chainguard/minio \
    -c 'mkdir -p /data/integration && minio server --address ":9000" --console-address ":9001" /data' >/dev/null

# Ollama: real local model health checks; no external model API key is needed.
docker run --detach \
    --name "$ollama_name" \
    --network "$network" \
    --network-alias ollama \
    --publish 127.0.0.1:11435:11434 \
    --env OLLAMA_HOST=0.0.0.0:11434 \
    ollama/ollama:latest >/dev/null

wait_for_mongo
wait_for_http "Qdrant" "http://127.0.0.1:16333/readyz" 90
wait_for_http "MinIO" "http://127.0.0.1:19000/minio/health/ready" 90
wait_for_http "Ollama" "http://127.0.0.1:11435/api/tags" 90

# Small disposable CI models. The integration job has a 30-minute timeout.
docker exec "$ollama_name" ollama pull qwen2.5:0.5b
docker exec "$ollama_name" ollama pull all-minilm:33m

# Test-only configuration. No repository production secrets are used.
umask 077
cat > "$env_file" <<'ENV'
APP_NAME=rag-system
APP_VERSION=ci-integration
LOG_LEVEL=INFO

FILE_ALLOWED_TYPES=["application/pdf","text/plain"]
FILE_MAX_SIZE=10
FILE_DEFAULT_CHUNK_SIZE=512

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=ci-minio-user
MINIO_SECRET_KEY=ci-minio-password-123
MINIO_BUCKET_NAME=rag-integration-assets
MINIO_SECURE=false

MONGODB_URL=mongodb://mongo:27017
MONGODB_DATABASE=rag_ci
MONGODB_TIMEOUT_MS=10000
MONGODB_CONNECT_TIMEOUT_MS=5000
MONGODB_MAX_POOL_SIZE=10
MONGODB_MIN_POOL_SIZE=0
MONGODB_WAIT_QUEUE_TIMEOUT_MS=5000

CHAT_MODELS={"ci-primary":{"provider":"ollama","model":"qwen2.5:0.5b"},"ci-fallback":{"provider":"ollama","model":"qwen2.5:0.5b"}}
CHAT_PRIMARY_MODEL=ci-primary
CHAT_FALLBACK_MODEL=ci-fallback
CHAT_FALLBACK_COOLDOWN_SECONDS=60

EMBEDDING_MODELS={"ci-embedding":{"provider":"ollama","model":"all-minilm:33m","dimension":384}}
EMBEDDING_MODEL=ci-embedding

OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_KEEP_ALIVE=120
MODEL_WARMUP_ENABLED=true
MODEL_WARMUP_TIMEOUT_SECONDS=180
LLM_REQUEST_TIMEOUT_SECONDS=180
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=16
LLM_CHAT_MAX_CONCURRENCY=1
LLM_EMBEDDING_MAX_CONCURRENCY=1
PROVIDER_RETRY_MAX_ATTEMPTS=1
PROVIDER_RETRY_INITIAL_DELAY_SECONDS=0.5
PROVIDER_RETRY_MAX_DELAY_SECONDS=1

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=rag-integration-test
QDRANT_VECTOR_SIZE=384
QDRANT_DISTANCE=cosine
QDRANT_REQUEST_TIMEOUT_SECONDS=30

LANGFUSE_PUBLIC_KEY=pk-lf-ci-integration
LANGFUSE_SECRET_KEY=sk-lf-ci-integration
LANGFUSE_BASE_URL=http://127.0.0.1:9
LANGFUSE_TRACING_ENVIRONMENT=ci
ENV

# Start the built image with its normal CMD. Lifespan stays enabled here.
docker run --detach \
    --init \
    --name "$api_name" \
    --network "$network" \
    --publish 127.0.0.1:18002:8001 \
    --env-file "$env_file" \
    "$IMAGE" >/dev/null

echo "Waiting for full application startup and health endpoint."
api_ready=0

for ((attempt = 1; attempt <= 180; attempt++)); do
    if curl --fail --silent --max-time 3 \
        http://127.0.0.1:18002/api/v1/health \
        --output "$health_file" 2>/dev/null; then
        if python3 - "$health_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    payload = json.load(file)

valid = (
    payload.get("API_Name") == "rag-system"
    and payload.get("API_Version") == "ci-integration"
)
sys.exit(0 if valid else 1)
PY
        then
            api_ready=1
            break
        fi
    fi

    container_state="$(docker inspect --format '{{.State.Status}}' "$api_name" 2>/dev/null || true)"
    if [[ "$container_state" != "running" ]]; then
        echo "::error::API container stopped before startup completed."
        break
    fi

    sleep 2
done

if (( api_ready != 1 )); then
    echo "::error::Full application startup or health check failed."
    exit 1
fi


curl --fail --silent --show-error --max-time 5 \
    http://127.0.0.1:16333/collections/rag-integration-test |
python3 -c '
import json
import sys
payload = json.load(sys.stdin)
if payload.get("status") != "ok" or not payload.get("result"):
    raise SystemExit("Qdrant integration-test collection was not found.")
print("Qdrant collection verified.")
'

echo "Full startup integration test passed."
