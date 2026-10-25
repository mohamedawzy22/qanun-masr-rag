APP_NAME="rag-system"
APP_VERSION="0.1.0"

FILE_ALLOWED_TYPES=["application/pdf", "text/plain"]
FILE_MAX_SIZE=10
FILE_DEFAULT_CHUNK_SIZE=512000

MONGODB_URL="mongodb://admin:admin@localhost:27007"
MONGODB_DATABASE="rag-system"

MONGODB_TIMEOUT_MS=10000
MONGODB_CONNECT_TIMEOUT_MS=5000
MONGODB_MAX_POOL_SIZE=50
MONGODB_MIN_POOL_SIZE=5
MONGODB_WAIT_QUEUE_TIMEOUT_MS=5000

OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

OLLAMA_BASE_URL="http://172.31.128.1:11434"
CHAT_PRIMARY_MODEL="qwen2.5"
CHAT_FALLBACK_MODEL="minimax"

CHAT_MODELS='{
    "minimax": {
        "provider": "openrouter",
        "model": "minimax/minimax-m3.1-flash:free"
    },
    "qwen2.5": {
        "provider": "ollama",
        "model": "qwen2.5:3b"
    },
    "qwen3": {
        "provider": "ollama",
        "model": "qwen3:8b"
    }
}'

EMBEDDING_MODEL="bge-m3"

EMBEDDING_MODELS='{
    "bge-m3": {
        "provider": "ollama",
        "model": "bge-m3",
        "dimension": 1024
    }
}'

QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY=""

QDRANT_COLLECTION_NAME="documents"
QDRANT_VECTOR_SIZE=1024
QDRANT_DISTANCE="cosine"
QDRANT_REQUEST_TIMEOUT_SECONDS=30


MODEL_WARMUP_ENABLED=true
MODEL_WARMUP_TIMEOUT_SECONDS=30
OLLAMA_KEEP_ALIVE=300

CHAT_FALLBACK_COOLDOWN_SECONDS=60

LLM_REQUEST_TIMEOUT_SECONDS=120
LLM_CHAT_MAX_CONCURRENCY=1
LLM_EMBEDDING_MAX_CONCURRENCY=2
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.0

PROVIDER_RETRY_MAX_ATTEMPTS=3
PROVIDER_RETRY_INITIAL_DELAY_SECONDS=0.5
PROVIDER_RETRY_MAX_DELAY_SECONDS=4


# -------------------------
# Langfuse
# -------------------------

LANGFUSE_PUBLIC_KEY=pk-lf-c6463f8131a46a70cc8329dfbe14b517
LANGFUSE_SECRET_KEY=sk-lf-eeceae90e6d7fbf569661bff087a4c8b2b4acc565473a8f3e88b9c0aa5b92c39
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_TRACING_ENVIRONMENT=development


# MinIO object storage for uploaded legal documents.
MINIO_ENDPOINT="localhost:9091"

# Replace these placeholders with your actual MinIO credentials.
MINIO_ACCESS_KEY="minio"
MINIO_SECRET_KEY="2b9392ab1edb4f6449b4fbe1757d7618"

MINIO_BUCKET_NAME="qanun-masr-assets"
MINIO_SECURE="false"
MONGODB_URL_DOCKER='mongodb://admin:admin@mongo:27017/rag-system?authSource=admin'
OLLAMA_BASE_URL_DOCKER="http://host.docker.internal:11434"
LANGFUSE_BASE_URL_DOCKER="http://langfuse-web:3000"

ONLINE_EVALUATION_INTERVAL_SECONDS=300
