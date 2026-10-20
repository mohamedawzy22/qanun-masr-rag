from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from services.vectordb.vector_db_enum import DistanceMetric

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


class ChatModelSettings(BaseModel):
    provider: str
    model: str


class EmbeddingModelSettings(BaseModel):
    provider: str
    model: str
    dimension: int


class AppConfig(BaseModel):
    name: str
    version: str
    log_level: str


class FileConfig(BaseModel):
    allowed_types: list[str]
    max_size_mb: int
    default_chunk_size: int


class DatabaseConfig(BaseModel):
    mongodb_url: str
    mongodb_database: str
    timeout_ms: int
    connect_timeout_ms: int
    max_pool_size: int
    min_pool_size: int
    wait_queue_timeout_ms: int


class LLMConfig(BaseModel):
    openrouter_api_key: str | None
    openrouter_base_url: str
    ollama_base_url: str | None
    ollama_keep_alive: int
    request_timeout_seconds: int
    temperature: float
    max_tokens: int
    chat_max_concurrency: int
    embedding_max_concurrency: int
    model_warmup_enabled: bool
    model_warmup_timeout_seconds: int
    chat_models: dict[str, ChatModelSettings]
    embedding_models: dict[str, EmbeddingModelSettings]
    chat_primary_model: str
    chat_fallback_model: str
    chat_fallback_cooldown_seconds: int
    embedding_model: str


class QdrantConfig(BaseModel):
    url: str
    api_key: str | None
    collection_name: str
    vector_size: int
    distance: DistanceMetric
    request_timeout_seconds: int


class RetryConfig(BaseModel):
    max_attempts: int
    initial_delay_seconds: float
    max_delay_seconds: float


class Setting(BaseSettings):
    # -------------------------
    # Application
    # -------------------------
    APP_NAME: str
    APP_VERSION: str
    LOG_LEVEL: str = "INFO"

    # -------------------------
    # Model runtime
    # -------------------------
    MODEL_WARMUP_ENABLED: bool = True
    MODEL_WARMUP_TIMEOUT_SECONDS: int = 30
    OLLAMA_KEEP_ALIVE: int = 300
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 300
    LLM_REQUEST_TIMEOUT_SECONDS: int = 120
    LLM_CHAT_MAX_CONCURRENCY: int = 1
    LLM_EMBEDDING_MAX_CONCURRENCY: int = 2

    # -------------------------
    # Retry
    # -------------------------
    PROVIDER_RETRY_MAX_ATTEMPTS: int = 3
    PROVIDER_RETRY_INITIAL_DELAY_SECONDS: float = 0.5
    PROVIDER_RETRY_MAX_DELAY_SECONDS: float = 4.0

    # -------------------------
    # MinIO object storage
    # -------------------------
    MINIO_ENDPOINT: str = "localhost:9091"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str = "qanun-masr-assets"
    MINIO_SECURE: bool = False

    # -------------------------
    # Files
    # -------------------------
    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    # -------------------------
    # MongoDB
    # -------------------------
    MONGODB_URL: str
    MONGODB_DATABASE: str
    MONGODB_TIMEOUT_MS: int = 10000
    MONGODB_CONNECT_TIMEOUT_MS: int = 5000
    MONGODB_MAX_POOL_SIZE: int = 50
    MONGODB_MIN_POOL_SIZE: int = 5
    MONGODB_WAIT_QUEUE_TIMEOUT_MS: int = 5000

    # -------------------------
    # LLM providers
    # -------------------------
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OLLAMA_BASE_URL: str | None = None

    # -------------------------
    # Models
    # -------------------------
    CHAT_MODELS: dict[str, ChatModelSettings]
    EMBEDDING_MODELS: dict[str, EmbeddingModelSettings]
    CHAT_PRIMARY_MODEL: str
    CHAT_FALLBACK_MODEL: str
    CHAT_FALLBACK_COOLDOWN_SECONDS: int = 60
    EMBEDDING_MODEL: str

    # -------------------------
    # Qdrant
    # -------------------------
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str
    QDRANT_VECTOR_SIZE: int
    QDRANT_DISTANCE: DistanceMetric
    QDRANT_REQUEST_TIMEOUT_SECONDS: int = 30

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
    )

    # -------------------------
    # Langfuse
    # -------------------------
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_TRACING_ENVIRONMENT: str = "development"

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.CHAT_PRIMARY_MODEL not in self.CHAT_MODELS:
            raise ValueError(
                f"Chat primary model '{self.CHAT_PRIMARY_MODEL}' is not configured"
            )
        if self.CHAT_FALLBACK_MODEL not in self.CHAT_MODELS:
            raise ValueError(
                f"Chat fallback model '{self.CHAT_FALLBACK_MODEL}' is not configured"
            )
        if self.CHAT_PRIMARY_MODEL == self.CHAT_FALLBACK_MODEL:
            raise ValueError("Chat primary and fallback models must be different")
        if self.EMBEDDING_MODEL not in self.EMBEDDING_MODELS:
            raise ValueError(
                f"Embedding model '{self.EMBEDDING_MODEL}' is not configured"
            )

        if not 0.0 <= self.LLM_TEMPERATURE <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0")
        if self.CHAT_FALLBACK_COOLDOWN_SECONDS < 0:
            raise ValueError("CHAT_FALLBACK_COOLDOWN_SECONDS cannot be negative")
        if self.LLM_MAX_TOKENS <= 0:
            raise ValueError("LLM_MAX_TOKENS must be greater than zero")
        if self.MODEL_WARMUP_TIMEOUT_SECONDS <= 0:
            raise ValueError("MODEL_WARMUP_TIMEOUT_SECONDS must be greater than zero")
        if self.OLLAMA_KEEP_ALIVE == 0 or self.OLLAMA_KEEP_ALIVE < -1:
            raise ValueError("OLLAMA_KEEP_ALIVE must be -1 or greater than zero")
        if self.LLM_REQUEST_TIMEOUT_SECONDS <= 0:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be greater than zero")
        if self.LLM_CHAT_MAX_CONCURRENCY <= 0:
            raise ValueError("LLM_CHAT_MAX_CONCURRENCY must be greater than zero")
        if self.LLM_EMBEDDING_MAX_CONCURRENCY <= 0:
            raise ValueError("LLM_EMBEDDING_MAX_CONCURRENCY must be greater than zero")

        if self.PROVIDER_RETRY_MAX_ATTEMPTS <= 0:
            raise ValueError("PROVIDER_RETRY_MAX_ATTEMPTS must be greater than zero")
        if self.PROVIDER_RETRY_INITIAL_DELAY_SECONDS < 0:
            raise ValueError("PROVIDER_RETRY_INITIAL_DELAY_SECONDS cannot be negative")
        if self.PROVIDER_RETRY_MAX_DELAY_SECONDS < 0:
            raise ValueError("PROVIDER_RETRY_MAX_DELAY_SECONDS cannot be negative")
        if (
            self.PROVIDER_RETRY_INITIAL_DELAY_SECONDS
            > self.PROVIDER_RETRY_MAX_DELAY_SECONDS
        ):
            raise ValueError(
                "PROVIDER_RETRY_INITIAL_DELAY_SECONDS "
                "cannot exceed "
                "PROVIDER_RETRY_MAX_DELAY_SECONDS"
            )

        if self.FILE_MAX_SIZE <= 0:
            raise ValueError("FILE_MAX_SIZE must be greater than zero")
        if self.FILE_DEFAULT_CHUNK_SIZE <= 0:
            raise ValueError("FILE_DEFAULT_CHUNK_SIZE must be greater than zero")

        if self.MONGODB_TIMEOUT_MS <= 0:
            raise ValueError("MONGODB_TIMEOUT_MS must be greater than zero")
        if self.MONGODB_CONNECT_TIMEOUT_MS <= 0:
            raise ValueError("MONGODB_CONNECT_TIMEOUT_MS must be greater than zero")
        if self.MONGODB_MAX_POOL_SIZE <= 0:
            raise ValueError("MONGODB_MAX_POOL_SIZE must be greater than zero")
        if self.MONGODB_MIN_POOL_SIZE < 0:
            raise ValueError("MONGODB_MIN_POOL_SIZE cannot be negative")
        if self.MONGODB_MIN_POOL_SIZE > self.MONGODB_MAX_POOL_SIZE:
            raise ValueError(
                "MONGODB_MIN_POOL_SIZE cannot exceed MONGODB_MAX_POOL_SIZE"
            )
        if self.MONGODB_WAIT_QUEUE_TIMEOUT_MS <= 0:
            raise ValueError("MONGODB_WAIT_QUEUE_TIMEOUT_MS must be greater than zero")

        if self.QDRANT_REQUEST_TIMEOUT_SECONDS <= 0:
            raise ValueError("QDRANT_REQUEST_TIMEOUT_SECONDS must be greater than zero")
        if self.QDRANT_VECTOR_SIZE <= 0:
            raise ValueError("QDRANT_VECTOR_SIZE must be greater than zero")
        embedding_dimension = self.EMBEDDING_MODELS[self.EMBEDDING_MODEL].dimension
        if embedding_dimension != self.QDRANT_VECTOR_SIZE:
            raise ValueError("Embedding dimension does not match Qdrant vector size")
        return self

    # ============================================================
    # Structured configuration
    # ============================================================
    @cached_property
    def app(self) -> AppConfig:
        return AppConfig(
            name=self.APP_NAME,
            version=self.APP_VERSION,
            log_level=self.LOG_LEVEL,
        )

    @cached_property
    def files(self) -> FileConfig:
        return FileConfig(
            allowed_types=self.FILE_ALLOWED_TYPES,
            max_size_mb=self.FILE_MAX_SIZE,
            default_chunk_size=self.FILE_DEFAULT_CHUNK_SIZE,
        )

    @cached_property
    def database(self) -> DatabaseConfig:
        return DatabaseConfig(
            mongodb_url=self.MONGODB_URL,
            mongodb_database=self.MONGODB_DATABASE,
            timeout_ms=self.MONGODB_TIMEOUT_MS,
            connect_timeout_ms=self.MONGODB_CONNECT_TIMEOUT_MS,
            max_pool_size=self.MONGODB_MAX_POOL_SIZE,
            min_pool_size=self.MONGODB_MIN_POOL_SIZE,
            wait_queue_timeout_ms=self.MONGODB_WAIT_QUEUE_TIMEOUT_MS,
        )

    @cached_property
    def llm(self) -> LLMConfig:
        return LLMConfig(
            openrouter_api_key=self.OPENROUTER_API_KEY,
            openrouter_base_url=self.OPENROUTER_BASE_URL,
            ollama_base_url=self.OLLAMA_BASE_URL,
            ollama_keep_alive=self.OLLAMA_KEEP_ALIVE,
            request_timeout_seconds=self.LLM_REQUEST_TIMEOUT_SECONDS,
            temperature=self.LLM_TEMPERATURE,
            max_tokens=self.LLM_MAX_TOKENS,
            chat_max_concurrency=self.LLM_CHAT_MAX_CONCURRENCY,
            embedding_max_concurrency=self.LLM_EMBEDDING_MAX_CONCURRENCY,
            model_warmup_enabled=self.MODEL_WARMUP_ENABLED,
            model_warmup_timeout_seconds=self.MODEL_WARMUP_TIMEOUT_SECONDS,
            chat_models=self.CHAT_MODELS,
            embedding_models=self.EMBEDDING_MODELS,
            chat_primary_model=self.CHAT_PRIMARY_MODEL,
            chat_fallback_model=self.CHAT_FALLBACK_MODEL,
            chat_fallback_cooldown_seconds=self.CHAT_FALLBACK_COOLDOWN_SECONDS,
            embedding_model=self.EMBEDDING_MODEL,
        )

    @cached_property
    def qdrant(self) -> QdrantConfig:
        return QdrantConfig(
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY,
            collection_name=self.QDRANT_COLLECTION_NAME,
            vector_size=self.QDRANT_VECTOR_SIZE,
            distance=self.QDRANT_DISTANCE,
            request_timeout_seconds=self.QDRANT_REQUEST_TIMEOUT_SECONDS,
        )

    @cached_property
    def retry(self) -> RetryConfig:
        return RetryConfig(
            max_attempts=self.PROVIDER_RETRY_MAX_ATTEMPTS,
            initial_delay_seconds=self.PROVIDER_RETRY_INITIAL_DELAY_SECONDS,
            max_delay_seconds=self.PROVIDER_RETRY_MAX_DELAY_SECONDS,
        )


def get_setting() -> Setting:
    return Setting()
