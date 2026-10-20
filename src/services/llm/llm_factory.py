from langfuse import Langfuse

from helpers.config import (
    ChatModelSettings,
    EmbeddingModelSettings,
    Setting,
)
from utils.logger import get_logger

from .chat_interface import ChatModel
from .embedding_interface import EmbeddingModel
from .providers.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
)
from .providers.openrouter import (
    OpenRouterChatModel,
    OpenRouterEmbeddingModel,
)

logger = get_logger(__name__)


class ModelFactory:
    def __init__(
        self,
        settings: Setting,
        langfuse: Langfuse,
    ):
        self.settings = settings
        self.langfuse = langfuse

    def create_chat_model(
        self,
        config: ChatModelSettings,
    ) -> ChatModel:
        logger.debug(
            "Creating chat model: provider=%s | model=%s",
            config.provider,
            config.model,
        )

        llm_config = self.settings.llm

        if config.provider == "ollama":
            if not llm_config.ollama_base_url:
                raise ValueError(
                    "OLLAMA_BASE_URL is required",
                )

            return OllamaChatModel(
                model_id=config.model,
                base_url=llm_config.ollama_base_url,
                temperature=llm_config.temperature,
                keep_alive=llm_config.ollama_keep_alive,
                timeout_seconds=llm_config.request_timeout_seconds,
                max_tokens=llm_config.max_tokens,
                max_concurrency=llm_config.chat_max_concurrency,
                retry_config=self.settings.retry,
                langfuse=self.langfuse,
            )

        if config.provider == "openrouter":
            if not self.settings.OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY is required",
                )

            return OpenRouterChatModel(
                model_id=config.model,
                api_key=self.settings.OPENROUTER_API_KEY,
                base_url=llm_config.openrouter_base_url,
                temperature=llm_config.temperature,
                timeout_seconds=llm_config.request_timeout_seconds,
                max_tokens=llm_config.max_tokens,
                max_concurrency=llm_config.chat_max_concurrency,
                retry_config=self.settings.retry,
                langfuse=self.langfuse,
            )

        raise ValueError(
            f"Unsupported chat provider: {config.provider}",
        )

    def create_embedding_model(
        self,
        config: EmbeddingModelSettings,
    ) -> EmbeddingModel:
        logger.debug(
            "Creating embedding model: provider=%s | model=%s | dimension=%s",
            config.provider,
            config.model,
            config.dimension,
        )

        llm_config = self.settings.llm

        if config.provider == "ollama":
            if not llm_config.ollama_base_url:
                raise ValueError(
                    "OLLAMA_BASE_URL is required",
                )

            return OllamaEmbeddingModel(
                model_id=config.model,
                base_url=llm_config.ollama_base_url,
                dimension=config.dimension,
                keep_alive=llm_config.ollama_keep_alive,
                timeout_seconds=llm_config.request_timeout_seconds,
                max_concurrency=llm_config.embedding_max_concurrency,
                retry_config=self.settings.retry,
            )

        if config.provider == "openrouter":
            if not self.settings.OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY is required",
                )

            return OpenRouterEmbeddingModel(
                model_id=config.model,
                api_key=self.settings.OPENROUTER_API_KEY,
                dimension=config.dimension,
                base_url=llm_config.openrouter_base_url,
                timeout_seconds=llm_config.request_timeout_seconds,
                max_concurrency=llm_config.embedding_max_concurrency,
                retry_config=self.settings.retry,
            )

        raise ValueError(
            f"Unsupported embedding provider: {config.provider}",
        )
