import asyncio

from utils.logger import get_logger

from .chat_interface import ChatModel
from .embedding_interface import EmbeddingModel
from .fallback_chat import FallbackChatModel
from .llm_factory import ModelFactory

logger = get_logger(__name__)


class ModelManager:
    def __init__(
        self,
        factory: ModelFactory,
    ):
        self.factory = factory

        self._chat_models: dict[str, ChatModel] = {}
        self._embedding_models: dict[str, EmbeddingModel] = {}

        self._selected_chat_model: ChatModel | None = None
        self._selected_embedding_model: EmbeddingModel | None = None

    def load_models(self) -> None:
        logger.info("Loading chat models...")

        for name, config in self.factory.settings.CHAT_MODELS.items():
            self._chat_models[name] = self.factory.create_chat_model(config)

            logger.info("Chat model loaded: %s", name)

        logger.info("Loading embedding models...")

        for name, config in self.factory.settings.EMBEDDING_MODELS.items():
            self._embedding_models[name] = self.factory.create_embedding_model(config)

            logger.info("Embedding model loaded: %s", name)

        self._build_selected_chat_model()
        self._build_selected_embedding_model()

        logger.info(
            "Models loaded successfully: chat=%s | embedding=%s",
            len(self._chat_models),
            len(self._embedding_models),
        )

    async def health_check_models(self) -> None:
        logger.info("Checking model health...")

        primary_name = self.factory.settings.CHAT_PRIMARY_MODEL
        fallback_name = self.factory.settings.CHAT_FALLBACK_MODEL
        embedding_name = self.factory.settings.EMBEDDING_MODEL

        primary_model = self.get_chat_model(primary_name)
        fallback_model = self.get_chat_model(fallback_name)
        embedding_model = self.get_embedding_model(embedding_name)

        primary_healthy = await self._check_model(
            name=primary_name,
            model=primary_model,
        )

        fallback_healthy = await self._check_model(
            name=fallback_name,
            model=fallback_model,
        )

        embedding_healthy = await self._check_model(
            name=embedding_name,
            model=embedding_model,
        )

        if not fallback_healthy:
            logger.error(
                "Fallback chat model is unavailable: %s",
                fallback_name,
            )

            raise RuntimeError(f"Fallback chat model '{fallback_name}' is unavailable")

        if not embedding_healthy:
            logger.error(
                "Embedding model is unavailable: %s",
                embedding_name,
            )

            raise RuntimeError(f"Embedding model '{embedding_name}' is unavailable")

        if not primary_healthy:
            logger.warning(
                "Primary chat model is unavailable. Starting with fallback: %s",
                fallback_name,
            )

            selected_chat_model = self.get_selected_chat_model()

            if isinstance(selected_chat_model, FallbackChatModel):
                selected_chat_model.set_primary_unavailable()

        logger.info("Model health check completed successfully")

    @staticmethod
    async def _check_model(
        name: str,
        model,
    ) -> bool:
        logger.debug(
            "Checking model health: %s",
            name,
        )

        try:
            return await asyncio.wait_for(
                model.health_check(),
                timeout=20,
            )

        except Exception:
            logger.exception(
                "Model health check failed: %s",
                name,
            )

            return False

    def _build_selected_chat_model(self) -> None:
        primary_name = self.factory.settings.CHAT_PRIMARY_MODEL
        fallback_name = self.factory.settings.CHAT_FALLBACK_MODEL
        cooldown_seconds = self.factory.settings.CHAT_FALLBACK_COOLDOWN_SECONDS

        primary_model = self.get_chat_model(primary_name)
        fallback_model = self.get_chat_model(fallback_name)

        self._selected_chat_model = FallbackChatModel(
            primary=primary_model,
            fallback=fallback_model,
            primary_name=primary_name,
            fallback_name=fallback_name,
            cooldown_seconds=cooldown_seconds,
            langfuse=self.factory.langfuse,
        )

        logger.info(
            "Selected chat models configured: primary=%s | fallback=%s",
            primary_name,
            fallback_name,
        )

    def _build_selected_embedding_model(self) -> None:
        embedding_name = self.factory.settings.EMBEDDING_MODEL

        self._selected_embedding_model = self.get_embedding_model(embedding_name)

        logger.info(
            "Selected embedding model configured: %s",
            embedding_name,
        )

    async def warm_up_models(self, timeout_seconds: int = 30) -> None:
        logger.info("Warming up selected chat and embedding models...")

        selected_chat = self.get_selected_chat_model()
        selected_embedding = self.get_selected_embedding_model()
        langfuse = self.factory.langfuse

        if langfuse is None:
            await self._warm_up_model(
                name="chat",
                model=selected_chat,
                timeout_seconds=timeout_seconds,
            )
        else:
            # Warm-up calls the provider's tracing code, so give it
            # an active observation instead of querying an empty context.
            with langfuse.start_as_current_observation(
                as_type="span",
                name="model.chat-warmup",
                input={
                    "model": self.factory.settings.CHAT_PRIMARY_MODEL,
                },
                metadata={
                    "operation": "startup_warmup",
                },
            ):
                await self._warm_up_model(
                    name="chat",
                    model=selected_chat,
                    timeout_seconds=timeout_seconds,
                )

        await self._warm_up_model(
            name="embedding",
            model=selected_embedding,
            timeout_seconds=timeout_seconds,
        )

        logger.info("Model warm-up completed successfully")

    @staticmethod
    async def _warm_up_model(
        name: str,
        model,
        timeout_seconds: int,
    ) -> None:
        logger.info("Warming up %s model", name)

        try:
            await asyncio.wait_for(
                model.warm_up(),
                timeout=timeout_seconds,
            )

        except Exception as exc:
            logger.exception(
                "Warm-up failed: %s",
                name,
            )
            raise RuntimeError(f"{name} model warm-up failed") from exc

        logger.info("%s model warm-up completed", name)

    async def close(self) -> None:
        logger.info("Closing model clients...")

        for name, model in self._chat_models.items():
            try:
                await model.close()
                logger.debug("Closed chat model: %s", name)
            except Exception:
                logger.exception(
                    "Failed to close chat model: %s",
                    name,
                )

        for name, model in self._embedding_models.items():
            try:
                await model.close()
                logger.debug(
                    "Closed embedding model: %s",
                    name,
                )
            except Exception:
                logger.exception(
                    "Failed to close embedding model: %s",
                    name,
                )

        logger.info("Model clients closed")

    def get_chat_model(self, name: str) -> ChatModel:
        logger.debug("Getting chat model: %s", name)

        if name not in self._chat_models:
            logger.error("Chat model not found: %s", name)
            raise ValueError(f"Chat model '{name}' is not configured")

        return self._chat_models[name]

    def get_selected_chat_model(self) -> ChatModel:
        if self._selected_chat_model is None:
            logger.error("Selected chat model is not initialized")
            raise RuntimeError("Selected chat model is not initialized")

        return self._selected_chat_model

    def get_embedding_model(self, name: str) -> EmbeddingModel:
        logger.debug("Getting embedding model: %s", name)

        if name not in self._embedding_models:
            logger.error("Embedding model not found: %s", name)
            raise ValueError(f"Embedding model '{name}' is not configured")

        return self._embedding_models[name]

    def get_selected_embedding_model(self) -> EmbeddingModel:
        if self._selected_embedding_model is None:
            logger.error("Selected embedding model is not initialized")
            raise RuntimeError("Selected embedding model is not initialized")

        return self._selected_embedding_model
