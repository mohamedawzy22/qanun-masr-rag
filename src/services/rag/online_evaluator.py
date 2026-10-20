"""Online RAG evaluation using local Ollama models and Ragas."""

from asyncio import Semaphore

from openai import AsyncOpenAI

from services.rag.evaluation_metrics import (
    EMBEDDING_MODEL,
    JUDGE_MODEL,
    build_metrics,
    score_sample,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class OnlineRAGEvaluator:
    """Evaluate generated RAG answers without requiring a Dataset."""

    def __init__(
        self,
        *,
        ollama_base_url: str,
        timeout_seconds: int = 120,
    ):
        if not ollama_base_url.strip():
            raise ValueError("OLLAMA_BASE_URL must not be empty")

        base_url = ollama_base_url.rstrip("/")

        # Ollama's OpenAI-compatible endpoints use /v1.
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        self.judge_model = JUDGE_MODEL
        self.embedding_model = EMBEDDING_MODEL

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=timeout_seconds,
            max_retries=1,
        )

        self.metrics = build_metrics(self.client)

        # Avoid overwhelming the local model when requests overlap.
        self._semaphore = Semaphore(1)

        logger.info(
            "Online RAG evaluator initialized: judge=%s | embeddings=%s | base_url=%s",
            self.judge_model,
            self.embedding_model,
            base_url,
        )

    async def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str = "",
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Return available Ragas scores and per-metric errors."""

        async with self._semaphore:
            return await score_sample(
                self.metrics,
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference,
            )

    async def close(self) -> None:
        """Release the asynchronous OpenAI-compatible client."""

        await self.client.close()
        logger.info("Online RAG evaluator closed")
