import asyncio
import random
from collections.abc import AsyncIterator
from contextlib import nullcontext

from langfuse import Langfuse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from core.retry import retry_async
from helpers.config import RetryConfig
from utils.logger import get_logger
from utils.metrics import (
    LLM_INPUT_TOKENS,
    LLM_OUTPUT_TOKENS,
    LLM_TOTAL_TOKENS,
)

from ..chat_interface import ChatModel
from ..embedding_interface import EmbeddingModel

logger = get_logger(__name__)


_RETRYABLE_STATUS_CODES = {
    408,
    409,
    429,
}


def _is_retryable_openrouter_error(
    exc: Exception,
) -> bool:
    """
    Return whether an OpenRouter/OpenAI SDK error is retryable.
    """

    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500

    return False


def _calculate_retry_delay(
    retry_config: RetryConfig,
    attempt: int,
) -> float:
    """
    Calculate exponential backoff with jitter.
    """

    exponential_delay = min(
        retry_config.initial_delay_seconds * (2 ** (attempt - 1)),
        retry_config.max_delay_seconds,
    )

    jitter = exponential_delay * 0.25 * random.random()

    return exponential_delay + jitter


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        temperature: float,
        timeout_seconds: int = 120,
        max_tokens: int = 300,
        max_concurrency: int = 1,
        retry_config: RetryConfig | None = None,
        langfuse: Langfuse | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.retry_config = retry_config
        self.langfuse = langfuse

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

        logger.info(
            "Initialized OpenRouter chat model: "
            "model=%s | temperature=%s | timeout=%ss | "
            "max_tokens=%s | max_concurrency=%s",
            self.model_id,
            self.temperature,
            self.timeout_seconds,
            self.max_tokens,
            self.max_concurrency,
        )

    def _extract_token_usage(
        self,
        response,
    ) -> dict[str, int] | None:
        """
        Extract token usage from an OpenRouter response.
        """

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return None

        input_tokens = getattr(
            usage,
            "prompt_tokens",
            0,
        )

        output_tokens = getattr(
            usage,
            "completion_tokens",
            0,
        )

        total_tokens = getattr(
            usage,
            "total_tokens",
            input_tokens + output_tokens,
        )

        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        }

    def _record_token_usage(
        self,
        response,
    ) -> bool:
        """
        Record token usage returned by OpenRouter.
        """

        usage = self._extract_token_usage(
            response,
        )

        if usage is None:
            return False

        LLM_INPUT_TOKENS.labels(
            provider="openrouter",
            model=self.model_id,
        ).inc(
            usage["input"],
        )

        LLM_OUTPUT_TOKENS.labels(
            provider="openrouter",
            model=self.model_id,
        ).inc(
            usage["output"],
        )

        LLM_TOTAL_TOKENS.labels(
            provider="openrouter",
            model=self.model_id,
        ).inc(
            usage["total"],
        )

        logger.debug(
            "OpenRouter token usage: model=%s | input=%s | output=%s | total=%s",
            self.model_id,
            usage["input"],
            usage["output"],
            usage["total"],
        )

        return True

    async def _generate_once(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int | None,
        system_prompt: str | None = None,
    ) -> tuple[str, dict[str, int] | None]:
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            extra_body={
                "usage": {"include": True},
                "reasoning": {"effort": "low"},
            },
        )

        usage = self._extract_token_usage(
            response,
        )

        self._record_token_usage(
            response,
        )

        if not response.choices:
            raise RuntimeError("OpenRouter returned no completion choices.")

        choice = response.choices[0]
        message = choice.message
        content = message.content

        # Support text content returned as either a string or text parts.
        if isinstance(content, list):
            content = "".join(
                part
                if isinstance(part, str)
                else (
                    part.get("text", "")
                    if isinstance(part, dict)
                    else getattr(part, "text", "") or ""
                )
                for part in content
            )

        if not isinstance(content, str) or not content.strip():
            finish_reason = getattr(choice, "finish_reason", None)
            has_reasoning = bool(
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_details", None)
            )

            logger.error(
                "OpenRouter returned empty visible output: "
                "model=%s | finish_reason=%s | has_reasoning=%s | "
                "response_id=%s",
                self.model_id,
                finish_reason,
                has_reasoning,
                getattr(response, "id", None),
            )

            raise RuntimeError(
                "OpenRouter returned empty visible output "
                f"(finish_reason={finish_reason})."
            )

        return content.strip(), usage

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        logger.info(
            "Generating text with OpenRouter model=%s",
            self.model_id,
        )

        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        trace_active = (
            self.langfuse is not None
            and self.langfuse.get_current_trace_id() is not None
        )

        observation_context = (
            self.langfuse.start_as_current_observation(
                as_type="generation",
                name="llm.generate",
                model=self.model_id,
                input={
                    "prompt_length": len(prompt),
                },
                model_parameters={
                    "temperature": effective_temperature,
                    "max_tokens": effective_max_tokens,
                },
            )
            if trace_active
            else nullcontext(None)
        )

        async with self._semaphore:
            with observation_context as generation:
                try:
                    if self.retry_config is None:
                        response_text, usage = await self._generate_once(
                            prompt=prompt,
                            temperature=effective_temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt,
                        )
                    else:
                        response_text, usage = await retry_async(
                            lambda: self._generate_once(
                                prompt=prompt,
                                temperature=effective_temperature,
                                max_tokens=max_tokens,
                                system_prompt=system_prompt,
                            ),
                            should_retry=_is_retryable_openrouter_error,
                            config=self.retry_config,
                            operation_name=(f"openrouter.chat:{self.model_id}"),
                        )

                    if generation is not None:
                        generation.update(
                            output=response_text,
                        )

                        if usage is not None:
                            generation.update(
                                usage_details=usage,
                            )

                    return response_text

                except APITimeoutError:
                    if generation is not None:
                        generation.update(
                            level="ERROR",
                            status_message="APITimeoutError",
                        )

                    logger.error(
                        "OpenRouter generation timed out: model=%s",
                        self.model_id,
                    )

                    raise

                except Exception as exc:
                    if generation is not None:
                        generation.update(
                            level="ERROR",
                            status_message=type(exc).__name__,
                        )

                    logger.exception(
                        "OpenRouter generation failed: model=%s",
                        self.model_id,
                    )

                    raise

    async def stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        logger.info(
            "Starting OpenRouter stream: model=%s",
            self.model_id,
        )

        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        trace_active = (
            self.langfuse is not None
            and self.langfuse.get_current_trace_id() is not None
        )

        observation_context = (
            self.langfuse.start_as_current_observation(
                as_type="generation",
                name="llm.generate",
                model=self.model_id,
                input={
                    "prompt_length": len(prompt),
                },
                model_parameters={
                    "temperature": effective_temperature,
                    "max_tokens": effective_max_tokens,
                },
            )
            if trace_active
            else nullcontext(None)
        )

        async with self._semaphore:
            yielded_content = False
            usage_recorded = False
            generated_chunks: list[str] = []
            usage: dict[str, int] | None = None

            attempts = (
                self.retry_config.max_attempts if self.retry_config is not None else 1
            )

            with observation_context as generation:
                try:
                    for attempt in range(
                        1,
                        attempts + 1,
                    ):
                        try:
                            response = await self.client.chat.completions.create(
                                model=self.model_id,
                                messages=messages,
                                temperature=effective_temperature,
                                max_tokens=effective_max_tokens,
                                stream=True,
                                extra_body={
                                    "usage": {"include": True},
                                    "reasoning": {"effort": "low"},
                                },
                            )

                            async for chunk in response:
                                if chunk.usage is not None:
                                    usage = self._extract_token_usage(
                                        chunk,
                                    )

                                    usage_recorded = self._record_token_usage(
                                        chunk,
                                    )
                                    continue

                                if not chunk.choices:
                                    continue

                                content = chunk.choices[0].delta.content

                                if content:
                                    yielded_content = True
                                    generated_chunks.append(content)
                                    yield content

                            logger.info(
                                "OpenRouter stream completed: model=%s",
                                self.model_id,
                            )

                            if generation is not None:
                                generation.update(
                                    output="".join(generated_chunks),
                                )

                                if usage is not None:
                                    generation.update(
                                        usage_details=usage,
                                    )

                            if not usage_recorded:
                                logger.debug(
                                    "OpenRouter stream completed without token usage: "
                                    "model=%s",
                                    self.model_id,
                                )

                            return

                        except Exception as exc:
                            should_stop_retrying = (
                                yielded_content
                                or self.retry_config is None
                                or attempt >= attempts
                                or not _is_retryable_openrouter_error(exc)
                            )

                            if should_stop_retrying:
                                raise

                            delay = _calculate_retry_delay(
                                retry_config=self.retry_config,
                                attempt=attempt,
                            )

                            logger.warning(
                                "Retrying OpenRouter stream: "
                                "model=%s | attempt=%s/%s | "
                                "retry_in=%.2fs",
                                self.model_id,
                                attempt,
                                attempts,
                                delay,
                            )

                            await asyncio.sleep(delay)

                except APITimeoutError:
                    if generation is not None:
                        generation.update(
                            output="".join(generated_chunks),
                            level="ERROR",
                            status_message="APITimeoutError",
                        )

                    logger.error(
                        "OpenRouter stream timed out: model=%s",
                        self.model_id,
                    )
                    raise

                except Exception as exc:
                    if generation is not None:
                        generation.update(
                            output="".join(generated_chunks),
                            level="ERROR",
                            status_message=type(exc).__name__,
                        )

                    logger.exception(
                        "OpenRouter stream failed: model=%s",
                        self.model_id,
                    )
                    raise

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(
                f"/model/{self.model_id}",
                cast_to=dict[str, object],
            )

            if not isinstance(response, dict) or not response.get("data"):
                raise RuntimeError(
                    "OpenRouter returned an invalid model metadata response."
                )

            logger.info(
                "OpenRouter chat model is healthy: %s",
                self.model_id,
            )
            return True

        except Exception:
            logger.exception(
                "OpenRouter chat model health check failed: %s",
                self.model_id,
            )
            return False

    async def close(self) -> None:
        await self.client.close()


class OpenRouterEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        dimension: int,
        base_url: str,
        timeout_seconds: int = 120,
        max_concurrency: int = 2,
        retry_config: RetryConfig | None = None,
    ):
        self.model_id = model_id
        self.dimension = dimension

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.retry_config = retry_config

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

        logger.info(
            "Initialized OpenRouter embedding model: "
            "model=%s | dimension=%s | "
            "timeout=%ss | max_concurrency=%s",
            self.model_id,
            self.dimension,
            self.timeout_seconds,
            self.max_concurrency,
        )

    async def _embed_once(
        self,
        text: str,
    ) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model_id,
            input=text,
        )

        embedding = response.data[0].embedding

        self._validate_dimension(
            embedding,
        )

        return embedding

    async def _embed_documents_once(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model_id,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        for embedding in embeddings:
            self._validate_dimension(
                embedding,
            )

        return embeddings

    def _validate_dimension(
        self,
        embedding: list[float],
    ) -> None:
        if len(embedding) != self.dimension:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected={self.dimension} | "
                f"actual={len(embedding)}",
            )

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        logger.info(
            "Creating embedding with OpenRouter model=%s",
            self.model_id,
        )

        async with self._semaphore:
            try:
                if self.retry_config is None:
                    return await self._embed_once(
                        text,
                    )

                return await retry_async(
                    lambda: self._embed_once(
                        text,
                    ),
                    should_retry=_is_retryable_openrouter_error,
                    config=self.retry_config,
                    operation_name=(f"openrouter.embed:{self.model_id}"),
                )

            except APITimeoutError:
                logger.error(
                    "OpenRouter embedding timed out: model=%s",
                    self.model_id,
                )

                raise

            except Exception:
                logger.exception(
                    "OpenRouter embedding failed: model=%s",
                    self.model_id,
                )

                raise

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        logger.info(
            "Creating document embeddings: model=%s | documents=%s",
            self.model_id,
            len(texts),
        )

        if not texts:
            return []

        async with self._semaphore:
            try:
                if self.retry_config is None:
                    return await self._embed_documents_once(
                        texts,
                    )

                return await retry_async(
                    lambda: self._embed_documents_once(
                        texts,
                    ),
                    should_retry=_is_retryable_openrouter_error,
                    config=self.retry_config,
                    operation_name=(f"openrouter.embed_documents:{self.model_id}"),
                )

            except APITimeoutError:
                logger.error(
                    "OpenRouter document embeddings timed out: model=%s",
                    self.model_id,
                )

                raise

            except Exception:
                logger.exception(
                    "OpenRouter document embeddings failed: model=%s",
                    self.model_id,
                )

                raise

    async def health_check(self) -> bool:
        try:
            await self.client.models.retrieve(
                self.model_id,
            )

            logger.info(
                "OpenRouter embedding model is healthy: %s",
                self.model_id,
            )

            return True

        except Exception:
            logger.exception(
                "OpenRouter embedding model health check failed: %s",
                self.model_id,
            )

            return False

    async def close(self) -> None:
        await self.client.close()
