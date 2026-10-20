import asyncio
import random
from contextlib import nullcontext

from langfuse import Langfuse
from ollama import AsyncClient, ResponseError

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
    429,
}


def _is_retryable_ollama_error(
    exc: Exception,
) -> bool:
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    if isinstance(exc, ResponseError):
        return exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500

    return False


class OllamaChatModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        temperature: float,
        keep_alive: int = 300,
        timeout_seconds: int = 120,
        max_tokens: int = 300,
        max_concurrency: int = 1,
        retry_config: RetryConfig | None = None,
        langfuse: Langfuse | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.max_tokens = max_tokens

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency

        self.retry_config = retry_config
        self.langfuse = langfuse

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncClient(
            host=base_url,
            timeout=timeout_seconds,
        )

        logger.info(
            "Initialized Ollama chat model: "
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
        input_tokens = response.get(
            "prompt_eval_count",
        )

        output_tokens = response.get(
            "eval_count",
        )

        if input_tokens is None and output_tokens is None:
            return None

        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0

        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }

    def _record_token_usage(
        self,
        response,
    ) -> bool:
        usage = self._extract_token_usage(
            response,
        )

        if usage is None:
            return False

        LLM_INPUT_TOKENS.labels(
            provider="ollama",
            model=self.model_id,
        ).inc(
            usage["input"],
        )

        LLM_OUTPUT_TOKENS.labels(
            provider="ollama",
            model=self.model_id,
        ).inc(
            usage["output"],
        )

        LLM_TOTAL_TOKENS.labels(
            provider="ollama",
            model=self.model_id,
        ).inc(
            usage["total"],
        )

        logger.debug(
            "Ollama token usage: model=%s | input=%s | output=%s | total=%s",
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
        options = {
            "temperature": temperature,
        }

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        options["num_predict"] = effective_max_tokens

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with asyncio.timeout(
            self.timeout_seconds,
        ):
            response = await self.client.chat(
                model=self.model_id,
                messages=messages,
                options=options,
                keep_alive=self.keep_alive,
            )

        usage = self._extract_token_usage(
            response,
        )

        self._record_token_usage(
            response,
        )

        return (
            response["message"]["content"],
            usage,
        )

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        logger.info(
            "Generating text with Ollama model=%s",
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
                            should_retry=_is_retryable_ollama_error,
                            config=self.retry_config,
                            operation_name=(f"ollama.chat:{self.model_id}"),
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

                except TimeoutError:
                    if generation is not None:
                        generation.update(
                            level="ERROR",
                            status_message="TimeoutError",
                        )

                    logger.error(
                        "Ollama generation timed out: model=%s",
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
                        "Ollama generation failed: model=%s",
                        self.model_id,
                    )
                    raise

    async def stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ):
        logger.info(
            "Starting Ollama stream: model=%s",
            self.model_id,
        )

        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        options = {
            "temperature": effective_temperature,
            "num_predict": effective_max_tokens,
        }

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
                            async with asyncio.timeout(
                                self.timeout_seconds,
                            ):
                                response = await self.client.chat(
                                    model=self.model_id,
                                    messages=messages,
                                    options=options,
                                    stream=True,
                                    keep_alive=self.keep_alive,
                                )

                                async for chunk in response:
                                    content = chunk["message"]["content"]

                                    if content:
                                        yielded_content = True
                                        generated_chunks.append(content)
                                        yield content

                                    if not usage_recorded:
                                        usage = self._extract_token_usage(
                                            chunk,
                                        )

                                        usage_recorded = self._record_token_usage(
                                            chunk,
                                        )

                            logger.info(
                                "Ollama stream completed: model=%s",
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

                            return

                        except Exception as exc:
                            should_stop_retrying = (
                                yielded_content
                                or self.retry_config is None
                                or attempt >= attempts
                                or not _is_retryable_ollama_error(exc)
                            )

                            if should_stop_retrying:
                                raise

                            exponential_delay = min(
                                self.retry_config.initial_delay_seconds
                                * (2 ** (attempt - 1)),
                                self.retry_config.max_delay_seconds,
                            )

                            jitter = exponential_delay * 0.25 * random.random()

                            delay = exponential_delay + jitter

                            logger.warning(
                                "Retrying Ollama stream: "
                                "model=%s | attempt=%s/%s | "
                                "retry_in=%.2fs",
                                self.model_id,
                                attempt,
                                attempts,
                                delay,
                            )

                            await asyncio.sleep(delay)

                except TimeoutError:
                    if generation is not None:
                        generation.update(
                            output="".join(generated_chunks),
                            level="ERROR",
                            status_message="TimeoutError",
                        )

                    logger.error(
                        "Ollama stream timed out: model=%s",
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
                        "Ollama stream failed: model=%s",
                        self.model_id,
                    )
                    raise

    async def health_check(self) -> bool:
        try:
            await self.client.show(
                self.model_id,
            )

            logger.info(
                "Ollama chat model is healthy: %s",
                self.model_id,
            )

            return True

        except Exception:
            logger.exception(
                "Ollama chat model health check failed: %s",
                self.model_id,
            )
            return False

    async def close(self) -> None:
        await self.client.close()


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        dimension: int,
        keep_alive: int = 300,
        timeout_seconds: int = 120,
        max_concurrency: int = 2,
        retry_config: RetryConfig | None = None,
    ):
        self.model_id = model_id
        self.dimension = dimension
        self.keep_alive = keep_alive

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency

        self.retry_config = retry_config

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncClient(
            host=base_url,
            timeout=timeout_seconds,
        )

        logger.info(
            "Initialized Ollama embedding model: "
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
        async with asyncio.timeout(
            self.timeout_seconds,
        ):
            response = await self.client.embed(
                model=self.model_id,
                input=text,
                keep_alive=self.keep_alive,
            )

        return response["embeddings"][0]

    async def _embed_documents_once(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        async with asyncio.timeout(
            self.timeout_seconds,
        ):
            response = await self.client.embed(
                model=self.model_id,
                input=texts,
                keep_alive=self.keep_alive,
            )

        return response["embeddings"]

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        logger.info(
            "Creating embedding with Ollama model=%s",
            self.model_id,
        )

        async with self._semaphore:
            try:
                if self.retry_config is None:
                    return await self._embed_once(text)

                return await retry_async(
                    lambda: self._embed_once(text),
                    should_retry=_is_retryable_ollama_error,
                    config=self.retry_config,
                    operation_name=(f"ollama.embed:{self.model_id}"),
                )

            except Exception:
                logger.exception(
                    "Ollama embedding failed: model=%s",
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
                    should_retry=_is_retryable_ollama_error,
                    config=self.retry_config,
                    operation_name=(f"ollama.embed_documents:{self.model_id}"),
                )

            except Exception:
                logger.exception(
                    "Ollama document embeddings failed: model=%s",
                    self.model_id,
                )
                raise

    async def health_check(self) -> bool:
        try:
            await self.client.show(
                self.model_id,
            )

            logger.info(
                "Ollama embedding model is healthy: %s",
                self.model_id,
            )

            return True

        except Exception:
            logger.exception(
                "Ollama embedding model health check failed: %s",
                self.model_id,
            )
            return False

    async def close(self) -> None:
        await self.client.close()
