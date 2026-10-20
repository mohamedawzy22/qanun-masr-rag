import time
from collections.abc import AsyncIterator
from contextlib import nullcontext

from langfuse import Langfuse

from utils.logger import get_logger

from .chat_interface import ChatModel

logger = get_logger(__name__)


class FallbackChatModel(ChatModel):
    def __init__(
        self,
        primary: ChatModel,
        fallback: ChatModel,
        primary_name: str,
        fallback_name: str,
        cooldown_seconds: int = 60,
        langfuse: Langfuse | None = None,
    ):
        self.primary = primary
        self.fallback = fallback

        self.primary_name = primary_name
        self.fallback_name = fallback_name

        self.cooldown_seconds = cooldown_seconds
        self.langfuse = langfuse

        self._primary_failed_at: float | None = None

    def _primary_available(self) -> bool:
        if self._primary_failed_at is None:
            return True

        elapsed = time.monotonic() - self._primary_failed_at

        if elapsed >= self.cooldown_seconds:
            self._primary_failed_at = None

            logger.info(
                "Primary chat model cooldown expired: %s",
                self.primary_name,
            )

            return True

        return False

    def _mark_primary_failed(self) -> None:
        self._primary_failed_at = time.monotonic()

        logger.warning(
            "Primary chat model marked unavailable: %s | cooldown=%ss",
            self.primary_name,
            self.cooldown_seconds,
        )

    def _mark_primary_success(self) -> None:
        if self._primary_failed_at is not None:
            logger.info(
                "Primary chat model recovered: %s",
                self.primary_name,
            )

        self._primary_failed_at = None

    def set_primary_unavailable(self) -> None:
        self._mark_primary_failed()

    def _start_fallback_observation(
        self,
        reason: str,
    ):
        trace_active = (
            self.langfuse is not None
            and self.langfuse.get_current_trace_id() is not None
        )

        if not trace_active:
            return nullcontext(None)

        return self.langfuse.start_as_current_observation(
            as_type="span",
            name="llm.fallback",
            input={
                "primary_model": self.primary_name,
                "fallback_model": self.fallback_name,
            },
            metadata={
                "reason": reason,
            },
        )

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        fallback_reason: str | None = None

        if self._primary_available():
            try:
                response = await self.primary.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )

                self._mark_primary_success()
                return response

            except Exception:  # noqa: BLE001
                self._mark_primary_failed()
                fallback_reason = "primary_error"

        else:
            fallback_reason = "primary_cooldown"

        with self._start_fallback_observation(
            reason=fallback_reason,
        ) as fallback_observation:
            try:
                response = await self.fallback.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )

                if fallback_observation is not None:
                    fallback_observation.update(
                        output={
                            "selected_model": self.fallback_name,
                            "reason": fallback_reason,
                        },
                    )

                return response

            except Exception as exc:
                if fallback_observation is not None:
                    fallback_observation.update(
                        level="ERROR",
                        status_message=type(exc).__name__,
                    )

                raise

    async def stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        fallback_reason: str | None = None

        if self._primary_available():
            yielded_content = False

            try:
                async for content in self.primary.stream(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                ):
                    yielded_content = True
                    yield content

                self._mark_primary_success()
                return

            except Exception:
                self._mark_primary_failed()

                if yielded_content:
                    raise

                fallback_reason = "primary_error"

        else:
            fallback_reason = "primary_cooldown"

        with self._start_fallback_observation(
            reason=fallback_reason,
        ) as fallback_observation:
            try:
                async for content in self.fallback.stream(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                ):
                    yield content

                if fallback_observation is not None:
                    fallback_observation.update(
                        output={
                            "selected_model": self.fallback_name,
                            "reason": fallback_reason,
                        },
                    )

            except Exception as exc:
                if fallback_observation is not None:
                    fallback_observation.update(
                        level="ERROR",
                        status_message=type(exc).__name__,
                    )

                raise

    async def health_check(self) -> bool:
        primary_healthy = await self.primary.health_check()
        fallback_healthy = await self.fallback.health_check()

        return primary_healthy or fallback_healthy

    async def warm_up(self) -> None:
        await self.generate(
            prompt="Reply with OK",
            temperature=0.0,
            max_tokens=2048,
        )

    async def close(self) -> None:
        await self.primary.close()
        await self.fallback.close()
