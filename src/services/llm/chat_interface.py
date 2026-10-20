from abc import ABC, abstractmethod


class ChatModel(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ):
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    async def warm_up(self) -> None:
        await self.generate(
            prompt="Reply with OK",
            temperature=0.0,
            max_tokens=1,
        )

    async def close(self) -> None:
        return None
