from abc import ABC, abstractmethod


class VectorDB(ABC):
    @abstractmethod
    async def create_collection(self) -> None:
        pass

    @abstractmethod
    async def delete_collection(self) -> None:
        pass

    @abstractmethod
    async def delete_by_project_id(
        self,
        project_id: str,
    ) -> None:
        """Delete indexed vectors belonging to one project."""

    @abstractmethod
    async def upsert(
        self,
        vectors: list[dict],
    ) -> None:
        pass

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        project_id: str,
        limit: int = 5,
    ) -> list[dict]:
        pass

    async def close(self) -> None:
        return None
