import asyncio

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from core.retry import retry_async
from helpers.config import RetryConfig
from services.vectordb.vector_db_enum import DistanceMetric
from services.vectordb.vector_db_interface import VectorDB
from utils.logger import get_logger

logger = get_logger(__name__)


def _is_retryable_qdrant_error(
    exc: Exception,
) -> bool:
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            ResponseHandlingException,
        ),
    ):
        return True

    if isinstance(exc, UnexpectedResponse):
        return (
            exc.status_code == 408 or exc.status_code == 429 or exc.status_code >= 500
        )

    return False


class QdrantVectorDB(VectorDB):
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        distance: DistanceMetric,
        timeout_seconds: int = 30,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.timeout_seconds = timeout_seconds
        self.retry_config = retry_config

        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

        logger.info(
            "Initialized Qdrant: "
            "collection=%s | vector_size=%s | "
            "distance=%s | timeout=%ss",
            collection_name,
            vector_size,
            distance.value,
            timeout_seconds,
        )

    async def close(self) -> None:
        logger.info(
            "Closing Qdrant client: collection=%s",
            self.collection_name,
        )

        await self.client.close()

    def _get_distance(self) -> models.Distance:
        distance_map = {
            DistanceMetric.COSINE: models.Distance.COSINE,
            DistanceMetric.EUCLID: models.Distance.EUCLID,
            DistanceMetric.DOT: models.Distance.DOT,
        }

        return distance_map[self.distance]

    async def _collection_exists(self) -> bool:
        return await self.client.collection_exists(
            collection_name=self.collection_name,
        )

    async def create_collection(self) -> None:
        if self.retry_config is None:
            exists = await self._collection_exists()
        else:
            exists = await retry_async(
                self._collection_exists,
                should_retry=_is_retryable_qdrant_error,
                config=self.retry_config,
                operation_name="qdrant.collection_exists",
            )

        if not exists:

            async def create():
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=self._get_distance(),
                    ),
                )

            if self.retry_config is None:
                await create()
            else:
                await retry_async(
                    create,
                    should_retry=_is_retryable_qdrant_error,
                    config=self.retry_config,
                    operation_name="qdrant.create_collection",
                )

            logger.info(
                "Qdrant collection created: %s",
                self.collection_name,
            )
        else:
            logger.info(
                "Qdrant collection already exists: %s",
                self.collection_name,
            )

        await self._create_payload_indexes()

    async def _create_payload_indexes(self) -> None:
        async def create_index():
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="project_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        if self.retry_config is None:
            await create_index()
        else:
            await retry_async(
                create_index,
                should_retry=_is_retryable_qdrant_error,
                config=self.retry_config,
                operation_name="qdrant.create_payload_index",
            )

    async def delete_collection(self) -> None:
        async def delete():
            await self.client.delete_collection(
                collection_name=self.collection_name,
            )

        if self.retry_config is None:
            await delete()
        else:
            await retry_async(
                delete,
                should_retry=_is_retryable_qdrant_error,
                config=self.retry_config,
                operation_name="qdrant.delete_collection",
            )

    async def delete_by_project_id(
        self,
        project_id: str,
    ) -> None:
        """Delete Qdrant points belonging to this project only."""
        project_filter = Filter(
            must=[
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id),
                ),
            ],
        )

        async def delete_points():
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=project_filter,
                ),
                wait=True,
            )

        if self.retry_config is None:
            await delete_points()
        else:
            await retry_async(
                delete_points,
                should_retry=_is_retryable_qdrant_error,
                config=self.retry_config,
                operation_name="qdrant.delete_project_points",
            )

        logger.info(
            "Deleted indexed vectors for project: project_id=%s | collection=%s",
            project_id,
            self.collection_name,
        )

    async def upsert(
        self,
        vectors: list[dict],
    ) -> None:
        if not vectors:
            return

        points = [
            models.PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload=item.get("payload", {}),
            )
            for item in vectors
        ]

        async def upsert_points():
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        if self.retry_config is None:
            await upsert_points()
        else:
            await retry_async(
                upsert_points,
                should_retry=_is_retryable_qdrant_error,
                config=self.retry_config,
                operation_name="qdrant.upsert",
            )

    async def search(
        self,
        vector: list[float],
        project_id: str,
        limit: int = 5,
    ) -> list[dict]:
        project_filter = Filter(
            must=[
                FieldCondition(
                    key="project_id",
                    match=MatchValue(
                        value=project_id,
                    ),
                )
            ]
        )

        async def query():
            return await self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=project_filter,
                limit=limit,
                with_payload=[
                    "chunk_id",
                    "project_id",
                    "asset_id",
                    "text",
                    "article_number",
                    "book",
                    "chapter",
                    "section",
                    "topic",
                    "text_ar",
                    "text_en",
                    "is_repealed",
                    "source_page",
                    "citation",
                    "source_file",
                    "chunk_metadata",
                ],
                with_vectors=False,
            )

        if self.retry_config is None:
            response = await query()
        else:
            response = await retry_async(
                query,
                should_retry=_is_retryable_qdrant_error,
                config=self.retry_config,
                operation_name="qdrant.query_points",
            )

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload or {},
            }
            for result in response.points
        ]
