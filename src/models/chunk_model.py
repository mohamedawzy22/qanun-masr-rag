from collections.abc import AsyncIterator

from bson.objectid import ObjectId
from pymongo import ASCENDING, InsertOne

from .base_data_model import BaseDataModel
from .db_schemes import DataChunk
from .enums.database_enum import DataBaseEnum


class ChunkModel(BaseDataModel):
    def __init__(
        self,
        db_client: object,
    ):
        super().__init__(
            db_client=db_client,
        )

        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHUNK_NAME.value]

    async def init_collection(self) -> None:
        """
        Initialize indexes required by chunk queries.

        The compound project_id + _id index supports the
        keyset pagination query used during indexing.
        """

        await self.collection.create_index(
            [
                (
                    "chunk_project_id",
                    ASCENDING,
                ),
                (
                    "_id",
                    ASCENDING,
                ),
            ],
            name="idx_chunk_project_id_id",
        )

    @staticmethod
    def _validate_batch_size(
        batch_size: int,
    ) -> None:
        if not isinstance(batch_size, int):
            raise TypeError(
                "Batch size must be an integer",
            )

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero",
            )

    @staticmethod
    def _validate_page(
        page_no: int,
        page_size: int,
    ) -> None:
        if not isinstance(page_no, int):
            raise TypeError(
                "Page number must be an integer",
            )

        if page_no <= 0:
            raise ValueError(
                "Page number must be greater than zero",
            )

        if not isinstance(page_size, int):
            raise TypeError(
                "Page size must be an integer",
            )

        if page_size <= 0:
            raise ValueError(
                "Page size must be greater than zero",
            )

    async def create_chunk(
        self,
        chunk: DataChunk,
    ) -> DataChunk:
        result = await self.collection.insert_one(
            chunk.model_dump(
                by_alias=True,
                exclude_unset=True,
            )
        )

        chunk.id = result.inserted_id

        return chunk

    async def get_chunk(
        self,
        chunk_id: str,
    ) -> DataChunk | None:
        if not ObjectId.is_valid(chunk_id):
            raise ValueError(
                f"Invalid chunk ID: {chunk_id}",
            )

        record = await self.collection.find_one(
            {
                "_id": ObjectId(chunk_id),
            },
            projection={
                "_id": 1,
                "chunk_text": 1,
                "chunk_metadata": 1,
                "chunk_order": 1,
                "chunk_project_id": 1,
                "chunk_asset_id": 1,
                "article_number": 1,
                "book": 1,
                "chapter": 1,
                "section": 1,
                "topic": 1,
                "text_ar": 1,
                "text_en": 1,
                "is_repealed": 1,
                "source_page": 1,
                "citation": 1,
            },
        )

        if record is None:
            return None

        return DataChunk(**record)

    async def insert_many_chunks(
        self,
        chunks: list[DataChunk],
        batch_size: int = 100,
    ) -> int:
        self._validate_batch_size(
            batch_size,
        )

        if not chunks:
            return 0

        for i in range(
            0,
            len(chunks),
            batch_size,
        ):
            batch = chunks[i : i + batch_size]

            operations = [
                InsertOne(
                    chunk.model_dump(
                        by_alias=True,
                        exclude_unset=True,
                    )
                )
                for chunk in batch
            ]

            await self.collection.bulk_write(
                operations,
            )

        return len(chunks)

    async def delete_chunks_by_project_id(
        self,
        project_id: ObjectId,
    ) -> int:
        result = await self.collection.delete_many(
            {
                "chunk_project_id": project_id,
            },
        )

        return result.deleted_count

    async def get_project_chunks(
        self,
        project_id: ObjectId,
        page_no: int = 1,
        page_size: int = 100,
    ) -> list[DataChunk]:
        self._validate_page(
            page_no=page_no,
            page_size=page_size,
        )

        records = (
            await self.collection.find(
                {
                    "chunk_project_id": project_id,
                },
                projection={
                    "_id": 1,
                    "chunk_text": 1,
                    "chunk_metadata": 1,
                    "chunk_order": 1,
                    "chunk_project_id": 1,
                    "chunk_asset_id": 1,
                    "article_number": 1,
                    "book": 1,
                    "chapter": 1,
                    "section": 1,
                    "topic": 1,
                    "text_ar": 1,
                    "text_en": 1,
                    "is_repealed": 1,
                    "source_page": 1,
                    "citation": 1,
                },
            )
            .sort(
                "_id",
                ASCENDING,
            )
            .skip(
                (page_no - 1) * page_size,
            )
            .limit(
                page_size,
            )
            .to_list(
                length=page_size,
            )
        )

        return [DataChunk(**record) for record in records]

    async def iter_project_chunks(
        self,
        project_id: ObjectId,
        batch_size: int = 32,
    ) -> AsyncIterator[list[DataChunk]]:
        """
        Yield project chunks in batches using keyset pagination.

        Uses the compound (chunk_project_id, _id)
        index to avoid large skip offsets.
        """

        self._validate_batch_size(
            batch_size,
        )

        last_id = None

        while True:
            query = {
                "chunk_project_id": project_id,
            }

            if last_id is not None:
                query["_id"] = {
                    "$gt": last_id,
                }

            records = (
                await self.collection.find(
                    query,
                    projection={
                        "_id": 1,
                        "chunk_text": 1,
                        "chunk_metadata": 1,
                        "chunk_order": 1,
                        "chunk_project_id": 1,
                        "chunk_asset_id": 1,
                        "article_number": 1,
                        "book": 1,
                        "chapter": 1,
                        "section": 1,
                        "topic": 1,
                        "text_ar": 1,
                        "text_en": 1,
                        "is_repealed": 1,
                        "source_page": 1,
                        "citation": 1,
                    },
                )
                .sort(
                    "_id",
                    ASCENDING,
                )
                .limit(
                    batch_size,
                )
                .to_list(
                    length=batch_size,
                )
            )

            if not records:
                break

            chunks = [DataChunk(**record) for record in records]

            yield chunks

            last_id = records[-1]["_id"]

            if len(records) < batch_size:
                break
