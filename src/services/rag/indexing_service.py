import time
from uuid import NAMESPACE_URL, uuid5

from langfuse import Langfuse

from models import RAGMessage
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from services.llm.embedding_interface import EmbeddingModel
from services.vectordb.vector_db_interface import VectorDB
from utils.logger import get_logger
from utils.metrics import (
    EMBEDDING_DURATION,
    INDEXING_BATCHES,
    INDEXING_CHUNKS,
    INDEXING_DURATION,
    INDEXING_FAILURES,
    VECTOR_STORE_DURATION,
)

logger = get_logger(__name__)


class IndexingService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDB,
        project_model: ProjectModel,
        chunk_model: ChunkModel,
        langfuse: Langfuse,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.project_model = project_model
        self.chunk_model = chunk_model
        self.langfuse = langfuse

        logger.info(
            "Indexing service initialized successfully",
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
    def _build_qdrant_point_id(
        chunk_id,
    ) -> str:
        """
        Generate a deterministic UUID for a MongoDB chunk ID.

        The same chunk ID always generates the same UUID,
        allowing repeated indexing to update the same Qdrant point.
        """

        return str(
            uuid5(
                NAMESPACE_URL,
                f"chunk:{chunk_id}",
            )
        )

    async def index(
        self,
        project_id: str,
        batch_size: int = 32,
        do_reset: bool = False,
    ) -> dict:
        """
        Generate embeddings for project chunks in batches
        and store each batch in Qdrant.
        """

        self._validate_batch_size(
            batch_size,
        )

        logger.info(
            "Starting indexing: project_id=%s | batch_size=%s | do_reset=%s",
            project_id,
            batch_size,
            do_reset,
        )

        indexing_start_time = time.perf_counter()

        with self.langfuse.start_as_current_observation(
            as_type="chain",
            name="rag.indexing",
            input={
                "project_id": project_id,
                "batch_size": batch_size,
                "do_reset": do_reset,
            },
            metadata={
                "project_id": project_id,
            },
        ) as indexing_observation:
            try:
                project = await self.project_model.get_project_or_create_one(
                    project_id=project_id,
                )

                logger.info(
                    "Resolved project_id=%s to ObjectId=%s",
                    project_id,
                    project.id,
                )

                if do_reset:
                    logger.warning(
                        "Resetting Qdrant index for project_id=%s",
                        project_id,
                    )

                    await self.vector_db.delete_by_project_id(
                        project_id=project_id,
                    )

                total_indexed_chunks = 0
                total_assets = set()
                total_batches = 0
                embedding_dimension = None

                async for chunks in self.chunk_model.iter_project_chunks(
                    project_id=project.id,
                    batch_size=batch_size,
                ):
                    # Keep repealed article records in MongoDB, but do not
                    # embed metadata-only repeal markers into the vector index.
                    searchable_chunks = [
                        chunk
                        for chunk in chunks
                        if not getattr(chunk, "is_repealed", False)
                    ]

                    skipped_repealed = len(chunks) - len(searchable_chunks)

                    if skipped_repealed:
                        logger.info(
                            "Excluded repealed articles from vector indexing: "
                            "project_id=%s | count=%s",
                            project_id,
                            skipped_repealed,
                        )

                    if not searchable_chunks:
                        continue

                    chunks = searchable_chunks
                    total_batches += 1

                    with self.langfuse.start_as_current_observation(
                        as_type="span",
                        name=f"rag.indexing.batch.{total_batches}",
                        input={
                            "batch_number": total_batches,
                            "chunk_count": len(chunks),
                        },
                        metadata={
                            "project_id": project_id,
                        },
                    ) as batch_observation:
                        logger.info(
                            "Processing batch %s: size=%s",
                            total_batches,
                            len(chunks),
                        )

                        texts = [chunk.chunk_text for chunk in chunks]

                        logger.info(
                            "Generating embeddings: model=%s | batch=%s | chunks=%s",
                            getattr(
                                self.embedding_model,
                                "model_id",
                                "unknown",
                            ),
                            total_batches,
                            len(texts),
                        )

                        # --------------------------------------------
                        # Embedding timing
                        # --------------------------------------------
                        embedding_start_time = time.perf_counter()

                        with self.langfuse.start_as_current_observation(
                            as_type="embedding",
                            name="rag.document-embedding",
                            input={
                                "document_count": len(texts),
                            },
                            metadata={
                                "project_id": project_id,
                                "model": getattr(
                                    self.embedding_model,
                                    "model_id",
                                    "unknown",
                                ),
                            },
                        ) as embedding_observation:
                            try:
                                embeddings = await self.embedding_model.embed_documents(
                                    texts,
                                )

                                batch_dimension = (
                                    len(embeddings[0]) if embeddings else 0
                                )

                                embedding_observation.update(
                                    output={
                                        "embedding_count": len(embeddings),
                                        "dimension": batch_dimension,
                                    },
                                )

                            except Exception as exc:
                                embedding_observation.update(
                                    level="ERROR",
                                    status_message=type(exc).__name__,
                                )
                                raise

                            finally:
                                EMBEDDING_DURATION.observe(
                                    time.perf_counter() - embedding_start_time,
                                )

                        if len(embeddings) != len(chunks):
                            raise RuntimeError(
                                "Number of embeddings does not match number of chunks",
                            )

                        if embeddings:
                            batch_dimension = len(
                                embeddings[0],
                            )

                            for embedding in embeddings:
                                if len(embedding) != batch_dimension:
                                    raise RuntimeError(
                                        "Embedding dimensions are inconsistent",
                                    )

                            if embedding_dimension is None:
                                embedding_dimension = batch_dimension

                            elif batch_dimension != embedding_dimension:
                                raise RuntimeError(
                                    "Embedding dimension changed between batches",
                                )

                            logger.info(
                                "Embeddings generated successfully: "
                                "batch=%s | count=%s | dimension=%s",
                                total_batches,
                                len(embeddings),
                                batch_dimension,
                            )

                        vectors = [
                            {
                                "id": self._build_qdrant_point_id(
                                    chunk.id,
                                ),
                                "vector": embedding,
                                "payload": {
                                    "chunk_id": str(chunk.id),
                                    "project_id": project_id,
                                    "asset_id": str(
                                        chunk.chunk_asset_id,
                                    ),
                                    "text": chunk.chunk_text,
                                    "article_number": chunk.article_number,
                                    "book": chunk.book,
                                    "chapter": chunk.chapter,
                                    "section": chunk.section,
                                    "topic": chunk.topic,
                                    "text_ar": chunk.text_ar,
                                    "text_en": chunk.text_en,
                                    "is_repealed": chunk.is_repealed,
                                    "source_page": chunk.source_page,
                                    "citation": chunk.citation,
                                    "source_file": (chunk.chunk_metadata.get("source")),
                                    "chunk_metadata": chunk.chunk_metadata,
                                },
                            }
                            for chunk, embedding in zip(
                                chunks,
                                embeddings,
                            )
                        ]

                        logger.info(
                            "Upserting vectors into Qdrant: "
                            "batch=%s | collection=%s | vectors=%s",
                            total_batches,
                            getattr(
                                self.vector_db,
                                "collection_name",
                                "unknown",
                            ),
                            len(vectors),
                        )

                        # --------------------------------------------
                        # Vector store timing
                        # --------------------------------------------
                        vector_store_start_time = time.perf_counter()

                        with self.langfuse.start_as_current_observation(
                            as_type="span",
                            name="rag.vector-upsert",
                            input={
                                "vector_count": len(vectors),
                            },
                            metadata={
                                "project_id": project_id,
                                "collection": getattr(
                                    self.vector_db,
                                    "collection_name",
                                    "unknown",
                                ),
                            },
                        ) as vector_upsert_observation:
                            try:
                                await self.vector_db.upsert(
                                    vectors,
                                )

                                vector_upsert_observation.update(
                                    output={
                                        "vector_count": len(vectors),
                                    },
                                )

                            except Exception as exc:
                                vector_upsert_observation.update(
                                    level="ERROR",
                                    status_message=type(exc).__name__,
                                )
                                raise

                            finally:
                                VECTOR_STORE_DURATION.observe(
                                    time.perf_counter() - vector_store_start_time,
                                )

                        # --------------------------------------------
                        # Successful batch metrics
                        # --------------------------------------------
                        total_indexed_chunks += len(vectors)

                        INDEXING_BATCHES.inc()

                        INDEXING_CHUNKS.inc(
                            len(vectors),
                        )

                        total_assets.update(chunk.chunk_asset_id for chunk in chunks)

                        batch_observation.update(
                            output={
                                "indexed_chunks": len(vectors),
                                "embedding_dimension": (
                                    batch_dimension if embeddings else 0
                                ),
                            },
                        )

                        logger.info(
                            "Batch indexing completed: batch=%s | indexed_chunks=%s",
                            total_batches,
                            total_indexed_chunks,
                        )

                if total_indexed_chunks == 0:
                    logger.info(
                        "No chunks found for project_id=%s",
                        project_id,
                    )

                    indexing_observation.update(
                        output={
                            "assets": 0,
                            "chunks": 0,
                            "batches": 0,
                        },
                    )

                    return {
                        "message": RAGMessage.NO_CHUNKS.value,
                        "project_id": project_id,
                        "assets": 0,
                        "chunks": 0,
                    }

                logger.info(
                    "Indexing completed successfully: "
                    "project_id=%s | batches=%s | assets=%s | chunks=%s",
                    project_id,
                    total_batches,
                    len(total_assets),
                    total_indexed_chunks,
                )

                indexing_observation.update(
                    output={
                        "assets": len(total_assets),
                        "chunks": total_indexed_chunks,
                        "batches": total_batches,
                    },
                )

                return {
                    "message": RAGMessage.INDEX_SUCCESS.value,
                    "project_id": project_id,
                    "assets": len(total_assets),
                    "chunks": total_indexed_chunks,
                }

            except Exception as exc:
                INDEXING_FAILURES.inc()

                indexing_observation.update(
                    level="ERROR",
                    status_message=type(exc).__name__,
                )

                logger.exception(
                    "Indexing failed for project_id=%s",
                    project_id,
                )

                raise

            finally:
                # --------------------------------------------
                # Total indexing duration
                # --------------------------------------------
                INDEXING_DURATION.observe(
                    time.perf_counter() - indexing_start_time,
                )
