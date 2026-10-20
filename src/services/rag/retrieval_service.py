import time

from langfuse import Langfuse

from services.llm.embedding_interface import EmbeddingModel
from services.vectordb.vector_db_interface import VectorDB
from utils.logger import get_logger
from utils.metrics import (
    RETRIEVAL_DURATION,
    RETRIEVAL_EMBEDDING_DURATION,
    RETRIEVAL_FAILURES,
    RETRIEVAL_REQUESTS,
    RETRIEVAL_RESULTS,
    RETRIEVAL_VECTOR_SEARCH_DURATION,
)

logger = get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDB,
        langfuse: Langfuse,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.langfuse = langfuse

        logger.info("Retrieval service initialized successfully")

    async def search(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve legal articles using vector similarity search."""

        if limit < 1:
            raise ValueError("Search limit must be greater than zero")

        # Retrieve a larger candidate pool for lightweight legal ranking.
        candidate_limit = limit

        logger.info(
            "Starting retrieval: project_id=%s | limit=%s | candidates=%s",
            project_id,
            limit,
            candidate_limit,
        )

        RETRIEVAL_REQUESTS.inc()
        retrieval_start_time = time.perf_counter()

        with self.langfuse.start_as_current_observation(
            as_type="retriever",
            name="rag.retrieval",
            input={
                "query": query,
                "limit": limit,
                "candidate_limit": candidate_limit,
            },
            metadata={
                "project_id": project_id,
            },
        ) as retrieval_observation:
            try:
                # Generate the query embedding.
                embedding_start_time = time.perf_counter()

                try:
                    with self.langfuse.start_as_current_observation(
                        as_type="embedding",
                        name="rag.query-embedding",
                        input={"query": query},
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
                            query_embedding = await self.embedding_model.embed_text(
                                query
                            )

                            embedding_observation.update(
                                output={
                                    "dimension": len(query_embedding),
                                },
                            )

                        except Exception as exc:
                            embedding_observation.update(
                                level="ERROR",
                                status_message=type(exc).__name__,
                            )
                            raise
                finally:
                    RETRIEVAL_EMBEDDING_DURATION.observe(
                        time.perf_counter() - embedding_start_time,
                    )

                logger.info(
                    "Query embedding generated: project_id=%s | dimension=%s",
                    project_id,
                    len(query_embedding),
                )

                # Search Qdrant using the original requested limit.
                vector_search_start_time = time.perf_counter()

                try:
                    with self.langfuse.start_as_current_observation(
                        as_type="span",
                        name="rag.vector-search",
                        input={
                            "requested_limit": limit,
                            "candidate_limit": candidate_limit,
                        },
                        metadata={
                            "project_id": project_id,
                            "vector_dimension": len(query_embedding),
                            "vector_db": getattr(
                                self.vector_db,
                                "collection_name",
                                "unknown",
                            ),
                        },
                    ) as vector_search_observation:
                        try:
                            results = await self.vector_db.search(
                                vector=query_embedding,
                                limit=candidate_limit,
                                project_id=project_id,
                            )

                            vector_search_observation.update(
                                output={
                                    "result_count": len(results),
                                    "scores": [
                                        result["score"]
                                        for result in results
                                        if "score" in result
                                    ],
                                },
                            )

                        except Exception as exc:
                            vector_search_observation.update(
                                level="ERROR",
                                status_message=type(exc).__name__,
                            )
                            raise
                finally:
                    RETRIEVAL_VECTOR_SEARCH_DURATION.observe(
                        time.perf_counter() - vector_search_start_time,
                    )

                candidate_count = len(results)

                # Qdrant already returns results by vector similarity.
                # Keep that order and let the generation prompt assess relevance.
                with self.langfuse.start_as_current_observation(
                    as_type="span",
                    name="rag.candidate-selection",
                    input={
                        "requested_limit": limit,
                        "candidate_count": candidate_count,
                    },
                    metadata={
                        "project_id": project_id,
                        "strategy": "vector_similarity",
                    },
                ) as selection_observation:
                    selection_observation.update(
                        output={
                            "returned_count": candidate_count,
                            "article_numbers": [
                                (item.get("payload") or {}).get("article_number")
                                for item in results
                            ],
                            "scores": [
                                float(item.get("score") or 0.0) for item in results
                            ],
                        },
                    )

                # Convert Qdrant payloads to application results.
                search_results = []

                for result in results:
                    payload = result.get("payload") or {}
                    text = payload.get("text")

                    if not text:
                        logger.warning(
                            "Search result has no text: project_id=%s | point_id=%s",
                            project_id,
                            result.get("id"),
                        )
                        continue

                    search_results.append(
                        {
                            "text": text,
                            "score": result["score"],
                            "article_number": payload.get("article_number"),
                            "book": payload.get("book"),
                            "chapter": payload.get("chapter"),
                            "section": payload.get("section"),
                            "topic": payload.get("topic"),
                            "text_ar": payload.get("text_ar"),
                            "text_en": payload.get("text_en"),
                            "is_repealed": payload.get("is_repealed"),
                            "source_page": payload.get("source_page"),
                            "citation": payload.get("citation"),
                            "source_file": payload.get("source_file"),
                        }
                    )

                RETRIEVAL_RESULTS.inc(len(search_results))

                retrieval_observation.update(
                    output={
                        "result_count": len(search_results),
                        "scores": [result["score"] for result in search_results],
                    },
                )

                logger.info(
                    "Retrieval completed successfully: project_id=%s | results=%s",
                    project_id,
                    len(search_results),
                )

                return search_results

            except Exception as exc:
                RETRIEVAL_FAILURES.inc()

                retrieval_observation.update(
                    level="ERROR",
                    status_message=type(exc).__name__,
                )

                logger.exception(
                    "Retrieval failed: project_id=%s",
                    project_id,
                )
                raise

            finally:
                RETRIEVAL_DURATION.observe(
                    time.perf_counter() - retrieval_start_time,
                )
