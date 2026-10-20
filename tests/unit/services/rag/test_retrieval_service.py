from unittest.mock import AsyncMock

import pytest

from services.rag.retrieval_service import RetrievalService


def make_service(embedding_model, vector_db, langfuse):
    service = RetrievalService.__new__(RetrievalService)
    service.embedding_model = embedding_model
    service.vector_db = vector_db
    service.langfuse = langfuse
    return service


@pytest.mark.anyio
async def test_search_embeds_query_and_returns_legal_metadata(
    sample_documents,
    langfuse,
):
    embedding_model = AsyncMock()
    vector_db = AsyncMock()
    embedding_model.embed_text.return_value = [0.1, 0.2, 0.3]

    vector_db.search.return_value = [
        {
            "id": f"point-{document['article_number']}",
            "payload": {
                key: value for key, value in document.items() if key != "score"
            },
            "score": document["score"],
        }
        for document in sample_documents
    ]

    service = make_service(embedding_model, vector_db, langfuse)

    results = await service.search(
        project_id="project1",
        query="What is contract formation?",
        limit=2,
    )

    embedding_model.embed_text.assert_awaited_once_with(
        "What is contract formation?",
    )
    vector_db.search.assert_awaited_once_with(
        vector=[0.1, 0.2, 0.3],
        limit=2,
        project_id="project1",
    )

    assert len(results) == 2
    assert results[0]["article_number"] == 147
    assert results[0]["text_ar"] == "ينشأ العقد بتوافق إرادتين."
    assert results[0]["citation"] == "Egyptian Civil Code, Article 147"
    assert results[0]["score"] == 0.91
    assert results[1]["article_number"] == 148


@pytest.mark.anyio
async def test_search_skips_results_without_text(langfuse):
    embedding_model = AsyncMock()
    vector_db = AsyncMock()
    embedding_model.embed_text.return_value = [0.1, 0.2]

    vector_db.search.return_value = [
        {"payload": {}, "score": 0.9},
        {
            "payload": {
                "text": "Valid legal article",
                "article_number": 147,
                "text_en": "Valid legal article",
            },
            "score": 0.8,
        },
    ]

    service = make_service(embedding_model, vector_db, langfuse)

    results = await service.search(
        project_id="project1",
        query="test",
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["text"] == "Valid legal article"
    assert results[0]["article_number"] == 147
    assert results[0]["score"] == 0.8


@pytest.mark.anyio
async def test_search_rejects_non_positive_limit():
    service = RetrievalService.__new__(RetrievalService)

    with pytest.raises(ValueError, match="greater than zero"):
        await service.search(
            project_id="project1",
            query="test",
            limit=0,
        )


def test_retrieval_service_constructor_initializes_dependencies(langfuse):
    embedding_model = object()
    vector_db = object()

    service = RetrievalService(
        embedding_model=embedding_model,
        vector_db=vector_db,
        langfuse=langfuse,
    )

    assert service.embedding_model is embedding_model
    assert service.vector_db is vector_db
    assert service.langfuse is langfuse
