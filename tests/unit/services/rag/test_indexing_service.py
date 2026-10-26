from unittest.mock import AsyncMock

import pytest

from services.rag.indexing_service import IndexingService


def make_service(
    embedding_model,
    vector_db,
    project_model,
    chunk_model,
    langfuse,
):
    service = IndexingService.__new__(IndexingService)
    service.embedding_model = embedding_model
    service.vector_db = vector_db
    service.project_model = project_model
    service.chunk_model = chunk_model
    service.langfuse = langfuse
    return service


def test_build_qdrant_point_id_is_deterministic():
    first = IndexingService._build_qdrant_point_id("chunk-123")
    second = IndexingService._build_qdrant_point_id("chunk-123")

    assert first == second
    assert isinstance(first, str)


def test_build_qdrant_point_id_changes_for_different_chunks():
    first = IndexingService._build_qdrant_point_id("chunk-1")
    second = IndexingService._build_qdrant_point_id("chunk-2")

    assert first != second


@pytest.mark.anyio
async def test_index_embeds_chunks_and_upserts_legal_metadata(
    sample_chunks,
    langfuse,
):
    embedding_model = AsyncMock()
    vector_db = AsyncMock()
    project_model = AsyncMock()
    chunk_model = AsyncMock()

    project_model.get_project_or_create_one.return_value = type(
        "Project",
        (),
        {"id": "mongo-project-1"},
    )()

    async def chunk_iterator(*, project_id, batch_size):
        assert project_id == "mongo-project-1"
        assert batch_size == 2
        yield sample_chunks

    chunk_model.iter_project_chunks = chunk_iterator
    embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    service = make_service(
        embedding_model,
        vector_db,
        project_model,
        chunk_model,
        langfuse,
    )

    result = await service.index(
        project_id="project1",
        batch_size=2,
    )

    project_model.get_project_or_create_one.assert_awaited_once_with(
        project_id="project1",
    )
    embedding_model.embed_documents.assert_awaited_once_with(
        [chunk.chunk_text for chunk in sample_chunks]
    )
    vector_db.upsert.assert_awaited_once()

    vectors = vector_db.upsert.await_args.args[0]
    assert len(vectors) == 2

    first_payload = vectors[0]["payload"]
    assert first_payload["chunk_id"] == "chunk-1"
    assert first_payload["project_id"] == "project1"
    assert first_payload["asset_id"] == "asset-1"
    assert first_payload["article_number"] == 147
    assert first_payload["text_ar"] == "النص العربي"
    assert first_payload["text_en"] == "English text"
    assert first_payload["citation"] == "Egyptian Civil Code, Article 147"
    assert first_payload["source_file"] == "civil_code.pdf"
    assert vectors[0]["vector"] == [0.1, 0.2, 0.3]

    assert result["project_id"] == "project1"
    assert result["assets"] == 1
    assert result["chunks"] == 2


@pytest.mark.anyio
async def test_index_includes_repealed_articles_and_upserts_metadata(
    langfuse,
):
    embedding_model = AsyncMock()
    embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
    ]

    vector_db = AsyncMock()
    project_model = AsyncMock()
    chunk_model = AsyncMock()

    project_model.get_project_or_create_one.return_value = type(
        "Project",
        (),
        {"id": "mongo-project-1"},
    )()

    repealed_chunk = type(
        "Chunk",
        (),
        {
            "id": "repealed-54",
            "chunk_text": "Repealed article",
            "chunk_asset_id": "asset-1",
            "chunk_metadata": {"source": "civil_code.pdf"},
            "article_number": 54,
            "book": "Book One",
            "chapter": "Chapter One",
            "section": "Section One",
            "topic": "Repealed provisions",
            "text_ar": "المادة 54 ملغاة",
            "text_en": "Article 54 repealed",
            "is_repealed": True,
            "source_page": 4,
            "citation": "Egyptian Civil Code, Article 54",
        },
    )()

    async def chunk_iterator(*, project_id, batch_size):
        yield [repealed_chunk]

    chunk_model.iter_project_chunks = chunk_iterator

    service = make_service(
        embedding_model,
        vector_db,
        project_model,
        chunk_model,
        langfuse,
    )

    result = await service.index(project_id="project1")

    embedding_model.embed_documents.assert_awaited_once_with(
        ["Repealed article"],
    )
    vector_db.upsert.assert_awaited_once()

    vectors = vector_db.upsert.await_args.args[0]
    assert len(vectors) == 1

    payload = vectors[0]["payload"]

    assert payload["article_number"] == 54
    assert payload["book"] == "Book One"
    assert payload["chapter"] == "Chapter One"
    assert payload["section"] == "Section One"
    assert payload["topic"] == "Repealed provisions"
    assert payload["text_ar"] == "المادة 54 ملغاة"
    assert payload["text_en"] == "Article 54 repealed"
    assert payload["is_repealed"] is True
    assert payload["source_page"] == 4
    assert payload["citation"] == "Egyptian Civil Code, Article 54"

    assert payload["source_file"] == "civil_code.pdf"
    assert vectors[0]["vector"] == [0.1, 0.2, 0.3]

    assert result["chunks"] == 1
    assert result["assets"] == 1


@pytest.mark.anyio
async def test_index_rejects_invalid_batch_size(langfuse):
    service = IndexingService.__new__(IndexingService)
    service.langfuse = langfuse

    with pytest.raises(ValueError, match="greater than zero"):
        await service.index(
            project_id="project1",
            batch_size=0,
        )


@pytest.mark.anyio
async def test_index_rejects_non_integer_batch_size(langfuse):
    service = IndexingService.__new__(IndexingService)
    service.langfuse = langfuse

    with pytest.raises(TypeError, match="must be an integer"):
        await service.index(
            project_id="project1",
            batch_size="invalid",
        )
