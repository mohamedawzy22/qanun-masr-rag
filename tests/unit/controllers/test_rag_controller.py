from unittest.mock import AsyncMock

import pytest

from controllers.rag_controller import RAGController


@pytest.mark.anyio
async def test_index_delegates_to_rag_service():
    rag_service = AsyncMock()
    expected = {
        "project_id": "project1",
        "chunks": 2,
    }
    rag_service.index.return_value = expected

    controller = RAGController(rag_service)

    result = await controller.index(
        project_id="project1",
        batch_size=16,
        do_reset=True,
    )

    rag_service.index.assert_awaited_once_with(
        project_id="project1",
        batch_size=16,
        do_reset=True,
    )
    assert result == expected


@pytest.mark.anyio
async def test_search_delegates_to_rag_service():
    rag_service = AsyncMock()
    expected = [{"text": "نص المادة", "score": 0.9}]
    rag_service.search.return_value = expected

    controller = RAGController(rag_service)

    result = await controller.search(
        project_id="project1",
        query="ما شروط العقد؟",
        limit=5,
    )

    rag_service.search.assert_awaited_once_with(
        project_id="project1",
        query="ما شروط العقد؟",
        limit=5,
    )
    assert result == expected


@pytest.mark.anyio
async def test_generate_delegates_to_rag_service():
    rag_service = AsyncMock()
    expected = {
        "answer": "الإجابة مستندة إلى النص القانوني.",
        "sources": [],
    }
    rag_service.generate.return_value = expected

    controller = RAGController(rag_service)

    result = await controller.generate(
        project_id="project1",
        query="ما شروط العقد؟",
        limit=5,
        user_id="user-1",
        session_id="session-1",
    )

    rag_service.generate.assert_awaited_once_with(
        project_id="project1",
        query="ما شروط العقد؟",
        limit=5,
        user_id="user-1",
        session_id="session-1",
    )
    assert result == expected
