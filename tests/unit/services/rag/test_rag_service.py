from unittest.mock import AsyncMock

import pytest

from services.rag.rag_service import RAGService


def make_service(retrieval_service, chat_model, langfuse):
    # AsyncMock creates unknown attributes dynamically. Disable this optional
    # method here so these tests exercise the regular generate() fallback.
    chat_model.generate_validated = None

    service = RAGService.__new__(RAGService)
    service.retrieval_service = retrieval_service
    service.chat_model = chat_model
    service.langfuse = langfuse
    service.online_evaluator = None
    service._evaluation_tasks = set()
    return service


@pytest.mark.anyio
async def test_generate_uses_retrieved_legal_documents_and_returns_sources(
    sample_documents,
    langfuse,
):
    retrieval_service = AsyncMock()
    chat_model = AsyncMock()

    retrieval_service.search.return_value = sample_documents
    chat_model.generate.return_value = (
        "Under Article 147, a contract is formed by agreement. "
        "Article 148 requires performance in good faith."
    )

    service = make_service(retrieval_service, chat_model, langfuse)

    result = await service.generate(
        project_id="project1",
        query="Explain contract formation and good faith.",
        limit=2,
    )

    retrieval_service.search.assert_awaited_once_with(
        project_id="project1",
        query="Explain contract formation and good faith.",
        limit=2,
    )
    chat_model.generate.assert_awaited_once()

    prompt = chat_model.generate.await_args.kwargs["prompt"]
    assert "Article 147" in prompt
    assert "Article 148" in prompt
    assert sample_documents[0]["text_en"] in prompt
    assert sample_documents[1]["text_en"] in prompt

    assert result["answer"].startswith("Under Article 147")
    assert len(result["sources"]) == 2
    assert [source["article_number"] for source in result["sources"]] == [
        147,
        148,
    ]
    assert result["sources"][0]["citation"] == ("Egyptian Civil Code, Article 147")
    assert result["sources"][0]["source_file"] == "civil_code.pdf"


@pytest.mark.anyio
async def test_generate_returns_no_context_answer_without_calling_llm(langfuse):
    retrieval_service = AsyncMock()
    chat_model = AsyncMock()
    retrieval_service.search.return_value = []

    service = make_service(retrieval_service, chat_model, langfuse)

    result = await service.generate(
        project_id="project1",
        query="ما شروط العقد؟",
        limit=5,
    )

    assert "لم أجد" in result["answer"]
    assert result["sources"] == []
    chat_model.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_generate_returns_english_no_context_answer(langfuse):
    retrieval_service = AsyncMock()
    chat_model = AsyncMock()
    retrieval_service.search.return_value = []

    service = make_service(retrieval_service, chat_model, langfuse)

    result = await service.generate(
        project_id="project1",
        query="What are the contract requirements?",
        limit=5,
    )

    assert "not contain enough information" in result["answer"]
    assert result["sources"] == []
    chat_model.generate.assert_not_awaited()


def test_build_sources_filters_articles_not_cited_in_answer(
    sample_documents,
):
    sources = RAGService._build_sources(
        sample_documents,
        answer="The rule appears in Article 148.",
    )

    assert len(sources) == 1
    assert sources[0]["article_number"] == 148


def test_build_sources_does_not_include_empty_legal_records():
    sources = RAGService._build_sources(
        [
            {
                "article_number": 54,
                "score": 0.9,
                "text_ar": "",
                "text_en": "",
                "is_repealed": True,
            }
        ],
    )

    assert sources == []


@pytest.mark.anyio
async def test_evidence_selection_returns_selected_candidate_ids(
    sample_documents,
    langfuse,
):
    retrieval_service = AsyncMock()
    chat_model = AsyncMock()
    chat_model.generate.return_value = '{"selected_ids": [2]}'

    service = make_service(retrieval_service, chat_model, langfuse)

    selected = await service._select_relevant_results(
        query="What is good faith?",
        results=sample_documents,
        limit=1,
    )

    assert selected == [sample_documents[1]]
    chat_model.generate.assert_awaited_once()


@pytest.mark.anyio
async def test_evidence_selection_falls_back_to_top_result_on_invalid_json(
    sample_documents,
    langfuse,
):
    retrieval_service = AsyncMock()
    chat_model = AsyncMock()
    chat_model.generate.return_value = "not valid JSON"

    service = make_service(retrieval_service, chat_model, langfuse)

    selected = await service._select_relevant_results(
        query="What is good faith?",
        results=sample_documents,
        limit=1,
    )

    assert selected == [sample_documents[0]]


# Additional coverage: prompt construction and online evaluation


@pytest.mark.anyio
async def test_index_and_search_delegate_to_their_services(langfuse):
    from unittest.mock import AsyncMock

    service = RAGService.__new__(RAGService)
    service.indexing_service = AsyncMock()
    service.retrieval_service = AsyncMock()
    service.langfuse = langfuse

    expected_index = {"project_id": "project1", "chunks": 3}
    expected_results = [{"article_number": 147, "score": 0.9}]
    service.indexing_service.index.return_value = expected_index
    service.retrieval_service.search.return_value = expected_results

    indexed = await service.index(
        project_id="project1",
        batch_size=8,
        do_reset=True,
    )
    results = await service.search(
        project_id="project1",
        query="ما شروط العقد؟",
        limit=4,
    )

    assert indexed == expected_index
    service.indexing_service.index.assert_awaited_once_with(
        project_id="project1",
        batch_size=8,
        do_reset=True,
    )
    assert results == expected_results
    service.retrieval_service.search.assert_awaited_once_with(
        project_id="project1",
        query="ما شروط العقد؟",
        limit=4,
    )


def test_query_language_and_prompt_document_formatting(sample_documents):
    assert RAGService._query_language("ما شروط العقد؟") == "ar"
    assert RAGService._query_language("What is a contract?") == "en"

    formatted = RAGService._format_prompt_document(
        {
            "citation": "Custom legal citation",
            "source_page": 12,
            "is_repealed": True,
            "text": "Extracted source text",
        },
        "en",
    )

    assert "Custom legal citation" in formatted
    assert "PDF page: 12" in formatted
    assert "marked as repealed" in formatted
    assert "Extracted source text" in formatted


def test_build_prompt_preserves_article_metadata(sample_documents):
    system_text, user_text, documents = RAGService._build_prompt(
        query="ما شروط العقد؟",
        results=sample_documents,
    )

    assert system_text
    assert "ما شروط العقد؟" in user_text
    assert len(documents) == 2
    assert documents[0]["article_number"] == 147
    assert documents[0]["citation"] == ("Egyptian Civil Code, Article 147")


def test_schedule_evaluation_returns_when_evaluator_or_trace_is_missing(
    langfuse,
    sample_documents,
):
    from types import SimpleNamespace

    service = RAGService.__new__(RAGService)
    service.langfuse = langfuse
    service.online_evaluator = None
    service._evaluation_tasks = set()

    service._schedule_evaluation(
        project_id="project1",
        query="Question",
        answer="Answer",
        results=sample_documents,
    )
    assert not service._evaluation_tasks

    service.online_evaluator = SimpleNamespace()
    service._schedule_evaluation(
        project_id="project1",
        query="Question",
        answer="Answer",
        results=sample_documents,
    )
    assert not service._evaluation_tasks


@pytest.mark.anyio
async def test_evaluate_and_score_records_scores_and_completed_status(
    sample_documents,
    langfuse,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    service = RAGService.__new__(RAGService)
    service.langfuse = langfuse
    service.online_evaluator = SimpleNamespace(
        judge_model="qwen2.5:3b",
        embedding_model="bge-m3:latest",
        evaluate=AsyncMock(
            return_value=(
                {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                },
                {},
            )
        ),
    )
    langfuse.create_score = Mock()

    await service._evaluate_and_score(
        trace_id="trace-1",
        project_id="project1",
        query="What are the contract requirements?",
        answer="Article 147 describes contract formation.",
        results=sample_documents,
    )

    service.online_evaluator.evaluate.assert_awaited_once()
    assert langfuse.create_score.call_count == 3
    assert langfuse.create_score.call_args_list[-1].kwargs["name"] == (
        "online_evaluation_status"
    )
    assert langfuse.create_score.call_args_list[-1].kwargs["value"] == ("completed")


@pytest.mark.anyio
async def test_evaluate_and_score_records_failed_status(
    sample_documents,
    langfuse,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    service = RAGService.__new__(RAGService)
    service.langfuse = langfuse
    service.online_evaluator = SimpleNamespace(
        judge_model="qwen2.5:3b",
        embedding_model="bge-m3:latest",
        evaluate=AsyncMock(side_effect=RuntimeError("Local evaluator unavailable")),
    )
    langfuse.create_score = Mock()

    await service._evaluate_and_score(
        trace_id="trace-2",
        project_id="project1",
        query="Question",
        answer="Answer",
        results=sample_documents,
    )

    assert langfuse.create_score.call_count == 1
    call = langfuse.create_score.call_args.kwargs
    assert call["name"] == "online_evaluation_status"
    assert call["value"] == "failed"


@pytest.mark.anyio
async def test_generate_uses_generate_validated_when_available(
    sample_documents,
    langfuse,
):
    from unittest.mock import AsyncMock

    retrieval_service = AsyncMock()
    chat_model = AsyncMock()
    retrieval_service.search.return_value = [sample_documents[0]]
    service = make_service(retrieval_service, chat_model, langfuse)
    chat_model.generate_validated = AsyncMock(
        return_value="The rule is stated in Article 147."
    )

    result = await service.generate(
        project_id="project1",
        query="What is the contract formation rule?",
        limit=1,
    )

    chat_model.generate_validated.assert_awaited_once()
    chat_model.generate.assert_not_awaited()
    assert result["sources"][0]["article_number"] == 147


@pytest.mark.anyio
async def test_generate_propagates_retrieval_errors(langfuse):
    from unittest.mock import AsyncMock

    retrieval_service = AsyncMock()
    chat_model = AsyncMock()
    retrieval_service.search.side_effect = RuntimeError("Qdrant unavailable")

    service = make_service(retrieval_service, chat_model, langfuse)

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        await service.generate(
            project_id="project1",
            query="What is the law?",
            limit=3,
        )

    chat_model.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_close_returns_when_no_evaluation_tasks_are_pending(langfuse):
    service = RAGService.__new__(RAGService)
    service.langfuse = langfuse
    service._evaluation_worker_task = None
    service._pending_evaluations = []

    await service.close(timeout_seconds=0.1)


def test_rag_service_constructor_initializes_dependencies(langfuse):
    chat_model = object()
    retrieval_service = object()
    indexing_service = object()
    online_evaluator = object()

    service = RAGService(
        chat_model=chat_model,
        retrieval_service=retrieval_service,
        indexing_service=indexing_service,
        langfuse=langfuse,
        online_evaluator=online_evaluator,
    )

    assert service.chat_model is chat_model
    assert service.retrieval_service is retrieval_service
    assert service.indexing_service is indexing_service
    assert service.langfuse is langfuse
    assert service.online_evaluator is online_evaluator
    assert service._pending_evaluations == []
    assert service._evaluation_worker_task is None
    assert service.evaluation_interval_seconds == 300


@pytest.mark.anyio
async def test_evaluate_and_score_records_partial_status_when_metric_is_missing(
    sample_documents,
    langfuse,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    service = RAGService.__new__(RAGService)
    service.langfuse = langfuse
    service.online_evaluator = SimpleNamespace(
        judge_model="qwen2.5:3b",
        embedding_model="bge-m3:latest",
        evaluate=AsyncMock(
            return_value=(
                {"answer_relevancy": 0.8},
                {"faithfulness": "Judge unavailable"},
            )
        ),
    )
    langfuse.create_score = Mock()

    await service._evaluate_and_score(
        trace_id="trace-partial-metric",
        project_id="project1",
        query="What is the legal rule?",
        answer="The answer.",
        results=sample_documents,
    )

    status_call = langfuse.create_score.call_args_list[-1]
    assert status_call.kwargs["name"] == "online_evaluation_status"
    assert status_call.kwargs["value"] == "partial"
    assert status_call.kwargs["metadata"]["missing_required_metrics"] == [
        "faithfulness"
    ]
    assert status_call.kwargs["metadata"]["failed_metrics"] == ["faithfulness"]


@pytest.mark.anyio
async def test_evaluate_and_score_records_partial_status_when_score_write_fails(
    sample_documents,
    langfuse,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    service = RAGService.__new__(RAGService)
    service.langfuse = langfuse
    service.online_evaluator = SimpleNamespace(
        judge_model="qwen2.5:3b",
        embedding_model="bge-m3:latest",
        evaluate=AsyncMock(
            return_value=(
                {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                },
                {},
            )
        ),
    )

    def create_score_side_effect(**kwargs):
        if kwargs.get("name") == "faithfulness":
            raise RuntimeError("Langfuse unavailable")

    langfuse.create_score = Mock(side_effect=create_score_side_effect)

    await service._evaluate_and_score(
        trace_id="trace-score-write-error",
        project_id="project1",
        query="What is the legal rule?",
        answer="The answer.",
        results=sample_documents,
    )

    status_call = langfuse.create_score.call_args_list[-1]
    assert status_call.kwargs["name"] == "online_evaluation_status"
    assert status_call.kwargs["value"] == "partial"
    assert status_call.kwargs["metadata"]["score_write_failures"] == ["faithfulness"]


@pytest.mark.anyio
async def test_evaluate_and_score_propagates_cancellation(
    sample_documents,
    langfuse,
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    service = RAGService.__new__(RAGService)
    service.langfuse = langfuse
    service.online_evaluator = SimpleNamespace(
        judge_model="qwen2.5:3b",
        embedding_model="bge-m3:latest",
        evaluate=AsyncMock(side_effect=asyncio.CancelledError()),
    )
    langfuse.create_score = Mock()

    with pytest.raises(asyncio.CancelledError):
        await service._evaluate_and_score(
            trace_id="trace-cancelled",
            project_id="project1",
            query="What is the legal rule?",
            answer="The answer.",
            results=sample_documents,
        )

    langfuse.create_score.assert_not_called()
