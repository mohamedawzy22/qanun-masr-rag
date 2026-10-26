from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.rag import evaluation_metrics


def make_metric(*, value=None, error=None):
    """Create a deterministic async metric for unit testing."""
    ascore = AsyncMock()

    if error is not None:
        ascore.side_effect = error
    else:
        ascore.return_value = SimpleNamespace(value=value)

    return SimpleNamespace(ascore=ascore)


def test_build_metrics_configures_llm_embeddings_and_prompts():
    client = object()
    llm = object()
    embeddings = object()
    faithfulness_metric = object()
    answer_relevancy_metric = MagicMock()

    with (
        patch.object(evaluation_metrics, "llm_factory") as llm_factory,
        patch.object(evaluation_metrics, "embedding_factory") as embedding_factory,
        patch.object(
            evaluation_metrics,
            "AnswerRelevancy",
            return_value=answer_relevancy_metric,
        ) as answer_relevancy_class,
        patch.object(
            evaluation_metrics,
            "Faithfulness",
            return_value=faithfulness_metric,
        ) as faithfulness_class,
    ):
        llm_factory.return_value = llm
        embedding_factory.return_value = embeddings

        metrics = evaluation_metrics.build_metrics(client)

    llm_factory.assert_called_once_with(
        evaluation_metrics.JUDGE_MODEL,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=2048,
    )
    embedding_factory.assert_called_once_with(
        provider="openai",
        model=evaluation_metrics.EMBEDDING_MODEL,
        client=client,
    )
    answer_relevancy_class.assert_called_once_with(
        llm=llm,
        embeddings=embeddings,
    )
    faithfulness_class.assert_called_once_with(llm=llm)

    assert metrics == {
        "faithfulness": faithfulness_metric,
        "answer_relevancy": answer_relevancy_metric,
    }

    prompt = answer_relevancy_metric.prompt
    assert "noncommittal=1" in prompt.instruction
    assert len(prompt.examples) == 2


@pytest.mark.anyio
async def test_score_sample_returns_both_scores():
    faithfulness = make_metric(value=0.91)
    relevancy = make_metric(value=0.84)

    scores, errors = await evaluation_metrics.score_sample(
        {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
        },
        question="ما شروط تنفيذ العقد؟",
        answer="يجب تنفيذ العقد طبقًا لما اشتمل عليه وبطريقة تتفق مع حسن النية.",
        contexts=["Article 148: A contract must be performed in good faith."],
        reference="This legacy argument is not used by the current metrics.",
    )

    assert scores == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.84,
    }
    assert errors == {}

    faithfulness.ascore.assert_awaited_once_with(
        user_input="ما شروط تنفيذ العقد؟",
        response="يجب تنفيذ العقد طبقًا لما اشتمل عليه وبطريقة تتفق مع حسن النية.",
        retrieved_contexts=["Article 148: A contract must be performed in good faith."],
    )
    relevancy.ascore.assert_awaited_once_with(
        user_input="ما شروط تنفيذ العقد؟",
        response="يجب تنفيذ العقد طبقًا لما اشتمل عليه وبطريقة تتفق مع حسن النية.",
    )


@pytest.mark.anyio
async def test_score_sample_rejects_blank_question_or_answer():
    faithfulness = make_metric(value=0.9)
    relevancy = make_metric(value=0.8)

    scores, errors = await evaluation_metrics.score_sample(
        {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
        },
        question="   ",
        answer="A non-empty answer",
        contexts=["Some legal context"],
    )

    assert scores == {}
    assert errors == {
        "faithfulness": "Missing question or answer.",
        "answer_relevancy": "Missing question or answer.",
    }
    faithfulness.ascore.assert_not_awaited()
    relevancy.ascore.assert_not_awaited()


@pytest.mark.anyio
async def test_score_sample_skips_faithfulness_without_contexts():
    faithfulness = make_metric(value=0.9)
    relevancy = make_metric(value=0.8)

    scores, errors = await evaluation_metrics.score_sample(
        {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
        },
        question="What does the contract require?",
        answer="The answer contains a definite statement.",
        contexts=[],
    )

    assert scores == {"answer_relevancy": 0.8}
    assert errors == {"faithfulness": "Skipped: no retrieved contexts."}
    faithfulness.ascore.assert_not_awaited()
    relevancy.ascore.assert_awaited_once()


@pytest.mark.anyio
async def test_score_sample_reports_missing_metric():
    relevancy = make_metric(value=0.82)

    scores, errors = await evaluation_metrics.score_sample(
        {"answer_relevancy": relevancy},
        question="What is the legal rule?",
        answer="The answer.",
        contexts=["A retrieved provision."],
    )

    assert scores == {"answer_relevancy": 0.82}
    assert errors == {"faithfulness": "Metric is not configured."}


@pytest.mark.anyio
async def test_score_sample_preserves_success_when_one_metric_fails():
    faithfulness = make_metric(error=RuntimeError("Judge unavailable"))
    relevancy = make_metric(value=0.78)

    scores, errors = await evaluation_metrics.score_sample(
        {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
        },
        question="What does the provision say?",
        answer="The provision sets out a legal obligation.",
        contexts=["Retrieved legal text."],
    )

    assert scores == {"answer_relevancy": 0.78}
    assert errors == {"faithfulness": "RuntimeError: Judge unavailable"}
    faithfulness.ascore.assert_awaited_once()
    relevancy.ascore.assert_awaited_once()
