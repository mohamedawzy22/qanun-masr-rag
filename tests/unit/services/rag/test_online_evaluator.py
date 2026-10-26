import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services.rag import online_evaluator


def make_evaluator(monkeypatch, base_url="http://localhost:11434"):
    """Construct the evaluator with deterministic client and metrics mocks."""
    client = SimpleNamespace(close=AsyncMock())
    metrics = {"faithfulness": object()}

    client_factory = Mock(return_value=client)
    metrics_factory = Mock(return_value=metrics)

    monkeypatch.setattr(
        online_evaluator,
        "AsyncOpenAI",
        client_factory,
    )
    monkeypatch.setattr(
        online_evaluator,
        "build_metrics",
        metrics_factory,
    )

    evaluator = online_evaluator.OnlineRAGEvaluator(
        ollama_base_url=base_url,
        timeout_seconds=37,
    )

    return evaluator, client, client_factory, metrics_factory, metrics


def test_evaluator_rejects_blank_ollama_url(monkeypatch):
    client_factory = Mock()
    monkeypatch.setattr(
        online_evaluator,
        "AsyncOpenAI",
        client_factory,
    )

    with pytest.raises(
        ValueError,
        match="OLLAMA_BASE_URL must not be empty",
    ):
        online_evaluator.OnlineRAGEvaluator(
            ollama_base_url="   ",
        )

    client_factory.assert_not_called()


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
    ],
)
def test_evaluator_normalizes_ollama_url(
    monkeypatch,
    base_url,
    expected_url,
):
    evaluator, client, client_factory, metrics_factory, metrics = make_evaluator(
        monkeypatch, base_url
    )

    client_factory.assert_called_once_with(
        base_url=expected_url,
        api_key="ollama",
        timeout=37,
        max_retries=1,
    )
    metrics_factory.assert_called_once_with(client)
    assert evaluator.client is client
    assert evaluator.metrics is metrics


@pytest.mark.anyio
async def test_evaluate_forwards_inputs_and_returns_scores(monkeypatch):
    evaluator, _, _, _, metrics = make_evaluator(monkeypatch)

    expected = (
        {"faithfulness": 0.9, "answer_relevancy": 0.8},
        {},
    )
    score_mock = AsyncMock(return_value=expected)
    monkeypatch.setattr(online_evaluator, "score_sample", score_mock)

    contexts = ["Article 147: legal text"]

    result = await evaluator.evaluate(
        question="What is the rule?",
        answer="The answer.",
        contexts=contexts,
        reference="legacy reference",
    )

    assert result == expected
    score_mock.assert_awaited_once_with(
        metrics,
        question="What is the rule?",
        answer="The answer.",
        contexts=contexts,
        reference="legacy reference",
    )


@pytest.mark.anyio
async def test_evaluate_limits_concurrent_scoring(monkeypatch):
    evaluator, _, _, _, _ = make_evaluator(monkeypatch)

    active_calls = 0
    peak_concurrency = 0

    async def fake_score_sample(metrics, **kwargs):
        nonlocal active_calls, peak_concurrency

        active_calls += 1
        peak_concurrency = max(peak_concurrency, active_calls)

        try:
            await asyncio.sleep(0.02)
            return {"answer_relevancy": 0.8}, {}
        finally:
            active_calls -= 1

    monkeypatch.setattr(
        online_evaluator,
        "score_sample",
        fake_score_sample,
    )

    results = await asyncio.gather(
        *(
            evaluator.evaluate(
                question=f"Question {index}",
                answer="An answer",
                contexts=["Legal context"],
            )
            for index in range(5)
        )
    )

    assert len(results) == 5
    assert all(scores == {"answer_relevancy": 0.8} for scores, _ in results)
    assert all(errors == {} for _, errors in results)
    assert peak_concurrency == 1


@pytest.mark.anyio
async def test_close_closes_async_openai_client(monkeypatch):
    evaluator, client, _, _, _ = make_evaluator(monkeypatch)

    await evaluator.close()

    client.close.assert_awaited_once()
