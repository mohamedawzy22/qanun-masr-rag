from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
)
from ragas.metrics.collections.answer_relevancy.util import (
    AnswerRelevanceInput,
    AnswerRelevanceOutput,
)

JUDGE_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "bge-m3:latest"

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
)


def build_metrics(client: AsyncOpenAI) -> dict[str, Any]:
    """Create the reference-free RAG evaluation metrics."""

    llm = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=2048,
    )

    embeddings = embedding_factory(
        provider="openai",
        model=EMBEDDING_MODEL,
        client=client,
    )

    answer_relevancy = AnswerRelevancy(
        llm=llm,
        embeddings=embeddings,
    )

    # Help the local model distinguish factual answers
    # from genuinely evasive or unknown answers.
    answer_relevancy.prompt.instruction = """
Generate a question that can be answered from the response.

Set noncommittal=1 only when the response is evasive,
vague, or says that the answer is unknown.

Set noncommittal=0 when the response contains definite
factual information, even if it does not answer every
part of the original question.
"""

    answer_relevancy.prompt.examples = [
        (
            AnswerRelevanceInput(
                response=("توفي توماس إديسون في 18 أكتوبر 1931، وكان عمره 84 عامًا.")
            ),
            AnswerRelevanceOutput(
                question="متى توفي توماس إديسون وكم كان عمره؟",
                noncommittal=0,
            ),
        ),
        (
            AnswerRelevanceInput(response="لا أعرف متى توفي توماس إديسون."),
            AnswerRelevanceOutput(
                question="متى توفي توماس إديسون؟",
                noncommittal=1,
            ),
        ),
    ]

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": answer_relevancy,
    }


async def score_sample(
    metrics: dict[str, Any],
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str = "",
) -> tuple[dict[str, float], dict[str, str]]:
    """Evaluate one RAG answer without a reference answer."""

    # Retain compatibility with existing callers.
    del reference

    inputs = {
        "faithfulness": {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        },
        "answer_relevancy": {
            "user_input": question,
            "response": answer,
        },
    }

    scores: dict[str, float] = {}
    errors: dict[str, str] = {}

    for name in METRIC_NAMES:
        metric = metrics.get(name)

        if metric is None:
            errors[name] = "Metric is not configured."
            continue

        if not question.strip() or not answer.strip():
            errors[name] = "Missing question or answer."
            continue

        if name == "faithfulness" and not contexts:
            errors[name] = "Skipped: no retrieved contexts."
            continue

        try:
            result = await metric.ascore(**inputs[name])
            scores[name] = float(result.value)
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"

    return scores, errors
