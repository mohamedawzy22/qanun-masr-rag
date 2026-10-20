from types import SimpleNamespace

import pytest


class FakeObservation:
    """Lightweight Langfuse observation for unit tests."""

    def __init__(self, **kwargs):
        self.arguments = kwargs
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuse:
    """Avoid connecting to a real Langfuse instance during tests."""

    def __init__(self):
        self.observations = []

    def start_as_current_observation(self, **kwargs):
        observation = FakeObservation(**kwargs)
        self.observations.append(observation)
        return observation

    def get_current_trace_id(self):
        return None

    def create_score(self, **kwargs):
        return None


@pytest.fixture
def anyio_backend():
    # Keep async unit tests on asyncio, without requiring Trio.
    return "asyncio"


@pytest.fixture
def langfuse():
    return FakeLangfuse()


@pytest.fixture
def sample_documents():
    return [
        {
            "text": "A contract is created when two parties agree.",
            "score": 0.91,
            "article_number": 147,
            "book": "General Provisions",
            "chapter": "Sources of Obligations",
            "section": None,
            "topic": "Contract formation",
            "text_ar": "ينشأ العقد بتوافق إرادتين.",
            "text_en": "A contract is created when two parties agree.",
            "is_repealed": False,
            "source_page": 10,
            "citation": "Egyptian Civil Code, Article 147",
            "source_file": "civil_code.pdf",
        },
        {
            "text": "A contract must be performed in good faith.",
            "score": 0.84,
            "article_number": 148,
            "book": "General Provisions",
            "chapter": "Effects of Obligations",
            "section": None,
            "topic": "Good faith",
            "text_ar": "يجب تنفيذ العقد طبقًا لحسن النية.",
            "text_en": "A contract must be performed in good faith.",
            "is_repealed": False,
            "source_page": 11,
            "citation": "Egyptian Civil Code, Article 148",
            "source_file": "civil_code.pdf",
        },
    ]


@pytest.fixture
def sample_chunks():
    return [
        SimpleNamespace(
            id="chunk-1",
            chunk_text="Legal article 147",
            chunk_asset_id="asset-1",
            article_number=147,
            book="General Provisions",
            chapter="Sources of Obligations",
            section=None,
            topic="Contract formation",
            text_ar="النص العربي",
            text_en="English text",
            is_repealed=False,
            source_page=10,
            citation="Egyptian Civil Code, Article 147",
            chunk_metadata={"source": "civil_code.pdf"},
        ),
        SimpleNamespace(
            id="chunk-2",
            chunk_text="Legal article 148",
            chunk_asset_id="asset-1",
            article_number=148,
            book="General Provisions",
            chapter="Effects of Obligations",
            section=None,
            topic="Good faith",
            text_ar="نص المادة",
            text_en="Article text",
            is_repealed=False,
            source_page=11,
            citation="Egyptian Civil Code, Article 148",
            chunk_metadata={"source": "civil_code.pdf"},
        ),
    ]
