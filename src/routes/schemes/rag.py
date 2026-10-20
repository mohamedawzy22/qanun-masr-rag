from pydantic import BaseModel, Field

QUERY_MAX_LENGTH = 4000
LIMIT_MAX_VALUE = 100


# --------------------------------------------------
# Shared request schema
# --------------------------------------------------
class RAGQueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=QUERY_MAX_LENGTH,
    )

    limit: int = Field(
        default=5,
        gt=0,
        le=LIMIT_MAX_VALUE,
    )


class SearchRequest(RAGQueryRequest):
    pass


class GenerateRequest(RAGQueryRequest):
    user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )


# --------------------------------------------------
# Shared retrieved legal article fields
# --------------------------------------------------
class LegalArticleResult(BaseModel):
    score: float

    article_number: int | None = None

    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None

    text_ar: str | None = None
    text_en: str | None = None

    is_repealed: bool | None = None

    source_page: int | None = None
    citation: str | None = None
    source_file: str | None = None


# --------------------------------------------------
# Search response
# --------------------------------------------------
class SearchResult(LegalArticleResult):
    """Search result without duplicated combined text."""


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# --------------------------------------------------
# Generate response
# --------------------------------------------------
class GenerateSource(LegalArticleResult):
    """Source supporting the generated answer."""


class GenerateResponse(BaseModel):
    answer: str
    sources: list[GenerateSource] = Field(
        default_factory=list,
    )
