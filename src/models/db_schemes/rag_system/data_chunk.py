from bson.objectid import ObjectId
from pydantic import BaseModel, Field


class DataChunk(BaseModel):
    """A stored chunk enriched with Egyptian legal article metadata."""

    id: ObjectId | None = Field(None, alias="_id")

    # Egyptian Civil Code metadata.
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id: ObjectId
    chunk_asset_id: ObjectId

    article_number: int | None = Field(default=None, gt=0)

    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None

    text_ar: str = ""
    text_en: str = ""

    is_repealed: bool = False

    source_page: int | None = Field(default=None, gt=0)
    citation: str | None = Field(default=None, min_length=1)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [
                    ("chunk_project_id", 1),
                    ("_id", 1),
                ],
                "name": "idx_chunk_project_id_id",
                "unique": False,
            },
        ]


class RetrievedDocument(BaseModel):
    text: str
    score: float
