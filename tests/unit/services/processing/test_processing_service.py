from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from bson import ObjectId

from models import ResponseSignal
from services.processing.processing_service import ProcessingService


def make_processing_service():
    project = SimpleNamespace(id=ObjectId())

    project_model = AsyncMock()
    project_model.get_project_or_create_one.return_value = project

    asset_model = AsyncMock()
    chunk_model = AsyncMock()
    project_storage = Mock()
    minio_storage = AsyncMock()

    service = ProcessingService(
        project_model=project_model,
        asset_model=asset_model,
        chunk_model=chunk_model,
        project_storage=project_storage,
        minio_storage=minio_storage,
    )

    return (
        service,
        project,
        project_model,
        asset_model,
        chunk_model,
        project_storage,
        minio_storage,
    )


def make_asset(*, asset_name="civil_code.pdf", object_key=True):
    config = {"storage_provider": "minio", "bucket": "legal-documents"}

    if object_key:
        config["object_key"] = f"projects/project1/assets/{asset_name}"

    return SimpleNamespace(
        id=ObjectId(),
        asset_name=asset_name,
        asset_config=config,
    )


async def fake_download(*, object_name, destination, bucket_name=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"%PDF test fixture")


@pytest.mark.anyio
async def test_process_returns_error_when_requested_file_does_not_exist():
    (
        service,
        project,
        _project_model,
        asset_model,
        chunk_model,
        _project_storage,
        minio_storage,
    ) = make_processing_service()

    asset_model.get_asset_record.return_value = None

    result = await service.process(
        project_id="project1",
        file_id="missing.pdf",
    )

    assert result == {"signal": ResponseSignal.FILE_ID_ERROR.value}
    asset_model.get_asset_record.assert_awaited_once_with(
        asset_project_id=project.id,
        asset_name="missing.pdf",
    )
    chunk_model.insert_many_chunks.assert_not_awaited()
    minio_storage.download_file.assert_not_awaited()


@pytest.mark.anyio
async def test_process_returns_error_when_project_has_no_files():
    (
        service,
        _project,
        _project_model,
        asset_model,
        chunk_model,
        _project_storage,
        minio_storage,
    ) = make_processing_service()

    asset_model.get_all_project_assets.return_value = []

    result = await service.process(project_id="project1")

    assert result == {"signal": ResponseSignal.NO_FILES_ERROR.value}
    chunk_model.insert_many_chunks.assert_not_awaited()
    minio_storage.download_file.assert_not_awaited()


@pytest.mark.anyio
async def test_process_converts_articles_and_preserves_repeal_records(
    monkeypatch,
):
    (
        service,
        project,
        _project_model,
        asset_model,
        chunk_model,
        _project_storage,
        minio_storage,
    ) = make_processing_service()

    asset = make_asset()
    asset_model.get_all_project_assets.return_value = [asset]
    minio_storage.download_file.side_effect = fake_download
    chunk_model.delete_chunks_by_project_id.return_value = 4
    chunk_model.insert_many_chunks.return_value = 2

    articles = [
        {
            "article_number": 147,
            "book": "General Provisions",
            "chapter": "Sources of Obligations",
            "section": None,
            "topic": "Contract formation",
            "text_ar": "ينشأ العقد بتوافق إرادتين.",
            "text_en": "A contract is created by agreement.",
            "is_repealed": False,
            "source_page": 10,
            "citation": "Egyptian Civil Code, Article 147",
        },
        {
            "article_number": 54,
            "book": "General Provisions",
            "chapter": "General Provisions",
            "text_ar": "",
            "text_en": "",
            "is_repealed": True,
            "source_page": None,
            "citation": "Egyptian Civil Code, Article 54",
        },
        {
            "article_number": "invalid",
            "text_ar": "نص",
            "text_en": "Text",
        },
        {
            "article_number": 149,
            "text_ar": "",
            "text_en": "",
            "is_repealed": False,
        },
    ]

    monkeypatch.setattr(
        ProcessingService,
        "_extract_articles",
        staticmethod(lambda _file_path: articles),
    )

    result = await service.process(
        project_id="project1",
        do_reset=1,
    )

    assert result == {
        "signal": ResponseSignal.PROCESSING_SUCCESS.value,
        "inserted_chunks": 2,
        "processed_files": 1,
    }

    chunk_model.delete_chunks_by_project_id.assert_awaited_once_with(
        project_id=project.id,
    )
    minio_storage.download_file.assert_awaited_once_with(
        object_name=asset.asset_config["object_key"],
        destination=minio_storage.download_file.await_args.kwargs["destination"],
        bucket_name="legal-documents",
    )
    chunk_model.insert_many_chunks.assert_awaited_once()

    chunks = chunk_model.insert_many_chunks.await_args.kwargs["chunks"]
    assert len(chunks) == 2
    assert [chunk.article_number for chunk in chunks] == [147, 54]
    assert chunks[0].chunk_order == 1
    assert chunks[0].text_en == "A contract is created by agreement."
    assert chunks[0].citation == "Egyptian Civil Code, Article 147"
    assert chunks[0].chunk_metadata["source"] == "civil_code.pdf"

    # Repealed articles remain stored as explicit metadata-only records.
    repealed = chunks[1]
    assert repealed.is_repealed is True
    assert repealed.text_en == ""
    assert repealed.text_ar == ""
    assert "Repealed" in repealed.chunk_text


@pytest.mark.anyio
async def test_process_skips_non_pdf_assets(monkeypatch):
    (
        service,
        _project,
        _project_model,
        asset_model,
        chunk_model,
        _project_storage,
        minio_storage,
    ) = make_processing_service()

    asset = make_asset(asset_name="notes.txt")
    asset_model.get_all_project_assets.return_value = [asset]
    minio_storage.download_file.side_effect = fake_download

    def extraction_must_not_run(_file_path):
        raise AssertionError("The PDF extractor must not run for TXT files")

    monkeypatch.setattr(
        ProcessingService,
        "_extract_articles",
        staticmethod(extraction_must_not_run),
    )

    result = await service.process(project_id="project1")

    assert result["inserted_chunks"] == 0
    assert result["processed_files"] == 0
    chunk_model.insert_many_chunks.assert_not_awaited()


def test_build_article_text_includes_citation_and_bilingual_content():
    text = ProcessingService._build_article_text(
        {
            "article_number": 147,
            "citation": "Egyptian Civil Code, Article 147",
            "book": "General Provisions",
            "text_ar": "النص العربي",
            "text_en": "English statutory text",
            "is_repealed": False,
        }
    )

    assert "Egyptian Civil Code, Article 147" in text
    assert "النص العربي" in text
    assert "English statutory text" in text
    assert "Article: 147" in text


def test_build_chunk_record_rejects_invalid_article_number():
    record = ProcessingService._build_chunk_record(
        article={"article_number": "not-a-number"},
        chunk_order=1,
        project_id=ObjectId(),
        asset_id=ObjectId(),
        file_id="civil_code.pdf",
    )

    assert record is None


def test_build_chunk_record_keeps_repealed_article_without_body():
    record = ProcessingService._build_chunk_record(
        article={
            "article_number": 54,
            "text_ar": "",
            "text_en": "",
            "is_repealed": True,
        },
        chunk_order=1,
        project_id=ObjectId(),
        asset_id=ObjectId(),
        file_id="civil_code.pdf",
    )

    assert record is not None
    assert record.article_number == 54
    assert record.is_repealed is True
    assert "Repealed" in record.chunk_text
