from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from bson import ObjectId
from fastapi import UploadFile
from starlette.datastructures import Headers

from core.exceptions import BadRequestError
from models import ResponseSignal
from services.storage.upload_service import UploadService


def make_upload_service(*, max_file_size_mb=10):
    project_model = AsyncMock()
    project_model.get_project_or_create_one.return_value = SimpleNamespace(
        id=ObjectId(),
    )

    asset_model = AsyncMock()

    async def return_asset(*, asset):
        return asset

    asset_model.create_asset.side_effect = return_asset

    project_storage = Mock()
    project_storage.sanitize_file_name.return_value = "civil_code.pdf"
    project_storage.generate_unique_file_name.return_value = "abc123_civil_code.pdf"

    minio_storage = AsyncMock()
    minio_storage.bucket_name = "legal-documents"

    service = UploadService(
        project_model=project_model,
        asset_model=asset_model,
        project_storage=project_storage,
        minio_storage=minio_storage,
        chunk_size=1024,
        max_file_size_mb=max_file_size_mb,
        allowed_file_types=["application/pdf"],
    )

    return (
        service,
        project_model,
        asset_model,
        project_storage,
        minio_storage,
    )


def make_file(
    content=b"%PDF-1.7 sample",
    *,
    filename="civil_code.pdf",
    content_type="application/pdf",
):
    return UploadFile(
        filename=filename,
        file=BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


@pytest.mark.anyio
async def test_upload_sends_object_to_minio_and_saves_metadata():
    (
        service,
        project_model,
        asset_model,
        project_storage,
        minio_storage,
    ) = make_upload_service()

    file = make_file()

    result = await service.upload(
        project_id="project1",
        file=file,
    )

    assert result == {
        "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
        "asset_name": "abc123_civil_code.pdf",
    }

    project_model.get_project_or_create_one.assert_awaited_once_with(
        project_id="project1",
    )
    minio_storage.upload_file.assert_awaited_once()

    upload_args = minio_storage.upload_file.await_args.kwargs
    assert upload_args["object_name"] == (
        "projects/project1/assets/abc123_civil_code.pdf"
    )
    assert upload_args["length"] == len(b"%PDF-1.7 sample")
    assert upload_args["content_type"] == "application/pdf"

    asset_model.create_asset.assert_awaited_once()
    asset = asset_model.create_asset.await_args.kwargs["asset"]
    assert asset.asset_name == "abc123_civil_code.pdf"
    assert asset.asset_config["storage_provider"] == "minio"
    assert asset.asset_config["bucket"] == "legal-documents"
    assert asset.asset_config["object_key"] == upload_args["object_name"]
    assert asset.asset_config["original_filename"] == "civil_code.pdf"

    minio_storage.delete_file.assert_not_awaited()
    project_storage.sanitize_file_name.assert_called_once_with(
        "civil_code.pdf",
    )


@pytest.mark.anyio
async def test_upload_rejects_unsupported_content_type():
    (
        service,
        project_model,
        asset_model,
        _project_storage,
        minio_storage,
    ) = make_upload_service()

    with pytest.raises(BadRequestError) as exc:
        await service.upload(
            project_id="project1",
            file=make_file(content_type="text/plain"),
        )

    assert exc.value.detail == ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
    project_model.get_project_or_create_one.assert_not_awaited()
    asset_model.create_asset.assert_not_awaited()
    minio_storage.upload_file.assert_not_awaited()


@pytest.mark.anyio
async def test_upload_rejects_file_over_configured_size_limit():
    (
        service,
        project_model,
        asset_model,
        _project_storage,
        minio_storage,
    ) = make_upload_service(max_file_size_mb=0)

    with pytest.raises(BadRequestError) as exc:
        await service.upload(
            project_id="project1",
            file=make_file(content=b"x"),
        )

    assert exc.value.detail == ResponseSignal.FILE_SIZE_EXCEEDED.value
    project_model.get_project_or_create_one.assert_not_awaited()
    asset_model.create_asset.assert_not_awaited()
    minio_storage.upload_file.assert_not_awaited()


@pytest.mark.anyio
async def test_upload_cleans_up_minio_object_after_upload_failure():
    (
        service,
        _project_model,
        asset_model,
        _project_storage,
        minio_storage,
    ) = make_upload_service()

    minio_storage.upload_file.side_effect = RuntimeError("MinIO unavailable")

    with pytest.raises(BadRequestError) as exc:
        await service.upload(
            project_id="project1",
            file=make_file(),
        )

    assert exc.value.detail == ResponseSignal.FILE_UPLOAD_FAILED.value
    minio_storage.delete_file.assert_awaited_once_with(
        object_name="projects/project1/assets/abc123_civil_code.pdf",
    )
    asset_model.create_asset.assert_not_awaited()


@pytest.mark.anyio
async def test_upload_cleans_up_minio_object_if_metadata_write_fails():
    (
        service,
        _project_model,
        asset_model,
        _project_storage,
        minio_storage,
    ) = make_upload_service()

    asset_model.create_asset.side_effect = RuntimeError("Mongo unavailable")

    with pytest.raises(BadRequestError) as exc:
        await service.upload(
            project_id="project1",
            file=make_file(),
        )

    assert exc.value.detail == ResponseSignal.FILE_UPLOAD_FAILED.value
    minio_storage.upload_file.assert_awaited_once()
    minio_storage.delete_file.assert_awaited_once_with(
        object_name="projects/project1/assets/abc123_civil_code.pdf",
    )
