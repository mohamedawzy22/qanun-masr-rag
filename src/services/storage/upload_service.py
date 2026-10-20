import os

from fastapi import UploadFile

from core.exceptions import BadRequestError
from models import ResponseSignal
from models.asset_model import AssetModel
from models.db_schemes import Asset
from models.enums.asset_type_enum import AssetTypeEnum
from models.project_model import ProjectModel
from services.storage.minio_storage import MinIOStorage
from services.storage.project_storage import ProjectStorage
from utils.logger import get_logger

logger = get_logger(__name__)


class UploadService:
    def __init__(
        self,
        project_model: ProjectModel,
        asset_model: AssetModel,
        project_storage: ProjectStorage,
        minio_storage: MinIOStorage,
        chunk_size: int,
        max_file_size_mb: int,
        allowed_file_types: list[str],
    ):
        self.project_model = project_model
        self.asset_model = asset_model
        self.project_storage = project_storage
        self.minio_storage = minio_storage
        self.chunk_size = chunk_size
        self.max_file_size = max_file_size_mb * 1048576
        self.allowed_file_types = set(allowed_file_types)

        logger.info("Upload service initialized with MinIO storage")

    def _validate_file(
        self,
        file: UploadFile | None,
    ) -> int:
        if file is None or not file.filename:
            raise BadRequestError(
                ResponseSignal.FILE_NOT_FOUND.value,
            )

        if file.content_type not in self.allowed_file_types:
            raise BadRequestError(
                ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value,
            )

        # Measure the actual received stream instead of trusting the filename.
        try:
            file.file.seek(0, os.SEEK_END)
            file_size = file.file.tell()
            file.file.seek(0)
        except (AttributeError, OSError) as exc:
            logger.warning("Could not determine uploaded file size")
            raise BadRequestError(
                ResponseSignal.FILE_UPLOAD_FAILED.value,
            ) from exc

        if file_size > self.max_file_size:
            raise BadRequestError(
                ResponseSignal.FILE_SIZE_EXCEEDED.value,
            )

        return file_size

    async def upload(
        self,
        project_id: str,
        file: UploadFile,
    ) -> dict:
        file_size = self._validate_file(file)

        project = await self.project_model.get_project_or_create_one(
            project_id=project_id,
        )

        sanitized_name = self.project_storage.sanitize_file_name(
            file.filename or "",
        )

        generated_file_name = self.project_storage.generate_unique_file_name(
            sanitized_name,
        )

        # Use an object key scoped to the project and its assets.
        object_key = f"projects/{project_id}/assets/{generated_file_name}"

        try:
            await self.minio_storage.upload_file(
                object_name=object_key,
                source=file.file,
                length=file_size,
                content_type=file.content_type or "application/octet-stream",
            )
        except Exception as exc:
            logger.exception(
                "Failed to upload object to MinIO: project_id=%s | asset_name=%s",
                project_id,
                generated_file_name,
            )

            # Best-effort cleanup if the remote operation partially succeeded.
            try:
                await self.minio_storage.delete_file(
                    object_name=object_key,
                )
            except Exception:
                logger.exception(
                    "Failed to clean up object after upload failure: %s",
                    object_key,
                )

            raise BadRequestError(
                ResponseSignal.FILE_UPLOAD_FAILED.value,
            ) from exc

        asset = Asset(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_name=generated_file_name,
            asset_size=file_size,
            asset_config={
                "storage_provider": "minio",
                "bucket": self.minio_storage.bucket_name,
                "object_key": object_key,
                "content_type": file.content_type,
                "original_filename": file.filename,
            },
        )

        try:
            asset_record = await self.asset_model.create_asset(
                asset=asset,
            )
        except Exception as exc:
            logger.exception(
                "Failed to save asset metadata: project_id=%s | asset_name=%s",
                project_id,
                generated_file_name,
            )

            try:
                await self.minio_storage.delete_file(
                    object_name=object_key,
                )
            except Exception:
                logger.exception(
                    "Failed to delete MinIO object after MongoDB failure: %s",
                    object_key,
                )

            raise BadRequestError(
                ResponseSignal.FILE_UPLOAD_FAILED.value,
            ) from exc

        logger.info(
            "File uploaded successfully: project_id=%s | asset_name=%s "
            "| size=%s | storage=minio",
            project_id,
            asset_record.asset_name,
            file_size,
        )

        # Preserve the existing upload response contract.
        return {
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "asset_name": asset_record.asset_name,
        }
