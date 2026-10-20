import asyncio
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error


class MinIOStorage:
    """Async wrapper around the synchronous MinIO Python SDK."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
    ):
        if not access_key or not secret_key:
            raise ValueError("MinIO access key and secret key must be configured")

        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    async def initialize(self) -> None:
        """Ensure the application bucket exists."""
        await asyncio.to_thread(self._ensure_bucket)

    def _ensure_bucket(self) -> None:
        if self.client.bucket_exists(
            bucket_name=self.bucket_name,
        ):
            return

        try:
            self.client.make_bucket(
                bucket_name=self.bucket_name,
            )
        except S3Error as exc:
            # Another application instance may have created it concurrently.
            if exc.code not in {
                "BucketAlreadyOwnedByYou",
                "BucketAlreadyExists",
            }:
                raise

    async def upload_file(
        self,
        *,
        object_name: str,
        source: BinaryIO,
        length: int,
        content_type: str,
    ) -> None:
        """Upload a seekable stream without creating a persistent local copy."""
        source.seek(0)

        await asyncio.to_thread(
            self.client.put_object,
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=source,
            length=length,
            content_type=content_type or "application/octet-stream",
        )

    async def download_file(
        self,
        *,
        object_name: str,
        destination: Path,
        bucket_name: str | None = None,
    ) -> None:
        """Download one object to a local processing path."""
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        await asyncio.to_thread(
            self.client.fget_object,
            bucket_name=bucket_name or self.bucket_name,
            object_name=object_name,
            file_path=str(destination),
        )

    async def delete_file(
        self,
        *,
        object_name: str,
        bucket_name: str | None = None,
    ) -> None:
        """Delete an object, for example when its MongoDB insert fails."""
        await asyncio.to_thread(
            self.client.remove_object,
            bucket_name=bucket_name or self.bucket_name,
            object_name=object_name,
        )
