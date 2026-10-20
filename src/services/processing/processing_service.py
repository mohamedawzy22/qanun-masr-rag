import asyncio
import tempfile
from pathlib import Path

from bson import ObjectId

from models import ResponseSignal
from models.asset_model import AssetModel
from models.chunk_model import ChunkModel
from models.db_schemes import DataChunk
from models.enums.asset_type_enum import AssetTypeEnum
from models.project_model import ProjectModel
from services.processing.document_processor import DocumentProcessor
from services.storage.minio_storage import MinIOStorage
from services.storage.project_storage import ProjectStorage
from utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingService:
    """
    Process Egyptian Civil Code PDFs using DocumentProcessor directly.

    The existing upload flow and database storage are preserved.
    Each legal article becomes one chunk, except long articles that
    are split at paragraph boundaries while retaining article metadata.
    """

    def __init__(
        self,
        project_model: ProjectModel,
        asset_model: AssetModel,
        chunk_model: ChunkModel,
        project_storage: ProjectStorage,
        minio_storage: MinIOStorage,
    ):
        self.project_model = project_model
        self.asset_model = asset_model
        self.chunk_model = chunk_model
        self.project_storage = project_storage
        self.minio_storage = minio_storage

        logger.info("Processing service initialized")

    @staticmethod
    def _extract_articles(file_path: Path) -> list[dict]:
        """
        Extract bilingual legal articles directly through DocumentProcessor.

        No subprocess, external extractor script, or temporary JSON file
        is required.
        """
        processor = DocumentProcessor(
            project_path=str(file_path.parent),
        )

        return processor.extract_legal_articles(file_path)

    @staticmethod
    def _build_article_text(article: dict) -> str:
        """Build searchable text while preserving legal context and citation."""

        article_number = article["article_number"]
        citation = article.get("citation") or (
            f"Egyptian Civil Code, Article {article_number}"
        )

        parts = [
            citation,
            f"رقم المادة: {article_number} | Article: {article_number}",
        ]

        metadata_labels = (
            ("book", "الكتاب / Book"),
            ("chapter", "الفصل / Chapter"),
            ("section", "القسم / Section"),
            ("topic", "الموضوع / Topic"),
        )

        for key, label in metadata_labels:
            value = article.get(key)
            if value:
                parts.append(f"{label}: {value}")

        article_part = article.get("article_part")
        article_parts = article.get("article_parts")

        if article_part is not None and article_parts is not None:
            parts.append(f"Article part: {article_part}/{article_parts}")

        if article.get("is_repealed", False):
            parts.append("حالة المادة: ملغاة. | Article status: Repealed.")
        else:
            text_ar = (article.get("text_ar") or "").strip()
            text_en = (article.get("text_en") or "").strip()

            if text_ar:
                parts.append(f"النص العربي:\n{text_ar}")

            if text_en:
                parts.append(f"English text:\n{text_en}")

        return "\n\n".join(parts).strip()

    @staticmethod
    def _build_chunk_record(
        article: dict,
        chunk_order: int,
        project_id: ObjectId,
        asset_id: ObjectId,
        file_id: str,
    ) -> DataChunk | None:
        """Convert an extracted article or article part into DataChunk."""

        try:
            article_number = int(article["article_number"])
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Skipping record with invalid article number: file=%s",
                file_id,
            )
            return None

        is_repealed = bool(article.get("is_repealed", False))

        text_ar = (article.get("text_ar") or "").strip()
        text_en = (article.get("text_en") or "").strip()

        # Keep explicit repeal records even when the original text is absent.
        if not is_repealed and not (text_ar or text_en):
            logger.warning(
                "Skipping empty legal article: file=%s | article=%s",
                file_id,
                article_number,
            )
            return None

        source_page = article.get("source_page")
        citation = article.get("citation") or (
            f"Egyptian Civil Code, Article {article_number}"
        )

        book = article.get("book") or None
        chapter = article.get("chapter") or None
        section = article.get("section") or None
        topic = article.get("topic") or None

        metadata = {
            "source": file_id,
            "article_number": article_number,
            "book": book,
            "chapter": chapter,
            "section": section,
            "topic": topic,
            "is_repealed": is_repealed,
            "source_page": source_page,
            "citation": citation,
        }

        # Preserve the relationship between multiple chunks of a long article.
        if article.get("article_part") is not None:
            metadata["article_part"] = article["article_part"]

        if article.get("article_parts") is not None:
            metadata["article_parts"] = article["article_parts"]

        normalized_article = {
            **article,
            "article_number": article_number,
            "text_ar": text_ar,
            "text_en": text_en,
            "citation": citation,
            "is_repealed": is_repealed,
        }

        return DataChunk(
            chunk_text=ProcessingService._build_article_text(normalized_article),
            chunk_metadata=metadata,
            chunk_order=chunk_order,
            chunk_project_id=project_id,
            chunk_asset_id=asset_id,
            article_number=article_number,
            book=book,
            chapter=chapter,
            section=section,
            topic=topic,
            text_ar=text_ar,
            text_en=text_en,
            is_repealed=is_repealed,
            source_page=source_page,
            citation=citation,
        )

    async def process(
        self,
        project_id: str,
        chunk_size: int = 100,
        overlap_size: int = 20,
        do_reset: int = 0,
        file_id: str | None = None,
    ) -> dict:
        """
        Preserve the current processing API.

        chunk_size and overlap_size remain accepted for compatibility.
        Legal chunk boundaries are controlled by DocumentProcessor,
        not by fixed-size character splitting.
        """

        logger.info(
            "Legal processing started: project_id=%s | do_reset=%s | file_id=%s",
            project_id,
            do_reset,
            file_id,
        )

        project = await self.project_model.get_project_or_create_one(
            project_id=project_id,
        )

        # Preserve the existing file-selection behavior.
        if file_id:
            asset_record = await self.asset_model.get_asset_record(
                asset_project_id=project.id,
                asset_name=file_id,
            )

            if asset_record is None:
                logger.warning(
                    "Requested file not found: project_id=%s | file_id=%s",
                    project_id,
                    file_id,
                )

                return {
                    "signal": ResponseSignal.FILE_ID_ERROR.value,
                }

            project_files = {
                asset_record.id: asset_record,
            }
        else:
            assets = await self.asset_model.get_all_project_assets(
                asset_project_id=project.id,
                asset_type=AssetTypeEnum.FILE.value,
            )

            project_files = {asset.id: asset for asset in assets}

        if not project_files:
            logger.warning(
                "No files found for processing: project_id=%s",
                project_id,
            )

            return {
                "signal": ResponseSignal.NO_FILES_ERROR.value,
            }

        if do_reset == 1:
            deleted_count = await self.chunk_model.delete_chunks_by_project_id(
                project_id=project.id,
            )

            logger.info(
                "Existing chunks deleted: project_id=%s | chunks=%s",
                project_id,
                deleted_count,
            )

        inserted_chunks = 0
        processed_files = 0

        for asset_id, asset_record in project_files.items():
            current_file_id = asset_record.asset_name
            temporary_directory = None

            try:
                asset_config = asset_record.asset_config or {}

                if asset_config.get("storage_provider") == "minio":
                    object_key = asset_config.get("object_key")

                    if not object_key:
                        raise ValueError(
                            f"MinIO asset is missing its object_key: {current_file_id}"
                        )

                    temporary_directory = tempfile.TemporaryDirectory(
                        prefix="qanun-masr-"
                    )
                    file_path = Path(temporary_directory.name) / current_file_id

                    await self.minio_storage.download_file(
                        object_name=object_key,
                        destination=file_path,
                        bucket_name=asset_config.get("bucket"),
                    )
                else:
                    # Backward compatibility for assets uploaded before
                    # switching to MinIO.
                    project_path = Path(
                        self.project_storage.get_project_path(
                            project_id=project_id,
                        )
                    )
                    file_path = project_path / current_file_id

                logger.info(
                    "Processing legal file: file_id=%s",
                    current_file_id,
                )

                if file_path.suffix.lower() != ".pdf":
                    logger.warning(
                        "Skipping unsupported legal source format: file=%s",
                        current_file_id,
                    )
                    continue

                if not file_path.is_file():
                    logger.error(
                        "Uploaded file is missing from storage: file=%s",
                        current_file_id,
                    )
                    continue

                # PDF extraction can block, so run it outside the event loop.
                articles = await asyncio.to_thread(
                    self._extract_articles,
                    file_path,
                )

                if not articles:
                    logger.warning(
                        "No legal articles extracted: file=%s",
                        current_file_id,
                    )
                    continue

                chunk_records: list[DataChunk] = []
                skipped_articles = 0

                for article in articles:
                    chunk = self._build_chunk_record(
                        article=article,
                        chunk_order=len(chunk_records) + 1,
                        project_id=project.id,
                        asset_id=asset_id,
                        file_id=current_file_id,
                    )

                    if chunk is None:
                        skipped_articles += 1
                        continue

                    chunk_records.append(chunk)

                if not chunk_records:
                    logger.warning(
                        "No valid article chunks to store: file=%s",
                        current_file_id,
                    )
                    continue

                inserted_count = await self.chunk_model.insert_many_chunks(
                    chunks=chunk_records,
                )

                inserted_chunks += inserted_count
                processed_files += 1

                logger.info(
                    "Legal file processed: file=%s | article_chunks=%s | skipped=%s",
                    current_file_id,
                    inserted_count,
                    skipped_articles,
                )

            except Exception:
                logger.exception(
                    "Legal processing failed: file=%s",
                    current_file_id,
                )
                continue

            finally:
                if temporary_directory is not None:
                    temporary_directory.cleanup()

        logger.info(
            "Legal processing completed: project_id=%s | files=%s | article_chunks=%s",
            project_id,
            processed_files,
            inserted_chunks,
        )

        return {
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": inserted_chunks,
            "processed_files": processed_files,
        }
