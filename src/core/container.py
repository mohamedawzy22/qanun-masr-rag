from langfuse import Langfuse
from pymongo import AsyncMongoClient

from controllers.rag_controller import RAGController
from helpers.config import Setting
from models.asset_model import AssetModel
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from services.llm.llm_factory import ModelFactory
from services.llm.llm_manager import ModelManager
from services.processing.processing_service import ProcessingService
from services.rag.indexing_service import IndexingService
from services.rag.online_evaluator import OnlineRAGEvaluator
from services.rag.rag_service import RAGService
from services.rag.retrieval_service import RetrievalService
from services.storage.minio_storage import MinIOStorage
from services.storage.project_storage import ProjectStorage
from services.storage.upload_service import UploadService
from services.vectordb.vectordb_factory import VectorDBFactory
from services.vectordb.vectordb_manager import VectorDBManager
from utils.logger import get_logger

logger = get_logger(__name__)


class ApplicationContainer:
    def __init__(
        self,
        settings: Setting,
    ):
        self.settings = settings

        self.mongo_conn: AsyncMongoClient | None = None
        self.db_client = None

        self.project_model: ProjectModel | None = None
        self.asset_model: AssetModel | None = None
        self.chunk_model: ChunkModel | None = None

        self.model_manager: ModelManager | None = None
        self.vector_db_manager: VectorDBManager | None = None

        self.project_storage: ProjectStorage | None = None
        self.minio_storage: MinIOStorage | None = None
        self.upload_service: UploadService | None = None

        self.retrieval_service: RetrievalService | None = None
        self.indexing_service: IndexingService | None = None
        self.processing_service: ProcessingService | None = None

        self.rag_service: RAGService | None = None
        self.rag_controller: RAGController | None = None

        self.langfuse: Langfuse | None = None
        self.online_evaluator: OnlineRAGEvaluator | None = None

    async def initialize(self) -> None:
        logger.info(
            "Initializing application dependencies",
        )

        self._initialize_storage()
        if self.minio_storage is None:
            raise RuntimeError("MinIO storage is not initialized")
        await self.minio_storage.initialize()
        self._initialize_langfuse()

        await self._initialize_mongodb()
        await self._initialize_models()
        await self._initialize_vector_db()
        await self._initialize_data_models()

        self._initialize_upload()
        self._initialize_processing()
        await self._initialize_rag()

        logger.info(
            "Application dependencies initialized",
        )

    def _initialize_storage(self) -> None:
        self.project_storage = ProjectStorage()
        self.minio_storage = MinIOStorage(
            endpoint=self.settings.MINIO_ENDPOINT,
            access_key=self.settings.MINIO_ACCESS_KEY,
            secret_key=self.settings.MINIO_SECRET_KEY,
            bucket_name=self.settings.MINIO_BUCKET_NAME,
            secure=self.settings.MINIO_SECURE,
        )

        logger.info(
            "Project storage initialized",
        )

    async def _initialize_mongodb(self) -> None:
        database_config = self.settings.database

        self.mongo_conn = AsyncMongoClient(
            database_config.mongodb_url,
            timeoutMS=database_config.timeout_ms,
            connectTimeoutMS=database_config.connect_timeout_ms,
            maxPoolSize=database_config.max_pool_size,
            minPoolSize=database_config.min_pool_size,
            waitQueueTimeoutMS=database_config.wait_queue_timeout_ms,
        )

        self.db_client = self.mongo_conn[database_config.mongodb_database]

        await self.mongo_conn.admin.command(
            "ping",
        )

        logger.info(
            "MongoDB initialized: database=%s | max_pool=%s | min_pool=%s",
            database_config.mongodb_database,
            database_config.max_pool_size,
            database_config.min_pool_size,
        )

    async def _initialize_models(self) -> None:

        if self.langfuse is None:
            raise RuntimeError(
                "Langfuse is not initialized",
            )

        model_factory = ModelFactory(
            self.settings,
            langfuse=self.langfuse,
        )

        self.model_manager = ModelManager(
            model_factory,
        )

        self.model_manager.load_models()

        await self.model_manager.health_check_models()

        if self.settings.MODEL_WARMUP_ENABLED:
            await self.model_manager.warm_up_models(
                timeout_seconds=(self.settings.MODEL_WARMUP_TIMEOUT_SECONDS),
            )

        logger.info(
            "LLM models initialized",
        )

    async def _initialize_vector_db(self) -> None:
        vector_db_factory = VectorDBFactory(
            self.settings,
        )

        self.vector_db_manager = VectorDBManager(
            vector_db_factory,
        )

        self.vector_db_manager.load_databases()

        vector_db = self.vector_db_manager.get_database(
            "default",
        )

        await vector_db.create_collection()

        logger.info(
            "Vector database initialized",
        )

    async def _initialize_data_models(self) -> None:
        if self.db_client is None:
            raise RuntimeError(
                "Database client is not initialized",
            )

        self.project_model = ProjectModel(
            db_client=self.db_client,
        )

        await self.project_model.init_collection()

        self.asset_model = AssetModel(
            db_client=self.db_client,
        )

        await self.asset_model.init_collection()

        self.chunk_model = ChunkModel(
            db_client=self.db_client,
        )

        await self.chunk_model.init_collection()

        logger.info(
            "MongoDB data models initialized",
        )

    def _initialize_upload(self) -> None:
        if self.project_model is None:
            raise RuntimeError(
                "Project model is not initialized",
            )

        if self.asset_model is None:
            raise RuntimeError(
                "Asset model is not initialized",
            )

        if self.project_storage is None:
            raise RuntimeError(
                "Project storage is not initialized",
            )

        if self.minio_storage is None:
            raise RuntimeError("MinIO storage is not initialized")

        self.upload_service = UploadService(
            project_model=self.project_model,
            asset_model=self.asset_model,
            project_storage=self.project_storage,
            minio_storage=self.minio_storage,
            chunk_size=self.settings.files.default_chunk_size,
            max_file_size_mb=self.settings.files.max_size_mb,
            allowed_file_types=self.settings.files.allowed_types,
        )

        logger.info(
            "Upload service initialized",
        )

    def _initialize_processing(self) -> None:
        if self.project_model is None:
            raise RuntimeError(
                "Project model is not initialized",
            )

        if self.asset_model is None:
            raise RuntimeError(
                "Asset model is not initialized",
            )

        if self.chunk_model is None:
            raise RuntimeError(
                "Chunk model is not initialized",
            )

        if self.project_storage is None:
            raise RuntimeError(
                "Project storage is not initialized",
            )

        if self.minio_storage is None:
            raise RuntimeError("MinIO storage is not initialized")

        self.processing_service = ProcessingService(
            project_model=self.project_model,
            asset_model=self.asset_model,
            chunk_model=self.chunk_model,
            project_storage=self.project_storage,
            minio_storage=self.minio_storage,
        )

        logger.info(
            "Processing service initialized",
        )

    async def _initialize_rag(self) -> None:
        if self.model_manager is None:
            raise RuntimeError(
                "Model manager is not initialized",
            )

        if self.vector_db_manager is None:
            raise RuntimeError(
                "Vector database manager is not initialized",
            )

        if self.project_model is None:
            raise RuntimeError(
                "Project model is not initialized",
            )

        if self.chunk_model is None:
            raise RuntimeError(
                "Chunk model is not initialized",
            )

        if self.langfuse is None:
            raise RuntimeError(
                "Langfuse is not initialized",
            )

        embedding_model = self.model_manager.get_selected_embedding_model()

        chat_model = self.model_manager.get_selected_chat_model()

        vector_db = self.vector_db_manager.get_database(
            "default",
        )

        self.retrieval_service = RetrievalService(
            embedding_model=embedding_model,
            vector_db=vector_db,
            langfuse=self.langfuse,
        )

        self.indexing_service = IndexingService(
            embedding_model=embedding_model,
            vector_db=vector_db,
            project_model=self.project_model,
            chunk_model=self.chunk_model,
            langfuse=self.langfuse,
        )

        # Online evaluation is required for generated RAG answers.
        if not self.settings.OLLAMA_BASE_URL:
            raise RuntimeError("OLLAMA_BASE_URL is required for online RAG evaluation")

        self.online_evaluator = OnlineRAGEvaluator(
            ollama_base_url=self.settings.OLLAMA_BASE_URL,
            timeout_seconds=self.settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

        self.rag_service = RAGService(
            chat_model=chat_model,
            retrieval_service=self.retrieval_service,
            indexing_service=self.indexing_service,
            langfuse=self.langfuse,
            online_evaluator=self.online_evaluator,
        )

        self.rag_controller = RAGController(
            rag_service=self.rag_service,
        )

        logger.info(
            "RAG dependencies initialized",
        )

    async def close(self) -> None:
        logger.info(
            "Closing application dependencies",
        )

        if self.rag_service is not None:
            await self.rag_service.close()

        if self.vector_db_manager is not None:
            await self.vector_db_manager.close()

        if self.online_evaluator is not None:
            await self.online_evaluator.close()

        if self.model_manager is not None:
            await self.model_manager.close()

        if self.mongo_conn is not None:
            await self.mongo_conn.close()

        if self.langfuse is not None:
            self.langfuse.shutdown()

        logger.info(
            "Application dependencies closed",
        )

    def _initialize_langfuse(self) -> None:
        public_key = self.settings.LANGFUSE_PUBLIC_KEY
        secret_key = self.settings.LANGFUSE_SECRET_KEY

        if not public_key or not secret_key:
            raise RuntimeError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required "
                "for online evaluation and score tracking"
            )

        self.langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=self.settings.LANGFUSE_BASE_URL,
            environment=self.settings.LANGFUSE_TRACING_ENVIRONMENT,
            release=self.settings.APP_VERSION,
            tracing_enabled=True,
        )

        logger.info(
            "Langfuse initialized: tracing=enabled | environment=%s",
            self.settings.LANGFUSE_TRACING_ENVIRONMENT,
        )
