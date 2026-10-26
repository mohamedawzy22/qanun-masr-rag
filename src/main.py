from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.container import ApplicationContainer
from core.exception_handlers import (
    application_error_handler,
    unhandled_exception_handler,
)
from core.exceptions import ApplicationError
from helpers.config import get_setting
from routes import base, data, rag
from utils.logger import get_logger, setup_logging
from utils.metrics import setup_metrics

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    settings = get_setting()

    setup_logging(settings.LOG_LEVEL)

    logger.info(
        "Starting %s version=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
    )

    container = ApplicationContainer(settings)

    try:
        await container.initialize()

        app.state.container = container
        app.state.mongo_conn = container.mongo_conn
        app.state.db_client = container.db_client
        app.state.model_manager = container.model_manager
        app.state.vector_db_manager = container.vector_db_manager
        app.state.rag_controller = container.rag_controller
        app.state.langfuse = container.langfuse

        yield

    finally:
        logger.info("Application shutdown started")

        await container.close()

        logger.info("Application shutdown completed")


app = FastAPI(
    lifespan=lifespan,
)

setup_metrics(app)

app.add_exception_handler(
    ApplicationError,
    application_error_handler,
)

app.add_exception_handler(
    Exception,
    unhandled_exception_handler,
)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(rag.rag_router)
