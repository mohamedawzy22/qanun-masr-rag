from typing import Annotated

from fastapi import APIRouter, Depends, Path

from controllers.rag_controller import RAGController
from core.dependencies import get_rag_controller
from routes.schemes.rag import (
    GenerateRequest,
    GenerateResponse,
    SearchRequest,
    SearchResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)


rag_router = APIRouter(
    prefix="/api/v1/rag",
    tags=["api_v1", "rag"],
)


ProjectId = Annotated[
    str,
    Path(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9]+$",
    ),
]


@rag_router.post("/index/{project_id}")
async def index(
    project_id: ProjectId,
    do_reset: bool = False,
    rag_controller: RAGController = Depends(
        get_rag_controller,
    ),
):
    logger.info(
        "RAG indexing started: project_id=%s",
        project_id,
    )

    return await rag_controller.index(
        project_id=project_id,
        do_reset=do_reset,
    )


@rag_router.post(
    "/search/{project_id}",
    response_model=SearchResponse,
)
async def search(
    project_id: ProjectId,
    search_request: SearchRequest,
    rag_controller: RAGController = Depends(
        get_rag_controller,
    ),
):
    results = await rag_controller.search(
        project_id=project_id,
        query=search_request.query,
        limit=search_request.limit,
    )

    return SearchResponse(
        query=search_request.query,
        results=results,
    )


@rag_router.post(
    "/generate/{project_id}",
    response_model=GenerateResponse,
)
async def generate(
    project_id: ProjectId,
    generate_request: GenerateRequest,
    rag_controller: RAGController = Depends(
        get_rag_controller,
    ),
):
    result = await rag_controller.generate(
        project_id=project_id,
        query=generate_request.query,
        limit=generate_request.limit,
        user_id=generate_request.user_id,
        session_id=generate_request.session_id,
    )

    return GenerateResponse(
        **result,
    )
