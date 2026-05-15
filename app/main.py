"""FastAPI entrypoint with thin HTTP routing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import load_settings
from app.embedder import OpenAiEmbedder
from app.explainer import ProductExplainer
from app.llm_client import LlmClientError, OpenAiLlmClient
from app.pipeline import NoResultsError, RecommendationPipeline
from app.query_parser import QueryParser
from app.schemas import ApiError, RecommendationRequest, RecommendationResponse
from app.vector_index import NumpyVectorIndex

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ApiError, "description": "Invalid shopper request."},
    404: {"model": ApiError, "description": "No matching products found."},
    502: {"model": ApiError, "description": "Upstream OpenAI request failed."},
    503: {"model": ApiError, "description": "Service is missing required configuration."},
}

ApiErrorCode = Literal["missing_api_key", "bad_request", "upstream_error", "no_results"]


def create_app(pipeline: RecommendationPipeline | None = None) -> FastAPI:
    """Create the FastAPI app with optional dependency injection for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline if pipeline is not None else _build_pipeline()
        yield

    app = FastAPI(title="Semantic Fashion Recommender", lifespan=lifespan)

    @app.post(
        "/recommendations",
        response_model=RecommendationResponse,
        responses=ERROR_RESPONSES,
        summary="Recommend fashion products",
    )
    def recommend(
        request: Request,
        body: RecommendationRequest,
    ) -> RecommendationResponse | JSONResponse:
        """Recommend fashion products for a natural-language shopper query."""

        active_pipeline = getattr(request.app.state, "pipeline", None)
        if active_pipeline is None:
            return _api_error(
                status_code=503,
                error="missing_api_key",
                message="Set OPENAI_API_KEY in .env before requesting recommendations.",
            )
        try:
            return active_pipeline.recommend(body)
        except NoResultsError:
            return _api_error(
                status_code=404,
                error="no_results",
                message="No matching products passed the recommendation filters.",
            )
        except (LlmClientError, openai.APIError):
            return _api_error(
                status_code=502,
                error="upstream_error",
                message="OpenAI could not complete the recommendation request.",
            )

    return app


def _build_pipeline() -> RecommendationPipeline | None:
    settings = load_settings()
    if settings.openai_api_key is None:
        return None

    llm_client = OpenAiLlmClient(
        api_key=settings.openai_api_key,
        model=settings.chat_model,
        fallback_model=settings.fallback_chat_model,
    )
    return RecommendationPipeline(
        settings=settings,
        query_parser=QueryParser(llm_client),
        embedder=OpenAiEmbedder(api_key=settings.openai_api_key, model=settings.embedding_model),
        vector_index=NumpyVectorIndex.load(settings.product_metadata_path, settings.embedding_matrix_path),
        explainer=ProductExplainer(llm_client),
    )


def _api_error(*, status_code: int, error: ApiErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ApiError(error=error, message=message).model_dump(),
    )


app = create_app()
