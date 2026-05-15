"""API contract smoke tests for the recommendation service."""

from __future__ import annotations

from typing import Any

import httpx
import openai
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from app.llm_client import LlmClientError
from app.main import app, create_app
from app.pipeline import NoResultsError
from app.schemas import ExplainedProduct, ProductIntent, ProductRecord, RecommendationRequest, RecommendationResponse


def product() -> ProductRecord:
    """Build one response-safe product for API fakes."""

    return ProductRecord(
        parent_asin="B000TEST",
        title="Comfort Walk Sandal",
        store="Comfort Walk",
        average_rating=4.7,
        rating_number=42,
        price=29.99,
        image_url=None,
        features=["Cushioned footbed"],
        normalized_title="comfort walk sandal",
        retrieval_text="Comfort Walk Sandal\n- Cushioned footbed",
    )


def intent() -> ProductIntent:
    """Build one parsed intent for API fakes."""

    return ProductIntent(
        original_query="comfortable summer sandals",
        occasion=None,
        season="summer",
        garment_types=["sandals"],
        attributes=["comfortable"],
        audience=None,
        price_preference=None,
        retrieval_text="comfortable summer sandals",
    )


def recommendation_response() -> RecommendationResponse:
    """Build a minimal successful recommendation response."""

    return RecommendationResponse(
        query="comfortable summer sandals",
        intent=intent(),
        results=[
            ExplainedProduct(
                product=product(),
                score=0.91,
                rank=1,
                why_this_matches="The cushioned footbed fits a comfortable summer sandal request.",
            )
        ],
    )


class FakePipeline:
    """Small pipeline double for route behavior tests."""

    def __init__(self, outcome: RecommendationResponse | Exception) -> None:
        self.outcome = outcome
        self.requests: list[RecommendationRequest] = []

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def api_connection_error() -> openai.APIConnectionError:
    """Create an SDK APIError subclass without making a network request."""

    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"))


def test_openapi_exposes_recommendations_contract() -> None:
    """The primary endpoint is visible and describes the locked response shape."""

    schema = app.openapi()
    operation = schema["paths"]["/recommendations"]["post"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/RecommendationResponse"
    )
    assert operation["responses"]["400"]["content"]["application/json"]["schema"]["$ref"].endswith("/ApiError")
    assert operation["responses"]["404"]["content"]["application/json"]["schema"]["$ref"].endswith("/ApiError")
    assert operation["responses"]["502"]["content"]["application/json"]["schema"]["$ref"].endswith("/ApiError")
    assert operation["responses"]["503"]["content"]["application/json"]["schema"]["$ref"].endswith("/ApiError")

    response_schema = schema["components"]["schemas"]["RecommendationResponse"]
    assert {"query", "intent", "results"}.issubset(response_schema["properties"])

    explained_product_schema = schema["components"]["schemas"]["ExplainedProduct"]
    assert "why_this_matches" in explained_product_schema["properties"]


def test_recommendation_request_trims_query_and_accepts_default_max_results() -> None:
    """Request validation keeps Swagger and route behavior aligned."""

    request = RecommendationRequest(query="  linen wedding guest dress  ")

    assert request.query == "linen wedding guest dress"
    assert request.max_results == 8


def test_invalid_max_results_is_rejected_before_route_execution() -> None:
    """Out-of-range max_results should fail Pydantic validation."""

    client = TestClient(app)
    response = client.post("/recommendations", json={"query": "summer sandals", "max_results": 21})

    assert response.status_code == 422


def test_openapi_loads_without_api_key_or_artifacts(monkeypatch) -> None:
    """Swagger should work even before the grader adds OPENAI_API_KEY."""

    def fail_if_artifacts_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("artifact loading should be skipped without an API key")

    monkeypatch.setattr(main, "load_settings", lambda: Settings(openai_api_key=None))
    monkeypatch.setattr(main.NumpyVectorIndex, "load", fail_if_artifacts_load)

    with TestClient(create_app()) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/recommendations" in response.json()["paths"]


def test_recommendations_returns_fake_pipeline_response() -> None:
    """A configured pipeline response should pass through the route unchanged."""

    pipeline = FakePipeline(recommendation_response())

    with TestClient(create_app(pipeline=pipeline)) as client:
        response = client.post(
            "/recommendations",
            json={"query": "comfortable summer sandals", "max_results": 1},
        )

    assert response.status_code == 200
    assert response.json()["query"] == "comfortable summer sandals"
    assert response.json()["results"][0]["why_this_matches"].startswith("The cushioned")
    assert pipeline.requests == [RecommendationRequest(query="comfortable summer sandals", max_results=1)]


def test_recommendations_returns_503_when_pipeline_is_missing(monkeypatch) -> None:
    """No API key should produce the typed service-unavailable response."""

    monkeypatch.setattr(main, "load_settings", lambda: Settings(openai_api_key=None))

    with TestClient(create_app()) as client:
        response = client.post("/recommendations", json={"query": "summer sandals"})

    assert response.status_code == 503
    assert response.json() == {
        "error": "missing_api_key",
        "message": "Set OPENAI_API_KEY in .env before requesting recommendations.",
    }


def test_no_results_error_maps_to_typed_404() -> None:
    """The pipeline no-results boundary should stay visible to shoppers."""

    with TestClient(create_app(pipeline=FakePipeline(NoResultsError("none")))) as client:
        response = client.post("/recommendations", json={"query": "summer sandals"})

    assert response.status_code == 404
    assert response.json()["error"] == "no_results"


def test_llm_client_error_maps_to_typed_502() -> None:
    """LLM parsing/explanation failures should surface as upstream failures."""

    with TestClient(create_app(pipeline=FakePipeline(LlmClientError("bad llm")))) as client:
        response = client.post("/recommendations", json={"query": "summer sandals"})

    assert response.status_code == 502
    assert response.json()["error"] == "upstream_error"


def test_openai_api_error_maps_to_typed_502() -> None:
    """Embedding or SDK API failures should surface as upstream failures."""

    with TestClient(create_app(pipeline=FakePipeline(api_connection_error()))) as client:
        response = client.post("/recommendations", json={"query": "summer sandals"})

    assert response.status_code == 502
    assert response.json()["error"] == "upstream_error"
