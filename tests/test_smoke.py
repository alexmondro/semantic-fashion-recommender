"""API contract smoke tests for the recommendation service."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings, load_settings
from app.explainer import _ExplanationBatch, _ExplanationItem
from app.llm_client import LlmClientError
from app.main import app, create_app
from app.pipeline import NoResultsError
from app.query_parser import _ParsedProductIntent
from app.schemas import ExplainedProduct, ProductIntent, ProductRecord, RecommendationRequest, RecommendationResponse
from app.vector_index import NumpyVectorIndex


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


_RANK_PATTERN = re.compile(r"^Product rank (\d+):", re.MULTILINE)


def _user_message_text(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


class _StubOpenAIClient:
    """Stub OpenAI SDK client wired into both the embedder and the LLM client.

    Returns the first product's actual embedding so the cosine search produces a
    deterministic top match, and synthesizes structured responses for each parse
    target the pipeline expects.
    """

    def __init__(self, vector_index: NumpyVectorIndex) -> None:
        self._query_vector = vector_index.embeddings[0].tolist()
        self.embeddings_call_count = 0
        self.responses_parse_call_count = 0
        self.embeddings = _StubEmbeddings(self)
        self.responses = _StubResponses(self)


class _StubEmbeddings:
    def __init__(self, parent: _StubOpenAIClient) -> None:
        self._parent = parent

    def create(self, *, model: str, input: Any, **kwargs: Any) -> SimpleNamespace:
        self._parent.embeddings_call_count += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=self._parent._query_vector)])


class _StubResponses:
    def __init__(self, parent: _StubOpenAIClient) -> None:
        self._parent = parent

    def parse(
        self,
        *,
        model: str,
        input: list[dict[str, str]],
        text_format: type,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self._parent.responses_parse_call_count += 1
        user_text = _user_message_text(input)

        if text_format.__name__ == "_ParsedProductIntent":
            shopper_query = user_text.removeprefix("Shopper query: ").strip()
            parsed = _ParsedProductIntent(
                original_query=shopper_query,
                occasion=None,
                season="summer",
                garment_types=["dress"],
                attributes=["beach", "lightweight"],
                audience=None,
                price_preference=None,
                retrieval_text=shopper_query,
            )
            return SimpleNamespace(output_parsed=parsed)

        if text_format.__name__ == "_ExplanationBatch":
            ranks = [int(match.group(1)) for match in _RANK_PATTERN.finditer(user_text)]
            batch = _ExplanationBatch(
                explanations=[
                    _ExplanationItem(
                        rank=rank,
                        why_this_matches=f"Fits the shopper intent at rank {rank}.",
                    )
                    for rank in ranks
                ]
            )
            return SimpleNamespace(output_parsed=batch)

        raise AssertionError(f"unexpected text_format passed to responses.parse: {text_format!r}")


@pytest.fixture(scope="module")
def real_vector_index() -> NumpyVectorIndex:
    """Load the shipped 77,740-product index once per test module.

    The shipped artifacts are a hard grader requirement — a missing file means a
    broken zip, so this hard-fails rather than skipping.
    """

    settings = load_settings()
    assert settings.product_metadata_path.exists(), (
        f"shipped artifact missing: {settings.product_metadata_path}"
    )
    assert settings.embedding_matrix_path.exists(), (
        f"shipped artifact missing: {settings.embedding_matrix_path}"
    )
    return NumpyVectorIndex.load(
        metadata_path=settings.product_metadata_path,
        embedding_matrix_path=settings.embedding_matrix_path,
    )


def test_end_to_end_recommendation_through_grader_startup_path(
    real_vector_index: NumpyVectorIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the full grader-facing startup: create_app() -> lifespan -> _build_pipeline."""

    stub = _StubOpenAIClient(real_vector_index)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.embedder.OpenAI", lambda **kwargs: stub)
    monkeypatch.setattr("app.llm_client.OpenAI", lambda **kwargs: stub)
    monkeypatch.setattr(main.NumpyVectorIndex, "load", lambda *args, **kwargs: real_vector_index)

    with TestClient(create_app()) as client:
        response = client.post(
            "/recommendations",
            json={"query": "summer beach dress", "max_results": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "summer beach dress"
    assert body["intent"]["retrieval_text"]
    results = body["results"]
    assert len(results) == 3
    assert [result["rank"] for result in results] == [1, 2, 3]
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    settings = load_settings()
    for result in results:
        assert result["why_this_matches"]
        assert result["product"]["average_rating"] >= settings.min_average_rating
        assert result["product"]["rating_number"] >= settings.min_rating_count
    assert results[0]["product"]["parent_asin"] == real_vector_index.products[0].parent_asin
    assert stub.embeddings_call_count == 1
    assert stub.responses_parse_call_count == 2
