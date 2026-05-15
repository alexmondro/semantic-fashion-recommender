"""Focused coverage for recommendation pipeline orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.config import Settings
from app.pipeline import NoResultsError, RecommendationPipeline
from app.schemas import ExplainedProduct, ProductIntent, ProductRecord, RankedProduct, RecommendationRequest


class FakeParser:
    """Parser double with a stable intent."""

    def __init__(self, parsed_intent: ProductIntent) -> None:
        self.parsed_intent = parsed_intent
        self.queries: list[str] = []

    def parse(self, query: str) -> ProductIntent:
        self.queries.append(query)
        return self.parsed_intent


class FakeEmbedder:
    """Embedder double that records retrieval text."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_text(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeIndex:
    """Vector index double with configurable ranked products."""

    def __init__(self, products: list[RankedProduct]) -> None:
        self.products = products
        self.calls: list[dict[str, Any]] = []

    def search(self, query_embedding: np.ndarray, *, top_k: int, candidate_pool_size: int) -> list[RankedProduct]:
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "candidate_pool_size": candidate_pool_size,
            }
        )
        return self.products


class FakeExplainer:
    """Explainer double that enriches ranked products without LLM calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[ProductIntent, list[RankedProduct]]] = []

    def explain(self, intent: ProductIntent, products: list[RankedProduct]) -> list[ExplainedProduct]:
        self.calls.append((intent, products))
        return [
            ExplainedProduct(
                product=ranked_product.product,
                score=ranked_product.score,
                rank=ranked_product.rank,
                why_this_matches="Matches the shopper request.",
            )
            for ranked_product in products
        ]


def intent() -> ProductIntent:
    """Build one parsed intent."""

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


def product(**updates: Any) -> ProductRecord:
    """Build one product for pipeline tests."""

    payload = {
        "parent_asin": "B000TEST",
        "title": "Comfort Walk Sandal",
        "store": "Comfort Walk",
        "average_rating": 4.6,
        "rating_number": 24,
        "price": None,
        "image_url": None,
        "features": ["Cushioned footbed"],
        "normalized_title": "comfort walk sandal",
        "retrieval_text": "Comfort Walk Sandal\n- Cushioned footbed",
    }
    payload.update(updates)
    return ProductRecord.model_validate(payload)


def ranked_product(product_record: ProductRecord, *, score: float = 0.91, rank: int = 1) -> RankedProduct:
    """Wrap a product in ranked search metadata."""

    return RankedProduct(product=product_record, score=score, rank=rank)


def pipeline_with(
    ranked_products: list[RankedProduct],
    *,
    settings: Settings | None = None,
) -> tuple[RecommendationPipeline, FakeParser, FakeEmbedder, FakeIndex, FakeExplainer]:
    """Build a recommendation pipeline from fakes."""

    parser = FakeParser(intent())
    embedder = FakeEmbedder()
    index = FakeIndex(ranked_products)
    explainer = FakeExplainer()
    pipeline = RecommendationPipeline(
        settings=settings or Settings(openai_api_key="test-key", candidate_pool_size=12),
        query_parser=parser,
        embedder=embedder,
        vector_index=index,
        explainer=explainer,
    )
    return pipeline, parser, embedder, index, explainer


def test_pipeline_uses_request_max_results_and_explains_filtered_products() -> None:
    """The orchestration should preserve the explicit pipeline stage contracts."""

    ranked = [ranked_product(product())]
    pipeline, parser, embedder, index, explainer = pipeline_with(ranked)

    response = pipeline.recommend(RecommendationRequest(query="comfortable summer sandals", max_results=3))

    assert parser.queries == ["comfortable summer sandals"]
    assert embedder.texts == ["comfortable summer sandals"]
    assert index.calls[0]["top_k"] == 3
    assert index.calls[0]["candidate_pool_size"] == 12
    assert explainer.calls == [(intent(), ranked)]
    assert response.query == "comfortable summer sandals"
    assert response.results[0].why_this_matches == "Matches the shopper request."


def test_pipeline_raises_no_results_when_quality_filter_drops_all_products() -> None:
    """Synthetic low-quality rows make the otherwise defensive 404 path reachable."""

    ranked = [
        ranked_product(product(parent_asin="B000LOWRATING", average_rating=3.9), rank=1),
        ranked_product(product(parent_asin="B000LOWCOUNT", rating_number=9), rank=2),
    ]
    pipeline, _parser, _embedder, _index, explainer = pipeline_with(ranked)

    with pytest.raises(NoResultsError):
        pipeline.recommend(RecommendationRequest(query="comfortable summer sandals"))

    assert explainer.calls == []
