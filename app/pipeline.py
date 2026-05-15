"""Recommendation orchestration for parse -> embed -> search -> explain."""

from __future__ import annotations

from app.config import Settings
from app.embedder import Embedder
from app.explainer import ProductExplainer
from app.query_parser import QueryParser
from app.schemas import RankedProduct, RecommendationRequest, RecommendationResponse
from app.vector_index import VectorIndex


class NoResultsError(RuntimeError):
    """Raised when retrieval cannot produce any shopper-safe recommendations."""


class RecommendationPipeline:
    """Coordinates typed pipeline stages without owning their implementations."""

    def __init__(
        self,
        *,
        settings: Settings,
        query_parser: QueryParser,
        embedder: Embedder,
        vector_index: VectorIndex,
        explainer: ProductExplainer,
    ) -> None:
        self.settings = settings
        self.query_parser = query_parser
        self.embedder = embedder
        self.vector_index = vector_index
        self.explainer = explainer

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Return explained recommendations for a shopper query."""

        intent = self.query_parser.parse(request.query)
        query_embedding = self.embedder.embed_text(intent.retrieval_text)
        ranked_products = self.vector_index.search(
            query_embedding,
            top_k=request.max_results,
            candidate_pool_size=self.settings.candidate_pool_size,
        )
        filtered_products = self._quality_filter(ranked_products)
        if not filtered_products:
            raise NoResultsError("No matching products passed the quality filter.")
        explained_products = self.explainer.explain(intent, filtered_products)
        return RecommendationResponse(
            query=request.query,
            intent=intent,
            results=explained_products,
        )

    def _quality_filter(self, products: list[RankedProduct]) -> list[RankedProduct]:
        # Mirrors the shipped offline corpus policy and guards artifact/config drift.
        return [
            product
            for product in products
            if product.product.rating_number >= self.settings.min_rating_count
            and product.product.average_rating >= self.settings.min_average_rating
        ]
