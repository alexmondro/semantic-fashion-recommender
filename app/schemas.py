"""Typed request, response, and pipeline contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints


ShopperQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class RecommendationRequest(BaseModel):
    """User-facing recommendation request."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "I need comfortable beach vacation sandals for summer",
            "max_results": 8,
        }
    })

    query: ShopperQuery = Field(description="Natural-language fashion shopping request.")
    max_results: int = Field(default=8, ge=1, le=20, description="Number of recommendations to return.")


class ProductIntent(BaseModel):
    """Structured shopping intent extracted from a natural-language query."""

    original_query: str
    occasion: str | None = None
    season: str | None = None
    garment_types: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    audience: str | None = None
    price_preference: str | None = None
    retrieval_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class ProductRecord(BaseModel):
    """Product metadata shipped with the precomputed embedding matrix."""

    parent_asin: str
    title: str
    store: str | None = None
    average_rating: float
    rating_number: int
    price: float | None = None
    image_url: HttpUrl | None = None
    features: list[str] = Field(default_factory=list)
    normalized_title: str | None = Field(default=None, exclude=True)
    retrieval_text: str = Field(exclude=True)


class RankedProduct(BaseModel):
    """Search result before the explanation pass."""

    product: ProductRecord
    score: float
    rank: int


class ExplainedProduct(BaseModel):
    """Search result enriched for a shopper."""

    product: ProductRecord
    score: float
    rank: int
    why_this_matches: str


class RecommendationResponse(BaseModel):
    """User-facing recommendation response."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "comfortable beach vacation sandals for summer",
            "intent": {
                "original_query": "comfortable beach vacation sandals for summer",
                "occasion": "beach vacation",
                "season": "summer",
                "garment_types": ["sandals"],
                "attributes": ["comfortable"],
                "audience": None,
                "price_preference": None,
                "retrieval_text": "comfortable summer beach sandals",
            },
            "results": [],
        }
    })

    query: str
    intent: ProductIntent
    results: list[ExplainedProduct]


class ApiError(BaseModel):
    """Stable error response body."""

    error: Literal["missing_api_key", "bad_request", "upstream_error", "no_results"]
    message: str
