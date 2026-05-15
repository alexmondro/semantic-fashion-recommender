"""One-sentence shopper explanations for retrieved products."""

from __future__ import annotations

from pydantic import BaseModel

from app.llm_client import LlmClient, LlmClientError
from app.schemas import ExplainedProduct, ProductIntent, ProductRecord, RankedProduct


FEATURE_LIMIT = 6
SYSTEM_PROMPT = """You write concise fashion recommendation explanations for shoppers.

Use only the supplied shopper intent and product facts.
Return exactly one explanation for each supplied product rank.
Each explanation must be exactly one sentence under 25 words.
Do not use bullets, quotes, ASINs, scores, URLs, or internal metadata.
Do not invent missing product details.
"""


class _ExplanationItem(BaseModel):
    """One structured explanation keyed to a search result rank."""

    rank: int
    why_this_matches: str


class _ExplanationBatch(BaseModel):
    """Structured explanation response for a result set."""

    explanations: list[_ExplanationItem]


class ProductExplainer:
    """Generate concise per-result explanations."""

    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client

    def explain(self, intent: ProductIntent, products: list[RankedProduct]) -> list[ExplainedProduct]:
        """Attach one shopper-facing reason to each ranked product."""

        if not products:
            return []

        batch = self.llm_client.parse_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_user_prompt(intent, products),
            response_model=_ExplanationBatch,
        )
        return _merge_explanations(products, batch)


def _user_prompt(intent: ProductIntent, products: list[RankedProduct]) -> str:
    return "\n".join(
        [
            "Shopper intent:",
            *_intent_lines(intent),
            "",
            "Products:",
            *_ranked_product_lines(products),
        ]
    )


def _ranked_product_lines(products: list[RankedProduct]) -> list[str]:
    lines: list[str] = []
    for ranked_product in products:
        if lines:
            lines.append("")
        lines.append(f"Product rank {ranked_product.rank}:")
        lines.extend(_product_lines(ranked_product.product))
    return lines


def _intent_lines(intent: ProductIntent) -> list[str]:
    lines = [f"- original_query: {intent.original_query}"]
    optional_fields = (
        ("occasion", intent.occasion),
        ("season", intent.season),
        ("garment_types", intent.garment_types),
        ("attributes", intent.attributes),
        ("audience", intent.audience),
        ("price_preference", intent.price_preference),
    )
    for name, value in optional_fields:
        if cleaned := _format_value(value):
            lines.append(f"- {name}: {cleaned}")
    return lines


def _product_lines(product: ProductRecord) -> list[str]:
    lines = [
        f"- title: {product.title}",
        f"- average_rating: {product.average_rating:.1f}",
        f"- rating_number: {product.rating_number}",
    ]
    if store := _clean_optional(product.store):
        lines.append(f"- store: {store}")
    if product.price is not None:
        lines.append(f"- price: ${product.price:.2f}")
    if features := _feature_lines(product.features):
        lines.append("- features:")
        lines.extend(features)
    return lines


def _feature_lines(features: list[str]) -> list[str]:
    cleaned = [_cleaned for feature in features if (_cleaned := feature.strip())]
    return [f"  - {feature}" for feature in cleaned[:FEATURE_LIMIT]]


def _merge_explanations(
    products: list[RankedProduct],
    batch: _ExplanationBatch,
) -> list[ExplainedProduct]:
    expected_ranks = [product.rank for product in products]
    explanations_by_rank: dict[int, str] = {}
    for explanation in batch.explanations:
        if explanation.rank in explanations_by_rank:
            raise LlmClientError(f"OpenAI explanation batch duplicated rank {explanation.rank}")
        cleaned = _clean_explanation(explanation.why_this_matches)
        if cleaned:
            explanations_by_rank[explanation.rank] = cleaned

    if sorted(explanations_by_rank) != sorted(expected_ranks):
        raise LlmClientError(
            "OpenAI explanation batch did not match product ranks: "
            f"expected {expected_ranks}, got {sorted(explanations_by_rank)}"
        )

    return [
        ExplainedProduct(
            product=ranked_product.product,
            score=ranked_product.score,
            rank=ranked_product.rank,
            why_this_matches=explanations_by_rank[ranked_product.rank],
        )
        for ranked_product in products
    ]


def _format_value(value: str | list[str] | None) -> str | None:
    if isinstance(value, list):
        cleaned = [item.strip() for item in value if item.strip()]
        return ", ".join(cleaned) if cleaned else None
    return _clean_optional(value)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_explanation(value: str) -> str:
    return " ".join(value.split())
