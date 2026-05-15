"""One-sentence shopper explanations for retrieved products."""

from __future__ import annotations

from app.llm_client import LlmClient
from app.schemas import ExplainedProduct, ProductIntent, ProductRecord, RankedProduct


FEATURE_LIMIT = 6
SYSTEM_PROMPT = """You write concise fashion recommendation explanations for shoppers.

Use only the supplied shopper intent and product facts.
Write exactly one sentence under 25 words.
Do not use bullets, quotes, ASINs, scores, URLs, or internal metadata.
Do not invent missing product details.
"""


class ProductExplainer:
    """Generate concise per-result explanations."""

    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client

    def explain(self, intent: ProductIntent, products: list[RankedProduct]) -> list[ExplainedProduct]:
        """Attach one shopper-facing reason to each ranked product."""

        return [_explain_one(self.llm_client, intent, product) for product in products]


def _explain_one(
    llm_client: LlmClient,
    intent: ProductIntent,
    ranked_product: RankedProduct,
) -> ExplainedProduct:
    explanation = llm_client.complete_text(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(intent, ranked_product.product),
    )
    return ExplainedProduct(
        product=ranked_product.product,
        score=ranked_product.score,
        rank=ranked_product.rank,
        why_this_matches=_clean_explanation(explanation),
    )


def _user_prompt(intent: ProductIntent, product: ProductRecord) -> str:
    return "\n".join(
        [
            "Shopper intent:",
            *_intent_lines(intent),
            "",
            "Product facts:",
            *_product_lines(product),
        ]
    )


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
