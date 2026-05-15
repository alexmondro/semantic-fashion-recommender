"""Focused coverage for shopper-facing product explanations."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from app.explainer import ProductExplainer
from app.llm_client import LlmClientError
from app.schemas import ExplainedProduct, ProductIntent, ProductRecord, RankedProduct


class FakeLlmClient:
    """Capture explanation prompts while returning canned structured responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def parse_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        response = self.responses.pop(0)
        if isinstance(response, BaseModel):
            return response
        return response_model.model_validate(response)

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("batch explanation should not request free-text completion")


def intent(**updates: Any) -> ProductIntent:
    payload = {
        "original_query": "comfortable black sandals for a beach vacation",
        "occasion": None,
        "season": None,
        "garment_types": [],
        "attributes": [],
        "audience": None,
        "price_preference": None,
        "retrieval_text": "internal semantic retrieval text",
    }
    payload.update(updates)
    return ProductIntent.model_validate(payload)


def product(**updates: Any) -> ProductRecord:
    payload = {
        "parent_asin": "B000TEST",
        "title": "Comfort Walk Black Sandals",
        "store": "Comfort Walk",
        "average_rating": 4.6,
        "rating_number": 42,
        "price": 29.99,
        "image_url": None,
        "features": ["Cushioned footbed", "Water-friendly straps"],
        "normalized_title": "comfort walk black sandals",
        "retrieval_text": "Comfort Walk Black Sandals\n- Cushioned footbed",
    }
    payload.update(updates)
    return ProductRecord.model_validate(payload)


def ranked(product_record: ProductRecord, *, score: float = 0.91, rank: int = 1) -> RankedProduct:
    return RankedProduct(product=product_record, score=score, rank=rank)


def test_explain_empty_products_skips_llm_calls() -> None:
    """No results means no explanation work."""

    fake_client = FakeLlmClient([])
    explainer = ProductExplainer(fake_client)

    explained = explainer.explain(intent(), [])

    assert explained == []
    assert fake_client.calls == []


def test_explain_preserves_ranked_products_and_normalizes_explanation_text() -> None:
    """The explanation pass should enrich results without reordering or rescoring."""

    first_product = product(parent_asin="B000FIRST", title="First Sandal")
    second_product = product(parent_asin="B000SECOND", title="Second Sandal")
    fake_client = FakeLlmClient(
        [
            {
                "explanations": [
                    {"rank": 1, "why_this_matches": "  Matches the beach need.\n"},
                    {"rank": 2, "why_this_matches": "Works for casual summer walking."},
                ]
            }
        ]
    )
    explainer = ProductExplainer(fake_client)

    explained = explainer.explain(
        intent(),
        [
            ranked(first_product, score=0.92, rank=1),
            ranked(second_product, score=0.87, rank=2),
        ],
    )

    assert [item.product for item in explained] == [first_product, second_product]
    assert [item.score for item in explained] == [0.92, 0.87]
    assert [item.rank for item in explained] == [1, 2]
    assert [item.why_this_matches for item in explained] == [
        "Matches the beach need.",
        "Works for casual summer walking.",
    ]
    assert all(isinstance(item, ExplainedProduct) for item in explained)
    assert len(fake_client.calls) == 1


def test_prompt_includes_public_intent_fields_and_omits_retrieval_text() -> None:
    """Prompt intent should be explicit while leaving internal retrieval text out."""

    shopper_intent = intent(
        occasion="beach vacation",
        season="summer",
        garment_types=["sandals"],
        attributes=["comfortable", "black"],
        audience="women",
        price_preference="under $50",
    )
    fake_client = FakeLlmClient(
        [{"explanations": [{"rank": 1, "why_this_matches": "Good match."}]}]
    )
    explainer = ProductExplainer(fake_client)

    explainer.explain(shopper_intent, [ranked(product())])

    prompt = fake_client.calls[0]["user_prompt"]
    assert "- original_query: comfortable black sandals for a beach vacation" in prompt
    assert "- occasion: beach vacation" in prompt
    assert "- season: summer" in prompt
    assert "- garment_types: sandals" in prompt
    assert "- attributes: comfortable, black" in prompt
    assert "- audience: women" in prompt
    assert "- price_preference: under $50" in prompt
    assert "retrieval_text" not in prompt
    assert "internal semantic retrieval text" not in prompt


def test_prompt_uses_product_facts_and_omits_missing_store_and_price() -> None:
    """Product facts should stay concrete and omit unavailable optional fields."""

    features = [f"Feature {number}" for number in range(1, 9)]
    fake_client = FakeLlmClient(
        [{"explanations": [{"rank": 1, "why_this_matches": "Good match."}]}]
    )
    explainer = ProductExplainer(fake_client)

    explainer.explain(
        intent(),
        [
            ranked(
                product(
                    title="Minimal Sandal",
                    store=None,
                    price=None,
                    features=features,
                )
            )
        ],
    )

    prompt = fake_client.calls[0]["user_prompt"]
    assert "- title: Minimal Sandal" in prompt
    assert "- average_rating: 4.6" in prompt
    assert "- rating_number: 42" in prompt
    assert "- store:" not in prompt
    assert "- price:" not in prompt
    assert "\n  - Feature 1\n  - Feature 2" in prompt
    assert "  - Feature 6" in prompt
    assert "Feature 7" not in prompt
    assert "Feature 8" not in prompt


def test_explain_rejects_batch_that_does_not_match_product_ranks() -> None:
    """Structured explanation output must align exactly with the ranked products."""

    fake_client = FakeLlmClient(
        [{"explanations": [{"rank": 99, "why_this_matches": "Wrong product."}]}]
    )
    explainer = ProductExplainer(fake_client)

    with pytest.raises(LlmClientError, match="did not match product ranks"):
        explainer.explain(intent(), [ranked(product())])
