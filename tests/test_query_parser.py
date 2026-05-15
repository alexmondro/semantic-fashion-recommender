"""Focused coverage for natural-language query parsing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.query_parser import QueryParser
from app.schemas import ProductIntent


class FakeLlmClient:
    """Capture parser prompts while returning a typed fake model response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        self.response_model: type[BaseModel] | None = None

    def parse_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_model = response_model
        return response_model.model_validate(self.payload)

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("query parsing should not request free-form completion")


def intent_payload(**updates: Any) -> dict[str, Any]:
    payload = {
        "original_query": "model rewrote this",
        "occasion": "beach vacation",
        "season": "summer",
        "garment_types": ["sandals"],
        "attributes": ["comfortable", "waterproof"],
        "audience": None,
        "price_preference": None,
        "retrieval_text": "comfortable waterproof summer sandals",
    }
    payload.update(updates)
    return payload


def test_parser_overwrites_original_query_with_raw_input() -> None:
    """The API response should preserve the caller's query, not the model echo."""

    query = "comfortable beach sandals"
    parser = QueryParser(FakeLlmClient(intent_payload()))

    intent = parser.parse(query)

    assert intent.original_query == query


def test_parser_falls_back_to_query_when_retrieval_text_is_blank() -> None:
    """The parser repairs blank retrieval text before public schema validation."""

    query = "black linen dress for a summer wedding"
    parser = QueryParser(FakeLlmClient(intent_payload(retrieval_text="   ")))

    intent = parser.parse(query)

    assert intent.retrieval_text == query


def test_parser_prompt_names_retrieval_intent_fields_and_includes_query() -> None:
    """Prompt wording should keep the model aligned with the retrieval schema."""

    query = "soft red cardigan for fall office outfits"
    fake_client = FakeLlmClient(intent_payload())
    parser = QueryParser(fake_client)

    intent = parser.parse(query)

    assert isinstance(intent, ProductIntent)
    assert fake_client.user_prompt is not None
    assert query in fake_client.user_prompt
    assert fake_client.system_prompt is not None
    system_prompt = fake_client.system_prompt.lower()
    for expected in ("retrieval", "occasion", "garment", "attributes"):
        assert expected in system_prompt
