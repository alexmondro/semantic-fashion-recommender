"""Natural-language shopping query to structured intent."""

from __future__ import annotations

from pydantic import BaseModel

from app.llm_client import LlmClient
from app.schemas import ProductIntent


SYSTEM_PROMPT = """You parse fashion shopping queries into structured retrieval intent.

Extract only shopper intent that is explicit or strongly implied.
Use null for unknown occasion, season, audience, or price preference.
Use empty arrays when no garment types or attributes are present.
Put colors, materials, fit, comfort, weather needs, style, vibe, and function words in attributes.
Write retrieval_text as a non-empty, concise phrase optimized for semantic retrieval.
"""


class _ParsedProductIntent(BaseModel):
    """Lenient intermediate schema so parser code can repair retrieval_text."""

    original_query: str
    occasion: str | None
    season: str | None
    garment_types: list[str]
    attributes: list[str]
    audience: str | None
    price_preference: str | None
    retrieval_text: str


class QueryParser:
    """Extract shopper intent with gpt-5-mini structured outputs."""

    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client

    def parse(self, query: str) -> ProductIntent:
        """Convert a free-form shopper query into retrieval-ready intent."""

        parsed = self.llm_client.parse_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_user_prompt(query),
            response_model=_ParsedProductIntent,
        )
        retrieval_text = parsed.retrieval_text.strip() or query
        return ProductIntent(
            original_query=query,
            occasion=_clean_optional(parsed.occasion),
            season=_clean_optional(parsed.season),
            garment_types=_clean_list(parsed.garment_types),
            attributes=_clean_list(parsed.attributes),
            audience=_clean_optional(parsed.audience),
            price_preference=_clean_optional(parsed.price_preference),
            retrieval_text=retrieval_text,
        )


def _user_prompt(query: str) -> str:
    return f"Shopper query: {query}"


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_list(values: list[str]) -> list[str]:
    return [cleaned for value in values if (cleaned := value.strip())]
