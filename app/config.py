"""Application settings read once at service startup.

This module is the single place for model names, artifact paths, and runtime
defaults so the FastAPI route and pipeline modules do not reach into
environment variables directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """Runtime configuration for the recommender service."""

    openai_api_key: str | None = Field(default=None, repr=False)
    chat_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    fallback_chat_model: str = "gpt-4o-mini"
    product_metadata_path: Path = PROJECT_ROOT / "data" / "products.jsonl"
    embedding_matrix_path: Path = PROJECT_ROOT / "data" / "embeddings.npy"
    candidate_pool_size: int = 40
    min_rating_count: int = 10
    min_average_rating: float = 4.0


def load_settings() -> Settings:
    """Load settings from environment and defaults."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    cleaned_api_key = api_key.strip() if api_key else None
    return Settings(openai_api_key=cleaned_api_key or None)
