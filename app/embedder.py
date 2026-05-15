"""Embedding provider boundary for runtime query embeddings."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from openai import OpenAI


class Embedder(Protocol):
    """Minimal embedding interface used by the recommendation pipeline."""

    def embed_text(self, text: str) -> np.ndarray:
        """Return a single normalized embedding vector for retrieval."""


class OpenAiEmbedder:
    """OpenAI implementation using text-embedding-3-small."""

    def __init__(self, api_key: str, model: str, *, client: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client or OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed the query or intent text for NumPy cosine search."""

        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float",
        )
        rows = list(getattr(response, "data", []) or [])
        if len(rows) != 1:
            raise ValueError("embedding response must contain exactly one row")

        embedding = getattr(rows[0], "embedding", None)
        if embedding is None:
            raise ValueError("embedding response row is missing embedding data")
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("embedding response must be a 1-dimensional vector")
        if not np.isfinite(vector).all():
            raise ValueError("embedding response contains non-finite values")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("embedding response must be nonzero")
        return vector / norm
