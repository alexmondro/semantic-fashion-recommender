"""Focused coverage for runtime query embeddings."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.embedder import OpenAiEmbedder


class FakeEmbeddings:
    """Small fake of the OpenAI embeddings resource."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    """Tiny OpenAI client fake with an embeddings resource."""

    def __init__(self, responses: list[Any]) -> None:
        self.embeddings = FakeEmbeddings(responses)


def embedding_response(vector: list[float]) -> SimpleNamespace:
    """Build a minimal SDK-like embedding response."""

    return SimpleNamespace(data=[SimpleNamespace(embedding=vector)])


def test_openai_embedder_reuses_injected_client_and_normalizes_vector() -> None:
    """The runtime embedder should not rebuild a client per query."""

    fake_client = FakeClient(
        [
            embedding_response([3.0, 4.0]),
            embedding_response([0.0, 2.0]),
        ]
    )
    embedder = OpenAiEmbedder(api_key="test-key", model="test-embedding-model", client=fake_client)

    first = embedder.embed_text("comfortable sandals")
    second = embedder.embed_text("linen dress")

    assert np.allclose(first, np.array([0.6, 0.8], dtype=np.float32))
    assert np.allclose(second, np.array([0.0, 1.0], dtype=np.float32))
    assert fake_client.embeddings.calls == [
        {
            "model": "test-embedding-model",
            "input": ["comfortable sandals"],
            "encoding_format": "float",
        },
        {
            "model": "test-embedding-model",
            "input": ["linen dress"],
            "encoding_format": "float",
        },
    ]


@pytest.mark.parametrize(
    "response, message",
    [
        (SimpleNamespace(data=[]), "exactly one row"),
        (embedding_response([]), "nonzero"),
        (embedding_response([float("nan")]), "non-finite"),
        (SimpleNamespace(data=[SimpleNamespace(embedding=[[1.0, 0.0]])]), "1-dimensional"),
    ],
)
def test_openai_embedder_rejects_unusable_embedding_response(response: Any, message: str) -> None:
    """Malformed embedding responses should fail before reaching vector search."""

    embedder = OpenAiEmbedder(
        api_key="test-key",
        model="test-embedding-model",
        client=FakeClient([response]),
    )

    with pytest.raises(ValueError, match=message):
        embedder.embed_text("comfortable sandals")
