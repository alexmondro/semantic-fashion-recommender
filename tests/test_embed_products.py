"""Smoke coverage for the offline embedding artifact generator."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.embed_products as embed_products


class FakeEmbeddings:
    """Small fake of the OpenAI embeddings resource."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, *, model: str, input: list[str], encoding_format: str) -> SimpleNamespace:
        """Return shuffled response rows to verify index-based ordering."""

        self.calls += 1
        rows = []
        for index, _text in reversed(list(enumerate(input))):
            vector = [0.0] * embed_products.DIMENSIONS
            vector[index] = 2.0
            rows.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(
            data=rows,
            usage=SimpleNamespace(prompt_tokens=len(input), total_tokens=len(input)),
        )


class FakeClient:
    """Tiny OpenAI client fake with timeout capture."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float) -> "FakeClient":
        self.timeouts.append(timeout)
        return self


def write_products(path, count: int) -> None:
    """Create a minimal products.jsonl fixture."""

    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"retrieval_text": f"product {index}"}) + "\n")


def test_embed_products_writes_normalized_matrix_and_manifest(monkeypatch, tmp_path) -> None:
    """A fake-client run exercises batching, row order, cleanup, and manifest fields."""

    products_path = tmp_path / "products.jsonl"
    output_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    partial_path = tmp_path / "embeddings.partial.npy"
    fake_client = FakeClient()
    write_products(products_path, 3)
    monkeypatch.setattr(embed_products, "create_openai_client", lambda: fake_client)

    embed_products.embed_products(
        products_path,
        output_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        partial_output_path=partial_path,
        max_inputs_per_request=2,
        request_timeout_seconds=12.0,
    )

    matrix = np.load(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert matrix.shape == (3, embed_products.DIMENSIONS)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)
    assert matrix[2, 0] == pytest.approx(1.0)
    assert manifest["model"] == embed_products.MODEL
    assert manifest["limit"] is None
    assert manifest["request_timeout_seconds"] == 12.0
    assert fake_client.timeouts == [12.0, 12.0]
    assert not checkpoint_path.exists()
    assert not partial_path.exists()


def test_limit_cannot_write_default_artifact() -> None:
    """Smoke runs must opt into a non-default output path."""

    with pytest.raises(ValueError, match="--limit cannot write"):
        embed_products.embed_products(
            embed_products.DEFAULT_INPUT,
            embed_products.DEFAULT_OUTPUT,
            limit=1,
        )


def test_embedding_response_retries_transient_errors(monkeypatch) -> None:
    """A transient API failure should retry before giving up on the batch."""

    class TransientError(Exception):
        status_code = 429

    class FlakyEmbeddings:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 1:
                raise TransientError("rate limited")
            return SimpleNamespace(data=[], usage=None)

    class FlakyClient:
        def __init__(self) -> None:
            self.embeddings = FlakyEmbeddings()

        def with_options(self, *, timeout: float) -> "FlakyClient":
            assert timeout == 9.0
            return self

    monkeypatch.setattr(embed_products.time, "sleep", lambda _seconds: None)
    client = FlakyClient()

    response = embed_products.create_embedding_response(client, ["hello"], 9.0)

    assert response.data == []
    assert client.embeddings.calls == 2
