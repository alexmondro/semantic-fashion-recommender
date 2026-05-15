"""Focused coverage for the local NumPy vector index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.vector_index import NumpyVectorIndex


def product_row(
    parent_asin: str,
    *,
    normalized_title: str | None = None,
    store: str | None = "Test Store",
) -> dict[str, Any]:
    """Build one response-safe product row for tiny artifact tests."""

    return {
        "parent_asin": parent_asin,
        "title": f"Product {parent_asin}",
        "store": store,
        "average_rating": 4.5,
        "rating_number": 20,
        "price": None,
        "image_url": None,
        "features": ["Soft fabric"],
        "normalized_title": normalized_title if normalized_title is not None else parent_asin.lower(),
        "retrieval_text": f"Product {parent_asin}\n- Soft fabric",
    }


def vector_with_score(score: float) -> np.ndarray:
    """Return a unit vector with the requested dot product against [1, 0]."""

    return np.array([score, (1.0 - score**2) ** 0.5], dtype=np.float32)


def write_artifacts(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    manifest_updates: dict[str, Any] | None = None,
    manifest_text: str | None = None,
    write_manifest: bool = True,
) -> tuple[Path, Path]:
    """Write tiny metadata, embedding, and manifest artifacts."""

    metadata_path = tmp_path / "products.jsonl"
    embedding_path = tmp_path / "embeddings.npy"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    np.save(embedding_path, embeddings.astype(np.float32))

    if write_manifest:
        manifest_path = tmp_path / "embedding_manifest.json"
        if manifest_text is not None:
            manifest_path.write_text(manifest_text, encoding="utf-8")
        else:
            manifest = {
                "product_count": len(rows),
                "dimensions": embeddings.shape[1],
                "normalized": True,
            }
            manifest.update(manifest_updates or {})
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    return metadata_path, embedding_path


def load_index(tmp_path: Path, rows: list[dict[str, Any]], embeddings: np.ndarray) -> NumpyVectorIndex:
    """Create and load a tiny index."""

    metadata_path, embedding_path = write_artifacts(tmp_path, rows, embeddings)
    return NumpyVectorIndex.load(metadata_path, embedding_path)


def test_loads_tiny_artifacts_and_returns_deterministic_score_order(tmp_path: Path) -> None:
    """Search sorts by score descending, then ASIN string ascending for ties."""

    rows = [
        product_row("B0002", normalized_title="tie two"),
        product_row("B0001", normalized_title="tie one"),
        product_row("A0001", normalized_title="best"),
    ]
    embeddings = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    index = load_index(tmp_path, rows, embeddings)

    results = index.search(np.array([1.0, 0.0]), top_k=3, candidate_pool_size=3)

    assert [result.product.parent_asin for result in results] == ["A0001", "B0001", "B0002"]
    assert [result.rank for result in results] == [1, 2, 3]


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    """A matrix without its manifest should fail loudly at load time."""

    rows = [product_row("A0001")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    metadata_path, embedding_path = write_artifacts(tmp_path, rows, embeddings, write_manifest=False)

    with pytest.raises(ValueError, match="manifest not found"):
        NumpyVectorIndex.load(metadata_path, embedding_path)


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    """Corrupt manifest JSON should not silently fall back to artifact guesses."""

    rows = [product_row("A0001")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    metadata_path, embedding_path = write_artifacts(tmp_path, rows, embeddings, manifest_text="{")

    with pytest.raises(ValueError, match="malformed JSON"):
        NumpyVectorIndex.load(metadata_path, embedding_path)


def test_manifest_count_mismatch_is_rejected(tmp_path: Path) -> None:
    """The metadata row count must match the manifest product count."""

    rows = [product_row("A0001")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    metadata_path, embedding_path = write_artifacts(
        tmp_path,
        rows,
        embeddings,
        manifest_updates={"product_count": 2},
    )

    with pytest.raises(ValueError, match="row count"):
        NumpyVectorIndex.load(metadata_path, embedding_path)


def test_manifest_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    """The embedding dimensions must match the manifest dimensions."""

    rows = [product_row("A0001")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    metadata_path, embedding_path = write_artifacts(
        tmp_path,
        rows,
        embeddings,
        manifest_updates={"dimensions": 3},
    )

    with pytest.raises(ValueError, match="shape"):
        NumpyVectorIndex.load(metadata_path, embedding_path)


def test_manifest_normalized_must_be_strict_true(tmp_path: Path) -> None:
    """Truthy values like 1 should not satisfy the normalized contract."""

    rows = [product_row("A0001")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    metadata_path, embedding_path = write_artifacts(
        tmp_path,
        rows,
        embeddings,
        manifest_updates={"normalized": 1},
    )

    with pytest.raises(ValueError, match="normalized"):
        NumpyVectorIndex.load(metadata_path, embedding_path)


def test_query_dimension_must_match_index(tmp_path: Path) -> None:
    """Wrong-size query vectors are rejected before scoring."""

    index = load_index(
        tmp_path,
        [product_row("A0001")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="dimension"):
        index.search(np.array([1.0, 0.0, 0.0]), top_k=1, candidate_pool_size=1)


def test_query_vector_must_be_nonzero(tmp_path: Path) -> None:
    """Zero vectors cannot be normalized for cosine search."""

    index = load_index(
        tmp_path,
        [product_row("A0001")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="nonzero"):
        index.search(np.array([0.0, 0.0]), top_k=1, candidate_pool_size=1)


def test_empty_normalized_title_falls_back_to_parent_asin(tmp_path: Path) -> None:
    """Empty normalized titles should not collapse unrelated products."""

    rows = [
        product_row("A0001", normalized_title=""),
        product_row("A0002", normalized_title=""),
        product_row("A0003", normalized_title=""),
    ]
    embeddings = np.vstack(
        [
            vector_with_score(1.0),
            vector_with_score(0.99),
            vector_with_score(0.98),
        ]
    )
    index = load_index(tmp_path, rows, embeddings)

    results = index.search(np.array([1.0, 0.0]), top_k=3, candidate_pool_size=3)

    assert [result.product.parent_asin for result in results] == ["A0001", "A0002", "A0003"]


def test_store_cap_uses_second_pass_backfill_in_candidate_order(tmp_path: Path) -> None:
    """Backfill skips selected rows and relaxes only store dominance."""

    rows = [
        product_row("A0001", normalized_title="one", store="One Store"),
        product_row("A0002", normalized_title="two", store="One Store"),
        product_row("A0003", normalized_title="three", store="One Store"),
        product_row("A0004", normalized_title="four", store="One Store"),
        product_row("B0001", normalized_title="five", store="Other Store"),
    ]
    embeddings = np.vstack(
        [
            vector_with_score(1.0),
            vector_with_score(0.99),
            vector_with_score(0.98),
            vector_with_score(0.97),
            vector_with_score(0.96),
        ]
    )
    index = load_index(tmp_path, rows, embeddings)

    results = index.search(np.array([1.0, 0.0]), top_k=4, candidate_pool_size=5)

    assert [result.product.parent_asin for result in results] == ["A0001", "A0002", "B0001", "A0003"]


def test_top_k_larger_than_index_is_clamped(tmp_path: Path) -> None:
    """Asking for more rows than exist should return every distinct candidate."""

    rows = [product_row("A0001"), product_row("A0002")]
    embeddings = np.vstack([vector_with_score(1.0), vector_with_score(0.99)])
    index = load_index(tmp_path, rows, embeddings)

    results = index.search(np.array([1.0, 0.0]), top_k=10, candidate_pool_size=1)

    assert [result.product.parent_asin for result in results] == ["A0001", "A0002"]


def test_candidate_pool_larger_than_index_is_clamped(tmp_path: Path) -> None:
    """The boundary-safe argpartition path handles pool size equal to n_rows."""

    rows = [product_row("A0001"), product_row("A0002"), product_row("A0003")]
    embeddings = np.vstack(
        [
            vector_with_score(1.0),
            vector_with_score(0.99),
            vector_with_score(0.98),
        ]
    )
    index = load_index(tmp_path, rows, embeddings)

    results = index.search(np.array([1.0, 0.0]), top_k=2, candidate_pool_size=10)

    assert [result.product.parent_asin for result in results] == ["A0001", "A0002"]
