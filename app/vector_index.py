"""Local NumPy vector index and result curation boundary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import ValidationError

from app.schemas import ProductRecord, RankedProduct


NORM_TOLERANCE = 1e-3
STORE_RESULT_CAP = 2
REQUIRED_MANIFEST_KEYS = frozenset({"product_count", "dimensions", "normalized"})


class VectorIndex(Protocol):
    """Search interface that could later be backed by FAISS or a remote index."""

    def search(self, query_embedding: np.ndarray, *, top_k: int, candidate_pool_size: int) -> list[RankedProduct]:
        """Return curated nearest products for a query embedding."""


class NumpyVectorIndex:
    """In-memory cosine search over shipped NumPy embeddings and metadata."""

    def __init__(self, products: list[ProductRecord], embeddings: np.ndarray) -> None:
        self.products = products
        self.embeddings = embeddings
        self.dimensions = embeddings.shape[1]

    @classmethod
    def load(cls, metadata_path: Path, embedding_matrix_path: Path) -> "NumpyVectorIndex":
        """Load product metadata and the precomputed embedding matrix."""

        manifest_path = embedding_matrix_path.parent / "embedding_manifest.json"
        manifest = _load_manifest(manifest_path)
        products = _load_products(metadata_path)
        embeddings = np.load(embedding_matrix_path, allow_pickle=False)
        _validate_artifacts(products, embeddings, manifest)
        return cls(products=products, embeddings=embeddings)

    def search(self, query_embedding: np.ndarray, *, top_k: int, candidate_pool_size: int) -> list[RankedProduct]:
        """Search, then diversify so shoppers see useful alternatives."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")

        query = _validated_query(query_embedding, self.dimensions)
        scores = self.embeddings @ query
        candidate_indices = self._candidate_indices(scores, top_k, candidate_pool_size)
        selected_indices = self._curate(candidate_indices, top_k)
        return [
            RankedProduct(product=self.products[index], score=float(scores[index]), rank=rank)
            for rank, index in enumerate(selected_indices, start=1)
        ]

    def _candidate_indices(
        self,
        scores: np.ndarray,
        top_k: int,
        candidate_pool_size: int,
    ) -> list[int]:
        """Return candidates sorted by score, then ASIN string for stable ties."""

        n_rows = len(self.products)
        effective_top_k = min(top_k, n_rows)
        effective_pool_size = min(max(candidate_pool_size, effective_top_k), n_rows)
        indices = np.argpartition(scores, -effective_pool_size)[-effective_pool_size:]
        return sorted(
            (int(index) for index in indices),
            key=lambda index: (-float(scores[index]), self.products[index].parent_asin),
        )

    def _curate(self, candidate_indices: list[int], top_k: int) -> list[int]:
        """Fill results in two passes, relaxing only the store cap in pass two."""

        selected: list[int] = []
        selected_set: set[int] = set()
        seen_titles: set[str] = set()
        store_counts: Counter[str] = Counter()
        target_count = min(top_k, len(self.products))

        self._fill_with_store_cap(
            candidate_indices,
            selected,
            selected_set,
            seen_titles,
            store_counts,
            target_count,
        )
        self._fill_without_store_cap(candidate_indices, selected, selected_set, seen_titles, target_count)
        return selected

    def _fill_with_store_cap(
        self,
        candidate_indices: list[int],
        selected: list[int],
        selected_set: set[int],
        seen_titles: set[str],
        store_counts: Counter[str],
        target_count: int,
    ) -> None:
        for index in candidate_indices:
            if len(selected) >= target_count:
                return
            product = self.products[index]
            dedup_key = _dedup_key(product)
            store_key = _store_key(product)
            if dedup_key in seen_titles or (store_key and store_counts[store_key] >= STORE_RESULT_CAP):
                continue
            _select_index(index, selected, selected_set, seen_titles, dedup_key)
            if store_key:
                store_counts[store_key] += 1

    def _fill_without_store_cap(
        self,
        candidate_indices: list[int],
        selected: list[int],
        selected_set: set[int],
        seen_titles: set[str],
        target_count: int,
    ) -> None:
        for index in candidate_indices:
            if len(selected) >= target_count:
                return
            product = self.products[index]
            dedup_key = _dedup_key(product)
            if index in selected_set or dedup_key in seen_titles:
                continue
            _select_index(index, selected, selected_set, seen_titles, dedup_key)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"embedding manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"embedding manifest is malformed JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"embedding manifest must be a JSON object: {path}")
    missing_keys = sorted(REQUIRED_MANIFEST_KEYS - manifest.keys())
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(f"embedding manifest missing required keys: {joined}")
    return manifest


def _load_products(path: Path) -> list[ProductRecord]:
    products: list[ProductRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                products.append(ProductRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"invalid product metadata at line {line_number}: {path}") from exc
    if not products:
        raise ValueError(f"no product metadata loaded from {path}")
    return products


def _validate_artifacts(
    products: list[ProductRecord],
    embeddings: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    product_count = _manifest_int(manifest, "product_count")
    dimensions = _manifest_int(manifest, "dimensions")
    if manifest.get("normalized") is not True:
        raise ValueError("embedding manifest normalized must be true")
    if len(products) != product_count:
        raise ValueError("product metadata row count does not match embedding manifest")
    if embeddings.ndim != 2:
        raise ValueError("embedding matrix must be 2-dimensional")
    if embeddings.dtype != np.float32:
        raise ValueError("embedding matrix must have dtype float32")
    if embeddings.shape != (product_count, dimensions):
        raise ValueError("embedding matrix shape does not match embedding manifest")
    _validate_spot_checked_rows(embeddings)


def _manifest_int(manifest: dict[str, Any], key: str) -> int:
    value = manifest[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"embedding manifest {key} must be a positive integer")
    return value


def _validate_spot_checked_rows(embeddings: np.ndarray) -> None:
    indices = sorted({0, embeddings.shape[0] // 2, embeddings.shape[0] - 1})
    sample = embeddings[indices]
    if not np.isfinite(sample).all():
        raise ValueError("embedding matrix spot-check contains non-finite values")
    norms = np.linalg.norm(sample, axis=1)
    if np.any(np.abs(norms - 1.0) > NORM_TOLERANCE):
        raise ValueError("embedding matrix spot-check found non-normalized rows")


def _validated_query(query_embedding: np.ndarray, dimensions: int) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query embedding must be a 1-dimensional vector")
    if query.shape[0] != dimensions:
        raise ValueError("query embedding dimension does not match index")
    if not np.isfinite(query).all():
        raise ValueError("query embedding contains non-finite values")
    norm = np.linalg.norm(query)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("query embedding must be nonzero")
    return query / norm


def _dedup_key(product: ProductRecord) -> str:
    return (product.normalized_title or "").strip() or product.parent_asin


def _store_key(product: ProductRecord) -> str:
    return (product.store or "").strip()


def _select_index(
    index: int,
    selected: list[int],
    selected_set: set[int],
    seen_titles: set[str],
    dedup_key: str,
) -> None:
    selected.append(index)
    selected_set.add(index)
    seen_titles.add(dedup_key)
