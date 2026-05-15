#!/usr/bin/env python3
"""Create the shipped embedding matrix for the eligible product metadata.

Input: data/products.jsonl.
Output: data/embeddings.npy aligned row-for-row with the metadata file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "products.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "embeddings.npy"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "embedding_manifest.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "embedding_checkpoint.json"
DEFAULT_PARTIAL_OUTPUT = PROJECT_ROOT / "data" / "embeddings.partial.npy"

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
MAX_INPUT_TOKENS = 8192
MAX_REQUEST_TOKENS = 300_000
DEFAULT_MAX_INPUTS_PER_REQUEST = 256
DEFAULT_MAX_ESTIMATED_TOKENS_PER_REQUEST = 200_000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
MAX_EMBEDDING_ATTEMPTS = 5
EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02
NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ProductText:
    """Embedding input paired with its original product row."""

    row_index: int
    text: str


@dataclass(frozen=True)
class EmbedConfig:
    """Resolved CLI settings for one embedding run."""

    input_path: Path
    output_path: Path
    manifest_path: Path
    checkpoint_path: Path
    partial_output_path: Path
    max_inputs_per_request: int
    max_estimated_tokens_per_request: int
    request_timeout_seconds: float
    limit: int | None


class TokenEstimator:
    """Estimate embedding request size without making API calls."""

    def __init__(self, model: str) -> None:
        self._encoding = self._load_encoding(model)

    @property
    def method(self) -> str:
        return "tiktoken" if self._encoding is not None else "chars_div_3"

    def estimate(self, text: str) -> int:
        """Return a conservative token estimate for one input string."""

        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, (len(text) + 2) // 3)

    @staticmethod
    def _load_encoding(model: str) -> Any | None:
        try:
            import tiktoken
        except ImportError:
            return None
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def sha256_file(path: Path) -> str:
    """Hash the exact bytes of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    """Return an installed package version without importing package internals."""

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_dotenv_file(path: Path) -> None:
    """Load .env when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path)


def create_openai_client() -> Any:
    """Create a modern OpenAI SDK client with a clear dependency error."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate embeddings.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Python SDK 1.x+ is required. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return OpenAI(api_key=api_key)


def load_product_texts(path: Path, limit: int | None) -> list[ProductText]:
    """Load retrieval text in artifact order."""

    products: list[ProductText] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if limit is not None and len(products) >= limit:
                break
            row = json.loads(line)
            text = row.get("retrieval_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"row {row_index} has empty retrieval_text")
            products.append(ProductText(row_index=row_index, text=text.strip()))
    if not products:
        raise ValueError(f"no products loaded from {path}")
    return products


def iter_batches(
    products: list[ProductText],
    token_estimator: TokenEstimator,
    max_inputs_per_request: int,
    max_estimated_tokens_per_request: int,
    start_row: int,
) -> list[list[ProductText]]:
    """Pack products into request batches bounded by count and token budget."""

    batches: list[list[ProductText]] = []
    current_batch: list[ProductText] = []
    current_tokens = 0
    for product in products[start_row:]:
        estimated_tokens = token_estimator.estimate(product.text)
        if estimated_tokens > MAX_INPUT_TOKENS:
            raise ValueError(
                f"row {product.row_index} estimates to {estimated_tokens:,} tokens; "
                f"max per embedding input is {MAX_INPUT_TOKENS:,}"
            )
        would_exceed_count = len(current_batch) >= max_inputs_per_request
        would_exceed_tokens = (
            current_batch
            and current_tokens + estimated_tokens > max_estimated_tokens_per_request
        )
        if would_exceed_count or would_exceed_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(product)
        current_tokens += estimated_tokens
    if current_batch:
        batches.append(current_batch)
    return batches


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Convert embeddings to unit vectors for cosine-as-dot-product search."""

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding response contains NaN or infinite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding response contains a zero vector")
    return embeddings / norms


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    """Read checkpoint JSON if a prior run exists."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_checkpoint(
    checkpoint: dict[str, Any],
    config: EmbedConfig,
    input_file_sha256: str,
    product_count: int,
) -> None:
    """Ensure resume state belongs to this exact artifact and config."""

    expected = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "input_path": str(config.input_path),
        "input_file_sha256": input_file_sha256,
        "product_count": product_count,
        "max_inputs_per_request": config.max_inputs_per_request,
        "max_estimated_tokens_per_request": config.max_estimated_tokens_per_request,
        "request_timeout_seconds": config.request_timeout_seconds,
        "limit": config.limit,
    }
    mismatches = [
        key for key, expected_value in expected.items() if checkpoint.get(key) != expected_value
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"checkpoint does not match this run: {joined}")
    completed_rows = checkpoint.get("completed_rows")
    if not isinstance(completed_rows, int) or not 0 <= completed_rows <= product_count:
        raise ValueError("checkpoint has invalid completed_rows")


def write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Persist checkpoint atomically enough for interrupted script resumes."""

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def default_checkpoint_for(output_path: Path) -> Path:
    """Choose the checkpoint path that matches the output target."""

    if output_path.resolve() == DEFAULT_OUTPUT.resolve():
        return DEFAULT_CHECKPOINT
    return output_path.with_suffix(".checkpoint.json")


def default_partial_output_for(output_path: Path) -> Path:
    """Choose the partial matrix path that matches the output target."""

    if output_path.resolve() == DEFAULT_OUTPUT.resolve():
        return DEFAULT_PARTIAL_OUTPUT
    return output_path.with_name(f"{output_path.stem}.partial.npy")


def create_or_resume_matrix(
    config: EmbedConfig,
    product_count: int,
    checkpoint: dict[str, Any] | None,
) -> tuple[np.memmap, int]:
    """Open the partial matrix and return it with the next row to fill."""

    if checkpoint is None:
        config.partial_output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            config.partial_output_path,
            mode="w+",
            dtype=np.float32,
            shape=(product_count, DIMENSIONS),
        )
        return matrix, 0
    if not config.partial_output_path.exists():
        raise ValueError(f"checkpoint exists but partial matrix is missing: {config.partial_output_path}")
    matrix = np.lib.format.open_memmap(config.partial_output_path, mode="r+")
    if matrix.shape != (product_count, DIMENSIONS) or matrix.dtype != np.float32:
        raise ValueError("partial matrix shape or dtype does not match checkpoint")
    return matrix, int(checkpoint["completed_rows"])


def close_memmap(matrix: np.memmap) -> None:
    """Flush and close a NumPy memmap handle before moving the backing file."""

    matrix.flush()
    mmap_handle = getattr(matrix, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def build_checkpoint(
    config: EmbedConfig,
    input_file_sha256: str,
    product_count: int,
    completed_rows: int,
    prompt_tokens: int,
    total_tokens: int,
) -> dict[str, Any]:
    """Build checkpoint state for the current completed row boundary."""

    return {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "input_path": str(config.input_path),
        "input_file_sha256": input_file_sha256,
        "product_count": product_count,
        "completed_rows": completed_rows,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "max_inputs_per_request": config.max_inputs_per_request,
        "max_estimated_tokens_per_request": config.max_estimated_tokens_per_request,
        "request_timeout_seconds": config.request_timeout_seconds,
        "limit": config.limit,
        "openai_sdk_version": package_version("openai"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def client_embedding_batch(
    client: Any,
    batch: list[ProductText],
    timeout_seconds: float,
) -> tuple[np.ndarray, int, int]:
    """Embed one batch and return vectors plus token usage."""

    response = create_embedding_response(client, [product.text for product in batch], timeout_seconds)
    ordered_data = sorted(response.data, key=lambda item: item.index)
    if len(ordered_data) != len(batch):
        raise ValueError("embedding response row count does not match request")
    embeddings = np.asarray([item.embedding for item in ordered_data], dtype=np.float32)
    if embeddings.shape != (len(batch), DIMENSIONS):
        raise ValueError(f"embedding response shape was {embeddings.shape}, expected {(len(batch), DIMENSIONS)}")
    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    total_tokens = getattr(usage, "total_tokens", prompt_tokens) if usage is not None else prompt_tokens
    return normalize_embeddings(embeddings), int(prompt_tokens or 0), int(total_tokens or 0)


def create_embedding_response(client: Any, input_texts: list[str], timeout_seconds: float) -> Any:
    """Call embeddings API with explicit timeout and transient retries."""

    last_error: Exception | None = None
    for attempt in range(1, MAX_EMBEDDING_ATTEMPTS + 1):
        try:
            request_client = client.with_options(timeout=timeout_seconds)
            return request_client.embeddings.create(
                model=MODEL,
                input=input_texts,
                encoding_format="float",
            )
        except Exception as exc:
            if not is_transient_openai_error(exc) or attempt == MAX_EMBEDDING_ATTEMPTS:
                raise
            last_error = exc
            sleep_seconds = min(2 ** attempt, 30)
            print(
                f"transient embedding error on attempt {attempt}/{MAX_EMBEDDING_ATTEMPTS}; "
                f"retrying in {sleep_seconds}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError("embedding request failed after retries") from last_error


def is_transient_openai_error(exc: Exception) -> bool:
    """Identify errors worth retrying during a long ingestion run."""

    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__
    return name in {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}


def embed_remaining_batches(
    config: EmbedConfig,
    products: list[ProductText],
    token_estimator: TokenEstimator,
    client: Any,
    matrix: np.memmap,
    completed_rows: int,
    input_file_sha256: str,
    prompt_tokens: int,
    total_tokens: int,
) -> tuple[int, int]:
    """Embed all rows after completed_rows and persist progress each batch."""

    batches = iter_batches(
        products,
        token_estimator,
        config.max_inputs_per_request,
        config.max_estimated_tokens_per_request,
        completed_rows,
    )
    print_batch_plan(len(products), completed_rows, len(batches), token_estimator.method)
    for batch_number, batch in enumerate(batches, start=1):
        start = batch[0].row_index
        end = batch[-1].row_index + 1
        embeddings, batch_prompt_tokens, batch_total_tokens = client_embedding_batch(
            client,
            batch,
            config.request_timeout_seconds,
        )
        matrix[start:end, :] = embeddings
        matrix.flush()
        completed_rows = end
        prompt_tokens += batch_prompt_tokens
        total_tokens += batch_total_tokens
        checkpoint = build_checkpoint(
            config,
            input_file_sha256,
            len(products),
            completed_rows,
            prompt_tokens,
            total_tokens,
        )
        write_checkpoint(config.checkpoint_path, checkpoint)
        print_batch_progress(batch_number, len(batches), completed_rows, len(products), batch_prompt_tokens, prompt_tokens)
    return prompt_tokens, total_tokens


def print_batch_plan(
    product_count: int,
    completed_rows: int,
    batch_count: int,
    token_estimation_method: str,
) -> None:
    """Log the work remaining before network calls begin."""

    print(
        f"Embedding {product_count - completed_rows:,}/{product_count:,} rows "
        f"in {batch_count:,} batches using {token_estimation_method} estimates.",
        file=sys.stderr,
    )


def print_batch_progress(
    batch_number: int,
    batch_count: int,
    completed_rows: int,
    product_count: int,
    batch_prompt_tokens: int,
    prompt_tokens: int,
) -> None:
    """Log one completed embeddings batch."""

    percent = (completed_rows / product_count) * 100
    cost = (prompt_tokens / 1_000_000) * EMBEDDING_PRICE_PER_MILLION_TOKENS
    print(
        f"batch={batch_number:,}/{batch_count:,} rows={completed_rows:,}/{product_count:,} "
        f"({percent:.1f}%) batch_tokens={batch_prompt_tokens:,} cost=${cost:.4f}",
        file=sys.stderr,
    )


def validate_final_matrix(path: Path, product_count: int) -> None:
    """Check final matrix contract before declaring the artifact complete."""

    matrix = np.load(path)
    if matrix.shape != (product_count, DIMENSIONS):
        raise ValueError(f"final matrix shape was {matrix.shape}, expected {(product_count, DIMENSIONS)}")
    if matrix.dtype != np.float32:
        raise ValueError(f"final matrix dtype was {matrix.dtype}, expected float32")
    if not np.isfinite(matrix).all():
        raise ValueError("final matrix contains NaN or infinite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
        raise ValueError("final matrix contains non-normalized rows")


def write_manifest(
    config: EmbedConfig,
    input_file_sha256: str,
    output_file_sha256: str,
    product_count: int,
    prompt_tokens: int,
    total_tokens: int,
    wall_clock_seconds: float,
) -> None:
    """Write reproducibility metadata for the generated embeddings."""

    estimated_cost = (prompt_tokens / 1_000_000) * EMBEDDING_PRICE_PER_MILLION_TOKENS
    manifest = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "product_count": product_count,
        "input_file_path": str(config.input_path),
        "input_file_sha256": input_file_sha256,
        "output_file_path": str(config.output_path),
        "output_file_sha256": output_file_sha256,
        "normalized": True,
        "max_inputs_per_request": config.max_inputs_per_request,
        "max_estimated_tokens_per_request": config.max_estimated_tokens_per_request,
        "request_timeout_seconds": config.request_timeout_seconds,
        "limit": config.limit,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "wall_clock_seconds": round(wall_clock_seconds, 3),
        "openai_sdk_version": package_version("openai"),
        "tiktoken_version": package_version("tiktoken"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finalize_artifacts(
    config: EmbedConfig,
    product_count: int,
    input_file_sha256: str,
    prompt_tokens: int,
    total_tokens: int,
    started_at: float,
) -> None:
    """Promote the partial matrix to final output and write the manifest."""

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.partial_output_path.replace(config.output_path)
    validate_final_matrix(config.output_path, product_count)
    output_file_sha256 = sha256_file(config.output_path)
    wall_clock_seconds = time.monotonic() - started_at
    write_manifest(
        config,
        input_file_sha256,
        output_file_sha256,
        product_count,
        prompt_tokens,
        total_tokens,
        wall_clock_seconds,
    )
    if config.checkpoint_path.exists():
        config.checkpoint_path.unlink()


def embed_products(
    product_metadata_path: Path,
    output_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    partial_output_path: Path = DEFAULT_PARTIAL_OUTPUT,
    max_inputs_per_request: int = DEFAULT_MAX_INPUTS_PER_REQUEST,
    max_estimated_tokens_per_request: int = DEFAULT_MAX_ESTIMATED_TOKENS_PER_REQUEST,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    limit: int | None = None,
) -> None:
    """Embed every eligible product and save a float32 NumPy matrix."""

    config = EmbedConfig(
        input_path=product_metadata_path,
        output_path=output_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        partial_output_path=partial_output_path,
        max_inputs_per_request=max_inputs_per_request,
        max_estimated_tokens_per_request=max_estimated_tokens_per_request,
        request_timeout_seconds=request_timeout_seconds,
        limit=limit,
    )
    validate_run_safety(config)
    run_embedding_job(config)


def validate_run_safety(config: EmbedConfig) -> None:
    """Reject settings that could accidentally replace the shipped artifact."""

    if config.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be greater than 0")
    if config.max_estimated_tokens_per_request > MAX_REQUEST_TOKENS:
        raise ValueError(
            f"--max-estimated-tokens-per-request must be <= {MAX_REQUEST_TOKENS:,}"
        )
    if config.limit is not None and config.output_path.resolve() == DEFAULT_OUTPUT.resolve():
        raise ValueError("--limit cannot write to the default final artifact; pass --output under /tmp.")


def run_embedding_job(config: EmbedConfig) -> None:
    """Run or resume embedding generation for the resolved config."""

    started_at = time.monotonic()
    input_file_sha256 = sha256_file(config.input_path)
    products = load_product_texts(config.input_path, config.limit)
    product_count = len(products)
    load_dotenv_file(PROJECT_ROOT / ".env")
    client = create_openai_client()
    token_estimator = TokenEstimator(MODEL)

    checkpoint = load_checkpoint(config.checkpoint_path)
    if checkpoint is not None:
        validate_checkpoint(checkpoint, config, input_file_sha256, product_count)
    matrix, completed_rows = create_or_resume_matrix(config, product_count, checkpoint)
    prompt_tokens = int((checkpoint or {}).get("prompt_tokens", 0))
    total_tokens = int((checkpoint or {}).get("total_tokens", 0))

    if completed_rows >= product_count:
        print("All rows are already embedded; finalizing artifact.", file=sys.stderr)
    else:
        prompt_tokens, total_tokens = embed_remaining_batches(
            config,
            products,
            token_estimator,
            client,
            matrix,
            completed_rows,
            input_file_sha256,
            prompt_tokens,
            total_tokens,
        )

    close_memmap(matrix)
    finalize_artifacts(
        config,
        product_count,
        input_file_sha256,
        prompt_tokens,
        total_tokens,
        started_at,
    )


def main() -> None:
    """CLI entrypoint for the one-time embedding step."""

    args = parse_args()
    output_path = args.output
    checkpoint_path = args.checkpoint or default_checkpoint_for(output_path)
    partial_output_path = args.partial_output or default_partial_output_for(output_path)
    embed_products(
        args.input,
        output_path,
        manifest_path=args.manifest,
        checkpoint_path=checkpoint_path,
        partial_output_path=partial_output_path,
        max_inputs_per_request=args.max_inputs_per_request,
        max_estimated_tokens_per_request=args.max_estimated_tokens_per_request,
        request_timeout_seconds=args.request_timeout_seconds,
        limit=args.limit,
    )


def positive_int(value: str) -> int:
    """Argparse type for positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse CLI options for offline embedding generation."""

    parser = argparse.ArgumentParser(description="Build the shipped product embedding matrix.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Filtered product JSONL artifact.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Final .npy matrix path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest JSON path.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume checkpoint JSON path.")
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=None,
        help="In-progress memmap .npy path.",
    )
    parser.add_argument(
        "--max-inputs-per-request",
        type=positive_int,
        default=DEFAULT_MAX_INPUTS_PER_REQUEST,
        help="Maximum input strings per embeddings request.",
    )
    parser.add_argument(
        "--max-estimated-tokens-per-request",
        type=positive_int,
        default=DEFAULT_MAX_ESTIMATED_TOKENS_PER_REQUEST,
        help="Conservative estimated token budget per embeddings request.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Timeout for each embeddings request.",
    )
    parser.add_argument("--limit", type=positive_int, default=None, help="Embed only the first N rows.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
