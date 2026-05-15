#!/usr/bin/env python3
"""Create the eligible 77,740-product metadata artifact.

Input: raw Amazon Fashion JSONL.GZ.
Output: response-safe JSONL metadata under data/products.jsonl.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas import ProductRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "meta_Amazon_Fashion.jsonl.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "products.jsonl"

MIN_RATING_COUNT = 10
MIN_AVERAGE_RATING = 4.0
PROGRESS_INTERVAL = 100_000

DETAIL_ALLOWLIST = (
    "Department",
    "Material",
    "Material Composition",
    "Fabric Type",
    "Outer Material",
    "Sole Material",
    "Care Instructions",
    "Closure Type",
    "Fit Type",
    "Style",
)


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"\([^)]*\)", " ", title)
    title = re.sub(r"\b(size|small|medium|large|x-large|xl|xxl|xs|s|m|l)\b", " ", title)
    title = re.sub(r"\b\d+(\.\d+)?\b", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split()[:12])


def _first_image_url(images: list[Any]) -> str | None:
    if not images or not isinstance(images[0], dict):
        return None
    first = images[0]
    return first.get("large") or first.get("hi_res") or first.get("thumb")


def _build_retrieval_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    title = row.get("title")
    if _has_text(title):
        parts.append(title.strip())
    for feature in _as_list(row.get("features")):
        if _has_text(feature):
            parts.append(f"- {feature.strip()}")
    store = row.get("store")
    if _has_text(store):
        parts.append(f"Store: {store.strip()}")
    raw_details = row.get("details")
    details = raw_details if isinstance(raw_details, dict) else {}
    for key in DETAIL_ALLOWLIST:
        value = details.get(key)
        if _has_text(value):
            parts.append(f"{key}: {value.strip()}")
    return "\n".join(parts)


def is_eligible(row: dict[str, Any]) -> bool:
    """Apply the locked eligibility policy from docs/eda_subset_policy.md."""

    if not _has_text(row.get("title")):
        return False
    if not _as_list(row.get("images")):
        return False
    if not _as_list(row.get("features")):
        return False
    rating_number = row.get("rating_number")
    average_rating = row.get("average_rating")
    if rating_number is None or average_rating is None:
        return False
    try:
        if int(rating_number) < MIN_RATING_COUNT:
            return False
        if float(average_rating) < MIN_AVERAGE_RATING:
            return False
    except (TypeError, ValueError):
        return False
    return True


def _build_record(row: dict[str, Any], image_url: str | None) -> ProductRecord:
    title = row["title"].strip()
    features = [f.strip() for f in _as_list(row.get("features")) if _has_text(f)]
    store_raw = row.get("store")
    store = store_raw.strip() if _has_text(store_raw) else None
    return ProductRecord(
        parent_asin=row["parent_asin"],
        title=title,
        store=store,
        average_rating=float(row["average_rating"]),
        rating_number=int(row["rating_number"]),
        price=float(row["price"]) if row.get("price") is not None else None,
        image_url=image_url,
        features=features,
        normalized_title=_normalize_title(title),
        retrieval_text=_build_retrieval_text(row),
    )


def to_product_record(row: dict[str, Any]) -> tuple[ProductRecord, bool]:
    """Validate and build a ProductRecord. Returns (record, image_url_dropped)."""

    image_candidate = _first_image_url(_as_list(row.get("images")))
    try:
        return _build_record(row, image_candidate), False
    except ValidationError as exc:
        if image_candidate is None or not all(
            err.get("loc") == ("image_url",) for err in exc.errors()
        ):
            raise
        return _build_record(row, None), True


def _to_artifact_dict(record: ProductRecord) -> dict[str, Any]:
    return {
        "parent_asin": record.parent_asin,
        "title": record.title,
        "store": record.store,
        "average_rating": record.average_rating,
        "rating_number": record.rating_number,
        "price": record.price,
        "image_url": str(record.image_url) if record.image_url is not None else None,
        "features": list(record.features),
        "normalized_title": record.normalized_title,
        "retrieval_text": record.retrieval_text,
    }


def filter_products(raw_dataset_path: Path, output_path: Path) -> None:
    """Apply the locked eligibility policy and write product metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    read_count = 0
    eligible_count = 0
    dropped_image_url = 0
    skipped_missing_asin = 0
    validation_failures = 0
    sample_failures: list[str] = []

    with gzip.open(raw_dataset_path, "rt", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            read_count += 1
            if read_count % PROGRESS_INTERVAL == 0:
                print(f"  read={read_count:,} eligible={eligible_count:,}", file=sys.stderr)
            row = json.loads(line)
            if not is_eligible(row):
                continue
            if not _has_text(row.get("parent_asin")):
                skipped_missing_asin += 1
                continue
            try:
                record, image_dropped = to_product_record(row)
            except ValidationError as exc:
                validation_failures += 1
                if len(sample_failures) < 3:
                    sample_failures.append(f"{row.get('parent_asin')}: {exc}")
                continue
            if image_dropped:
                dropped_image_url += 1
            output_handle.write(json.dumps(_to_artifact_dict(record), ensure_ascii=False))
            output_handle.write("\n")
            eligible_count += 1

    print(
        f"Wrote {output_path}: read={read_count:,} eligible={eligible_count:,} "
        f"dropped_image_url={dropped_image_url:,} skipped_missing_asin={skipped_missing_asin:,} "
        f"validation_failures={validation_failures:,}",
        file=sys.stderr,
    )
    for failure in sample_failures:
        print(f"  validation_failure_sample: {failure}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the eligible product metadata artifact.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the gzipped Amazon Fashion JSONL metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSONL file to write with the eligible product metadata.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the one-time filtering step."""

    args = parse_args()
    filter_products(args.input, args.output)


if __name__ == "__main__":
    main()
