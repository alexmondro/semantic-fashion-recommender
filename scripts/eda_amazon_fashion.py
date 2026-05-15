#!/usr/bin/env python3
"""Profile the Amazon Fashion metadata and lock the Step 1 subset policy.

This script intentionally uses only the Python standard library. It streams the
gzipped JSONL file, produces aggregate counts, and writes a concise Markdown note
that can be reused in the README.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EMBEDDING_DIMENSIONS = 1536
BYTES_PER_FLOAT32 = 4
RANDOM_SEED = 20260506
SAMPLE_LIMIT = 8

CATEGORY_KEYWORDS = {
    "women": ("women", "women's", "womens", "woman", "ladies", "lady"),
    "men": ("men", "men's", "mens", "man"),
    "kids": ("girls", "boys", "kids", "children", "child", "toddler", "baby"),
    "shoes": ("shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandals", "heels"),
    "accessories": (
        "bag",
        "purse",
        "wallet",
        "hat",
        "cap",
        "belt",
        "scarf",
        "sunglasses",
        "watch",
        "jewelry",
    ),
    "tops": ("shirt", "shirts", "top", "tops", "tee", "blouse", "sweater", "hoodie", "jacket"),
    "bottoms": ("pants", "jeans", "shorts", "skirt", "leggings", "trousers"),
    "dresses": ("dress", "dresses", "gown"),
}


@dataclass
class CandidatePool:
    name: str
    count: int = 0
    price_count: int = 0
    stores: Counter[str] = field(default_factory=Counter)
    categories: Counter[str] = field(default_factory=Counter)
    normalized_titles: Counter[str] = field(default_factory=Counter)
    metadata_bytes: list[int] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)


def has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"\([^)]*\)", " ", title)
    title = re.sub(r"\b(size|small|medium|large|x-large|xl|xxl|xs|s|m|l)\b", " ", title)
    title = re.sub(r"\b\d+(\.\d+)?\b", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split()[:12])


def detect_categories(title: str) -> list[str]:
    padded_title = f" {title.lower()} "
    matches = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(f" {keyword} " in padded_title for keyword in keywords):
            matches.append(category)
    return matches or ["uncategorized"]


def quantile(values: list[float], percentile: float) -> float:
    if not values:
        return 0
    index = round((len(values) - 1) * percentile)
    return values[index]


def compact_product(row: dict[str, Any]) -> dict[str, Any]:
    images = as_list(row.get("images"))
    first_image = images[0] if images and isinstance(images[0], dict) else {}
    return {
        "parent_asin": row.get("parent_asin"),
        "title": row.get("title"),
        "store": row.get("store"),
        "average_rating": row.get("average_rating"),
        "rating_number": row.get("rating_number"),
        "price": row.get("price"),
        "feature_count": len(as_list(row.get("features"))),
        "description_count": len(as_list(row.get("description"))),
        "image": first_image.get("large") or first_image.get("hi_res") or first_image.get("thumb"),
    }


def update_sample(pool: CandidatePool, row: dict[str, Any], rng: random.Random) -> None:
    compact = compact_product(row)
    if len(pool.samples) < SAMPLE_LIMIT:
        pool.samples.append(compact)
        return
    replacement_index = rng.randrange(pool.count)
    if replacement_index < SAMPLE_LIMIT:
        pool.samples[replacement_index] = compact


def candidate_matches(row: dict[str, Any]) -> dict[str, bool]:
    title_ok = has_text(row.get("title"))
    images_ok = bool(as_list(row.get("images")))
    features_ok = bool(as_list(row.get("features")))
    description_ok = bool(as_list(row.get("description")))
    rating_number = int(row.get("rating_number") or 0)
    average_rating = float(row.get("average_rating") or 0)
    base = title_ok and images_ok

    return {
        "base_title_image": base,
        "preferred_rn10_avg4_features": base and features_ok and rating_number >= 10 and average_rating >= 4.0,
        "relaxed_rn5_avg4_features": base and features_ok and rating_number >= 5 and average_rating >= 4.0,
        "strict_rn25_avg4_features": base and features_ok and rating_number >= 25 and average_rating >= 4.0,
        "fallback_rn10_avg4_features_or_description": (
            base and (features_ok or description_ok) and rating_number >= 10 and average_rating >= 4.0
        ),
        "rn10_features_no_avg_floor": base and features_ok and rating_number >= 10,
    }


def record_pool_match(pool: CandidatePool, row: dict[str, Any], rng: random.Random) -> None:
    pool.count += 1
    if row.get("price") is not None:
        pool.price_count += 1
    title = row.get("title") or ""
    pool.stores.update([row.get("store") or "Unknown"])
    pool.categories.update(detect_categories(title))
    pool.normalized_titles.update([normalize_title(title)])
    if len(pool.metadata_bytes) < 500:
        pool.metadata_bytes.append(len(json.dumps(compact_product(row), ensure_ascii=False).encode("utf-8")))
    update_sample(pool, row, rng)


def percent(part: int | float, whole: int | float) -> str:
    return f"{(part / whole * 100):.2f}%" if whole else "0.00%"


def mib(byte_count: float) -> float:
    return byte_count / (1024 * 1024)


def render_pool_table(pools: dict[str, CandidatePool], total_rows: int) -> str:
    lines = [
        "| Candidate pool | Rows | Dataset share | Price present | Top store share | Top duplicate-title share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pool in pools.values():
        top_store_count = pool.stores.most_common(1)[0][1] if pool.stores else 0
        top_title_count = pool.normalized_titles.most_common(1)[0][1] if pool.normalized_titles else 0
        lines.append(
            f"| `{pool.name}` | {pool.count:,} | {percent(pool.count, total_rows)} | "
            f"{percent(pool.price_count, pool.count)} | {percent(top_store_count, pool.count)} | "
            f"{percent(top_title_count, pool.count)} |"
        )
    return "\n".join(lines)


def render_artifact_estimates(preferred_pool: CandidatePool) -> str:
    avg_metadata_bytes = (
        sum(preferred_pool.metadata_bytes) / len(preferred_pool.metadata_bytes)
        if preferred_pool.metadata_bytes
        else 900
    )
    corpus_size = preferred_pool.count
    metadata_mib = mib(avg_metadata_bytes * corpus_size)
    embedding_mib = mib(corpus_size * EMBEDDING_DIMENSIONS * BYTES_PER_FLOAT32)
    combined_mib = metadata_mib + embedding_mib
    return "\n".join([
        "| Corpus size | Metadata estimate | Embedding matrix estimate | Combined startup memory estimate |",
        "|---:|---:|---:|---:|",
        f"| {corpus_size:,} | {metadata_mib:.1f} MiB | {embedding_mib:.1f} MiB | {combined_mib:.1f} MiB |",
    ])


def render_samples(pool: CandidatePool) -> str:
    lines = []
    for sample in pool.samples:
        title = (sample["title"] or "").replace("|", "\\|")
        lines.append(
            f"- `{sample['parent_asin']}` · {title} · rating "
            f"{sample['average_rating']} ({sample['rating_number']} reviews) · store: {sample['store']}"
        )
    return "\n".join(lines)


def write_markdown(
    output_path: Path,
    total_rows: int,
    present: Counter[str],
    rating_numbers: list[int],
    average_ratings: list[float],
    feature_lengths: list[int],
    description_lengths: list[int],
    image_counts: list[int],
    pools: dict[str, CandidatePool],
) -> None:
    preferred = pools["preferred_rn10_avg4_features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rating_numbers.sort()
    average_ratings.sort()
    feature_lengths.sort()
    description_lengths.sort()
    image_counts.sort()

    content = f"""# Step 1 EDA: Amazon Fashion Subset Policy

## Decision

Ship the **full {preferred.count:,}-product eligible pool** with this filter:

- `title` is present
- `images` is non-empty
- `features` is non-empty
- `rating_number >= 10`
- `average_rating >= 4.0`

No downsampling. The full eligible pool fits in memory and ships in the zip without trouble. Keeping every product gives better category breadth for the stretch outfit-composition mode and removes the need to defend a sampling strategy. Treat `price` as optional display data because it is sparse. Use **NumPy cosine search**, not FAISS, because the embedding matrix is small enough for simple in-memory search.

## Raw Dataset Profile

Total rows: **{total_rows:,}**

| Field | Present | Share |
|---|---:|---:|
| `title` | {present["title"]:,} | {percent(present["title"], total_rows)} |
| `features` | {present["features"]:,} | {percent(present["features"], total_rows)} |
| `description` | {present["description"]:,} | {percent(present["description"], total_rows)} |
| `images` | {present["images"]:,} | {percent(present["images"], total_rows)} |
| `price` | {present["price"]:,} | {percent(present["price"], total_rows)} |
| `average_rating` | {present["average_rating"]:,} | {percent(present["average_rating"], total_rows)} |
| `rating_number` | {present["rating_number"]:,} | {percent(present["rating_number"], total_rows)} |
| `store` | {present["store"]:,} | {percent(present["store"], total_rows)} |
| `parent_asin` | {present["parent_asin"]:,} | {percent(present["parent_asin"], total_rows)} |

## Distribution Snapshot

| Metric | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|
| `rating_number` | {quantile(rating_numbers, 0.50):.0f} | {quantile(rating_numbers, 0.75):.0f} | {quantile(rating_numbers, 0.90):.0f} | {quantile(rating_numbers, 0.95):.0f} |
| `average_rating` | {quantile(average_ratings, 0.50):.1f} | {quantile(average_ratings, 0.75):.1f} | {quantile(average_ratings, 0.90):.1f} | {quantile(average_ratings, 0.95):.1f} |
| feature count | {quantile(feature_lengths, 0.50):.0f} | {quantile(feature_lengths, 0.75):.0f} | {quantile(feature_lengths, 0.90):.0f} | {quantile(feature_lengths, 0.95):.0f} |
| description count | {quantile(description_lengths, 0.50):.0f} | {quantile(description_lengths, 0.75):.0f} | {quantile(description_lengths, 0.90):.0f} | {quantile(description_lengths, 0.95):.0f} |
| image count | {quantile(image_counts, 0.50):.0f} | {quantile(image_counts, 0.75):.0f} | {quantile(image_counts, 0.90):.0f} | {quantile(image_counts, 0.95):.0f} |

## Candidate Pools

{render_pool_table(pools, total_rows)}

The preferred filter avoids the weakest products without collapsing inventory. Requiring `price` would discard too much, and relying on `description` adds little because descriptions are rare.

## Shopper Usefulness

Preferred-pool category signals:

| Category signal | Count |
|---|---:|
"""
    for category, count in preferred.categories.most_common():
        content += f"| {category} | {count:,} |\n"

    content += f"""
Top preferred-pool stores:

| Store | Count |
|---|---:|
"""
    for store, count in preferred.stores.most_common(10):
        safe_store = str(store).replace("|", "\\|")
        content += f"| {safe_store} | {count:,} |\n"

    content += f"""
Deterministic sample from the preferred pool:

{render_samples(preferred)}

## Artifact Size Estimate

{render_artifact_estimates(preferred)}

## Step 5 Decision

Use **NumPy cosine search** for the first build. The full {preferred.count:,}-product corpus needs about **{mib(preferred.count * EMBEDDING_DIMENSIONS * BYTES_PER_FLOAT32):.1f} MiB** for a `float32` 1536-dimensional embedding matrix. This is acceptable for a local take-home microservice, keeps dependencies minimal, and avoids introducing FAISS for a corpus that fits comfortably in memory.

## Follow-On Inputs For Step 6

- Final corpus: **{preferred.count:,} products** (full eligible pool, no downsampling)
- Final filter: `title` + `images` + `features` + `rating_number >= 10` + `average_rating >= 4.0`
- Retrieval text fields: `title`, `features`, `store`, selected `details` when available
- Response-safe fields: `parent_asin`, `title`, `store`, `average_rating`, `rating_number`, optional `price`, first usable image URL
- Diversity consideration for later search: cap repeated normalized titles/stores in post-filtering so recommendations do not look like color/size variants of the same item
"""
    output_path.write_text(content, encoding="utf-8")


def profile_dataset(input_path: Path, output_path: Path) -> None:
    rng = random.Random(RANDOM_SEED)
    total_rows = 0
    present: Counter[str] = Counter()
    rating_numbers: list[int] = []
    average_ratings: list[float] = []
    feature_lengths: list[int] = []
    description_lengths: list[int] = []
    image_counts: list[int] = []
    pools = {
        name: CandidatePool(name=name)
        for name in (
            "base_title_image",
            "preferred_rn10_avg4_features",
            "relaxed_rn5_avg4_features",
            "strict_rn25_avg4_features",
            "fallback_rn10_avg4_features_or_description",
            "rn10_features_no_avg_floor",
        )
    }

    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            total_rows += 1
            row = json.loads(line)
            title = row.get("title")
            features = as_list(row.get("features"))
            description = as_list(row.get("description"))
            images = as_list(row.get("images"))

            if has_text(title):
                present["title"] += 1
            if features:
                present["features"] += 1
            if description:
                present["description"] += 1
            if images:
                present["images"] += 1
            if row.get("price") is not None:
                present["price"] += 1
            if row.get("average_rating") is not None:
                present["average_rating"] += 1
                average_ratings.append(float(row["average_rating"]))
            if row.get("rating_number") is not None:
                present["rating_number"] += 1
                rating_numbers.append(int(row["rating_number"]))
            if row.get("store"):
                present["store"] += 1
            if row.get("parent_asin"):
                present["parent_asin"] += 1

            feature_lengths.append(len(features))
            description_lengths.append(len(description))
            image_counts.append(len(images))

            for name, matched in candidate_matches(row).items():
                if matched:
                    record_pool_match(pools[name], row, rng)

    write_markdown(
        output_path=output_path,
        total_rows=total_rows,
        present=present,
        rating_numbers=rating_numbers,
        average_ratings=average_ratings,
        feature_lengths=feature_lengths,
        description_lengths=description_lengths,
        image_counts=image_counts,
        pools=pools,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 1 EDA for Amazon Fashion metadata.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("meta_Amazon_Fashion.jsonl.gz"),
        help="Path to the gzipped Amazon Fashion JSONL metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/eda_subset_policy.md"),
        help="Markdown file to write with the EDA decision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_dataset(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
