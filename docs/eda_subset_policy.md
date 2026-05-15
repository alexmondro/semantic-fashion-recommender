# Step 1 EDA: Amazon Fashion Subset Policy

## Decision

Ship the **full 77,740-product eligible pool** with this filter:

- `title` is present
- `images` is non-empty
- `features` is non-empty
- `rating_number >= 10`
- `average_rating >= 4.0`

No downsampling. The full eligible pool fits in memory at ~480 MiB combined and ships in the zip without trouble. Keeping all 77,740 products gives better category breadth for the stretch outfit-composition mode and removes the need to defend a sampling strategy. Treat `price` as optional display data because it is sparse. Use **NumPy cosine search**, not FAISS, because the embedding matrix is small enough for simple in-memory search.

## Raw Dataset Profile

Total rows: **826,108**

| Field | Present | Share |
|---|---:|---:|
| `title` | 826,050 | 99.99% |
| `features` | 463,074 | 56.05% |
| `description` | 59,289 | 7.18% |
| `images` | 826,107 | 100.00% |
| `price` | 50,249 | 6.08% |
| `average_rating` | 826,108 | 100.00% |
| `rating_number` | 826,108 | 100.00% |
| `store` | 799,267 | 96.75% |
| `parent_asin` | 826,108 | 100.00% |

## Distribution Snapshot

| Metric | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|
| `rating_number` | 4 | 10 | 24 | 44 |
| `average_rating` | 4.0 | 4.7 | 5.0 | 5.0 |
| feature count | 1 | 2 | 3 | 5 |
| description count | 0 | 0 | 0 | 1 |
| image count | 5 | 7 | 7 | 8 |

## Candidate Pools

| Candidate pool | Rows | Dataset share | Price present | Top store share | Top duplicate-title share |
|---|---:|---:|---:|---:|---:|
| `base_title_image` | 826,049 | 99.99% | 6.08% | 3.35% | 0.02% |
| `preferred_rn10_avg4_features` | 77,740 | 9.41% | 16.90% | 0.61% | 0.07% |
| `relaxed_rn5_avg4_features` | 127,571 | 15.44% | 13.29% | 0.71% | 0.05% |
| `strict_rn25_avg4_features` | 37,183 | 4.50% | 23.51% | 0.78% | 0.10% |
| `fallback_rn10_avg4_features_or_description` | 78,511 | 9.50% | 17.11% | 0.61% | 0.07% |
| `rn10_features_no_avg_floor` | 137,179 | 16.61% | 12.51% | 0.77% | 0.06% |

The preferred filter avoids the weakest products without collapsing inventory. Requiring `price` would discard too much, and relying on `description` adds little because descriptions are rare.

## Shopper Usefulness

Preferred-pool category signals:

| Category signal | Count |
|---|---:|
| women | 30,828 |
| accessories | 17,287 |
| uncategorized | 16,484 |
| men | 13,966 |
| tops | 12,922 |
| kids | 10,149 |
| dresses | 8,025 |
| bottoms | 6,795 |
| shoes | 2,391 |

Top preferred-pool stores:

| Store | Count |
|---|---:|
| Disney | 476 |
| Nike | 466 |
| GRACE KARIN | 455 |
| Bioworld | 451 |
| Unknown | 388 |
| Ekouaer | 276 |
| Verdusa | 261 |
| Outerstuff | 254 |
| Romwe | 239 |
| adidas | 209 |

Deterministic sample from the preferred pool:

- `B01F0II846` · REINDEAR Comics Movie Deadpool Logo Metal Pendant Keychain US Seller (Style #2) · rating 4.3 (24 reviews) · store: REINDEAR Mekomy
- `B0914R36BX` · Doctor Unicorn Soft Hooded Rainbow Bathrobe Sleepwear for Girls (Rainbow Galaxy Dots, 5-6 Years) · rating 4.5 (15 reviews) · store: Doctor Unicorn
- `B07CG5PHJG` · Moebao Frozen Little Girls' 2Pcs Suit Cartoon Shirt and Skirt Set,130,Blue · rating 4.1 (13 reviews) · store: Moebao
- `B06ZXQXP3H` · HIKA Women's Casual Elegant A Line Short Cap Sleeve Round Neck Dress (Medium, Yellow) · rating 4.1 (22 reviews) · store: HIKA
- `B09V1M3BNH` · SHEFIT Flex Sports Bra for Women, Medium Impact Sports Bra, Dusty Purple, 2X (2Luxe) · rating 4.2 (20 reviews) · store: SHEFIT
- `B07PYQ7233` · Lywjyb Birdgot Movie Inspired Bracelet Godmother Thank You Gift Godmother Proposal Jewelry Baptism Gift · rating 4.7 (102 reviews) · store: Lywjyb Birdgot
- `B08KZSRT9L` · CrocSee Leather Bands Compatible with Fitbit Versa 3/ Sense Fitness Smartwatch, Slim Top Grain Leather Replacement Strap for Women, Brown · rating 4.7 (16 reviews) · store: CrocSee
- `B083HGNXBH` · Gloria Vanderbilt Women's Amanda Basic Jean Short, Coral Pink, 12 · rating 4.7 (11 reviews) · store: Gloria Vanderbilt

## Artifact Size Estimate

| Corpus size | Metadata estimate | Embedding matrix estimate | Combined startup memory estimate |
|---:|---:|---:|---:|
| 77,740 | 24.3 MiB | 455.5 MiB | 479.8 MiB |

## Step 5 Decision

Use **NumPy cosine search** for the first build. The full 77,740-product corpus needs about **455.5 MiB** for a `float32` 1536-dimensional embedding matrix. This is acceptable for a local take-home microservice, keeps dependencies minimal, and avoids introducing FAISS for a corpus that fits comfortably in memory.

## Follow-On Inputs For Step 6

- Final corpus: **77,740 products** (full eligible pool, no downsampling)
- Final filter: `title` + `images` + `features` + `rating_number >= 10` + `average_rating >= 4.0`
- Retrieval text fields: `title`, `features`, `store`, selected `details` when available
- Response-safe fields: `parent_asin`, `title`, `store`, `average_rating`, `rating_number`, optional `price`, first usable image URL
- Diversity consideration for later search: cap repeated normalized titles/stores in post-filtering so recommendations do not look like color/size variants of the same item

## README Trade-Offs And Decisions

Use this section as source material for the README's "Key design decisions and trade-offs" section. The assignment specifically asks the README to explain setup, sample usage, and key decisions; these notes cover the data and retrieval decisions that should be included alongside the API instructions and sample response.

| Decision | Picked | Considered and rejected | Why it fits the assignment | Revisit at scale |
|---|---|---|---|---|
| Dataset size | Ship the full 77,740-product eligible pool | Using all 826K products; downsampling to ~25K | The recruiter said the full 826K dataset is unnecessary, but the eligible pool itself is small enough to ship in full (~480 MiB combined). Skipping the downsample step removes a defense burden and gives stretch outfit-composition mode meaningful category depth. | Move to FAISS or shard indexes if the eligible pool grows beyond what fits comfortably in memory. |
| Quality filter | Require title, image, features, `rating_number >= 10`, and `average_rating >= 4.0` | Looser `rating_number >= 5`; stricter `rating_number >= 25`; requiring `price` | This favors products with enough social proof and enough product text for semantic retrieval, without collapsing inventory variety. | Tune thresholds by category, since niche items may have fewer reviews than common apparel. |
| Price handling | Include `price` only when present | Requiring price for eligibility; hiding price entirely | Price is present on only 6.08% of rows, so requiring it would discard too much useful inventory. Keeping it optional preserves shopper value when available. | Use a fresher commerce feed or price service if this moved beyond the static metadata dataset. |
| Text used for embeddings | Build retrieval text from title, features, store, and selected details | Description-first retrieval; title-only retrieval | Descriptions exist for only 7.18% of rows. Features are much more common and provide material, closure, fit, and care signals that help natural-language matching. | Add taxonomy normalization or product-type extraction if recommendations need tighter category control. |
| Vector index | Local NumPy cosine search | FAISS; hosted vector databases | A 77,740 `float32` embedding matrix is about 455.5 MiB, which is acceptable for a local take-home microservice. NumPy keeps the grader install simple and avoids unnecessary infrastructure. | Move to FAISS or a managed vector index only when latency, memory, or corpus size requires it. |
| Result curation | Plan for diversity post-filtering by normalized title/store | Returning raw top-K cosine neighbors | Raw semantic top-K can surface near-duplicate size/color variants. Diversity filtering better matches a real shopper's expectation of useful alternatives. | Add category-aware diversification and business rules if merchandising requirements appear. |

README framing by evaluation criterion:

- **Customer acumen:** The subset filters out low-signal products, keeps images almost universally available, treats price honestly as optional, and reserves room for result diversity so shoppers do not see a wall of duplicates.
- **Creativity:** The data choice supports natural-language intent matching because product text includes features, not just keyword titles. It also leaves room for the explanation pass to cite concrete product attributes.
- **Strategic thinking:** The project deliberately spends complexity on the semantic pipeline rather than on infrastructure, model comparisons, or processing all 826K rows.
- **Clarity:** The EDA produces one explicit policy that Step 6 can implement directly: embed every product in the preferred eligibility pool.
- **Modularity:** The subset policy is independent of the embedding provider and vector index implementation, so the data-prep script, search module, and LLM modules can stay cleanly separated.

Suggested README wording:

> I intentionally did not embed the full 826K-product corpus. A streaming EDA pass found 77,740 products with a title, image, non-empty features, at least 10 ratings, and an average rating of 4.0 or higher — and the eligible pool itself was small enough to ship in full without downsampling. Price is included when available but not required because it appears in only 6.08% of records. At this size, local NumPy cosine search over precomputed `text-embedding-3-small` vectors is simpler and sufficient; FAISS or a hosted vector database would be premature for the take-home constraints.
