# OpenAI Take-Home — Semantic Fashion Recommender

## Project context
Five-business-day take-home assignment for an OpenAI role. Build a semantic recommendation microservice over the Amazon Fashion metadata dataset (826K products, shipped as `meta_Amazon_Fashion.jsonl.gz` in this directory). Submission is a zip uploaded to Ashby and must run out-of-the-box for the grader with only an API key added.

**Evaluated on:** customer acumen, creativity, strategic thinking, code clarity, modularity. Reviewed first by an internal tool (CodexCLI), then by a human.

**Background docs in this directory:**
- `assignment-brief.md` — full email from the recruiter
- `Take Home Project_ Data Set.pdf` — dataset schema (note: PDF lists `bought_together` as `list` and `price` as `float`, but in the JSONL both are frequently `null` — code defensively)

---

## Locked scope

### In scope (required, build first)
- FastAPI HTTP service with a recommendation endpoint
- Natural-language query → structured intent (LLM call with structured outputs)
- Semantic retrieval over a pre-embedded product subset
- Per-result one-sentence "why this matches" explanation (LLM call)
- Architecture diagram (PDF or JPEG)
- README covering install, sample query + output, design trade-offs
- Pre-computed embeddings shipped in the zip — grader must not pay to embed on first run

### Stretch (only after core ships)
- **Outfit composition mode** — LLM splits a query into garment slots (top, bottom, footwear, etc.), retrieves per slot, returns a composed outfit. The base architecture must leave this possible without rework.
- **Lightweight React UI** on top of the FastAPI service. If added, it must be pre-built and shipped as static assets — no `npm install` step for the grader. Skip if FastAPI's `/docs` Swagger UI is sufficient.

### Out of scope (do not build — refuse drift toward these)
- Auth, sessions, user accounts
- Multi-turn conversation
- Image / CLIP / multimodal embeddings
- Hosted vector DBs (Pinecone, Weaviate, Chroma cloud, etc.)
- Docker, k8s, CI/CD, any deploy infrastructure
- Comprehensive test suite — one smoke test is the bar
- Personalization or recommendation history
- A/B comparisons of multiple embedding or chat models
- Caching, rate limiting, observability, monitoring
- Using all 826K products (recruiter explicitly said this isn't required)

---

## Architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Service shape | FastAPI + Uvicorn | Matches "microservice" framing; auto-generated `/docs` Swagger UI is the built-in interactive demo |
| Embedding model | `text-embedding-3-small` (1536 dim) | ~$0.02/M tokens, strong quality for short product text |
| Chat model | **Locked: `gpt-5-mini`** | Verified 2026-05-06 against official OpenAI docs. Mini tier handles structured JSON parsing and short explanations cheaply and quickly |
| Structured outputs | **Locked: use `gpt-5-mini` Structured Outputs** on the Responses API. Prefer the current Python SDK's parsed structured-output helper with a Pydantic intent model; fallback to manual `json_schema` response format only if SDK helper friction appears. If account/model access blocks `gpt-5-mini`, use `gpt-4o-mini` with the same schema as an emergency fallback. | Avoids regex on free-text LLM responses; fallback preserves schema adherence without changing architecture |
| Vector index | **Locked: local NumPy cosine search** | Ships in the zip; no external infra; EDA confirms 77,740 × 1536 `float32` embeddings (~456 MiB) are acceptable in memory |
| Dataset subset | **Locked: ship the full eligible pool of 77,740 products with `title`, non-empty `images`, non-empty `features`, `rating_number >= 10`, and `average_rating >= 4.0`. No downsampling.** | The eligible pool fits in memory (~480 MiB combined) and ships in the zip without trouble. Keeping all 77,740 products gives better category breadth for stretch outfit composition and removes the need to defend a sampling strategy |
| LLM roles | (a) query → structured intent; (b) per-result one-line explanation | Highest leverage for the "customer acumen" signal |

**Model verification note:** Task 2 is closed. Do not reopen model selection unless `gpt-5-mini` access fails during implementation. The implementation target is `gpt-5-mini` for both structured intent parsing and short per-result explanations, with `text-embedding-3-small` for embeddings.

---

## Design principles (what graders are evaluating)

The recruiter named five dimensions. Internalize these — they determine **how** each board item is built, not what gets built.

### Customer acumen — built for a real shopper
- **Looks like:** rating/quality filter so junk doesn't surface; result diversity (no 10 near-duplicates); graceful empty-result handling; response shape gives a shopper enough to decide (title, image, price-if-available, explanation).
- **Anti-pattern:** raw `top_k` cosine results with no curation, dumped as-is.

### Creativity — value beyond the literal brief
- **Looks like:** structured intent extraction (occasion, season, vibe) before retrieval; per-result LLM explanation; outfit composition mode (stretch); a curated set of example queries showcased in the README.
- **Anti-pattern:** novelty that doesn't serve retrieval (e.g., sentiment scoring that goes nowhere).

### Strategic thinking — deliberate trade-offs, defended in writing
- **Looks like:** README explicitly cuts scope and explains why; a "what I didn't build" section; a "next steps with more time" list; cost-awareness against the $30 budget.
- **Anti-pattern:** "I built everything I could think of in 5 days."

### Clarity — readable top-to-bottom
- **Looks like:** self-documenting names; type hints throughout; Pydantic models on every API surface; no function over ~30 lines; file names match contents; no `utils.py` dumping ground.
- **Anti-pattern:** dense one-liners, abbreviation-heavy variables, mixed concerns per file.

### Modularity — components are swappable
- **Looks like:** embedding provider, vector index, and LLM client each behind a thin interface; pipeline stages (`parse → embed → search → explain`) are independent modules with explicit I/O contracts; one config module read at startup.
- **Anti-pattern:** OpenAI calls inline in the FastAPI route; NumPy operations mixed into explanation logic; a single `main.py` doing everything.

The README's "Design decisions & trade-offs" section is the artifact that proves all five. For each non-obvious choice, write: **what you picked, what you considered and rejected, and (optional) what you'd revisit at scale.**

---

## Data pipeline

```
Raw JSONL (826K products)
   │
   ▼  offline, one-time: filter
Eligible subset (77,740, JSON)
   │
   ▼  offline: embed via text-embedding-3-small
Embeddings + metadata  ──► shipped in the zip
   │
   ▼  loaded on service start
In-memory vector index

Runtime per query:
  user query
    → gpt-5-mini structured parse → {occasion, garment_types, attributes, ...}
    → embed intent string
    → top-K cosine search
    → gpt-5-mini explanation pass (one short sentence per result)
    → JSON response
```

---

## Tech stack
- Python 3.11+
- FastAPI, Uvicorn
- OpenAI Python SDK (latest)
- NumPy (FAISS only if needed)
- Pydantic for request/response schemas
- python-dotenv for `OPENAI_API_KEY`

---

## Hard constraints
- Single zip submission via Ashby
- Grader runs: `pip install -r requirements.txt` → set `OPENAI_API_KEY` in `.env` → `uvicorn app.main:app` (or equivalent). No other steps.
- Pre-computed embeddings ship in the repo so first run does not require embedding 77,740 products
- Stay well within the $30 OpenAI credit (real cost should be < $5 across all dev)
- 5 business days from email receipt (received 2026-05-05)

---

## Scope-creep guard
Before adding anything not in **In scope** or **Stretch** above, stop and ask. The default answer to "should we also add X?" is **no**. Common temptations to refuse:
- "Let's add a small auth layer" — no
- "Let's compare two embedding models" — no, document the choice in the README
- "Let's containerize it" — no, plain Python only
- "Let's add proper test coverage" — no, one smoke test is the target
- "Let's add a caching layer for repeated queries" — no
- "Let's process more of the dataset" — no, justify the subset in the README

---

## Project board

This is the project's pseudo-Jira and the source of truth for what's left and what's in flight.

**Status convention:**
- `[ ]` not started
- `[~]` in progress
- `[x]` done

**Maintenance rule (applies to Claude and to the user):** flip an item's marker the moment its state changes. Move to `[~]` when work begins on it; move to `[x]` the moment it's complete. Do not batch updates at end of session — update inline as work happens.

Items 1 and 4 can run in parallel. Everything else is roughly sequential.

### Git history / commit hygiene
- Keep the working tree clean between finished board items whenever practical.
- Make commits in project-board order. If one work session completes multiple board items, split the result into logical commits that map to those items instead of making one broad implementation commit.
- Commit generated artifacts with the script or source change that produced them when the grader needs those artifacts out-of-the-box. Example: `data/products.jsonl` belongs with the filter script; `data/embeddings.npy` and `data/embedding_manifest.json` belong with the embedding script.
- Keep local assistant/editor state, caches, virtualenvs, raw source data, and submission zips out of commits unless explicitly requested.
- Before each commit, review `git status --short` and `git diff --cached --stat` to confirm the staged files match the intended scope.
- Use concise imperative commit subjects that name the shipped behavior, not the tool that produced it.
- Do not rewrite existing git history after commits have been created unless the user explicitly asks for a history rewrite.

### Pre-build (decisions / unknowns)
- [x] 1. EDA on the 826K dataset — confirm filter hypothesis, lock corpus size
- [x] 2. Verify `gpt-5-mini` supports structured outputs in the current SDK; pick a fallback if not
- [x] 3. Lock directory structure (including thin interfaces for embedder, vector index, LLM client — modularity hinge)
- [x] 4. Lock API contract (request, response, error schemas)
- [x] 5. NumPy vs. FAISS — falls out of EDA result

### Build — data prep (offline scripts, run once)
- [x] 6. Filter script → eligible subset JSON (no downsampling)
- [x] 7. Embedding script → run once, save artifacts shipped in the repo

### Build — service
- [x] 8. Vector search module (load embeddings, cosine top-K, plus diversity / quality post-filtering for customer acumen)
- [x] 9. Query-parsing module (gpt-5-mini structured output → intent object)
- [x] 10. Explanation module (gpt-5-mini → one-line "why this matches")
- [x] 11. FastAPI app wiring it all together
- [x] 12. One end-to-end smoke test

### Wrap
- [x] 13. Architecture diagram (PDF or JPEG)
- [x] 14. README (install, curated sample queries + outputs, design trade-offs, what was cut and why, "next steps with more time")
- [ ] 15. Out-of-the-box verification on a clean venv
- [ ] 16. Zip + submit via Ashby

### Stretch (only if 1–16 done with time left)
- [ ] 17. Outfit composition mode
- [ ] 18. React UI on top

---

## Working preferences
- Be terse in responses; this user wants tight communication
- Confirm scope changes before implementing them
- Track non-trivial work with TodoWrite
- When in doubt about a model API surface (`gpt-5-mini`, structured outputs, etc.), verify against current OpenAI docs before writing code rather than guessing
