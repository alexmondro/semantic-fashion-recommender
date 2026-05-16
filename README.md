# Semantic Fashion Recommender

A small FastAPI service that turns a natural-language fashion request into
semantic product recommendations from the Amazon Fashion metadata dataset.

The service uses an LLM to parse shopper intent, searches a shipped local
embedding index, and returns product matches with one short explanation per
result.

## What It Does

The API accepts a shopper query such as:

```text
I need comfortable sandals for a beach vacation this summer
```

It returns:

- the parsed shopping intent
- ranked product recommendations
- product details such as title, image, rating, review count, and price when available
- one sentence explaining why each product matches the request

## Quick Start

Use Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

Start the service:

```bash
uvicorn app.main:app
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

The product metadata and embeddings are already included under `data/`, so the
grader should not need to run the offline embedding script before using the
service.

## API Usage

### `POST /recommendations`

Request body:

```json
{
  "query": "comfortable beach vacation sandals for summer",
  "max_results": 3
}
```

Fields:

- `query`: natural-language shopping request, 3 to 500 characters
- `max_results`: number of recommendations to return, from 1 to 20, default `8`

Example `curl` request:

```bash
curl -X POST "http://127.0.0.1:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"query":"comfortable beach vacation sandals for summer","max_results":3}'
```

Example response shape:

```json
{
  "query": "comfortable beach vacation sandals for summer",
  "intent": {
    "original_query": "comfortable beach vacation sandals for summer",
    "occasion": "beach vacation",
    "season": "summer",
    "garment_types": ["sandals"],
    "attributes": ["comfortable"],
    "audience": null,
    "price_preference": null,
    "retrieval_text": "comfortable summer beach vacation sandals"
  },
  "results": [
    {
      "product": {
        "parent_asin": "B000TEST",
        "title": "Comfort Walk Sandal",
        "store": "Comfort Walk",
        "average_rating": 4.7,
        "rating_number": 42,
        "price": 29.99,
        "image_url": null,
        "features": ["Cushioned footbed"]
      },
      "score": 0.91,
      "rank": 1,
      "why_this_matches": "The cushioned footbed fits a comfortable summer sandal request."
    }
  ]
}
```

Errors use a stable JSON shape:

```json
{
  "error": "missing_api_key",
  "message": "Set OPENAI_API_KEY in .env before requesting recommendations."
}
```

Common error cases:

- `422`: request validation failed
- `503`: `OPENAI_API_KEY` is missing
- `404`: no products passed the recommendation filters
- `502`: OpenAI could not complete the parse, embedding, or explanation request

## Example Queries

Good demo queries:

- `comfortable beach vacation sandals for summer`
- `a wedding guest dress that feels elegant but not too formal`
- `comfortable black shoes for standing at work all day`
- `lightweight summer travel clothes that pack easily`
- `a casual outfit for a warm weekend brunch`

## How It Works

Runtime flow:

1. The shopper sends a natural-language query to `/recommendations`.
2. `gpt-5-mini` converts the query into a structured intent with occasion, season, garment types, attributes, and retrieval text.
3. `text-embedding-3-small` embeds the retrieval text.
4. A local NumPy index searches the shipped product embeddings with cosine similarity.
5. The service filters and curates results so low-quality products and near-duplicates are less likely to dominate the response.
6. `gpt-5-mini` writes one concise explanation for each returned product.

The FastAPI route stays thin. Parsing, embedding, vector search, explanation,
and pipeline orchestration live in separate modules so each piece can be
replaced without rewriting the whole service.

## Data And Embeddings

The original Amazon Fashion metadata file has 826,108 rows. The shipped product
pool has 77,740 products selected with this policy:

- title is present
- at least one image is present
- features are present
- `rating_number >= 10`
- `average_rating >= 4.0`

This keeps enough inventory breadth for useful recommendations while filtering
out products with weak product text or limited social proof.

Price is included when available, but it is not required. In the raw dataset,
price appears on only 6.08 percent of rows, so requiring it would discard too
many otherwise useful products.

Embeddings are precomputed with `text-embedding-3-small` and stored in
`data/embeddings.npy` as a normalized `float32` matrix. The matching product
metadata lives in `data/products.jsonl`, and `data/embedding_manifest.json`
records the embedding model, dimensions, token count, and artifact hashes.

The embedding run used 6,452,989 input tokens with an estimated embedding cost
of about $0.13. The grader should only pay for the query-time parse, query
embedding, and explanation calls.

## Design Decisions And Trade-Offs

| Decision | Picked | Why |
|---|---|---|
| Service interface | FastAPI endpoint | The assignment asks for a function, CLI, or API. FastAPI gives a clear microservice shape and built-in Swagger docs. |
| Chat model | `gpt-5-mini` | It is the locked project choice for structured intent parsing and short explanations. |
| Embedding model | `text-embedding-3-small` | It is inexpensive and strong enough for short product retrieval text. |
| Corpus size | Full 77,740-product eligible pool | The full eligible pool fits locally and avoids defending a random downsample. |
| Vector search | Local NumPy cosine search | The matrix is about 456 MiB, so a hosted vector database or FAISS would add complexity without clear benefit for this prototype. |
| Structured parsing | Pydantic structured outputs | This avoids regex parsing of free-text LLM output and keeps the API contract stable. |
| Price handling | Optional display field | Price is sparse in the dataset, so requiring it would hurt recall. |
| Tests | Focused smoke and module tests | The take-home rewards clarity and end-to-end behavior more than a broad production test suite. |

## What I Did Not Build

I intentionally kept the project narrow. These are out of scope for this
take-home:

- authentication, sessions, or user accounts
- personalization or recommendation history
- hosted vector databases
- Docker, Kubernetes, or deployment infrastructure
- model bake-offs across several LLMs or embedding models
- caching, rate limiting, monitoring, or observability
- a React UI
- outfit composition mode

Those choices keep the submission focused on the core semantic recommendation
workflow and make it easier for the grader to run locally.

## Next Steps With More Time

- Add outfit composition mode by splitting a query into garment slots and retrieving per slot.
- Add category-aware diversification so result sets balance product type, store, and style.
- Improve price and availability with a fresher commerce feed.
- Add a small static UI if the Swagger demo is not enough for a product walkthrough.
- Move to FAISS or a managed vector index only if corpus size, memory, or latency requires it.

## Project Layout

```text
app/
  main.py           FastAPI entrypoint
  pipeline.py       parse, embed, search, explain orchestration
  query_parser.py   structured intent parsing
  embedder.py       OpenAI embedding client
  vector_index.py   local NumPy cosine search and result curation
  explainer.py      one-sentence product explanations
  schemas.py        Pydantic API and pipeline contracts
data/
  products.jsonl
  embeddings.npy
  embedding_manifest.json
scripts/
  filter_products.py
  embed_products.py
docs/
  eda_subset_policy.md
tests/
  test_smoke.py
```

## Verification

Run the smoke test:

```bash
python -m pytest tests/test_smoke.py
```

Run the full test suite:

```bash
python -m pytest
```

The smoke test exercises the grader-facing startup path with the shipped
artifacts while stubbing OpenAI calls, so it can verify the service wiring
without spending API credits.
