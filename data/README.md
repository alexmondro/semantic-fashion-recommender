# Data Artifacts

This directory is reserved for files shipped with the final zip:

- `products.jsonl` — filtered, response-safe product metadata for the 77,740 eligible products.
- `embeddings.npy` — `float32` `text-embedding-3-small` matrix aligned row-for-row with `products.jsonl`.
- `embedding_manifest.json` — reproducibility metadata for the generated embedding matrix.

The grader should not need to run embedding generation on first service startup.
