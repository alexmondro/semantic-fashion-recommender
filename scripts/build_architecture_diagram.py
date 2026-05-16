"""Render the architecture diagram for docs/architecture.pdf.

Local-only build tool. NOT a runtime dependency of the FastAPI service
and intentionally absent from requirements.txt. The committed PDF is
the grader-facing deliverable; this script exists for reproducibility.

Prerequisites (author machine only):
    pip install graphviz
    brew install graphviz   # or: apt-get install graphviz

Run from the repo root:
    python scripts/build_architecture_diagram.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import graphviz

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs"
OUTPUT_STEM = "architecture"

# Edge color used everywhere an OpenAI API boundary is crossed.
OPENAI_EDGE_COLOR = "#c2410c"
OPENAI_NODE_FILL = "#fed7aa"

ARTIFACT_FILL = "#e0e7ff"
OFFLINE_FILL = "#f3f4f6"
STARTUP_FILL = "#dbeafe"
RUNTIME_FILL = "#dcfce7"


def main() -> int:
    if shutil.which("dot") is None:
        sys.stderr.write(
            "error: graphviz 'dot' binary not found on PATH.\n"
            "       install with `brew install graphviz` (macOS) or "
            "`apt-get install graphviz` (linux), then rerun.\n"
        )
        return 1

    dot = graphviz.Digraph("architecture", format="pdf")
    dot.attr(rankdir="TB", splines="spline", nodesep="0.4", ranksep="0.55")
    dot.attr("node", fontname="Helvetica", fontsize="11", shape="box", style="rounded,filled")
    dot.attr("edge", fontname="Helvetica", fontsize="9")
    dot.attr(label="Semantic Fashion Recommender — Architecture", labelloc="t", fontsize="16")

    _add_openai_cluster(dot)
    _add_offline_band(dot)
    _add_artifacts_band(dot)
    _add_startup_band(dot)
    _add_runtime_band(dot)
    _add_cross_band_edges(dot)
    _add_legend(dot)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered = dot.render(
        filename=OUTPUT_STEM,
        directory=str(OUTPUT_DIR),
        cleanup=True,
    )
    print(f"wrote {rendered}")
    return 0


def _add_openai_cluster(dot: graphviz.Digraph) -> None:
    with dot.subgraph(name="cluster_openai") as c:
        c.attr(label="OpenAI API (external)", style="rounded,filled", fillcolor="#fff7ed", fontsize="12")
        c.node(
            "oai_responses",
            "Responses API\ngpt-5-mini\n(fallback: gpt-4o-mini)",
            fillcolor=OPENAI_NODE_FILL,
        )
        c.node(
            "oai_embeddings",
            "Embeddings API\ntext-embedding-3-small\n(1536 dim, L2-normalized)",
            fillcolor=OPENAI_NODE_FILL,
        )


def _add_offline_band(dot: graphviz.Digraph) -> None:
    with dot.subgraph(name="cluster_offline") as c:
        c.attr(label="Offline data prep — one-time, scripts ship in repo", style="rounded,filled", fillcolor=OFFLINE_FILL, fontsize="12")
        c.node("raw_dataset", "meta_Amazon_Fashion.jsonl.gz\n(826K products, raw)", shape="cylinder", fillcolor="#e5e7eb")
        c.node("filter_script", "scripts/filter_products.py\ntitle + images + features +\nrating_number ≥ 10 + avg ≥ 4.0", fillcolor="white")
        c.node("embed_script", "scripts/embed_products.py\nbatched embed via OpenAI", fillcolor="white")
        c.edge("raw_dataset", "filter_script")
        c.edge("filter_script", "embed_script", label=" products.jsonl ")


def _add_artifacts_band(dot: graphviz.Digraph) -> None:
    with dot.subgraph(name="cluster_artifacts") as c:
        c.attr(label="Shipped artifacts (committed to repo, zipped for grader)", style="rounded,filled", fillcolor="#eef2ff", fontsize="12")
        c.node("art_products", "data/products.jsonl\n77,740 eligible products", shape="note", fillcolor=ARTIFACT_FILL)
        c.node("art_embeddings", "data/embeddings.npy\n77,740 × 1536 float32, ~456 MiB", shape="note", fillcolor=ARTIFACT_FILL)
        c.node("art_manifest", "data/embedding_manifest.json\nmodel + hash + cost", shape="note", fillcolor=ARTIFACT_FILL)


def _add_startup_band(dot: graphviz.Digraph) -> None:
    with dot.subgraph(name="cluster_startup") as c:
        c.attr(label="Service startup — FastAPI lifespan", style="rounded,filled", fillcolor=STARTUP_FILL, fontsize="12")
        c.node("config", "app/config.py\nload_settings() — .env, paths, model IDs", fillcolor="white")
        c.node("vector_index", "app/vector_index.py::NumpyVectorIndex\nloaded once into memory at startup", fillcolor="white")


def _add_runtime_band(dot: graphviz.Digraph) -> None:
    with dot.subgraph(name="cluster_runtime") as c:
        c.attr(label="Per-request flow — POST /recommendations", style="rounded,filled", fillcolor=RUNTIME_FILL, fontsize="12")
        c.node("client", "Shopper client\n(curl / Swagger /docs)", shape="oval", fillcolor="white")
        c.node("api", "app/main.py\nFastAPI route handler", fillcolor="white")
        c.node("pipeline", "app/pipeline.py\nRecommendationPipeline.recommend()", fillcolor="white")
        c.node("parser", "app/query_parser.py\nQueryParser.parse()", fillcolor="white")
        c.node("llm_client_parse", "app/llm_client.py\nOpenAiLlmClient.parse_structured()", fillcolor="white")
        c.node("embedder", "app/embedder.py\nOpenAiEmbedder.embed_text()", fillcolor="white")
        c.node("search", "app/vector_index.py\ncosine top-K → dedup → store cap", fillcolor="white")
        c.node("filter_stage", "app/pipeline.py::_quality_filter\nrating thresholds", fillcolor="white")
        c.node("explainer", "app/explainer.py\nProductExplainer.explain()", fillcolor="white")
        c.node("llm_client_explain", "app/llm_client.py\nOpenAiLlmClient.parse_structured()", fillcolor="white")

        c.edge("client", "api", label=" RecommendationRequest ")
        c.edge("api", "pipeline", label=" RecommendationRequest ")
        c.edge("pipeline", "parser", label=" query: str ")
        c.edge("parser", "llm_client_parse")
        c.edge("llm_client_parse", "pipeline", label=" ProductIntent ", style="dashed", constraint="false")
        c.edge("pipeline", "embedder", label=" intent.retrieval_text ")
        c.edge("embedder", "pipeline", label=" np.ndarray[1536] ", style="dashed", constraint="false")
        c.edge("pipeline", "search", label=" query_embedding ")
        c.edge("search", "filter_stage", label=" list[RankedProduct] ")
        c.edge("filter_stage", "explainer", label=" filtered list[RankedProduct] ")
        c.edge("explainer", "llm_client_explain")
        c.edge("llm_client_explain", "explainer", label=" list[ExplainedProduct] ", style="dashed", constraint="false")
        c.edge("explainer", "pipeline", label=" list[ExplainedProduct] ", style="dashed", constraint="false")
        c.edge("pipeline", "api", label=" RecommendationResponse ", style="dashed", constraint="false")
        c.edge("api", "client", label=" RecommendationResponse / ApiError ", style="dashed", constraint="false")


def _add_cross_band_edges(dot: graphviz.Digraph) -> None:
    openai_edge = {"color": OPENAI_EDGE_COLOR, "penwidth": "2", "fontcolor": OPENAI_EDGE_COLOR}

    # Offline embed crosses to OpenAI once per artifact build.
    dot.edge("embed_script", "oai_embeddings", label=" offline batch embed ", **openai_edge)
    dot.edge("embed_script", "art_embeddings", label=" .npy ")
    dot.edge("embed_script", "art_manifest", label=" manifest ")
    dot.edge("filter_script", "art_products", label=" .jsonl ")

    # Startup wiring.
    dot.edge("art_products", "vector_index", label=" load ")
    dot.edge("art_embeddings", "vector_index", label=" load ")
    dot.edge("config", "vector_index", label=" paths ", style="dotted")

    # Runtime crosses to OpenAI — three external phases per request.
    dot.edge("llm_client_parse", "oai_responses", label=" 1. parse intent ", **openai_edge)
    dot.edge("embedder", "oai_embeddings", label=" 2. embed query ", **openai_edge)
    dot.edge("llm_client_explain", "oai_responses", label=" 3. explain results ", **openai_edge)

    # In-memory index used by the search stage.
    dot.edge("vector_index", "search", label=" in-memory ", style="dotted")


def _add_legend(dot: graphviz.Digraph) -> None:
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", style="rounded,filled", fillcolor="white", fontsize="11")
        c.node("legend_openai", "OpenAI API call", fillcolor=OPENAI_NODE_FILL)
        c.node("legend_artifact", "Shipped artifact", shape="note", fillcolor=ARTIFACT_FILL)
        c.node("legend_module", "App module", fillcolor="white")
        c.edge("legend_module", "legend_openai", label="external", color=OPENAI_EDGE_COLOR, penwidth="2", fontcolor=OPENAI_EDGE_COLOR)
        c.edge("legend_module", "legend_artifact", label="data", style="dotted")


if __name__ == "__main__":
    sys.exit(main())
