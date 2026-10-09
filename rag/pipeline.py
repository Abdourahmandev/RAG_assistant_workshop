"""Embedding, retrieval, and answer orchestration for the workshop."""

# Sequence accepts lists and other ordered chunk collections.
from collections.abc import Callable, Sequence
# perf_counter measures elapsed time without depending on wall-clock changes.
from time import perf_counter
# Any describes dictionaries whose values contain strings, IDs, numbers, etc.
from typing import Any

# NumPy combines embedding batches into one matrix for Chroma indexing.
import numpy as np

# The HTTP client calls Hugging Face; this module controls when those calls occur.
from rag.hf_client import HuggingFaceClient
# The vector-store search helper queries the already-indexed Chroma collection.
from rag.vector_store import search


def embed_chunks(
    chunks: Sequence[dict[str, Any]],
    client: HuggingFaceClient,
    batch_size: int = 8,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, float]:
    """Embed passage titles and texts in bounded batches and return elapsed time."""
    # These checks catch empty input or an invalid batch size before any API call.
    if not chunks:
        raise ValueError("Provide at least one chunk to embed.")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")

    # Include titles so embeddings carry document context as well as passage text.
    index_texts = [
        f"title: {chunk['title']} | text: {chunk['text']}" for chunk in chunks
    ]
    # Bound request size to keep payloads manageable and avoid endpoint limits.
    start = perf_counter()
    batches = []
    for offset in range(0, len(index_texts), batch_size):
        batches.append(client.embed(index_texts[offset : offset + batch_size]))
        if progress is not None:
            progress(min(offset + batch_size, len(index_texts)), len(index_texts))
    # Every response must be a 2-D matrix, have rows for its input texts, and use
    # the same embedding dimension before concatenation.
    dimensions = {batch.shape[1] for batch in batches if batch.ndim == 2}
    if (
        any(batch.ndim != 2 for batch in batches)
        or any(batch.shape[0] == 0 for batch in batches)
        or len(dimensions) != 1
        or sum(batch.shape[0] for batch in batches) != len(chunks)
    ):
        raise ValueError("Embedding batches returned inconsistent vector shapes.")

    # Stack the validated batches in original chunk order; row i still maps to
    # chunk i. Return elapsed embedding time for KPI reporting.
    vectors = np.vstack(batches)
    return vectors, perf_counter() - start


def retrieve(
    question: str,
    collection: Any,
    client: HuggingFaceClient,
    k: int = 3,
) -> tuple[list[dict[str, Any]], float]:
    """Embed a search query and retrieve its top-k passages."""
    # Validate the query and index state before making a paid/network request.
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The question is empty")
    if collection.count() == 0:
        raise ValueError("The collection is empty: run the indexing step first.")

    # The embedding model uses this task prefix to distinguish query vectors
    # from document vectors during semantic search.
    start = perf_counter()
    query_vector = client.embed(
        ["task: search result | query: " + question]
    )[0]
    # Search Chroma using the one query vector; returned hits are ranked by
    # semantic similarity and retain their original source metadata.
    hits = search(collection, query_vector, top_k=k)
    return hits, perf_counter() - start


def ask(
    question: str,
    collection: Any,
    client: HuggingFaceClient,
    k: int = 3,
) -> dict[str, Any]:
    """Retrieve evidence and generate an answer, preserving timing and usage data."""
    # Keep retrieval and generation behind one call for notebook use, while also
    # returning retrieval-specific and total timing separately.
    start = perf_counter()
    passages, retrieval_time = retrieve(question, collection, client, k)
    # Only retrieved passages are provided as context; the client instructs the
    # model to answer from this evidence and cite passage IDs.
    result = client.generate(question, passages)
    return {
        "question": question,
        "passages": passages,
        "retrieval_s": retrieval_time,
        **result,
        "total_s": perf_counter() - start,
    }
