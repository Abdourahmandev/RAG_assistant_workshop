"""Persistent ChromaDB helpers for indexing and retrieving workshop passages."""

# hashlib creates fingerprints that identify the exact index inputs.
import hashlib
# json serializes fingerprint inputs in a stable, deterministic order.
import json
# Path keeps database locations portable across operating systems.
from pathlib import Path
# perf_counter measures elapsed time for database import operations.
from time import perf_counter
# Any allows these helpers to work with Chroma collection/client objects without
# tying every function to one concrete ChromaDB implementation class.
from typing import Any

# ChromaDB stores vectors and metadata on disk and searches by vector similarity.
import chromadb
# NumPy validates and converts embedding arrays and computes vector norms.
import numpy as np
# Settings configures the Chroma client; telemetry is disabled for the local lab.
from chromadb.config import Settings


def collection_fingerprint(
    chunks: list[dict[str, Any]],
    model: str,
    endpoint: str,
    revision: str | None,
    size: int,
    overlap: int,
    vectors: np.ndarray,
) -> str:
    """Return a stable digest for the content, embedding deployment, and vectors."""
    # A collection should be reused only when its passages and the process used
    # to produce their vectors are unchanged. Include settings that affect either.
    signature = {
        "chunks": chunks,
        "model": model,
        "backend": "huggingface-tei",
        "endpoint": endpoint,
        "model_revision": revision,
        "size": size,
        "overlap": overlap,
        "document_prefix": "title: {title} | text: {text}",
        # Hash the actual vector bytes as a guard against changed output even if
        # a model identifier and deployment revision remain the same.
        "vectors_sha256": hashlib.sha256(np.asarray(vectors).tobytes()).hexdigest(),
    }
    # Sorted JSON keys make equivalent dictionaries serialize identically.
    serialized = json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    # A short digest is enough for a readable Chroma collection name.
    return hashlib.sha256(serialized).hexdigest()[:16]


def open_database(db_path: Path):
    """Open a persistent ChromaDB database at the requested directory."""
    # PersistentClient stores state on disk so students can close and reopen the
    # notebook without downloading/re-indexing the vectors every time.
    return chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )


def open_collection(client: Any, name: str):
    """Open or create a persistent cosine-similarity collection without clearing it."""
    # get_or_create makes reruns safe: it reuses this named collection instead of
    # deleting previous data. Vectors are supplied by our HF client below.
    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def index_chunks(
    collection: Any,
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
) -> float:
    """Upsert passages and supplied embeddings, returning the import duration."""
    # Chroma expects ordinary Python floats. Normalize the dtype before validating.
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("Embeddings must be a valid two-dimensional numeric array.") from None

    # Refuse a partial, malformed, NaN, infinite, or zero-vector index up front;
    # a bad vector can otherwise make retrieval unreliable or fail inside Chroma.
    if (
        not chunks
        or matrix.ndim != 2
        or matrix.shape[0] != len(chunks)
        or matrix.shape[1] == 0
        or not np.isfinite(matrix).all()
        or not (np.linalg.norm(matrix, axis=1) > 0).all()
    ):
        raise ValueError("Provide one finite, nonzero embedding for every chunk.")

    # Chroma requires one unique key per record; upsert uses these IDs to replace
    # records with the same ID rather than creating duplicates.
    ids = [chunk["id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Chunk IDs must be unique.")

    # Store text, vectors, and searchable/displayable metadata together.
    start = perf_counter()
    collection.upsert(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=matrix.tolist(),
        # Keep source page, title, and word offsets separate from the text so
        # callers can inspect where a hit came from.
        metadatas=[
            {
                "source": chunk["source"],
                "title": chunk["title"],
                "start": chunk["start"],
                "end": chunk["end"],
            }
            for chunk in chunks
        ],
    )
    # Return just the Chroma import duration for workshop KPI reporting.
    return perf_counter() - start


def search(
    collection: Any,
    query_vector: np.ndarray,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return the nearest passages with Chroma cosine distances as similarities."""
    # k controls result count, so reject invalid values before querying Chroma.
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    # A search in an empty collection cannot return evidence.
    if collection.count() == 0:
        raise ValueError("The collection is empty: index chunks before searching.")

    # Convert one question embedding to a finite nonzero float32 vector, matching
    # the document-vector representation used during indexing.
    try:
        vector = np.asarray(query_vector, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("The query embedding must be a valid numeric vector.") from None
    if (
        vector.ndim != 1
        or vector.size == 0
        or not np.isfinite(vector).all()
        or np.linalg.norm(vector) == 0
    ):
        raise ValueError("The query embedding must be a finite nonzero vector.")

    # Chroma returns cosine distance plus the stored text/metadata. Ask only for
    # fields used below and cap n_results to the number of indexed records.
    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Convert Chroma's nested response into simple passage dictionaries used by
    # pipeline.py, evaluation.py, and the notebook.
    hits = []
    for ident, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "id": ident,
                "text": text,
                "source": metadata["source"],
                "title": metadata["title"],
                # Chroma reports cosine distance (lower is closer); convert it
                # to cosine similarity (higher is closer) for easier reading.
                "similarity": 1.0 - distance,
            }
        )
    return hits
