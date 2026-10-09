"""Shared preparation, resumable embeddings and dialogue for both applications."""

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
from typing import Any
from uuid import uuid4
from weakref import finalize

import numpy as np

from rag.chunking import chunk_documents, verify_chunks
from rag.config import Settings, load_settings
from rag.corpus import load_documents
from rag.evaluation import build_report, compute_metrics, load_test_cases, run_evaluation, save_results
from rag.hf_client import HuggingFaceClient
from rag.pipeline import embed_chunks, retrieve
from rag.vector_store import collection_fingerprint, index_chunks, open_collection, open_database

PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class CorpusPlan:
    path: Path
    documents: list[dict[str, str]]
    chunks: list[dict[str, Any]]
    chunk_size: int
    overlap: int
    signature: str


@dataclass(frozen=True)
class Progress:
    message: str
    completed: int = 0
    total: int = 0


def inspect_pdf(path: str | Path, chunk_size: int = 80, overlap: int = 15) -> CorpusPlan:
    """Read and validate the corpus locally, without loading a token or calling HF."""
    path = Path(path).expanduser().resolve()
    documents = load_documents(path)
    chunks = chunk_documents(documents, chunk_size, overlap)
    verify_chunks(documents, chunks, chunk_size, overlap)
    signature = _digest({"chunks": chunks, "size": chunk_size, "overlap": overlap})
    return CorpusPlan(path, documents, chunks, chunk_size, overlap, signature)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def safe_error(exc: Exception) -> str:
    """Display errors without ever echoing a Hugging Face bearer token."""
    return re.sub(r"hf_[A-Za-z0-9_-]+", "[token redacted]", str(exc))


def _atomic_write(path: Path, writer: Callable) -> None:
    """Commit a cache file only after its full content has been written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _CachedEmbeddings:
    def __init__(self, client, directory: Path, identity: dict, force: bool):
        self.client = client
        self.hits = 0
        root = directory / _digest(identity)
        root.mkdir(parents=True, exist_ok=True)
        pointer = root / "current.json"
        generation = None
        if not force:
            try:
                candidate = json.loads(pointer.read_text(encoding="utf-8"))["generation"]
                if isinstance(candidate, str) and re.fullmatch(r"[a-f0-9]{32}", candidate):
                    generation = candidate
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if generation is None:
            generation = uuid4().hex
            # A rebuild selects a fresh generation before the first batch, so
            # an interruption cannot mix old vectors with newly computed ones.
            _atomic_write(pointer, lambda stream: stream.write(
                json.dumps({"generation": generation}).encode("utf-8")
            ))
        self.directory = root / generation
        self.directory.mkdir(exist_ok=True)

    @staticmethod
    def _valid(matrix, rows):
        return (
            isinstance(matrix, np.ndarray)
            and matrix.ndim == 2 and matrix.shape[0] == rows and matrix.shape[1] > 0
            and np.isfinite(matrix).all()
            and (np.linalg.norm(matrix, axis=1) > 0).all()
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        path = self.directory / (_digest(texts) + ".npy")
        try:
            matrix = np.load(path, allow_pickle=False)
            if isinstance(matrix, np.lib.npyio.NpzFile):
                matrix.close()
            if self._valid(matrix, len(texts)):
                self.hits += len(texts)
                return matrix
        except (OSError, ValueError, EOFError, TypeError):
            pass
        matrix = np.asarray(self.client.embed(texts), dtype=np.float32)
        if not self._valid(matrix, len(texts)):
            raise ValueError("The server returned invalid embeddings.")
        _atomic_write(path, lambda stream: np.save(stream, matrix, allow_pickle=False))
        return matrix


@dataclass
class Assistant:
    plan: CorpusPlan
    client: HuggingFaceClient = field(repr=False)
    collection: Any = field(repr=False)
    revision: str | None
    embedding_seconds: float
    import_seconds: float
    cached_passages: int
    database: Any = field(repr=False)
    _cleanup: Any = field(init=False, repr=False)

    def __post_init__(self):
        # Streamlit sessions can disappear when a tab closes; release this
        # client's database reference when the assistant is garbage-collected.
        self._cleanup = finalize(self, self.database.close)

    def close(self):
        self._cleanup()

    def answer(self, question: str, history: list[dict[str, str]] | None = None, k: int = 3) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a question.")
        if len(question) > 2000:
            raise ValueError("The question must contain at most 2,000 characters.")
        question = question.strip()
        dialogue = [
            {"role": item["role"], "content": item["content"][:1600]}
            for item in (history or [])[-6:]
            if item.get("role") in ("user", "assistant")
            and isinstance(item.get("content"), str) and item["content"].strip()
        ]
        # Recent dialogue helps resolve short follow-up questions. It is only
        # search context, never documentary evidence for the generated answer.
        search_query = question
        if dialogue and len(question) < 300:
            recent = "\n".join(item["content"][:600] for item in dialogue[-2:])
            search_query += "\nConversation context:\n" + recent
        started = perf_counter()
        passages, retrieval_s = retrieve(search_query, self.collection, self.client, k=k)
        result = self.client.generate(question, passages, history=dialogue, language="fr")
        return {
            "question": question, "passages": passages, "retrieval_s": retrieval_s,
            **result, "total_s": perf_counter() - started,
        }

    def evaluate(self, output_dir: Path, k: int = 3) -> tuple[Path, Path, dict]:
        """Optional workshop evaluation; never part of a chat rerun."""
        tests = load_test_cases(self.plan.chunks)
        results = run_evaluation(tests, self.collection, self.client, k=k)
        metrics, kpis = compute_metrics(results, k, self.embedding_seconds, self.import_seconds)
        report = build_report(
            self.client.settings, self.revision, self.plan.chunk_size,
            self.plan.overlap, k, self.collection.name, self.embedding_seconds,
            self.import_seconds, results, {}, metrics,
        )
        report["embedding_cache"] = {
            "reused_passages": self.cached_passages,
            "total_passages": len(self.plan.chunks),
            "timing_note": "Embedding preparation time includes cache reads; cached vectors are not recomputed.",
        }
        json_path, csv_path = save_results(output_dir, report, metrics, secret=self.client.settings.hf_token)
        return json_path, csv_path, kpis


def prepare_assistant(
    plan: CorpusPlan,
    *,
    settings: Settings | None = None,
    cache_dir: Path = PROJECT_DIR / ".rag_cache",
    database_dir: Path = PROJECT_DIR / "base_vectorielle",
    batch_size: int = 8,
    force_rebuild: bool = False,
    progress: Callable[[Progress], None] | None = None,
    client: HuggingFaceClient | None = None,
) -> Assistant:
    """Check services, resume completed batches, then create a complete index."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    notify = progress or (lambda event: None)
    settings = settings or load_settings(PROJECT_DIR / ".env")
    client = client or HuggingFaceClient(settings)
    notify(Progress("Checking models; an idle endpoint may take several minutes to start."))
    info = client.check_models()
    revision = info["embedding_revision"]
    cached_client = _CachedEmbeddings(client, Path(cache_dir), {
        "corpus": plan.signature, "model": settings.embed_model,
        "endpoint": settings.embed_endpoint, "revision": revision,
        "prefix": "title: {title} | text: {text}", "cache_format": 1,
    }, force_rebuild)
    notify(Progress("Preparing embeddings", 0, len(plan.chunks)))

    def report_batch(completed, total):
        notify(Progress(
            f"Passages prepared: {completed}/{total} · {cached_client.hits} from cache",
            completed, total,
        ))

    vectors, embedding_seconds = embed_chunks(
        plan.chunks, cached_client, batch_size=batch_size, progress=report_batch,
    )
    notify(Progress("Saving the document index", len(plan.chunks), len(plan.chunks)))
    fingerprint = collection_fingerprint(
        plan.chunks, settings.embed_model, settings.embed_endpoint, revision,
        plan.chunk_size, plan.overlap, vectors,
    )
    database = open_database(Path(database_dir))
    try:
        collection = open_collection(database, "tp_rag_app_" + fingerprint)
        import_seconds = index_chunks(collection, plan.chunks, vectors)
        if collection.count() != len(plan.chunks):
            raise RuntimeError("The index is incomplete. Run preparation again.")
        notify(Progress("Ready to chat", len(plan.chunks), len(plan.chunks)))
        return Assistant(plan, client, collection, revision, embedding_seconds, import_seconds,
                         cached_client.hits, database)
    except BaseException:
        database.close()
        raise
