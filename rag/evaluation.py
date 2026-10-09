"""Evaluation cases, KPI calculations, and secret-safe workshop reports."""

# datetime produces unique UTC report names; timezone avoids local-time ambiguity.
from datetime import datetime, timezone
# importlib.metadata reads installed package versions for reproducibility reports.
from importlib.metadata import version
# json serializes final report data.
import json
# Path handles the output directory and generated filenames portably.
from pathlib import Path
# re extracts citation IDs from generated answers.
import re
# sys records the Python interpreter version used for an experiment.
import sys
# perf_counter measures execution durations.
from time import perf_counter
# Any supports the structured dictionaries passed between notebook and functions.
from typing import Any

# NumPy averages manual binary review scores.
import numpy as np
# pandas builds readable tables and computes summary statistics.
import pandas as pd

# Settings lets the report record safe configuration such as model names and URLs;
# the token is intentionally omitted.
from rag.config import Settings
# Evaluation uses the same retrieval implementation as interactive RAG questions.
from rag.pipeline import retrieve


# Standard exact fallback answer. Metrics can count exact abstention separately.
ABSTENTION = "I cannot find this information in the documents."

# These questions match the 18-page Revenu Quebec privacy policy supplied as
# A-6.002.pdf. Update the questions and evidence when replacing that document.
# The five supported questions have evidence tied to specific PDF pages; the two
# unsupported questions check whether the model abstains instead of inventing facts.
_TEST_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "T1",
        "question": "What does the privacy policy define as confidential information?",
        "expected": (
            "Information that concerns an individual or a business and allows "
            "that individual or business to be identified."
        ),
        "evidence": (
            "Les renseignements confidentiels sont ceux qui concernent un individu "
            "ou une entreprise et permettent de l’identifier"
        ),
        "source": "a-6_002_p0004",
    },
    {
        "id": "T2",
        "question": "How often do staff renew their confidentiality commitments?",
        "expected": (
            "Staff renew their confidentiality commitments every year."
        ),
        "evidence": (
            "Celui-ci signe des engagements de confidentialité et les renouvelle chaque année"
        ),
        "source": "a-6_002_p0005",
    },
    {
        "id": "T3",
        "question": "From whom does Revenu Québec collect confidential information directly?",
        "expected": "It collects confidential information directly from the taxpayer.",
        "evidence": (
            "Nous recueillons les renseignements confidentiels directement auprès de vous"
        ),
        "source": "a-6_002_p0006",
    },
    {
        "id": "T4",
        "question": "How does Revenu Québec handle unsolicited information?",
        "expected": (
            "It handles unsolicited information in accordance with strict "
            "confidentiality principles."
        ),
        "evidence": (
            "Lorsque nous recevons des informations non sollicitées, nous les "
            "traitons conformément aux principes rigoureux de confidentialité"
        ),
        "source": "a-6_002_p0007",
    },
    {
        "id": "T5",
        "question": "What is the purpose of the automated system used in recruitment?",
        "expected": (
            "It performs an initial screening of applications to identify "
            "candidates matching the required profile."
        ),
        "evidence": "ce système automatisé permet un premier tri des candidatures",
        "source": "a-6_002_p0008",
    },
    {
        "id": "T6",
        "question": "What is the Wi-Fi password for the tax administration?",
        "expected": ABSTENTION,
        "evidence": None,
        "source": None,
    },
    {
        "id": "T7",
        "question": "What meal will the cafeteria serve tomorrow?",
        "expected": ABSTENTION,
        "evidence": None,
        "source": None,
    },
)

# These categories keep human review focused on factual correctness, evidence use,
# and safe abstention, rather than treating automatic retrieval metrics as accuracy.
_REVIEW_CRITERIA = (
    "correct_and_complete",
    "faithful_to_sources",
    "appropriate_abstention",
)
# Field names that must never appear in saved evaluation data.
_SECRET_FIELD_NAMES = {
    "token",
    "hf_token",
    "access_token",
    "authorization",
    "api_key",
    "apikey",
}


def load_test_cases(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the seven workshop questions with evidence passage identifiers."""
    cases = []
    # Copy each template so callers can safely enrich or edit a returned case.
    for test_case in _TEST_CASES:
        case = dict(test_case)
        # Map the annotated source phrase to actual chunk IDs after chunking; IDs
        # depend on chunk size and overlap, so they should not be hard-coded here.
        case["relevant_ids"] = [
            chunk["id"]
            for chunk in chunks
            if case["evidence"] is not None
            and chunk["source"] == case["source"]
            and case["evidence"] in chunk["text"]
        ]
        # A missing expected phrase usually means the source PDF changed or the
        # evidence annotation no longer matches, so stop before evaluating KPIs.
        if case["evidence"] is not None and not case["relevant_ids"]:
            raise ValueError(f"Evidence not found in a chunk: {case['id']}")
        cases.append(case)
    return cases


def compute_metrics(
    results: list[dict[str, Any]],
    k: int,
    embedding_time: float,
    import_time: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute retrieval, technical, latency, and usage KPIs separately."""
    # Require data and valid k so means and ranks below have defined meaning.
    if not results:
        raise ValueError("At least one evaluation result is required.")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")

    # Convert each model run into one flat row; pandas can then aggregate it.
    rows = []
    for result in results:
        # Retrieval IDs are compared with evidence IDs; rank is one-based to match
        # how people read the displayed search results.
        passages = result.get("passages", [])
        ids = [passage["id"] for passage in passages]
        ranks = [
            index
            for index, identifier in enumerate(ids, 1)
            if identifier in result["relevant_ids"]
        ]
        # Citation syntax is [chunk_id]. This checks formatting and known IDs,
        # not whether the cited passage truly supports the claim.
        cited_ids = re.findall(r"\[([^\[\]]+)\]", result.get("answer", ""))
        # Cases without annotated evidence are outside the known document corpus.
        answerable = result["evidence"] is not None
        rows.append(
            {
                "id": result["id"],
                "answerable": answerable,
                "success": result.get("error") is None,
                "hit_1": (
                    int(bool(ranks) and ranks[0] == 1) if answerable else None
                ),
                "hit_k": int(bool(ranks)) if answerable else None,
                "rr_k": (1 / ranks[0] if ranks else 0) if answerable else None,
                "citation_format_ok": bool(cited_ids) and set(cited_ids).issubset(ids),
                "exact_abstention": result.get("answer", "").strip() == ABSTENTION,
                "retrieval_s": result.get("retrieval_s"),
                "generation_s": result.get("generation_s"),
                "total_s": result["total_s"],
                "tokens_s": result.get("tokens_s"),
            }
        )

    # Separate answerable and unanswerable cases so abstention is not confused
    # with retrieval failure. Failed calls remain in technical-success counts.
    metrics = pd.DataFrame(rows)
    answerable_cases = metrics[metrics["answerable"]]
    out_of_corpus = metrics[~metrics["answerable"]]
    successful = metrics[metrics["success"]]
    # Aggregate retrieval quality, operational reliability, timing, and usage.
    # They are intentionally separate: a fast or successful answer is not
    # necessarily factually correct, which still needs human review.
    kpis = {
        "Technical success": f"{int(metrics['success'].sum())}/{len(metrics)}",
        "Hit@1 (answerable)": answerable_cases["hit_1"].mean(),
        f"Hit@{k} (answerable)": answerable_cases["hit_k"].mean(),
        f"MRR@{k} (answerable)": answerable_cases["rr_k"].mean(),
        "Citations present and known (answerable)": answerable_cases[
            "citation_format_ok"
        ].mean(),
        "Exact abstention outside the corpus": out_of_corpus[
            "exact_abstention"
        ].mean(),
        "Median latency, successful calls (s)": successful["total_s"].median(),
        "p95 latency, successful calls (s)": successful["total_s"].quantile(0.95),
        "Median retrieval, successful calls (s)": successful["retrieval_s"].median(),
        "Median generation, successful calls (s)": successful["generation_s"].median(),
        "Median observed API throughput, successful calls (tokens/s)": successful[
            "tokens_s"
        ].median(),
        "Corpus embeddings (s)": embedding_time,
        "ChromaDB import (s)": import_time,
    }
    return metrics, kpis


def score_manual_review(
    review: dict[str, dict[str, int | None]],
) -> dict[str, dict[str, str | int]]:
    """Summarize manual 1/0/None grades and show how many cases were reviewed."""
    # 1 means pass, 0 means fail, and None means not reviewed. Report denominators
    # so an unreviewed case is never silently counted as a pass.
    scores = {}
    for criterion in _REVIEW_CRITERIA:
        expected_grades = [
            grades[criterion]
            for grades in review.values()
            if criterion in grades
        ]
        # Reject scores outside the workshop's binary scale.
        if any(grade is not None and grade not in (0, 1) for grade in expected_grades):
            raise ValueError(f"Invalid manual grade for {criterion}; use 1, 0, or None.")
        grades = [grade for grade in expected_grades if grade is not None]
        scores[criterion] = {
            "score": f"{np.mean(grades):.0%}" if grades else "not evaluated",
            "reviewed": len(grades),
            "total": len(expected_grades),
        }
    return scores


def answer_review_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Format answer and evidence fields for the notebook's manual review table."""
    # Put the generated answer beside its expected answer and retrieved passages,
    # allowing students to inspect evidence rather than judging answer text alone.
    return [
        {
            "id": result["id"],
            "question": result["question"],
            "expected": result["expected"],
            "actual": result.get("answer") or result.get("error") or "",
            "passages": "\n\n".join(
                f"[{passage['id']}] {passage['text']}"
                for passage in result.get("passages", [])
            ),
        }
        for result in results
    ]


def run_evaluation(
    test_cases: list[dict[str, Any]],
    collection: Any,
    client: Any,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Run each evaluation case and retain expected inference failures as results."""
    # Reject an empty suite rather than returning a success-shaped empty result.
    if not test_cases:
        raise ValueError("At least one evaluation case is required.")

    # Run cases sequentially to keep the endpoint workload predictable and make
    # per-question elapsed times easy to interpret.
    results = []
    for test_case in test_cases:
        start = perf_counter()
        result = {
            **test_case,
            "answer": "",
            "passages": [],
            "error": None,
        }
        # Retrieval time and evidence are retained even if generation then fails;
        # this distinction is important when interpreting search versus model errors.
        try:
            passages, retrieval_time = retrieve(
                test_case["question"],
                collection,
                client,
                k=k,
            )
            result.update(
                {
                    "passages": passages,
                    "retrieval_s": retrieval_time,
                }
            )
            result.update(client.generate(test_case["question"], passages))
        # Expected API/data errors are recorded per case so later cases can still
        # run. Unexpected programming errors are not swallowed.
        except (RuntimeError, ValueError) as exc:
            result["error"] = str(exc)
        result["total_s"] = perf_counter() - start
        results.append(result)
    return results


def build_report(
    settings: Settings,
    embedding_revision: str | None,
    chunk_size: int,
    overlap: int,
    k: int,
    collection_name: str,
    embedding_time: float,
    import_time: float,
    results: list[dict[str, Any]],
    manual_review: dict[str, dict[str, int | None]],
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble reproducibility metadata without serializing credentials."""
    # Record enough context to understand a run later, but deliberately never add
    # settings.hf_token or any Authorization header to the report.
    return {
        "python": sys.version,
        "backend": "huggingface-dedicated",
        "libraries": {
            package: version(package)
            for package in ("chromadb", "numpy", "pandas", "requests")
        },
        "models": {
            "generation": settings.chat_model,
            "embedding": settings.embed_model,
            "embedding_revision": embedding_revision,
        },
        "endpoints": {
            "generation": settings.chat_endpoint,
            "embedding": settings.embed_endpoint,
        },
        "parameters": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "k": k,
            "collection": collection_name,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 180,
            "enable_thinking": False,
        },
        "preparation_s": {
            "embeddings": embedding_time,
            "import": import_time,
        },
        "conditions": (
            "Sequential API calls; timings include network, queueing and retry "
            "delays. Shared endpoint load may vary."
        ),
        "results": results,
        "manual_review": manual_review,
        "metrics": json.loads(metrics.to_json(orient="records")),
    }


def _contains_secret_material(value: Any, secret: str | None) -> bool:
    # Walk nested JSON-like structures recursively. Reject both suspicious field
    # names (for example "token") and the exact secret value wherever it appears.
    if isinstance(value, dict):
        return any(
            str(key).casefold() in _SECRET_FIELD_NAMES
            or _contains_secret_material(item, secret)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_material(item, secret) for item in value)
    return bool(secret and isinstance(value, str) and secret in value)


def save_results(
    output_dir: Path,
    report: dict[str, Any],
    metrics: pd.DataFrame,
    secret: str | None = None,
) -> tuple[Path, Path]:
    """Write timestamped JSON/CSV results after rejecting secret-bearing reports."""
    # Check both output structures before creating files. This is defense in depth
    # in case a notebook cell accidentally adds a credential to a result.
    if _contains_secret_material(report, secret) or _contains_secret_material(
        metrics.to_dict(orient="records"),
        secret,
    ):
        raise ValueError("Refusing to save report containing secret material.")

    # Create the destination only after the safety check succeeds.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # UTC timestamps sort consistently and include microseconds to avoid filename
    # collisions when repeated runs finish within the same second.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    safe_report = dict(report)
    safe_report["date_utc"] = stamp

    json_file = output_dir / f"essai_{stamp}.json"
    csv_file = output_dir / f"mesures_{stamp}.csv"
    # JSON retains the complete run; CSV is a compact metrics table. UTF-8-SIG
    # improves direct opening of French text in spreadsheet software.
    json_file.write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    metrics.to_csv(csv_file, index=False, encoding="utf-8-sig")
    return json_file, csv_file
