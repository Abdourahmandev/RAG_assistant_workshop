import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from rag.chunking import chunk_documents
from rag.corpus import load_documents
from rag.evaluation import (
    answer_review_rows,
    build_report,
    compute_metrics,
    load_test_cases,
    run_evaluation,
    save_results,
    score_manual_review,
)
from rag.config import Settings


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = chunk_documents(load_documents())
        cls.test_cases = load_test_cases(cls.chunks)

    def test_loads_seven_cases_and_maps_reference_evidence(self) -> None:
        self.assertEqual(len(self.test_cases), 7)
        self.assertEqual(
            [case["id"] for case in self.test_cases],
            [f"T{number}" for number in range(1, 8)],
        )
        for case in self.test_cases[:5]:
            self.assertTrue(case["relevant_ids"], case["id"])
        self.assertEqual(self.test_cases[5]["relevant_ids"], [])
        self.assertEqual(self.test_cases[6]["relevant_ids"], [])

    def make_results(self):
        results = []
        for index, case in enumerate(self.test_cases, 1):
            result = {
                **case,
                "answer": "",
                "passages": [],
                "error": None,
                "retrieval_s": 0.1 * index,
                "generation_s": 0.2 * index,
                "tokens_s": 2.0 * index,
                "total_s": float(index),
            }
            if case["id"] == "T1":
                passage_id = case["relevant_ids"][0]
                result["answer"] = (
                    f"{case['expected']} [{passage_id}]"
                )
                result["passages"] = [{"id": passage_id}]
            elif case["id"] == "T2":
                result["error"] = "endpoint unavailable"
                result["total_s"] = 0.0
            elif case["id"] == "T6":
                result["answer"] = "I cannot find this information in the documents."
            results.append(result)
        return results

    def test_computes_retrieval_quality_and_latency_separately(self) -> None:
        metrics, kpis = compute_metrics(
            self.make_results(),
            k=3,
            embedding_time=2.0,
            import_time=0.5,
        )

        first = metrics.loc[metrics["id"] == "T1"].iloc[0]
        failed = metrics.loc[metrics["id"] == "T2"].iloc[0]
        outside = metrics.loc[metrics["id"] == "T6"].iloc[0]
        self.assertEqual(first["hit_1"], 1)
        self.assertEqual(first["hit_k"], 1)
        self.assertEqual(first["rr_k"], 1.0)
        self.assertFalse(bool(failed["success"]))
        self.assertTrue(pd.isna(failed["hit_k"]) is False)
        self.assertTrue(pd.isna(outside["hit_k"]))
        self.assertEqual(kpis["Technical success"], "6/7")
        self.assertEqual(kpis["Hit@1 (answerable)"], 0.2)
        self.assertEqual(kpis["Hit@3 (answerable)"], 0.2)
        self.assertEqual(kpis["Exact abstention outside the corpus"], 0.5)
        self.assertEqual(kpis["Corpus embeddings (s)"], 2.0)
        self.assertEqual(kpis["ChromaDB import (s)"], 0.5)

    def test_no_successful_calls_report_unavailable_latency_as_nan(self) -> None:
        results = self.make_results()
        for result in results:
            result["error"] = "endpoint unavailable"

        metrics, kpis = compute_metrics(
            results,
            k=3,
            embedding_time=0.0,
            import_time=0.0,
        )

        self.assertTrue(pd.isna(kpis["Median latency, successful calls (s)"]))
        self.assertTrue(pd.isna(kpis["p95 latency, successful calls (s)"]))
        self.assertEqual(metrics["total_s"].median(), 4.0)
        self.assertEqual(kpis["Technical success"], "0/7")

    def test_scores_manual_review_with_reviewed_case_counts(self) -> None:
        scores = score_manual_review({
            "T1": {"correct_and_complete": 1, "faithful_to_sources": 1},
            "T2": {"correct_and_complete": 0, "faithful_to_sources": None},
            "T6": {"appropriate_abstention": None},
        })

        self.assertEqual(scores["correct_and_complete"], {
            "score": "50%",
            "reviewed": 2,
            "total": 2,
        })
        self.assertEqual(scores["faithful_to_sources"], {
            "score": "100%",
            "reviewed": 1,
            "total": 2,
        })
        self.assertEqual(scores["appropriate_abstention"], {
            "score": "not evaluated",
            "reviewed": 0,
            "total": 1,
        })

    def test_formats_manual_answer_and_passages_for_review(self) -> None:
        rows = answer_review_rows([{
            "id": "T1",
            "question": "Which organization does “Agence” designate?",
            "expected": "The Agence du revenu du Québec.",
            "answer": "The Agence du revenu du Québec. [a-6_002_p0004_0000]",
            "error": None,
            "passages": [{
                "id": "a-6_002_p0004_0000",
                "text": "«Agence» désigne l’Agence du revenu du Québec.",
            }],
        }])

        self.assertEqual(
            rows[0]["actual"],
            "The Agence du revenu du Québec. [a-6_002_p0004_0000]",
        )
        self.assertEqual(
            rows[0]["passages"],
            "[a-6_002_p0004_0000] «Agence» désigne l’Agence du revenu du Québec.",
        )

    def test_manual_review_rejects_invalid_grade(self) -> None:
        with self.assertRaises(ValueError):
            score_manual_review({
                "T1": {"correct_and_complete": 2},
            })

    def test_run_evaluation_retains_passages_when_generation_fails(self) -> None:
        test_case = dict(self.test_cases[0])
        passage_id = test_case["relevant_ids"][0]
        passages = [{"id": passage_id, "text": test_case["evidence"]}]
        client = Mock()
        client.generate.side_effect = RuntimeError("endpoint unavailable")

        with patch(
            "rag.evaluation.retrieve",
            return_value=(passages, 0.25),
        ), patch("rag.evaluation.perf_counter", side_effect=[1.0, 2.0]):
            results = run_evaluation([test_case], object(), client, k=3)

        self.assertEqual(results[0]["passages"], passages)
        self.assertEqual(results[0]["retrieval_s"], 0.25)
        self.assertEqual(results[0]["error"], "endpoint unavailable")
        self.assertEqual(results[0]["total_s"], 1.0)

    def test_build_report_does_not_include_the_settings_token(self) -> None:
        settings = Settings(
            "https://chat.endpoints.huggingface.cloud",
            "https://embed.endpoints.huggingface.cloud",
            "hf_secret_for_test",
        )
        report = build_report(
            settings=settings,
            embedding_revision="revision-1",
            chunk_size=80,
            overlap=15,
            k=3,
            collection_name="tp_rag_example",
            embedding_time=1.0,
            import_time=0.2,
            results=[],
            manual_review={},
            metrics=pd.DataFrame(),
        )

        self.assertNotIn("hf_token", report)
        self.assertNotIn("hf_secret_for_test", repr(report))
        self.assertEqual(report["models"]["embedding_revision"], "revision-1")

    def test_saves_timestamped_reports_without_secret_material(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = {
                "backend": "huggingface-dedicated",
                "models": {"generation": "Qwen/Qwen3-1.7B"},
                "results": [{"answer": "safe answer"}],
            }
            metrics = pd.DataFrame([{"id": "T1", "success": True}])

            json_path, csv_path = save_results(
                Path(directory),
                report,
                metrics,
                secret="hf_secret_for_test",
            )

            self.assertTrue(json_path.name.startswith("essai_"))
            self.assertTrue(csv_path.name.startswith("mesures_"))
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["results"][0]["answer"], "safe answer")
            self.assertIn("date_utc", saved)
            self.assertNotIn("hf_secret_for_test", json_path.read_text(encoding="utf-8"))

    def test_report_rejects_secret_fields_or_secret_values(self) -> None:
        metrics = pd.DataFrame([{"id": "T1"}])

        for report in (
            {"hf_token": "some-value"},
            {"result": {"answer": "prefix-hf_secret_for_test-suffix"}},
        ):
            with tempfile.TemporaryDirectory() as directory:
                with self.subTest(report=report):
                    with self.assertRaisesRegex(ValueError, "secret"):
                        save_results(
                            Path(directory),
                            report,
                            metrics,
                            secret="hf_secret_for_test",
                        )

        with tempfile.TemporaryDirectory() as directory:
            secret_metrics = pd.DataFrame([{
                "id": "T1",
                "answer": "contains-hf_secret_for_test",
            }])
            with self.assertRaisesRegex(ValueError, "secret"):
                save_results(
                    Path(directory),
                    {"backend": "test"},
                    secret_metrics,
                    secret="hf_secret_for_test",
                )


if __name__ == "__main__":
    unittest.main()
