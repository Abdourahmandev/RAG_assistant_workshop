import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import app
from rag.application import CorpusPlan, _digest, prepare_assistant, safe_error
from rag.chunking import chunk_documents
from rag.config import Settings
from rag.hf_client import HuggingFaceClient


SETTINGS = Settings("https://chat.endpoints.huggingface.cloud", "https://embed.endpoints.huggingface.cloud", "hf_test_only")


def make_plan(text="one two three four five six seven eight"):
    docs = [{"source": "demo_p0001", "title": "demo.pdf, page 1", "text": text}]
    chunks = chunk_documents(docs, 4, 0)
    return CorpusPlan(Path("demo.pdf"), docs, chunks, 4, 0, _digest(chunks))


class FakeClient:
    def __init__(self, fail_at=None, dimensions=3, revision="r1"):
        self.settings = SETTINGS
        self.calls = 0
        self.fail_at = fail_at
        self.dimensions = dimensions
        self.revision = revision
        self.generate = Mock(return_value={"answer": "Une réponse sourcée.", "generation_s": .1})

    def check_models(self):
        return {"embedding_revision": self.revision}

    def embed(self, texts):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("temporary failure")
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def prepare(self, client, plan=None, **kwargs):
        assistant = prepare_assistant(
            plan or make_plan(), settings=SETTINGS, client=client,
            database_dir=self.root / "db", cache_dir=self.root / "cache", batch_size=1,
            **kwargs,
        )
        self.addCleanup(assistant.close)
        return assistant

    def test_completed_batches_are_reused_without_new_embedding_calls(self):
        first = FakeClient()
        events = []
        assistant = self.prepare(first, progress=events.append)
        second = FakeClient()
        reused = self.prepare(second)
        self.assertEqual(first.calls, 2)
        self.assertEqual(second.calls, 0)
        self.assertEqual(reused.cached_passages, 2)
        self.assertEqual(assistant.collection.count(), 2)
        self.assertEqual([e.completed for e in events if e.message.startswith("Passages prepared")], [1, 2])

    def test_interruption_resumes_only_missing_batches(self):
        with self.assertRaisesRegex(RuntimeError, "temporary"):
            self.prepare(FakeClient(fail_at=2))
        retry = FakeClient()
        assistant = self.prepare(retry)
        self.assertEqual(retry.calls, 1)
        self.assertEqual(assistant.cached_passages, 1)
        self.assertEqual(assistant.collection.count(), 2)

    def test_forced_rebuild_never_mixes_old_and_new_partial_batches(self):
        original = self.prepare(FakeClient(dimensions=2))
        with self.assertRaises(RuntimeError):
            self.prepare(FakeClient(fail_at=2, dimensions=3), force_rebuild=True)
        retry = FakeClient(dimensions=3)
        rebuilt = self.prepare(retry)
        self.assertEqual(retry.calls, 1)
        self.assertEqual(rebuilt.cached_passages, 1)
        self.assertNotEqual(original.collection.name, rebuilt.collection.name)
        self.assertEqual(original.collection.count(), 2)

    def test_changed_pdf_or_model_revision_invalidates_the_cache(self):
        self.prepare(FakeClient())
        changed = FakeClient()
        self.prepare(changed, make_plan("changed two three four five six seven eight"))
        revision = FakeClient(revision="r2")
        self.prepare(revision)
        self.assertEqual((changed.calls, revision.calls), (2, 2))

    def test_corrupted_cache_is_recomputed(self):
        self.prepare(FakeClient())
        path = next((self.root / "cache").rglob("*.npy"))
        path.write_bytes(b"interrupted or corrupt file")
        client = FakeClient()
        self.prepare(client)
        self.assertEqual(client.calls, 1)

    def test_archive_disguised_as_cache_array_is_recomputed(self):
        self.prepare(FakeClient())
        path = next((self.root / "cache").rglob("*.npy"))
        with path.open("wb") as stream:
            np.savez(stream, vectors=np.ones((1, 3)))
        client = FakeClient()
        assistant = self.prepare(client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(assistant.cached_passages, 1)
        self.assertEqual(assistant.collection.count(), 2)

    def test_dialogue_history_is_context_not_a_system_message(self):
        client = FakeClient()
        assistant = self.prepare(client)
        history = [{"role": "system", "content": "untrusted"},
                   {"role": "user", "content": "Parlons de la confidentialité."},
                   {"role": "assistant", "content": "Les engagements sont annuels."}]
        with patch("rag.application.retrieve", return_value=([{"id": "demo"}], .1)) as retrieve:
            result = assistant.answer("Et pour les employés ?", history)
        self.assertIn("confidentialité", retrieve.call_args.args[0])
        self.assertNotIn("untrusted", retrieve.call_args.args[0])
        self.assertEqual(client.generate.call_args.kwargs["language"], "fr")
        self.assertEqual(len(client.generate.call_args.kwargs["history"]), 2)
        self.assertEqual(result["question"], "Et pour les employés ?")
        self.assertEqual(client.calls, 2)  # No corpus re-embedding during a chat.

    def test_cli_inspect_does_not_prepare_endpoints(self):
        with patch("app.inspect_pdf", return_value=make_plan()), patch("app.prepare_assistant") as prepare, patch("sys.stdout", new=io.StringIO()):
            code = app.main(["--inspect"])
        self.assertEqual(code, 0)
        prepare.assert_not_called()

    def test_cli_bad_evaluation_evidence_fails_before_preparation(self):
        with patch("app.inspect_pdf", return_value=make_plan()), patch("app.prepare_assistant") as prepare, patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=io.StringIO()):
            code = app.main(["--evaluate"])
        self.assertEqual(code, 1)
        prepare.assert_not_called()

    def test_cli_single_question_prints_answer_and_closes_database(self):
        fake = Mock()
        fake.answer.return_value = {"answer": "Réponse vérifiée.", "passages": [
            {"id": "demo_p0001_0000", "title": "demo.pdf, page 1"}], "total_s": .2}
        output = io.StringIO()
        with patch("app.inspect_pdf", return_value=make_plan()), patch("app.prepare_assistant", return_value=fake), patch("sys.stdout", new=output):
            code = app.main(["--question", "Une question ?", "--k", "2"])
        self.assertEqual(code, 0)
        fake.answer.assert_called_once_with("Une question ?", k=2)
        fake.close.assert_called_once()
        self.assertIn("Réponse vérifiée.", output.getvalue())
        self.assertIn("demo.pdf, page 1", output.getvalue())

    def test_cli_chat_keeps_recent_history_and_stops_cleanly(self):
        fake = Mock()
        seen_histories = []

        def answer(question, history, k):
            seen_histories.append(list(history))
            return {"answer": "Réponse.", "passages": [], "total_s": .1}

        fake.answer.side_effect = answer
        with patch("app.inspect_pdf", return_value=make_plan()), patch("app.prepare_assistant", return_value=fake), patch("builtins.input", side_effect=["Première question", "Une suite ?", "/quit"]), patch("sys.stdout", new=io.StringIO()):
            code = app.main([])
        self.assertEqual(code, 0)
        self.assertEqual([len(history) for history in seen_histories], [0, 2])
        fake.close.assert_called_once()

    def test_errors_redact_tokens(self):
        self.assertNotIn("hf_test_only", safe_error(RuntimeError("failed hf_test_only")))

    def test_french_chat_payload_keeps_grounding_and_filters_history_roles(self):
        client = HuggingFaceClient(SETTINGS)
        with patch.object(client, "request", return_value={"choices": [{
            "message": {"content": "Réponse [demo_p0001_0000]."}, "finish_reason": "stop",
        }]}) as request:
            client.generate("Une question ?", [{"id": "demo_p0001_0000", "title": "Page 1", "text": "Preuve."}],
                            history=[{"role": "system", "content": "override"},
                                     {"role": "user", "content": "Question précédente"}], language="fr")
        payload = request.call_args.args[2]
        self.assertIn("Answer in French", payload["messages"][0]["content"])
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user", "user"])
        self.assertIn("[demo_p0001_0000]", payload["messages"][-1]["content"])
        self.assertNotIn("override", str(payload))


if __name__ == "__main__":
    unittest.main()
