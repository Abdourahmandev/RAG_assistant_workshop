import unittest
from unittest.mock import Mock, patch

import numpy as np

from rag.pipeline import ask, embed_chunks, retrieve


class FakeClient:
    def __init__(self) -> None:
        self.embedding_calls = []
        self.generated = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedding_calls.append(texts)
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def generate(self, question, passages):
        self.generated.append((question, passages))
        return {
            "answer": "The lab closes at 5 p.m. [hours_0000]",
            "generation_s": 0.5,
            "tokens_s": 10.0,
            "generated_tokens": 5,
            "prompt_tokens": 20,
        }


class EmbedChunksTests(unittest.TestCase):
    def test_embeds_document_prefix_in_batches_and_measures_time(self) -> None:
        client = FakeClient()
        chunks = [
            {"title": f"title-{index}", "text": f"text-{index}"}
            for index in range(9)
        ]

        with patch("rag.pipeline.perf_counter", side_effect=[10.0, 11.25]):
            vectors, elapsed = embed_chunks(chunks, client, batch_size=4)

        self.assertEqual(vectors.shape, (9, 2))
        self.assertEqual(elapsed, 1.25)
        self.assertEqual([len(call) for call in client.embedding_calls], [4, 4, 1])
        self.assertEqual(
            client.embedding_calls[0][0],
            "title: title-0 | text: text-0",
        )

    def test_rejects_empty_chunks_and_invalid_batch_size(self) -> None:
        client = FakeClient()

        with self.assertRaises(ValueError):
            embed_chunks([], client)
        with self.assertRaises(ValueError):
            embed_chunks([{"title": "x", "text": "y"}], client, batch_size=0)


class RetrievalAndAskTests(unittest.TestCase):
    def test_retrieve_embeds_prefixed_question_and_returns_hits(self) -> None:
        client = FakeClient()
        collection = Mock()
        collection.count.return_value = 2
        hits = [{"id": "hours_0000", "text": "Open until 5 p.m."}]

        with patch("rag.pipeline.search", return_value=hits) as search_mock, patch(
            "rag.pipeline.perf_counter",
            side_effect=[5.0, 5.4],
        ):
            result, elapsed = retrieve("When does it close?", collection, client, k=1)

        self.assertEqual(result, hits)
        self.assertAlmostEqual(elapsed, 0.4)
        self.assertEqual(
            client.embedding_calls,
            [["task: search result | query: When does it close?"]],
        )
        np.testing.assert_array_equal(
            search_mock.call_args.args[1],
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(search_mock.call_args.kwargs["top_k"], 1)

    def test_retrieve_rejects_invalid_question_k_and_empty_collection(self) -> None:
        client = FakeClient()
        collection = Mock()
        collection.count.return_value = 1

        with self.assertRaisesRegex(ValueError, "positive integer"):
            retrieve("question", collection, client, k=0)
        with self.assertRaisesRegex(ValueError, "empty"):
            retrieve("  ", collection, client)
        collection.count.return_value = 0
        with self.assertRaisesRegex(ValueError, "empty"):
            retrieve("question", collection, client)

    def test_ask_combines_retrieval_generation_and_timing(self) -> None:
        client = FakeClient()
        collection = Mock()
        collection.count.return_value = 1
        passages = [{"id": "hours_0000", "text": "Open until 5 p.m."}]

        with patch("rag.pipeline.search", return_value=passages), patch(
            "rag.pipeline.perf_counter",
            side_effect=[1.0, 1.1, 1.3, 2.5],
        ):
            result = ask("When does it close?", collection, client, k=1)

        self.assertEqual(result["question"], "When does it close?")
        self.assertEqual(result["passages"], passages)
        self.assertEqual(result["answer"], "The lab closes at 5 p.m. [hours_0000]")
        self.assertAlmostEqual(result["retrieval_s"], 0.2)
        self.assertAlmostEqual(result["total_s"], 1.5)
        self.assertEqual(client.generated, [("When does it close?", passages)])

    def test_ask_propagates_generation_failure(self) -> None:
        client = FakeClient()
        client.generate = Mock(side_effect=RuntimeError("model failed"))
        collection = Mock()
        collection.count.return_value = 1

        with patch("rag.pipeline.search", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                ask("question", collection, client)


if __name__ == "__main__":
    unittest.main()
