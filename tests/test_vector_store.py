import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.vector_store import (
    collection_fingerprint,
    index_chunks,
    open_collection,
    open_database,
    search,
)


CHUNKS = [
    {
        "id": "first_0000",
        "source": "first",
        "title": "First passage",
        "start": 0,
        "end": 2,
        "text": "alpha beta",
    },
    {
        "id": "second_0000",
        "source": "second",
        "title": "Second passage",
        "start": 0,
        "end": 2,
        "text": "gamma delta",
    },
]
VECTORS = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class CollectionFingerprintTests(unittest.TestCase):
    def fingerprint(self, chunks=CHUNKS, vectors=VECTORS, **overrides):
        values = {
            "model": "embedding-model",
            "endpoint": "https://embed.endpoints.huggingface.cloud",
            "revision": "revision-1",
            "size": 80,
            "overlap": 15,
        }
        values.update(overrides)
        return collection_fingerprint(
            chunks,
            values["model"],
            values["endpoint"],
            values["revision"],
            values["size"],
            values["overlap"],
            vectors,
        )

    def test_fingerprint_is_stable_for_same_inputs(self) -> None:
        self.assertEqual(self.fingerprint(), self.fingerprint())

    def test_fingerprint_changes_when_index_inputs_change(self) -> None:
        changed_chunks = [dict(CHUNKS[0], text="changed"), CHUNKS[1]]
        changed_vectors = np.asarray([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32)

        self.assertNotEqual(self.fingerprint(), self.fingerprint(chunks=changed_chunks))
        self.assertNotEqual(self.fingerprint(), self.fingerprint(vectors=changed_vectors))
        self.assertNotEqual(
            self.fingerprint(),
            self.fingerprint(revision="revision-2"),
        )


class PersistentVectorStoreTests(unittest.TestCase):
    def test_upsert_round_trip_search_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "chroma"
            client = open_database(db_path)
            collection = open_collection(client, "tp_rag_test")
            reopened_client = None
            try:
                index_chunks(collection, CHUNKS, VECTORS)
                index_chunks(collection, CHUNKS, VECTORS)

                self.assertEqual(collection.count(), 2)
                sample = collection.get(
                    ids=["first_0000"],
                    include=["documents", "embeddings", "metadatas"],
                )
                self.assertEqual(sample["documents"][0], "alpha beta")
                self.assertEqual(sample["metadatas"][0]["source"], "first")
                np.testing.assert_allclose(sample["embeddings"][0], VECTORS[0])

                hits = search(collection, np.asarray([1.0, 0.0]), top_k=2)
                self.assertEqual(
                    [hit["id"] for hit in hits],
                    ["first_0000", "second_0000"],
                )
                self.assertEqual(hits[0]["text"], "alpha beta")
                self.assertEqual(hits[0]["source"], "first")
                self.assertAlmostEqual(hits[0]["similarity"], 1.0, places=5)

                reopened_client = open_database(db_path)
                reopened = open_collection(reopened_client, "tp_rag_test")
                self.assertEqual(reopened.count(), 2)
            finally:
                if reopened_client is not None:
                    reopened_client.close()
                client.close()

    def test_index_rejects_mismatched_or_invalid_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = open_database(Path(directory))
            collection = open_collection(client, "tp_rag_test")
            try:
                with self.assertRaises(ValueError):
                    index_chunks(collection, CHUNKS, VECTORS[:1])
                with self.assertRaises(ValueError):
                    index_chunks(
                        collection,
                        CHUNKS,
                        np.asarray([[np.nan, 0], [0, 1]]),
                    )
            finally:
                client.close()

    def test_search_rejects_invalid_top_k_and_empty_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = open_database(Path(directory))
            collection = open_collection(client, "tp_rag_test")
            try:
                with self.assertRaises(ValueError):
                    search(collection, np.asarray([1.0, 0.0]), top_k=0)
                with self.assertRaisesRegex(ValueError, "empty"):
                    search(collection, np.asarray([1.0, 0.0]), top_k=1)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
