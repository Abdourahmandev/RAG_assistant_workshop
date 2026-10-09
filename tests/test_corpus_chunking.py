import unittest
from pathlib import Path

from rag.chunking import chunk_documents, verify_chunks
from rag.corpus import DEFAULT_PDF_PATH, load_documents, normalize_text


class CorpusTests(unittest.TestCase):
    def test_loads_pdf_pages_with_unique_sources_and_normalized_text(self) -> None:
        documents = load_documents()

        self.assertEqual(DEFAULT_PDF_PATH.name, "A-6.002.pdf")
        self.assertEqual(len(documents), 18)
        self.assertTrue(all("_p" in document["source"] for document in documents))
        self.assertEqual(
            len({document["source"] for document in documents}),
            len(documents),
        )
        self.assertTrue(all(document["text"].strip() for document in documents))
        self.assertTrue(
            all("  " not in document["text"] for document in documents)
        )
        self.assertEqual(
            documents[3]["source"],
            "a-6_002_p0004",
        )
        self.assertIn(
            "Les renseignements confidentiels sont ceux qui concernent un individu",
            documents[3]["text"],
        )

    def test_keeps_all_reference_evidence_in_chunks(self) -> None:
        evidence = (
            ("a-6_002_p0004", "une entreprise et permettent de l’identifier"),
            (
                "a-6_002_p0005",
                "des engagements de confidentialité et les renouvelle chaque année",
            ),
            (
                "a-6_002_p0006",
                "Nous recueillons les renseignements confidentiels directement auprès de vous",
            ),
            (
                "a-6_002_p0007",
                "Lorsque nous recevons des informations non sollicitées",
            ),
            ("a-6_002_p0008", "ce système automatisé permet un premier tri des candidatures"),
        )
        documents = load_documents()
        chunks = chunk_documents(documents, size=80, overlap=15)

        for source, phrase in evidence:
            with self.subTest(source=source):
                self.assertTrue(
                    any(
                        chunk["source"] == source and phrase in chunk["text"]
                        for chunk in chunks
                    )
                )

    def test_rejects_missing_pdf_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_documents(Path("missing-workshop-document.pdf"))

    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  line one\n\tline   two "), "line one line two")


class ChunkingTests(unittest.TestCase):
    def test_chunks_keep_expected_fields_and_word_offsets(self) -> None:
        documents = [{
            "source": "demo",
            "title": "Demo",
            "text": "one two three four five six seven",
        }]

        chunks = chunk_documents(documents, size=4, overlap=2)

        self.assertEqual(
            chunks,
            [
                {
                    "id": "demo_0000",
                    "source": "demo",
                    "title": "Demo",
                    "start": 0,
                    "end": 4,
                    "text": "one two three four",
                },
                {
                    "id": "demo_0002",
                    "source": "demo",
                    "title": "Demo",
                    "start": 2,
                    "end": 6,
                    "text": "three four five six",
                },
                {
                    "id": "demo_0004",
                    "source": "demo",
                    "title": "Demo",
                    "start": 4,
                    "end": 7,
                    "text": "five six seven",
                },
            ],
        )

    def test_chunks_cover_every_word_and_repeat_exact_overlap(self) -> None:
        documents = [{
            "source": "demo",
            "title": "Demo",
            "text": " ".join(f"word{i}" for i in range(23)),
        }]

        chunks = chunk_documents(documents, size=8, overlap=3)
        covered = {
            position
            for chunk in chunks
            for position in range(chunk["start"], chunk["end"])
        }

        self.assertEqual(covered, set(range(23)))
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(previous["end"] - following["start"], 3)
            self.assertEqual(
                previous["text"].split()[-3:],
                following["text"].split()[:3],
            )

    def test_chunk_ids_are_unique_and_size_is_bounded(self) -> None:
        chunks = chunk_documents(load_documents(), size=50, overlap=10)

        self.assertEqual(len({chunk["id"] for chunk in chunks}), len(chunks))
        self.assertTrue(
            all(0 < len(chunk["text"].split()) <= 50 for chunk in chunks)
        )

    def test_invalid_chunk_parameters_raise_value_error(self) -> None:
        documents = [{"source": "demo", "title": "Demo", "text": "some text"}]

        for size, overlap in ((0, 0), (4, -1), (4, 4), (3, 5)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_documents(documents, size=size, overlap=overlap)

    def test_verifies_coverage_and_rejects_missing_passages(self) -> None:
        documents = [{
            "source": "demo",
            "title": "Demo",
            "text": "one two three four five six",
        }]
        chunks = chunk_documents(documents, size=4, overlap=2)

        verify_chunks(documents, chunks, size=4, overlap=2)
        with self.assertRaisesRegex(ValueError, "cover every word"):
            verify_chunks(documents, chunks[:1], size=4, overlap=2)


if __name__ == "__main__":
    unittest.main()
