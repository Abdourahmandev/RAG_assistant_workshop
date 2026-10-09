"""Word-window chunking for workshop documents."""

# Mapping/Sequence let callers pass ordinary dicts/lists or compatible
# read-only collections while keeping this function independent of PDF details.
from collections.abc import Mapping, Sequence
# Any describes passage metadata whose fields have different value types.
from typing import Any


def chunk_documents(
    documents: Sequence[Mapping[str, str]],
    size: int = 80,
    overlap: int = 15,
) -> list[dict[str, Any]]:
    """Split documents into overlapping word windows with stable passage IDs."""
    # Reject bools explicitly because bool is a subclass of int in Python.
    # Overlap must be smaller than the window or the chunking loop cannot advance.
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or size <= 0
        or overlap < 0
        or overlap >= size
    ):
        raise ValueError("size must be positive and overlap must be in [0, size).")

    passages: list[dict[str, Any]] = []
    # Process each PDF page independently so a chunk never crosses source pages.
    for document in documents:
        # Word splitting is deliberately simple and transparent for this lab.
        words = document["text"].split()
        # Advancing by size-overlap repeats context at each boundary while still
        # making forward progress.
        step = size - overlap
        # start/end are word offsets; end is exclusive, just like Python slices.
        for start in range(0, len(words), step):
            end = min(start + size, len(words))
            # The source and starting word offset make the chunk ID reproducible.
            # Store the original page metadata for citations and database filters.
            passages.append(
                {
                    "id": f"{document['source']}_{start:04d}",
                    "source": document["source"],
                    "title": document["title"],
                    "start": start,
                    "end": end,
                    "text": " ".join(words[start:end]),
                }
            )
            # Stop after the final partial window; otherwise the next start would
            # create an unnecessary empty chunk.
            if end == len(words):
                break
    return passages


def verify_chunks(
    documents: Sequence[Mapping[str, str]],
    passages: Sequence[Mapping[str, Any]],
    size: int,
    overlap: int,
) -> None:
    """Raise ValueError if generated passages lose words or alter overlap."""
    # Validate parameters here too, because callers may invoke verification
    # independently from chunk_documents.
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size).")

    # Check each original page separately and restore document order using offsets.
    for document in documents:
        pieces = sorted(
            (
                passage
                for passage in passages
                if passage["source"] == document["source"]
            ),
            key=lambda passage: passage["start"],
        )
        # Set coverage catches both missing word positions and chunks with
        # incorrect offsets, even if their displayed text looks plausible.
        covered = {
            position
            for piece in pieces
            for position in range(piece["start"], piece["end"])
        }
        expected = set(range(len(document["text"].split())))
        if covered != expected:
            raise ValueError(
                f"Chunks do not cover every word in source {document['source']}."
            )
        # Adjacent chunks must overlap by exactly the requested number of words.
        for previous, following in zip(pieces, pieces[1:]):
            if previous["end"] - following["start"] != overlap:
                raise ValueError(
                    f"Chunks have an invalid overlap in source {document['source']}."
                )
            if overlap and (
                previous["text"].split()[-overlap:]
                != following["text"].split()[:overlap]
            ):
                raise ValueError(
                    f"Chunk text does not match offsets in source {document['source']}."
                )
