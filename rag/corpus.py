"""PDF document loading and text normalization for the RAG workshop."""

# Path resolves the bundled PDF independent of the notebook's current directory.
from pathlib import Path
# re turns the PDF filename into a predictable ID prefix for citations.
import re

# pypdf extracts selectable text from PDF pages locally; scanned image-only pages
# need OCR and are not handled by this workshop loader.
from pypdf import PdfReader


# Keep the default document next to the project root, one directory above rag/.
DEFAULT_PDF_PATH = Path(__file__).resolve().parent.parent / "A-6.002.pdf"


def normalize_text(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    # PDF extraction often inserts line breaks and repeated spaces. Collapsing
    # them produces consistent input for word-based chunking and evidence checks.
    return " ".join(text.split())


def load_documents(pdf_path: str | Path | None = None) -> list[dict[str, str]]:
    """Extract one normalized, page-addressable document for each PDF page."""
    # A caller may supply a different PDF; otherwise use the workshop document.
    path = Path(pdf_path) if pdf_path is not None else DEFAULT_PDF_PATH
    # Fail clearly rather than silently returning an empty corpus if the file was
    # moved or omitted from the project folder.
    if not path.is_file():
        raise FileNotFoundError(f"PDF document not found: {path}")

    # PdfReader exposes pages in file order and extracts their selectable text.
    reader = PdfReader(path)
    documents = []
    # Convert punctuation in the filename to underscores to create a compact,
    # stable citation prefix (A-6.002.pdf -> a-6_002).
    source_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).lower()
    # Treat each page as its own source so students can trace a retrieved chunk
    # to the exact page in the original PDF.
    for page_number, page in enumerate(reader.pages, start=1):
        # A page may have no extractable text; skip empty pages but preserve the
        # original page number in source IDs for every page that is retained.
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        # Keep both a machine-readable ID and a human-readable title. Chunking
        # carries these into passage metadata and generated citations.
        documents.append(
            {
                "source": f"{source_prefix}_p{page_number:04d}",
                "title": f"{path.name}, page {page_number}",
                "text": text,
            }
        )

    # An all-empty result commonly means the PDF is scanned or text extraction
    # failed. Make that visible instead of letting later stages index nothing.
    if not documents:
        raise ValueError(f"No extractable text found in PDF: {path}")
    return documents
