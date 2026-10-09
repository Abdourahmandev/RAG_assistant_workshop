# RAG workshop with Hugging Face — Python and Streamlit

Build a document assistant with [`app.py`](app.py) in the terminal and
[`streamlit_app.py`](streamlit_app.py) in the browser. Both applications use the
same [`rag/`](rag/) package. No notebook or Jupyter server is required.

The software, slides, presenter guide and technical messages are in English.
The source PDF and the interactive RAG questions and answers remain in French.
The evaluation suite includes English questions to test retrieval across languages.

## Presentation materials

- [`RAG_Workshop_Guided_EN.pptx`](RAG_Workshop_Guided_EN.pptx):
  21 slides that alternate RAG concepts, the selected strategies and guided practice,
  including cloning the project from GitHub.
  Every slide includes a speaking script in the presenter notes.
- [`RAG_Workshop_Presenter_Guide_EN.pdf`](RAG_Workshop_Presenter_Guide_EN.pdf):
  a demonstration run card, setup commands, strategy explanations and the script
  for every slide. Use this guide with the 21-slide deck.

Notebooks are not required or included in this repository.

## Requirements

- Git, Python 3.12 or 3.13, and an internet connection.
- A chat endpoint serving `Qwen/Qwen3-1.7B`.
- An embedding endpoint serving `google/embeddinggemma-300m`.
- A Hugging Face token authorized to call both endpoints.
- A PDF containing selectable text; this application does not perform OCR.

The supplied `A-6.002.pdf` currently contains Revenu Québec's 18-page privacy
policy in French. It produces 110 passages using 80 words per passage and an
overlap of 15 words. Chroma stores the vectors and source passages locally.

## Clone the project

Open a terminal in the folder where you keep your projects:

```bash
git clone https://github.com/Abdourahmandev/RAG_assistant_workshop.git
cd RAG_assistant_workshop
```

Then follow the setup for your operating system below. The repository includes
the code, sample PDF, tests and presentation materials. Configure your own
`.env` privately. The first processing run creates the embedding cache and local
vector database; allow time for this before the workshop.

## Windows PowerShell setup

Run commands from the project folder:

```powershell
py -3.13 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pip check
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Use `py -3.12` if that is your installed version. If `.venv` already exists,
keep it and start with the `$py` assignment. Always use this interpreter to
avoid conflicts with packages installed in the computer's global Python.
`pypdf[crypto]` includes support for the source PDF's AES encryption with an
empty opening password.

Fill in the real `.env` privately using the values supplied by the instructor:

```dotenv
HF_CHAT_ENDPOINT=https://<chat>.endpoints.huggingface.cloud
HF_EMBED_ENDPOINT=https://<embed>.endpoints.huggingface.cloud
HF_TOKEN=<token-shared-privately>
```

Use deployment root URLs, not Hub model pages. Do not append `/v1` or `/embed`;
the client adds the API routes. The exact name is `HF_EMBED_ENDPOINT`, with a final `T`.

```powershell
& $py app.py --inspect
& $py app.py --prepare-only
& $py -m streamlit run streamlit_app.py
```

If Streamlit is already running, open its page instead of starting a second instance.

## macOS or Linux setup

Use Python 3.12 or 3.13 from the project folder:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
test -f .env || cp .env.example .env
```

Fill in `.env`, then run:

```bash
./.venv/bin/python app.py --inspect
./.venv/bin/python app.py --prepare-only
./.venv/bin/python -m streamlit run streamlit_app.py
```

The reference validation was performed on Windows. On macOS or Linux, use
`./.venv/bin/python` in place of the PowerShell interpreter path in later examples.

## Streamlit demonstration

1. Open **http://127.0.0.1:8501**, select `A-6.002.pdf`, and confirm 18 pages and 110 passages.
2. Click **Start processing**. Preparation checks the endpoints, resumes cached
   embedding batches and builds the index used by this interface.
3. Ask: **À quelle fréquence le personnel renouvelle-t-il ses engagements de confidentialité ?**
4. The expected answer is **chaque année**. Expand **View retrieved passages**
   and verify page 5, passage `a-6_002_p0005_0195`.
5. Click **New conversation**, then ask: **Quel est le mot de passe du Wi-Fi de Revenu Québec ?**
6. The expected refusal is **Je ne trouve pas cette information dans les documents.**

The interface stays local to this computer. Each question reuses the prepared
index. Recent dialogue helps interpret follow-up questions but is not evidence.
Changing the PDF or chunk settings clears the conversation and requires preparation.
**New conversation** clears the dialogue while keeping the index ready.

## Terminal interface

```powershell
& $py app.py
```

This prepares the document and opens a conversation in the terminal. Answers
are in French and include source identifiers. Use `/new` to clear the dialogue,
`/quit` to exit, or `Ctrl+C` to stop.

```powershell
# Inspect a PDF without calling Hugging Face.
& $py app.py --inspect

# Ask one question about a different PDF.
& $py app.py --pdf "my_document.pdf" --question "Que dit ce document ?"

# Prepare the cache and CLI index, then exit.
& $py app.py --prepare-only

# Run the seven evaluation questions and save reports.
& $py app.py --evaluate --k 3
```

`--chunk-size`, `--overlap`, `--batch-size` and `--k` control passage length,
overlap, embedding batches and retrieved passage count. Defaults are 80, 15, 8 and 3.
Evaluation questions and expected evidence are defined in `rag/evaluation.py`.
Update them when using another corpus for evaluation; ordinary chat does not need them.

## Caching, persistence and recovery

Both interfaces share `.rag_cache/`. Each completed embedding batch is saved
immediately, so a restart resumes missing batches. Tokens are never written to
this cache. The cache identity includes the document, chunk settings, model,
endpoint and model revision when the server exposes one.

If a model is redeployed at the same URL without an exposed revision, select
**Recompute embeddings** or use `--force-rebuild`. This creates a fresh cache
generation. After an interruption, retry without forcing again to resume the
new generation. Streamlit automatically clears this checkbox when preparation starts.

The local indexes are separate: `base_vectorielle/cli/` and
`base_vectorielle/streamlit/`. Embedded Chroma processes must not write to the
same index simultaneously. Run only one instance of each interface.
See the [Chroma constraints](https://cookbook.chromadb.dev/core/system_constraints/).

## Evaluation and checks

```powershell
& $py -m unittest discover -s tests -v
& $py app.py --evaluate --k 3
```

Offline tests use simulated HTTP responses and temporary databases. They do
not require a real token or make inference requests. Live evaluation creates
timestamped JSON and CSV reports in `resultats/`; human quality scores still
need to be assigned.

The recorded reference run passed 73 automated tests and completed 7/7 live
evaluation calls. The correct evidence appeared in the top three passages for
4/5 factual questions; both out-of-document questions produced the expected
abstention. These are separate measures on a small test set, not general accuracy.
Comparing `--k 3` with `--k 5` is an experiment, not a promised improvement.

Before the session, check dependencies, inspect the document, prepare the cache
and test a real question. If an endpoint is unavailable, use the recorded
application views on slides 14 and 17 and the source quotation on slide 18,
clearly identifying them as recorded examples. Stop Streamlit with `Ctrl+C` in its terminal. Manage the
Hugging Face endpoints separately; closing the local application does not pause them.

## Credentials and sharing

`.env.example` contains no secrets. The real `.env` is private and ignored by Git.
Share the token separately, keep it out of screenshots and archives, and revoke
the workshop token when the session ends. The required source passages and
questions are sent to the hosted endpoints; use an appropriate demonstration
document. Hosted endpoints must be accessible and may incur costs.

Distribute `app.py`, `streamlit_app.py`, `.streamlit/`, `rag/`, `README.md`,
`requirements.txt`, `.env.example`, the source PDF and the presentation materials.
Include `tests/` if participants will run the checks. Do not distribute `.env`,
`.venv/`, `.rag_cache/`, `base_vectorielle/` or historical `resultats/` data.
The old notebooks and Ollama variant are not included.
