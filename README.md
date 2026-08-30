# Glass Box RAG

[![tests](https://github.com/enescaglarr/glass-box-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/enescaglarr/glass-box-rag/actions/workflows/tests.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)

A RAG (Retrieval-Augmented Generation) customer support assistant. An inbound customer query is embedded, matched against a FAISS vector store built from a real customer-service dataset, and the closest historical responses are handed to Gemini as context. The model returns three things — an urgency score, a category, and a drafted reply — which a human agent can accept or regenerate with feedback.

It is a **human-in-the-loop drafting tool**, not an autonomous bot: the agent always sees the retrieved source records next to the generated draft — and, in four panels under every answer, the whole trace of how it was produced.

Most RAG demos show you the answer. This one shows you why: which rows were retrieved and how close they actually were, which were fetched and discarded, the exact prompt that got assembled, and whether the retrieval was confident enough to trust. [`demo.md`](demo.md) records what that visibility turned up — eight queries measured before and after a round of fixes, including the ones that did not get fixed.

---

## Quick Start

For anyone who just wants the commands — see the [Detailed Setup Guide](#detailed-setup-guide) below for explanations, quota notes, and troubleshooting.

```bash
# 1. Clone and install
git clone https://github.com/enescaglarr/glass-box-rag.git
cd glass-box-rag
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# open .env and paste your Gemini API key from https://aistudio.google.com/apikey

# 3. Build the vector store (embeds a sample of the dataset, takes a few minutes)
python build_index.py

# 4. Run
streamlit run demo.py
```

Opens at http://localhost:8501.

That is the whole app — the trace of each answer is built into it. `make_report.py` is only needed if you want the same trace for a fixed set of queries as a standalone static page to share:

```bash
python make_report.py --serve      # http://localhost:8000
```

---

## How It Works

```
Customer query
      │
      ▼
Gemini embedding (gemini-embedding-001, 768-dim, task_type=retrieval_query)
      │
      ▼
FAISS IndexFlatL2 search  ──►  top-3 most similar historical instructions
      │
      ▼
Their `response` texts become the context for the prompt
      │
      ▼
Gemini 3.5 Flash Lite returns:
      1. Urgency (1–5)
      2. Category (sales / product / operations …)
      3. A drafted reply grounded in the retrieved examples
      │
      ▼
Agent reviews ──► Accept, or write feedback ──► Regenerate (point 3 only)
```

Vectors are L2-normalised before indexing, so FAISS's L2 distance ranks identically to cosine similarity. Lower distance = more similar.

---

## Detailed Setup Guide

### 1. Install prerequisites

| Tool | Check if installed | Install if missing |
|---|---|---|
| Python 3.12+ | `python3 --version` | [python.org/downloads](https://python.org/downloads) |
| Git | `git --version` | [git-scm.com/downloads](https://git-scm.com/downloads) |

**Python 3.12 or newer.** `numpy 2.5` is the binding constraint: it publishes no wheels for 3.11 or below, so the pinned set cannot be installed there. CI verifies 3.12 and 3.13 on every push.

### 2. Clone and install dependencies

```bash
git clone https://github.com/enescaglarr/glass-box-rag.git
cd glass-box-rag
python3 -m venv .venv
```

Activate the virtual environment (repeat this every time you open a new terminal):

```bash
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows (cmd / PowerShell)
```

Then:

```bash
pip install -r requirements.txt
```

This installs `faiss-cpu`, `google-genai`, `streamlit`, `pandas`, `numpy`, and `python-dotenv` into the virtual environment only — your system Python is untouched.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | What it is | Required? |
|---|---|---|
| `GEMINI_API_KEY` | Your Google AI Studio key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Yes |

`.env` is gitignored — your real key never gets committed. `.env.example` (with a placeholder) is the template that ships in the repo.

`src/helper.py` also accepts `GOOGLE_API_KEY` as a fallback name, so either works.

The SDK is **`google-genai`** (`from google import genai`), not the retired `google-generativeai`. If you find older tutorials calling `genai.configure()` or `genai.GenerativeModel()`, they are for the dead package — this project builds one `genai.Client()` in `helper.py` and calls `client.models.embed_content()` / `client.models.generate_content()`.

### 4. Build the vector store

```bash
python build_index.py
```

This samples the dataset, embeds each instruction with the Gemini embedding API, and writes two files:

- `vector_store/faiss_index.index` — the FAISS index
- `vector_store/indexed_rows.csv` — the exact rows that went into it, so the app looks up the same records the index holds

**Why a sample and not all 26,872 rows?** The free-tier embedding quota is **1,000 requests per day**, and every row costs one request. Indexing the whole (deduplicated) dataset would take about 25 days of quota. Sampling evenly per intent keeps all 27 intents reachable by the retriever while fitting inside a single day.

| Command | Vectors | Fits free daily quota? |
|---|---|---|
| `python build_index.py` | 405 (15/intent, default) | Yes |
| `python build_index.py --per-intent 35` | 945 | Yes, just barely |
| `python build_index.py --all` | 24,635 | No — ~25 days, or a paid tier |

### 5. Run the application

```bash
streamlit run demo.py
```

Enter a query such as *"i cant cancel my order and i need my money back"*, press **Get response from internal dataset and Run LLM**, and you'll see the retrieved dataset rows followed by the LLM's draft. If the draft isn't right, type feedback and press **Regenerate Response** — only the drafted reply is regenerated; the urgency and category are preserved.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: No API key found` | `.env` missing or the key name is wrong | Run `cp .env.example .env` and paste your key into `GEMINI_API_KEY` |
| Streamlit shows *"Vector store not found"* | `build_index.py` hasn't been run yet | Run `python build_index.py` |
| The app shows *"Out of quota"* | The 500/day or 15/minute generation cap is used up | Wait a minute, or until the daily reset. The app catches this and explains it instead of showing a traceback |
| `429 ResourceExhausted ... embed_content_free_tier_requests` | Daily/minute embedding quota hit | The build script retries automatically with the delay the API asks for. If it persists, you've hit the 1,000/day cap — wait for reset or lower `--per-intent` |
| `429 ... generate_content_free_tier_requests` | LLM daily cap hit | `gemini-3.5-flash-lite` allows 500 requests/day. If you switched `LLM_MODEL` to a non-lite Flash model, the cap drops to **20/day** — switch back |
| `404 models/... is not found` | Using a retired model name | Model names are centralised at the top of `src/helper.py`. Run `client.models.list()` to see what your key can reach |
| `AttributeError: module 'google.genai' has no attribute 'configure'` | Following a tutorial written for the retired `google-generativeai` package | This project uses `google-genai`: build a `genai.Client(api_key=...)` and call `client.models.*` |
| Retrieval returns unrelated intents | Index built with too small a sample | Rebuild with a higher `--per-intent` |
| `Could not find a version that satisfies the requirement numpy==...` | Python older than 3.12 | Check `python3 --version`. numpy 2.5 ships no wheels below 3.12, so the pinned set needs 3.12 or newer |
| `bad interpreter: .../python: no such file or directory` | The project folder was moved after the venv was created — venv console scripts hardcode an absolute path | Either run everything through the module form (`python -m streamlit run demo.py`, `python -m pip install ...`) or recreate the venv: `rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |

---

## Description of Files and Folders

**`src/helper.py`**
All the modular functions the project needs: `embed_texts` (batched, rate-limited, retries on 429), `create_embeddings`, `create_index`, `semantic_similarity`, and `call_llm`. Model names and quota constants live at the top as the single source of truth, so the index and the queries can never drift apart.

**`build_index.py`**
CLI script that samples the dataset, embeds it, and writes the FAISS index plus the matching row CSV. Accepts `--per-intent`, `--all`, and `--seed`.

**`demo.py`**
The Streamlit app: query box, the answer (urgency, category, drafted reply), the four trace panels, and the accept / feedback / regenerate loop.

**`src/trace.py`**
`trace_query()` — runs one query through the pipeline and records every intermediate step: the query vector's head and norm, the retrieved rows with distances and cosine similarities, the assembled prompt (whole, and split into literal / interpolated spans), the raw output, its three parsed fields, and per-stage timings. Both `demo.py` and `make_report.py` call it, so the live app and the static report can never disagree.

**`tests/`**
Unit tests for the pure logic — retrieval selection, output parsing, prompt assembly, normalisation, and the eval scoring itself. They make no API calls and need no key, so they run in CI in about a second. `pytest`.

**`.github/workflows/tests.yml`**
Runs `pytest` on Python 3.10 and 3.13 on every push. No secrets configured, by design.

**`eval.py`**
The evaluation suite. Runs eleven queries whose correct intents are known and scores retrieval hit rate, whether the right intent ranked first, whether urgency lands in a sensible band, whether confidence stays honest on unanswerable queries, language matching, and placeholder leaks. Exits non-zero on failure, so it works in CI. `--only T1 T6` to iterate, `--json out.json` for raw results.

**`accepted_replies.csv`**
Written at runtime. Every draft an agent accepts is appended here with its query, urgency, category, confidence and the model that produced it. Downloadable from the sidebar. Gitignored.

**`LICENSE`**
MIT for the code. The dataset keeps its own CDLA-Sharing-1.0 licence.

**`make_report.py`**
Runs `trace_query()` over the fixed `QUERIES` list and renders the results as a standalone static page — the **RAG Trace Explorer**. Serves it over `http.server`. Accepts `--serve`, `--port`, and `--cached`.

**`report_template/`**
The report's `base.html`, `style.css`, and `app.js`. Edit these to change the report's design — `make_report.py` copies them into `report/` and injects the recorded run as `data.js`.

**`report/`**
Generated output of `make_report.py` (gitignored). Also holds `trace.json`, the recorded run that `--cached` replays.

**`Customer_Support_Training_Dataset/`**
The Bitext customer-support dataset (`.csv`) and its documentation (`dataset.md`).

**`vector_store/`**
Where the generated FAISS index and its row CSV land. Contents are gitignored — rebuild with `build_index.py`.

**`requirements.txt`**
Python dependencies.

**`.env.example`**
Template for the environment variables the app needs. Copy to `.env` and fill in your own key.

---

## Dataset

[**Bitext — Customer Service Tagged Training Dataset for LLM-based Virtual Assistants**](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

Open source under the **Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)**. Free to use, share, and modify with attribution; derivatives must be shared under the same license. © Bitext Innovations, 2023–2024.

It is a **synthetic** dataset generated with NLP/NLG tooling and automated labelling — not scraped from real customer conversations, so there is no PII in it.

Each row has five fields: `flags`, `instruction` (the customer's request), `category`, `intent`, and `response` (the expected assistant reply).

### Analysis Results

Run on the full CSV. These figures come from the exploratory analysis this project previously kept in a notebook.

**Size**

| Metric | Value |
|---|---|
| Rows | 26,872 |
| Unique instructions | 24,635 (2,237 exact duplicates) |
| Unique responses | 26,870 |
| Categories | 11 |
| Intents | 27 |
| Total tokens | ~3.57M |
| Avg. words per instruction | 8.7 |
| Avg. words per response | 104.8 |

Instructions are short and responses are long — which is exactly the shape RAG wants: cheap to embed, rich to use as context.

**Category distribution** — uneven at the category level, even at the intent level:

| Category | Rows | Intents |
|---|---|---|
| ACCOUNT | 5,986 | 6 |
| ORDER | 3,988 | 4 |
| REFUND | 2,992 | 3 |
| INVOICE | 1,999 | 2 |
| CONTACT | 1,999 | 2 |
| PAYMENT | 1,998 | 2 |
| FEEDBACK | 1,997 | 2 |
| DELIVERY | 1,994 | 2 |
| SHIPPING | 1,970 | 2 |
| SUBSCRIPTION | 999 | 1 |
| CANCEL | 950 | 1 |

Every intent has between 950 and 1,000 rows (mean 995). The category imbalance is purely a function of how many intents each category contains — ACCOUNT looks 6× bigger than CANCEL only because it holds 6 intents to CANCEL's 1.

**This is why `build_index.py` samples per *intent*, not per category** — sampling per category would over-represent ACCOUNT and starve the single-intent categories.

**Language variation flags** — the share of rows carrying each tag (rows carry several):

| Flag | Meaning | Share |
|---|---|---|
| B | Basic syntactic structure | 100.0% |
| L | Semantic variation (synonyms) | 89.7% |
| Q | Colloquial ("can u cancel my ordr") | 33.4% |
| I | Interrogative structure | 29.2% |
| Z | Errors and typos | 19.7% |
| M | Morphological variation | 18.3% |
| C | Coordinated structure | 9.8% |
| K | Keyword mode ("cancel order") | 8.3% |
| E | Abbreviations | 7.0% |
| P | Politeness variation | 4.9% |
| W | Offensive language | 4.8% |
| N | Negation | 1.7% |

Roughly a third of the corpus is colloquial and a fifth contains deliberate typos. That is the point: the retriever is trained against messy input rather than clean prose, which is what real inbound support queries look like. See `Customer_Support_Training_Dataset/dataset.md` for the full tag reference and the 30 entity placeholders (`{{Order Number}}`, `{{Customer Support Email}}`, …).

---

## Reading a Trace

Every answer in the app comes with four collapsed panels showing how it was produced:

1. **The query as a vector** — the first 8 of 768 dimensions and the L2 norm, showing the normalisation step.
2. **What the retriever pulled back** — the three nearest rows with cosine similarity, L2 distance, intent, category, and Bitext flags, plus the full response text handed to the model. The panel header says whether all three hits agree on one intent or split across several.
3. **The assembled prompt** — the real prompt string, so it is obvious which part is the fixed template and which part came from your data.
4. **The raw model output** — before it gets parsed into urgency / category / reply.

The header line gives per-stage timings. The most useful number is the best cosine similarity: on these 270 vectors, a clean match lands around 0.85–0.89 against a single intent, while a query with no good match drops to ~0.71 and splits across intents. That is a threshold you can act on.

### The standalone report

`make_report.py` runs the same `trace_query()` over a fixed list of queries and renders them as a static page you can share without an API key — useful for a demo or writeup.

```bash
python make_report.py --serve              # trace, build, serve on :8000
python make_report.py                      # trace and build only
python make_report.py --cached --serve     # replay the last run, zero API calls
python make_report.py --serve --port 8080
```

Each traced query is broken into four stages:

1. **The query as a vector** — the first 8 of 768 dimensions and the L2 norm, showing the normalisation step.
2. **What the retriever pulled back** — the three nearest rows with cosine similarity, L2 distance, intent, category, and Bitext flags, plus the full response text that gets handed to the model.
3. **The assembled prompt** — the real prompt string with the interpolated customer query and the retrieved-response list highlighted separately, so it is obvious which part is the template and which part came from your data.
4. **What came back** — the urgency score, category, and drafted reply, parsed apart, with per-stage timings.

A live run costs 4 embedding requests and 4 generation requests. `--cached` rebuilds the page from `report/trace.json` and costs nothing, which is what you want while editing `report_template/`.

The queries traced are defined in the `QUERIES` list at the top of `make_report.py` — edit it to trace your own. The page keeps no copy of the prompt template: `trace_query()` emits the prompt already split into literal and interpolated spans, and `app.js` just renders them.

---

## Tests

Two layers, deliberately separate.

```bash
pip install -r requirements-dev.txt
pytest                            # 57 unit tests, ~1s, no API key needed
python eval.py                    # 11 end-to-end cases, ~1 min, spends quota
```

`pytest` covers the logic that can be checked without a model: the tie-break and intent
deduplication in `select_hits`, the three-field output parsing, prompt assembly, vector
normalisation, and `eval.py`'s own scoring. The Gemini client is built on first use rather
than at import, so the module loads and the tests run without credentials — which is why
the CI workflow configures no secrets.

Several tests document a bug that actually happened rather than a hypothetical one:

- `test_parts_join_to_the_whole` — the prompt was once rewritten in `helper.py` while a
  second copy lived in `trace.py`, and the app's prompt panel displayed something that was
  not what got sent. There is now one definition, and this test fails if that changes.
- `test_prefers_the_fuller_response_when_scores_tie` — a Turkish query retrieved three rows
  0.002 apart and picked the one whose stored reply was a 146-character non-answer.
- `test_one_row_per_intent` — a three-part request had every retrieval slot filled by one
  intent.
- `test_reply_keeps_its_own_numbered_list` — replies now contain numbered steps, and the
  output format is itself numbered.

---

## Evaluation

```bash
python eval.py                  # all cases, prints a report, exits non-zero on failure
python eval.py --only T5 T10    # while iterating
python eval.py --json out.json  # raw results too
```

Eleven queries with known-correct intents, scored on six axes:

| | |
|---|---|
| Retrieval | did an expected intent come back, and did it rank first? |
| Confidence | does an unanswerable query avoid being reported as a high-confidence match? |
| Urgency | does the score land in the band a human reviewer would accept? |
| Language | is the reply in the language the customer wrote in? |
| Hygiene | any `{{placeholders}}` left in the reply? |
| Separation | do answerable and unanswerable queries occupy disjoint cosine ranges? |

Current state, on the 270-vector index:

```
Retrieval hit rate        8/8
Correct intent ranked #1  7/8
Urgency in expected band  11/11
Confidence honest         11/11
Replied in query language 11/11
Placeholder leaks         0

Cosine separation: answerable 0.831-0.886 | unanswerable 0.732-0.801
```

Three cases (T5, T7, T10) have **no** correct answer in the corpus. They are there to check
the pipeline says so rather than inventing a match. T10 — *"i was charged twice"* — earns
its place: every `payment_issue` row in the dataset is about a payment that **failed**, not
one that went through twice, so the honest response is low confidence.

A full run costs 11 embedding and 11 generation requests.

---

## Models and Free-Tier Quotas

Model names are set once at the top of `src/helper.py`:

```python
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL       = "models/gemini-3.5-flash-lite"
EMBEDDING_DIM   = 768
```

Free-tier limits are what drive both choices:

| Model | Role | RPM | TPM | **Requests/day** |
|---|---|---|---|---|
| `gemini-embedding-001` | Retrieval | 100 | 30K | **1,000** |
| `gemini-3.5-flash-lite` | Generation | 15 | 250K | **500** |
| `gemini-3.5-flash` | *(not used)* | 5 | 250K | **20** |
| `gemini-2.5-flash` | *(not used)* | 5 | 250K | **20** |

Calls go through the `google-genai` SDK. A batched `embed_content` accepts at most **100** texts per request (a 400 otherwise); `embed_texts` sends 50 and paces itself, retrying on `ClientError` with `code == 429` using the delay the API names in its message.

The daily cap is the binding constraint, not the per-minute rate. The non-lite Flash models allow only **20 requests per day**, which is unusable for a support desk — hence `flash-lite`, which allows 500. Check your own limits at [ai.dev/rate-limit](https://ai.dev/rate-limit); they vary by account and change over time.

`gemini-embedding-001` outputs 3,072 dimensions by default. This project requests 768 instead, which shrinks the index ~4× with negligible retrieval loss. Truncated Gemini embeddings are **not** unit length, so `helper.py` normalises them explicitly before indexing and searching.

`embed_texts` batches 50 texts per call and paces itself to stay under the per-minute quota, retrying with the delay the API returns when a 429 does come back.

---

## Notes and Limitations

- **The index only covers what you embedded.** With the default sample, retrieval draws from 405 of 26,872 rows. Queries far from any sampled instruction will return weak matches.
- **Responses contain placeholders.** Retrieved text carries entity tokens like `{{Order Number}}` and `{{Website URL}}`, which flow into the generated draft. In production these would be filled from your CRM before the agent sees them.
- **The regenerate loop re-sends the full previous response** in the prompt, so feedback rounds cost more tokens than the first call.
- **Rebuild the index whenever you change `EMBEDDING_MODEL` or `EMBEDDING_DIM`** — an index built with one embedding configuration cannot be searched with another.
- **pandas 3.0 changed `groupby().apply()`** to drop the grouping column from the frames it passes in. `build_index.py` builds its per-intent sample with an explicit loop instead, so it behaves the same on pandas 2 and 3.
