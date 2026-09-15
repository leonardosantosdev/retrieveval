# RetrievEval

Benchmark retrieval strategies on **your** corpus, locally, without a paid API.

RAG systems are usually built on an assumption — "vector search is probably
good enough" — that nobody ever measures. RetrievEval answers one question with
numbers instead:

> Given my corpus and a labeled evaluation set, which retrieval configuration
> gives me the best quality/latency trade-off?

It compares BM25, dense, hybrid (RRF) and cross-encoder reranking through one
interface, reports Recall@K, MRR, nDCG@K and latency for each, and writes a
JSON artifact recording the configuration that produced them.

It is not a RAG framework. There is no chatbot, no answer generation, no LLM
in the loop. Retrieval quality only.

---

## What this actually does (read this if RAG is new to you)

A RAG system answers a question by first **retrieving** a handful of relevant
chunks of text from your documents, then handing those chunks to an LLM to
write the answer. If the wrong chunks get retrieved, the LLM writes a
confident, wrong answer no matter how good the LLM is. RetrievEval only tests
that first step — it never calls an LLM, never generates an answer, and
**nothing here is trained or fine-tuned.**

There are a few ways to do "retrieval," and none is universally best — it
depends on your documents and how people phrase questions about them:

- **BM25** — classic keyword search. Fast, no model needed, excellent at exact
  terms, error codes, and identifiers. Misses paraphrases (a query about
  "cancelling" won't match a document that only says "terminate").
- **Dense** — searches by *meaning* using an embedding model, so a paraphrase
  can match even with zero shared words. Slower, and occasionally too fuzzy.
- **Hybrid** — runs both and merges the two rankings (via a technique called
  Reciprocal Rank Fusion), aiming to get the strengths of each.
- **Reranker** — an extra, slower pass: take the top candidates from any of the
  above and have a second, more careful model re-sort them. Usually the most
  accurate option, and the slowest.

RetrievEval runs your documents and your test questions through all four and
reports, in numbers, which one actually works best for *your* case — instead
of guessing.

**The evaluation dataset (`eval.json`) is not training data.** No model
learns from it and no weights change. It works more like a quiz with an
answer key that *you* write by hand: a list of realistic questions, and which
document should answer each one. RetrievEval runs those questions against each
strategy and grades the results against your answer key. See
[Evaluation dataset](#2-evaluation-dataset) below for the format.

---

## Install

```bash
git clone <repository-url>
cd retrieveval
python -m venv .venv && source .venv/bin/activate
pip install -e .

# only if your corpus contains Word documents
pip install -e ".[office]"
```

Python 3.10+. The first run downloads two small open-source models (~120 MB
total) from Hugging Face; after that everything runs offline.

`.docx` support is an optional extra so the base install stays lean; `.txt`,
`.md`, `.pdf`, `.html` and `.htm` work out of the box.

## Try it on the bundled example

```bash
retrieveval evaluate \
  --corpus ./examples/documents \
  --dataset ./examples/eval.json \
  --config ./retrieveval.example.yaml
```

```text
Retrieval quality (best per metric in bold)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Strategy          ┃   R@1 ┃   R@5 ┃  R@10 ┃   MRR ┃   N@1 ┃   N@5 ┃  N@10 ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ BM25              │ 0.694 │ 0.972 │ 1.000 │ 0.863 │ 0.778 │ 0.869 │ 0.881 │
│ Dense             │ 0.861 │ 0.972 │ 1.000 │ 0.963 │ 0.944 │ 0.946 │ 0.957 │
│ Hybrid            │ 0.750 │ 1.000 │ 1.000 │ 0.907 │ 0.833 │ 0.916 │ 0.916 │
│ Hybrid + reranker │ 0.861 │ 1.000 │ 1.000 │ 0.972 │ 0.944 │ 0.975 │ 0.975 │
└───────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

Latency (Index is a one-off cost; the rest is per query)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Strategy          ┃ Index ┃    Avg ┃    p50 ┃    p95 ┃    Max ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ BM25              │ 3.0ms │ 0.06ms │ 0.06ms │ 0.09ms │ 0.10ms │
│ Dense             │ 8.05s │  9.3ms │  9.1ms │   10ms │   11ms │
│ Hybrid            │ 8.06s │  9.3ms │  9.4ms │   10ms │   11ms │
│ Hybrid + reranker │ 12.3s │   26ms │   27ms │   29ms │   33ms │
└───────────────────┴───────┴────────┴────────┴────────┴────────┘

  Highest nDCG@10: Hybrid + reranker (0.975), p95 29ms
  Lowest p95 latency: BM25 (0.09ms), nDCG@10 0.881
  Highest nDCG@10 without reranking: Dense (0.957), p95 10ms
```

That example corpus is deliberately keyword-heavy support documentation, and
BM25 does well on it — which is the point. On a different corpus the ordering
changes, and finding that out is the whole exercise.

RetrievEval names leaders on stated metrics. It will not tell you which
strategy to ship: that depends on your latency budget, your corpus growth and
what happens downstream, none of which this tool can see.

---

## Your own corpus

### 1. Documents

Point `--corpus` at a directory, read recursively; hidden files are ignored.

| Extension | Notes |
|---|---|
| `.txt`, `.md` | Read as UTF-8. |
| `.pdf` | Text layer only — no OCR. |
| `.html`, `.htm` | Prose is extracted; `<script>` and `<style>` are dropped. Covers Confluence, Notion and SharePoint exports. |
| `.docx` | Paragraphs and table content, in document order. Needs `pip install -e ".[office]"`. Legacy `.doc` is a different format and is not supported. |

```text
documents/
├── billing.md
├── policies/
│   └── refunds.docx
├── faq/
│   └── cancellation.md
└── api/authentication.pdf
```

A document's **id** is its corpus-relative path — `faq/cancellation.md`. That
is what you write in your labels.

### 2. Evaluation dataset

This is the answer key mentioned above — a JSON array. Each entry needs a
query and the documents that should be retrieved for it. Nothing is trained
on this file; it is only used to grade each strategy's search results.

```json
[
  {
    "query": "How do I cancel my subscription?",
    "relevant_documents": ["faq/cancellation.md"]
  },
  {
    "query": "Why does authentication fail after token expiration?",
    "relevant_documents": ["api/authentication.pdf", "troubleshooting.md"],
    "notes": "optional; `id` is allowed too"
  }
]
```

Writing this by hand is the real work, and RetrievEval does not generate it for
you — a golden set the tool invented would measure the tool, not your system.
Thirty honest queries beat three hundred guessed ones.

`relevant_chunks` is also accepted, holding chunk ids such as
`faq/cancellation.md::chunk_2`. Chunk ids depend on your chunking settings, so
chunk-level labels have to be regenerated when those change; document-level
labels do not, which is why they are the default.

### 3. Configuration

Every key is optional. See [`retrieveval.example.yaml`](retrieveval.example.yaml).

```yaml
chunking:
  size: 500        # characters, not tokens
  overlap: 50

retrievers: [bm25, dense, hybrid]

dense:
  model: sentence-transformers/all-MiniLM-L6-v2
  cache_dir: .retrieveval_cache

hybrid:
  rrf_k: 60        # RRF smoothing constant
  depth: 100       # candidates per retriever before fusing

reranker:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  candidates: 50   # shortlist handed to the cross-encoder
  applies_to: [hybrid]

evaluation:
  k: [1, 5, 10]
```

A misspelled key is an error, not a silent default — otherwise you would
benchmark something other than what you configured.

---

## Commands

```bash
retrieveval evaluate --corpus DIR --dataset FILE [--config FILE] [--output DIR]
retrieveval index    --corpus DIR [--config FILE]
```

`index` only builds and caches the corpus embeddings. Embedding is the slow
part of a first run, and it is worth doing deliberately rather than discovering
it halfway through a benchmark.

Useful flags: `--no-save` (skip the JSON/Markdown artifacts), `--skip-unsupported`
(ignore files whose type cannot be read).

---

## How the numbers are computed

**Recall@K** — the fraction of a query's relevant documents that appear within
the top K retrieved *chunks*. **MRR** — the mean of `1 / rank` of the first
relevant result. **nDCG@K** — ranking quality, weighting hits near the top more
heavily, normalised so 1.0 is the best achievable ranking.

One rule matters when reading these: **a result counts as relevant only the
first time its document appears.** A retriever returning five chunks of the
same relevant document has found one relevant document, not five. Repeated
chunks are graded zero but keep their rank positions, so near-duplicate results
correctly cost you nDCG.

Latency is measured per query, after one untimed warm-up query that absorbs
one-off lazy initialisation. Index construction is timed separately and
reported apart, because building an index offline and serving a query online
are different performance problems. Index times overlap between strategies by
design — hybrid reuses the BM25 and dense indexes — so they should not be added
up.

### Results artifacts

Each run writes two files under the same timestamp, `results/<timestamp>.json`
and `results/<timestamp>.md`:

- **`.json`** — the machine-readable record: metrics, all latency statistics, a
  per-query breakdown, and the full configuration (chunk size and overlap,
  embedding model, RRF constant, reranker model and candidate pool), plus the
  Python version, platform and whether CUDA was available. A number without its
  configuration is not reproducible, and a latency without its hardware is not
  comparable. This is what a later run or script should read.
- **`.md`** — the same summary printed to the console, saved as a standalone
  document: the quality and latency tables plus the named leaders. Meant for
  reading later or sharing, without needing to re-run anything or parse JSON.

---

## Assumptions and limitations

Worth knowing before you trust a number:

- **Relevance is binary.** A document is relevant or it is not; there are no
  graded judgments.
- **Chunk size is in characters, not tokens.** This keeps chunking independent
  of whichever embedding model is configured, so every retriever is compared on
  identical chunks. Convert roughly at ~4 characters per token.
- **Recall@K counts documents found within K chunks**, not within K documents.
  It answers "if I put the top K chunks in my context window, do the right
  documents make it in?".
- **BM25 does no stemming and drops no stopwords.** Identifiers, error codes and
  accented words survive intact, which is where a lexical baseline earns its
  place — but "invoice" will not match "invoices".
- **Dense search is exact brute force**, a matrix product over every chunk. It
  is correct at any size but linear in corpus size, so its latency here is not
  what an approximate index (HNSW, IVF) would give you at millions of chunks.
  These figures compare *strategies*, not vector database implementations.
- **Latency depends on your machine**, above all on whether a GPU was used. The
  JSON records this; the console table does not.
- **p95 over a small query set is an interpolation** between your two or three
  slowest queries. The sample count is recorded next to it.
- **Your evaluation set bounds everything.** Metrics computed over 10 queries
  are noise. Incomplete labels — a document that *is* relevant but is not
  listed — are punished as false positives and systematically understate
  quality.
- **No OCR.** A scanned, image-only PDF yields no text and is reported as an
  error rather than silently indexed as empty.
- **Spreadsheets, email and presentations are not supported, deliberately.**
  A spreadsheet is not prose: cutting it into 500-character chunks produces
  semantically meaningless fragments and a benchmark number that looks valid
  but means nothing. If you need tabular data retrieved, serialise the rows
  into sentences first — that is a corpus-preparation decision, not something a
  loader should guess at. Email needs thread and quoted-reply handling, and
  `.pptx` text arrives as fragmented bullets with no guaranteed reading order.
  Supporting any of these half-heartedly would be worse than not supporting
  them.
- **A first run needs network access** to download models. After that, nothing
  external is required.

---

## Development

```bash
pip install -e ".[dev]"
pytest                    # everything
pytest -m "not slow"      # skips tests that need local models
```

The metric implementations are tested against small rankings whose expected
values are derived by hand in the test source, so their behaviour is auditable
rather than assumed.

### Layout

```text
src/retrieveval/
├── cli.py                  # commands, and the promise of clear errors
├── config.py               # YAML -> validated configuration
├── models.py               # Document, Chunk, SearchResult, EvalQuery
├── ingestion/              # loading documents (txt/md/pdf/html/docx), chunking them
├── retrieval/              # base interface, bm25, dense, hybrid, reranker, factory
├── evaluation/             # dataset, judgments, metrics, runner, results
└── reporting/              # console tables and the JSON artifact
```

Every strategy implements the same `Retriever` interface — `index`, `search`,
`describe` — and the benchmark runner uses nothing else. Adding a retrieval
strategy means writing that interface and registering it in the factory; no
change to evaluation or reporting is required.

## License

MIT.
