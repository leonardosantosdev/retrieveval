# RetrievEval — Project Context

## Purpose

RetrievEval is a local-first, open-source toolkit for developers building Retrieval-Augmented Generation (RAG) systems who need to determine which retrieval strategy works best for their own corpus and query patterns.

The project focuses exclusively on **retrieval quality**. It does not aim to be a chatbot framework, an LLM orchestration platform, or a production RAG application.

The core question RetrievEval answers is:

> Given my corpus and a labeled evaluation dataset, which retrieval configuration gives me the best quality/latency trade-off?

The toolkit should make retrieval measurable instead of relying on manual spot checks or assumptions such as "vector search is probably good enough."

---

## Target User

A developer already building or evaluating a RAG system.

They have:

1. A corpus of documents.
2. A set of evaluation queries.
3. Ground-truth relevance labels indicating which documents or chunks should be retrieved for each query.

They want to compare retrieval approaches before choosing a production configuration.

---

## Core User Flow

The intended workflow is CLI-first and fully local.

```bash
git clone <repository-url>
cd retrieveval
pip install -e .
```

The user provides a corpus:

```text
documents/
├── billing.md
├── cancellation.md
├── authentication.md
└── troubleshooting.md
```

They provide an evaluation dataset:

```json
[
  {
    "query": "How do I cancel my subscription?",
    "relevant_documents": ["cancellation.md"]
  },
  {
    "query": "Why does authentication fail after token expiration?",
    "relevant_documents": ["authentication.md", "troubleshooting.md"]
  }
]
```

They configure the benchmark:

```yaml
chunking:
  size: 500
  overlap: 50

retrievers:
  - dense
  - bm25
  - hybrid

reranker:
  enabled: true

evaluation:
  k:
    - 1
    - 5
    - 10
```

Then run:

```bash
retrieveval evaluate   --corpus ./documents   --dataset ./eval.json   --config ./retrieveval.yaml
```

Example output:

```text
Retrieval benchmark

Strategy                  Recall@5    MRR    nDCG@10    Avg latency
Dense                       0.81      0.70      0.76        41ms
BM25                        0.74      0.64      0.69        18ms
Hybrid                      0.89      0.79      0.84        57ms
Hybrid + reranker           0.93      0.86      0.90       143ms
```

The toolkit should help the developer understand the trade-off rather than blindly declare one strategy universally superior.

---

## V1 Scope

V1 is the complete intended scope for this project. Keep it focused.

### 1. Corpus ingestion

Support a small set of common text-oriented formats.

Initial supported formats:

- `.txt`
- `.md`
- `.pdf`

Responsibilities:

- Read documents.
- Preserve document identity/source.
- Extract text.
- Pass text into the configured chunking strategy.

The project does not need OCR.

---

### 2. Chunking

V1 should support configurable fixed/recursive text chunking.

Required parameters:

- chunk size
- overlap

Each produced chunk must retain enough metadata to identify its source document.

Example internal representation:

```python
Chunk(
    id="authentication.md::chunk_3",
    document_id="authentication.md",
    text="...",
    position=3,
)
```

The architecture should make it possible to add chunking strategies later, but V1 only needs the strategies necessary to meaningfully benchmark size and overlap.

---

### 3. Dense retrieval

Implement semantic retrieval using local embeddings.

Requirements:

- No paid API is required.
- Use a local embedding model suitable for sentence/document retrieval.
- Embed corpus chunks.
- Embed evaluation queries.
- Rank chunks by vector similarity.
- Return top-k results with scores.

The project should not require a hosted vector database.

Persistence/caching of embeddings is desirable so repeated benchmark runs do not recompute unchanged data.

---

### 4. BM25 retrieval

Implement lexical retrieval using BM25.

Purpose:

- Provide a strong keyword-based baseline.
- Perform well for exact terms, identifiers, names, error codes, and queries with strong lexical overlap.

Return ranked top-k chunks and scores through the same retriever interface used by dense retrieval.

---

### 5. Hybrid retrieval

Implement a retrieval strategy combining BM25 and dense retrieval.

The default fusion approach should be **Reciprocal Rank Fusion (RRF)** rather than attempting to directly compare raw BM25 and cosine-similarity scores.

Conceptually:

```text
query
 ├── BM25 ranking
 └── dense ranking
         ↓
       RRF
         ↓
   fused ranking
```

Hybrid retrieval must expose the same common retriever interface.

---

### 6. Cross-encoder reranking

Support an optional local cross-encoder reranker.

Conceptually:

```text
retriever
    ↓
top N candidates
    ↓
cross-encoder
    ↓
reranked top K
```

The candidate pool size and final top-k should be configurable.

Reranking must be benchmarked separately because it normally improves relevance at the cost of additional latency.

No paid reranking API should be required.

---

### 7. Retrieval evaluation

The project must implement retrieval evaluation from a labeled dataset.

Required metrics:

#### Recall@K

Measures how many relevant items were successfully retrieved within the first K results.

#### MRR — Mean Reciprocal Rank

Measures how early the first relevant result appears in the ranking.

#### nDCG@K — Normalized Discounted Cumulative Gain

Measures ranking quality while giving more importance to relevant results appearing near the top.

The metric implementations must have unit tests using small deterministic examples where the expected result can be calculated manually.

---

### 8. Latency measurement

For every retrieval configuration, measure query latency.

At minimum report:

- average latency
- p50 latency
- p95 latency

Embedding/index creation time should be reported separately from query-time retrieval latency.

This separation is important because offline indexing and online query serving have different performance concerns.

---

### 9. Comparison report

The CLI should produce a human-readable comparison table.

Example:

```text
Strategy                  Recall@5    MRR    nDCG@10    p95
Dense                       0.81      0.70      0.76     52ms
BM25                        0.74      0.64      0.69     24ms
Hybrid                      0.89      0.79      0.84     73ms
Hybrid + reranker           0.93      0.86      0.90    181ms
```

Also save machine-readable results, preferably JSON.

Example:

```text
results/
└── 2026-09-10T134500Z.json
```

The report should not claim that a strategy is universally best.

It may identify:

- highest retrieval quality
- lowest latency
- best quality among non-reranked strategies

Avoid opaque automated recommendations in V1.

---

## Ground Truth Model

The evaluation dataset is a first-class input.

V1 does **not** need to generate a golden dataset automatically.

The user is responsible for defining queries and relevant documents/chunks.

Minimum schema:

```json
{
  "query": "How do I cancel my subscription?",
  "relevant_documents": ["cancellation.md"]
}
```

The internal design should keep relevance judgments separate from retrieved results.

Document-level relevance is sufficient for the initial implementation. Chunk-level labels may be supported if they do not significantly complicate the design.

---

## CLI

CLI is the primary interface.

Recommended commands:

```bash
retrieveval index
retrieveval evaluate
```

A single-command experience is also acceptable:

```bash
retrieveval evaluate   --corpus ./documents   --dataset ./eval.json
```

Prefer a small, predictable CLI over many subcommands.

The CLI must provide clear validation errors for:

- missing corpus
- unsupported files
- malformed evaluation dataset
- missing relevant documents
- invalid configuration
- unavailable local model

---

## Architecture Principles

### Common interfaces

Retrievers should implement a common abstraction.

Conceptual example:

```python
class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None:
        ...

    def search(self, query: str, k: int) -> list[SearchResult]:
        ...
```

This allows BM25, dense, hybrid, and reranked retrieval to be evaluated by the same benchmark runner.

### Separation of concerns

Keep these responsibilities separate:

```text
ingestion
    ↓
chunking
    ↓
indexing
    ↓
retrieval
    ↓
reranking
    ↓
evaluation
    ↓
reporting
```

Evaluation code should not know implementation details of a specific retriever.

### Reproducibility

A benchmark result should include the configuration that generated it.

Record at least:

- timestamp
- chunk size
- overlap
- embedding model
- retrieval strategy
- top-k
- hybrid/RRF configuration
- reranker model
- candidate pool size
- metric values
- latency values

---

## Local-First / Zero-Cost Constraint

The complete V1 must be usable without paid APIs.

Requirements:

- local embeddings
- local BM25
- local reranker
- local corpus
- local evaluation
- no required OpenAI/Anthropic/Cohere/etc. key
- no required cloud account

An internet connection may be needed initially to download open-source models/packages.

After dependencies and models are available locally, normal benchmark execution should not depend on paid external services.

---

## Suggested Technology Direction

Language:

- Python

CLI:

- Typer or equivalent

Dense embeddings:

- Sentence Transformers or equivalent local embedding library

BM25:

- a maintained BM25 implementation or a small well-tested implementation

Hybrid fusion:

- Reciprocal Rank Fusion (RRF)

Reranking:

- a local cross-encoder model compatible with Sentence Transformers or an equivalent open-source implementation

Metrics:

- implement the core retrieval metrics directly where practical so their behavior is explicit and testable

Testing:

- pytest

Configuration:

- YAML

Results:

- console table + JSON artifact

Exact dependencies are implementation decisions, but avoid large frameworks when a small focused dependency is sufficient.

Do not introduce LangChain or LlamaIndex merely as orchestration layers unless they solve a concrete problem that cannot be handled cleanly by the project's own small abstractions.

---

## Explicit Non-Goals

Do not add these to V1:

- chatbot UI
- frontend
- answer generation
- LLM agent
- tool calling
- prompt engineering framework
- RAGAS-style answer evaluation
- automatic golden-dataset generation
- hosted API
- authentication
- multi-tenancy
- Terraform
- Azure/AWS/GCP deployment
- Kubernetes
- Dagster/Airflow
- Iceberg
- Kafka
- production monitoring platform
- multiple vector databases
- plugin ecosystem

These may be valid ideas independently, but they dilute this project's purpose.

---

## Quality Requirements

The project should be treated as reusable developer tooling rather than as a demo.

Required:

- clean package structure
- type hints
- deterministic unit tests where possible
- metric tests with manually verifiable examples
- integration tests covering a small end-to-end corpus
- useful CLI errors
- reproducible configuration
- no secrets required
- no hard-coded user-specific paths
- README focused on the problem and usage
- documented assumptions and limitations

Avoid unnecessary abstractions and premature extensibility.

---

## Suggested Repository Structure

```text
retrieveval/
├── pyproject.toml
├── README.md
├── LICENSE
├── PROJECT_CONTEXT.md
├── retrieveval.example.yaml
├── src/
│   └── retrieveval/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── ingestion/
│       │   ├── loaders.py
│       │   └── chunking.py
│       ├── retrieval/
│       │   ├── base.py
│       │   ├── bm25.py
│       │   ├── dense.py
│       │   ├── hybrid.py
│       │   └── reranker.py
│       ├── evaluation/
│       │   ├── dataset.py
│       │   ├── metrics.py
│       │   └── runner.py
│       └── reporting/
│           └── report.py
├── tests/
│   ├── unit/
│   └── integration/
└── examples/
    ├── documents/
    └── eval.json
```

This structure is guidance, not a requirement. Prefer changing it when implementation experience reveals a simpler design.

---

## Definition of Done

RetrievEval V1 is complete when a new developer can:

1. Clone the repository.
2. Install it locally.
3. Point it at a directory containing supported documents.
4. Provide a labeled evaluation dataset.
5. Run one CLI command.
6. Benchmark dense, BM25, hybrid, and hybrid + reranker retrieval.
7. See Recall@K, MRR, nDCG@K, and latency for each configuration.
8. Receive reproducible machine-readable results.
9. Run the entire workflow without a paid API or cloud service.

The project is successful if it helps a developer make a more informed retrieval decision for their own RAG system.

---

## Guiding Principle

Do not optimize the project for demonstrating technologies.

Optimize it for answering one question well:

> **Which retrieval strategy works best for this corpus and these queries, and what does that improvement cost in latency?**
