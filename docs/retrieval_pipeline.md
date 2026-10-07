# Retrieval pipeline

```
ingest:   Markdown ─► structure-aware chunker ─► contextual header ─► embed ─► FAISS (dense) + SQLite (payload)
                                                                         └──► BM25 postings (sparse)

query:    message ─► embed ─► FAISS top-20 ──┐
                 └─► BM25 top-20 ────────────┼─► RRF fusion ─► rerank ─► filter/dedup/compact ─► cited context
                     (per collection:        │
                      knowledge, memories) ──┘
```

Code: [`memagent/retrieval/`](../memagent/retrieval) and [`memagent/context/optimizer.py`](../memagent/context/optimizer.py).

## 1. Chunking — [`chunking.py`](../memagent/retrieval/chunking.py)

- **Structure first.** Documents are split on Markdown headings, so a chunk never straddles two sections and every
  chunk knows its *section path* (e.g. `Deployments > Rollback`). That path is the attribution unit.
- **Sentence packing.** Within a section, whole sentences (and list items) are packed greedily up to
  `chunk_target_tokens = 160`; sentences are never split.
- **Overlap.** The next chunk begins with the trailing ~30 tokens of whole sentences from the previous one, so a fact
  that spans a boundary is retrievable from either side.
- **Contextual headers.** What gets *embedded and BM25-indexed* is `"{title} | {section}\n{text}"`; what is *shown*
  to the model is just the text. Short chunks whose body is ambiguous ("It completes in under one minute") become
  findable by their heading.
- **Stable ids.** `chunk_id = "{file}:{position}:{hash6}"`, so re-ingesting unchanged content upserts in place.

The sample corpus (9 documents) yields 46 chunks of 20–104 tokens (mean 47): the docs have short sections, so most
chunks are one section. Longer documents are packed and overlapped as described.

## 2. Embedding — [`embeddings.py`](../memagent/embeddings.py)

Two interchangeable embedders behind one interface, both returning L2-normalised vectors so inner product equals
cosine similarity:

| Embedder | When | Notes |
|---|---|---|
| `HashingEmbedder` (default) | offline, zero downloads | hashes word unigrams (w=1.0), 6-char stems (0.6), word bigrams (0.8) and character 3–5-grams (0.25) into 1,024 signed buckets with sub-linear TF; adds a small synonym expansion (e.g. *rollback ↔ revert/undo*, *secret ↔ credential/password*). Fit-free, so new memories embed instantly and the index never needs rebuilding. |
| `SentenceTransformerEmbedder` | `MEMAGENT_EMBEDDER=sentence-transformers` | `all-MiniLM-L6-v2` dense neural embeddings (`pip install sentence-transformers`). |

Character n-grams give tolerance to morphology and typos ("deploying" ↔ "deployment"); bigrams give some phrase
sensitivity. The hashing model is a deliberate trade-off for portability; it is lexical-semantic rather than truly
semantic, which the BM25 + reranker stages partly compensate for.

## 3. Indexing and search — [`vector_store.py`](../memagent/retrieval/vector_store.py), [`bm25.py`](../memagent/retrieval/bm25.py), [`index.py`](../memagent/retrieval/index.py)

- **Dense.** FAISS `IndexIDMap2(IndexFlatIP)`: exact cosine search with delete-by-id (needed for memory
  supersession). Payloads (text + metadata JSON) are in SQLite under the same integer id. The index is written to
  disk after every write, so a crash never loses committed memories.
- **Sparse.** Okapi BM25 (k1 = 1.4, b = 0.75) with stopword removal and plural folding. It rescues exact
  identifiers that dense vectors blur: `NIMBUS_DB_POOL`, `E429`, `--to-time`.
- **Fusion.** Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)` over the two ranked lists. RRF needs no score
  calibration between the two retrievers, which have incomparable scales.
- **Collections.** `knowledge` and `memories` are separate `HybridIndex` instances over the same embedder; a query
  can search either or both.

## 4. Ranking — [`retriever.py`](../memagent/retrieval/retriever.py)

The fused top candidates are re-scored *as (query, passage) pairs*:

| Feature | Weight | What it captures |
|---|---|---|
| dense cosine (rescaled) | 0.35 | semantic closeness |
| BM25, normalised within the pool | 0.20 | exact term match |
| IDF-weighted query-term coverage (with synonyms and stems) | 0.35 | does the passage address *all* parts of the question, with rare terms ("downtime") counting more than common ones ("data")? |
| heading match | 0.10 | does the section title name the topic? |

When an LLM is configured (`ANTHROPIC_API_KEY`), the top 12 candidates are also scored 0–10 by the model and the
final score is `0.7 × LLM + 0.3 × cross-feature`; any LLM error falls back to the cross-feature score. For memories,
the score is further multiplied by `0.75 + 0.25 × prior` (importance × recency).

Effect on the 36-question eval set ([metrics](metrics.md)): dense-only and BM25-only each reach Hit@1 0.861 and
share only 3 of their 5 top-1 misses; RRF alone does not improve Hit@1; the reranker raises Hit@1 to 0.889 and MRR to 0.931.

## 5. Context retrieval and filtering

Reranked candidates pass through the optimiser (details in [memory_design.md](memory_design.md#context-optimisation)):
absolute and relative relevance floors, containment / hash / near-duplicate removal (also against upfront context),
top-k selection and extractive compaction under a token budget. On the eval set this keeps answer support at 97%
while removing 83% of the retrieved tokens and raising context precision from 29% (raw top-k) to 73%.

## 6. Source attribution

Attribution is carried end to end rather than reconstructed afterwards:

1. Every chunk stores `source`, `title`, `section`, `position` at ingestion; every memory stores `kind`,
   `session_id` and timestamps.
2. `Candidate.citation` renders these as `deployments.md § Rollback` or `memory:decision (session s-…, date)`.
3. The context builder tags each context item: `[S1]…[Sn]` for just-in-time results, `[P1]…` for pinned upfront
   knowledge, `[M]` for long-term memory. The system prompt requires inline tags on every factual claim.
4. `AgentResponse.render()` appends a **Sources** block listing only the tags the answer actually used, with the
   file, section and relevance score:

```text
After 14 days of stable production traffic on Nimbus, stop the v1 virtual machines and archive their configuration. [S1]

Sources:
  [S1] migration_guide.md § Step 7: Decommission v1  (relevance 0.50)
```

5. Assistant messages are stored in session memory with the list of sources they cited (`messages.meta`), so
   attribution survives into history.
