# Multi-Tier Memory & Retrieval Agent

An agent that keeps working context across long, multi-session tasks by combining **four memory tiers**,
**hybrid retrieval-augmented generation** (FAISS dense + BM25 sparse, fused and reranked), a **hybrid
upfront / just-in-time loading strategy**, and a **context optimiser** that filters, de-duplicates and compacts
everything before it reaches the model. Every answer carries visible source attribution, and an evaluation
harness measures retrieval accuracy, context relevance, memory quality and response consistency.

It runs fully offline out of the box (no API key, no model downloads) and switches to Claude for generation and
reranking when `ANTHROPIC_API_KEY` is set.

![Architecture](docs/img/architecture.png)

## Quick start

```bash
git clone <this repo> && cd memory-rag-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/demo_multi_session.py   # 3-session scenario  -> docs/sample_multi_session.md
python scripts/run_eval.py             # all metrics         -> docs/metrics.md, results/metrics.json
pytest -q                              # 17 tests
python scripts/chat.py --user alice    # interactive chat (/task, /trace, /memory, /context, /end)
```

To use Claude for generation and LLM reranking:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_MODEL=claude-sonnet-5-5     # optional override
python scripts/chat.py --llm anthropic
python scripts/run_eval.py --llm anthropic
```

## Results (offline, deterministic)

| | |
|---|---|
| Retrieval Hit@1 / Recall@5 / MRR (36 labelled questions) | **0.889 / 0.972 / 0.931** |
| Context precision: raw top-k → after optimiser | **28.5% → 72.5%**, answer support unchanged at 97.2% |
| Retrieved-context tokens removed by the optimiser | **83%** |
| Long-term memory fact recall@3 after three sessions (fresh agent) | **100%** (7/7 probes) |
| Stale value returned after the user changed a preference | **0%** |
| Same question asked again in a new session by a new agent instance | **100%** consistent and correct |
| Paraphrased question in a new session | **91.7%** consistent and correct |
| Cross-session continuity checks | **6/6** |
| Knowledge searches needed in a 19-turn, 3-session task | **4** (3 served from upfront context, 12 needed none) |

Full report with ablations, per-question misses and limitations: **[docs/metrics.md](docs/metrics.md)**.

## How the requirements map to the code

| Requirement | Where | What to look at |
|---|---|---|
| 1. Working memory | [`memagent/memory/working.py`](memagent/memory/working.py) | active task, workflow state (plan / step / vars), recent tool outputs; item cap + token budget with priority-aware eviction; snapshot/restore across sessions |
| 2. Session memory | [`memagent/memory/session.py`](memagent/memory/session.py) | SQLite history, rolling-summary compaction, user preferences, multi-session task progress with step tracking |
| 3. Long-term memory | [`memagent/memory/long_term.py`](memagent/memory/long_term.py), [`retrieval/vector_store.py`](memagent/retrieval/vector_store.py) | FAISS + SQLite persistent store; memory extraction; dedup-on-write; slot supersession with history; importance × recency ranking; episode consolidation |
| 4. Retrieval system | [`memagent/retrieval/`](memagent/retrieval) | structure-aware chunking, embeddings, dense + BM25 search, RRF fusion, cross-feature / LLM reranking, citations `file § section` |
| 5. Hybrid retrieval strategy | [`memagent/agent.py`](memagent/agent.py) — `_load_upfront()`, `plan_retrieval()` | upfront: prefs, tasks, previous sessions, top memories, task-pinned knowledge; JIT planner per turn |
| 6. Context optimisation | [`memagent/context/optimizer.py`](memagent/context/optimizer.py), [`context/builder.py`](memagent/context/builder.py) | absolute + relative relevance floors, containment / hash / near-duplicate removal, extractive compaction, sectioned token budget |
| 7. Memory evaluation | [`memagent/evaluation/`](memagent/evaluation), [`scripts/run_eval.py`](scripts/run_eval.py) | retrieval accuracy (+ ablation), context relevance (+ ablation), memory quality, response consistency, continuity, efficiency |

## Deliverables

| Deliverable | Location |
|---|---|
| Source code for the complete agent | [`memagent/`](memagent), [`scripts/`](scripts), [`tests/`](tests) |
| Architecture diagram (memory tiers + retrieval pipeline) | [`docs/img/architecture.svg`](docs/img/architecture.svg) (PNG alongside) |
| Memory design documentation | [`docs/memory_design.md`](docs/memory_design.md) |
| Retrieval pipeline description (chunking, embedding, ranking, attribution) | [`docs/retrieval_pipeline.md`](docs/retrieval_pipeline.md) |
| Sample multi-session interaction | [`docs/sample_multi_session.md`](docs/sample_multi_session.md) (+ [`results/transcript.json`](results/transcript.json), an assembled prompt in [`results/example_prompt.txt`](results/example_prompt.txt)) |
| Performance metrics | [`docs/metrics.md`](docs/metrics.md), [`results/metrics.json`](results/metrics.json) |

## What a turn looks like

```text
you> How long until I can turn off the v1 machines?

agent> After 14 days of stable production traffic on Nimbus, stop the v1 virtual machines and
       archive their configuration. [S1]

Sources:
  [S1] migration_guide.md § Step 7: Decommission v1  (relevance 0.50)
  · just-in-time retrieval · context 890 tok · 12 ms
```

`[S#]` marks a chunk retrieved just-in-time, `[P#]` a chunk pinned upfront for the current task, `[M]` a long-term
memory. Each turn also returns a `trace` with the retrieval plan, candidate counts, reranker used, what the optimiser
filtered or merged, memory writes, completed task steps and a per-section token breakdown of the prompt.

## Scenario used for the demo and continuity evaluation

Priya migrates the `orders-api` service from a legacy platform to "Nimbus" (a fictional internal platform whose
documentation is the knowledge corpus in [`data/knowledge/`](data/knowledge)) across three sessions. **Each session
is run by a brand-new `MemoryAgent` instance**, so continuity can only come from persistence:

- **Session 1** — she introduces herself and the project, the agent creates a 7-step task, she states preferences
  (Python, concise) and her region, asks about the base image (answered from pinned upfront knowledge without a
  search), finishes step 1 and records a decision.
- **Session 2** — "where did we leave off?" is answered from restored task state; she completes two more steps by
  saying so, asks two knowledge questions, and **changes her language preference to Go** and sets a deadline.
- **Session 3** — the agent recalls everything (with Go, not Python), recalls the session-1 decision, tracks two
  more steps and names the correct next step.

## Repository layout

```
memagent/
  agent.py                 orchestrator: session lifecycle, hybrid loading, per-turn loop
  config.py                every tunable in one place
  embeddings.py            hashed n-gram embedder (default) / sentence-transformers
  llm.py                   Claude client + deterministic offline generator
  memory/  working.py  session.py  long_term.py
  retrieval/  chunking.py  vector_store.py  bm25.py  index.py  retriever.py
  context/  optimizer.py  builder.py
  evaluation/  scenario.py  runner.py  metrics.py
data/knowledge/            9 Markdown docs (46 chunks)
data/eval/                 retrieval_qa.json (36 labelled Qs), consistency.json (12 Q + paraphrase pairs)
scripts/                   ingest.py  chat.py  demo_multi_session.py  run_eval.py
docs/                      architecture, memory design, retrieval pipeline, sample session, metrics
results/                   metrics.json, transcript.json, example_prompt.txt
tests/                     17 pytest tests
```

## Design choices worth knowing

- **Offline by default.** The default embedder is a fit-free hashed n-gram model and the default generator is
  extractive, so everything runs and every number reproduces with no network. Both are swappable
  (`MEMAGENT_EMBEDDER=sentence-transformers`, `ANTHROPIC_API_KEY`). The offline generator is a baseline for exercising
  the memory and retrieval machinery, not a substitute for a language model.
- **Rule-based memory extraction.** Deterministic and testable; the extraction step is one function and can be
  replaced by an LLM extractor.
- **Persistence everywhere it matters.** Tiers 2–4 are on disk (SQLite + FAISS, written on every update); tier 1 is
  checkpointed into the task at session end.
- **Honest evaluation.** Stores are rebuilt from scratch on every evaluation run; ablations show what each stage
  adds; misses are listed. The QA set is small and author-written, so treat absolute numbers as optimistic.

## License

MIT
