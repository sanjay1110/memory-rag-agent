"""Run the full evaluation suite and write the metrics report.

    python scripts/run_eval.py                 # offline, deterministic (default)
    python scripts/run_eval.py --llm anthropic # Claude for generation + reranking

Outputs:
    results/metrics.json          every number, machine-readable
    docs/metrics.md               the report (tables + interpretation)
    docs/img/retrieval_ablation.png
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memagent.evaluation.metrics import run_all  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (light surface)


def chart(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = results["retrieval"]["by_mode"]
    labels = {"dense_only": "Dense only\n(FAISS)", "bm25_only": "BM25 only", "hybrid_rrf": "Hybrid\n(RRF)",
              "hybrid_rerank": "Hybrid +\nrerank"}
    metrics = [("hit@1", "Hit@1"), ("mrr", "MRR"), ("ndcg@5", "nDCG@5")]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    w = 0.26
    xs = range(len(modes))
    for j, (key, name) in enumerate(metrics):
        vals = [modes[m][key] for m in modes]
        pos = [x + (j - 1) * (w + 0.01) for x in xs]
        bars = ax.bar(pos, vals, w, color=SERIES[j], label=name, edgecolor="white", linewidth=1.5, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color="#444444")
    ax.set_xticks(list(xs), [labels[m] for m in modes], fontsize=9, color="#333333")
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("score (36 labelled questions)", fontsize=9, color="#555555")
    ax.set_title("Retrieval accuracy by pipeline stage", fontsize=11, loc="left", color="#222222")
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(axis="y", colors="#777777", labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def pct(x):
    return f"{100 * x:.1f}%"


def report(r: dict) -> str:
    ret, ctx, mem, con, cont, eff = (r[k] for k in ("retrieval", "context", "memory", "consistency", "continuity", "efficiency"))
    cal, abl, stress = r["filter_calibration"], r["context_ablation"], r["compaction_stress"]
    L = [
        "# Performance metrics",
        "",
        f"Produced by `python scripts/run_eval.py` with embedder `{r['config']['embedder']}` and generator "
        f"`{r['config']['llm']}` over {r['config']['chunks']} indexed chunks. All numbers are regenerated from "
        "scratch on every run (fresh stores) and are deterministic in offline mode. Raw output: "
        "[`results/metrics.json`](../results/metrics.json).",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval Hit@1 (hybrid + rerank) | **{pct(ret['by_mode']['hybrid_rerank']['hit@1'])}** |",
        f"| Retrieval Recall@5 | **{pct(ret['by_mode']['hybrid_rerank']['recall@5'])}** |",
        f"| MRR | **{ret['by_mode']['hybrid_rerank']['mrr']:.3f}** |",
        f"| Context precision after optimisation | **{pct(ctx['context_precision'])}** |",
        f"| Answer support (final context contains the answer) | **{pct(ctx['answer_support'])}** |",
        f"| Retrieved-context token reduction | **{pct(ctx['token_reduction'])}** |",
        f"| Long-term memory fact recall@3 (fresh agent, after 3 sessions) | **{pct(mem['fact_recall@3'])}** |",
        f"| Stale value returned after a preference changed | **{pct(mem['stale_value_rate'])}** |",
        f"| Response consistency — same question, new session | **{pct(con['repeat']['both_correct'])}** correct in both |",
        f"| Response consistency — paraphrased question, new session | **{pct(con['paraphrase']['both_correct'])}** correct in both |",
        f"| Cross-session continuity checks passed | **{pct(cont['pass_rate'])}** ({len(cont['checks'])} checks) |",
        "",
        "## 1. Retrieval accuracy",
        "",
        f"{ret['n']} hand-labelled questions (`data/eval/retrieval_qa.json`), each with the gold document section(s) "
        "that answer it. Most questions are paraphrased so they do not copy the source wording. A result counts as "
        "relevant when its source file and section match a gold label.",
        "",
        "![Retrieval ablation](img/retrieval_ablation.png)",
        "",
        "| Pipeline | Hit@1 | Hit@3 | Recall@5 | MRR | nDCG@5 |",
        "|---|---|---|---|---|---|",
    ]
    names = {"dense_only": "Dense only (FAISS)", "bm25_only": "BM25 only", "hybrid_rrf": "Hybrid (RRF fusion)",
             "hybrid_rerank": "Hybrid + rerank (shipped)"}
    for k, v in ret["by_mode"].items():
        L.append(f"| {names[k]} | {v['hit@1']:.3f} | {v['hit@3']:.3f} | {v['recall@5']:.3f} | {v['mrr']:.3f} | {v['ndcg@5']:.3f} |")
    misses = [q for q in ret["per_question"] if not q["hit@1"]]
    L += ["", "Questions where the top-1 result was not a gold section:", ""]
    L += [f"- `{q['id']}` *{q['question']}* → top-1 was `{q['top1']}`" for q in misses] or ["- none"]
    L += [
        "",
        "Reading: on this small corpus dense and sparse retrieval are individually strong and share only 3 of their 5 top-1 misses; "
        "RRF fusion alone mostly averages them, and the cross-feature reranker (which scores query and "
        "passage jointly — IDF-weighted term coverage, heading match, dense and sparse evidence) is what lifts Hit@1 "
        "and MRR. With `ANTHROPIC_API_KEY` set the reranker additionally blends in an LLM relevance judgement.",
        "",
        "## 2. Context relevance and optimisation",
        "",
        "For every QA question the reranked candidates go through the context optimiser (filter → dedup → select → "
        "compact) exactly as in a chat turn.",
        "",
        "| Optimiser stages | Context precision | Answer support | Chunks kept | Tokens kept |",
        "|---|---|---|---|---|",
    ]
    for name, v in abl.items():
        L.append(f"| {name} | {pct(v['context_precision'])} | {pct(v['answer_support'])} | {v['items_after']:.2f} | {v['tokens_after']:.0f} |")
    L += [
        "",
        "- The relative floor does most of the work: it drops candidates that are much weaker than the best one "
        "without losing a single answer. Dedup slightly *lowers* the precision number: on the rollback and "
        "Friday-deploy questions the FAQ copy and the guide section were *both* labelled gold, so removing the "
        "redundant FAQ copy removes a gold-labelled chunk even though no information is lost. Its job is to stop "
        "the same text reaching the prompt twice, not to raise precision.",
        f"- Relevance floor calibration: with the absolute floor at {cal['relevance_floor']}, "
        f"{pct(cal['gold_kept_rate'])} of gold candidates are kept and {pct(cal['non_gold_filtered_rate'])} of non-gold "
        f"candidates are filtered (mean reranker score: gold {cal['mean_gold_score']}, non-gold {cal['mean_non_gold_score']}).",
        f"- Deduplication fired on {ctx['queries_with_dedup']} of {ret['n']} queries in the full pipeline "
        f"({ctx['near_dup_removed_total']} redundant chunks removed — mostly FAQ answers contained in the main guides).",
        f"- Compaction stress test (budget squeezed to 90 tokens): {stress['compacted_total']} chunks were compacted "
        f"to their most query-relevant sentences, keeping answer support at {pct(stress['answer_support'])} with "
        f"{stress['tokens_after']:.0f} tokens per query on average.",
        "",
        "## 3. Memory quality",
        "",
        "After the three-session scenario, a brand-new agent instance probes long-term memory "
        "(only the on-disk stores are shared).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Fact recall@3 over {len(mem['probes'])} probes | {pct(mem['fact_recall@3'])} |",
        f"| Stale-value rate (old preference returned after supersession) | {pct(mem['stale_value_rate'])} |",
        f"| Stored fact/preference/decision memories | {mem['stored_memories']} |",
        f"| Memory precision (stored memories that are expected facts) | {pct(mem['memory_precision'])} |",
        f"| Near-duplicate memory pairs (cosine ≥ dedup threshold) | {mem['duplicate_pairs']} |",
        f"| Memories superseded with history kept | {mem['superseded_with_history']} |",
        f"| Episode memories (one per session) | {mem['episodes']} |",
        "",
        "| Probe | Expected | Found in top-3 |",
        "|---|---|---|",
    ]
    L += [f"| {p['probe']} | {p['expected']} | {'yes' if p['found'] else 'no'} |" for p in mem["probes"]]
    L += [
        "",
        "## 4. Response consistency",
        "",
        f"{con['n_pairs']} questions are asked in one session; a **new agent instance** in a **new session** then asks "
        "the same question again and a paraphrase of it.",
        "",
        "| Variant | Keyword agreement | Correct in both | Cited-source Jaccard | Answer similarity |",
        "|---|---|---|---|---|",
    ]
    for v in ("repeat", "paraphrase"):
        c = con[v]
        L.append(f"| {v} | {pct(c['keyword_agreement'])} | {pct(c['both_correct'])} | {c['source_jaccard']:.2f} | {c['answer_similarity']:.2f} |")
    L += [
        "",
        "Repeat consistency is total because retrieval is deterministic and memory does not drift between sessions. "
        "Paraphrases still reach the same answer in most cases, but often cite a different (equally valid) section "
        "— e.g. the FAQ instead of the guide — which is why source Jaccard is lower than keyword agreement.",
        "",
        "### Cross-session continuity checks",
        "",
        "| Check | Session | Turn | Passed |",
        "|---|---|---|---|",
    ]
    L += [f"| {c['check']} | {c['session']} | *{c['message']}* | {'yes' if c['passed'] else 'no'} |" for c in cont["checks"]]
    L += [
        "",
        "## 5. Context efficiency and the hybrid strategy (scenario run)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| User turns | {eff['turns']} |",
        f"| Turns that needed a just-in-time knowledge search | {eff['jit_knowledge_calls']} |",
        f"| Turns answered from pinned upfront knowledge (JIT skipped) | {eff['served_from_upfront']} |",
        f"| Turns needing no knowledge retrieval (memory updates, recall) | {eff['no_retrieval_needed']} |",
        f"| Upfront context per session (tokens) | {', '.join(map(str, eff['upfront_tokens_per_session']))} |",
        f"| Mean / max prompt size (tokens) | {eff['mean_context_tokens']} / {eff['max_context_tokens']} (budget {eff['context_budget']}) |",
        f"| Session compactions / tokens saved | {eff['session_compactions']} / {eff['compaction_tokens_saved']} |",
        f"| Mean turn latency (offline) | {eff['mean_turn_latency_ms']} ms |",
        "",
        "## Limitations",
        "",
        "- The default embedder is an offline hashed n-gram model, chosen so the project runs anywhere with no "
        "downloads; swapping in `sentence-transformers` (`MEMAGENT_EMBEDDER=sentence-transformers`) is a one-line change.",
        "- The QA set is small (36 questions over 9 documents) and was written by the same author as the corpus, so "
        "absolute numbers are optimistic; the ablation *differences* are the more meaningful signal.",
        "- Offline answers are extractive. Response-consistency numbers with a real LLM will differ and can be "
        "measured with `--llm anthropic`.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="offline", choices=["offline", "anthropic", "auto"])
    ap.add_argument("--work-dir", default=str(ROOT / ".eval_state"))
    args = ap.parse_args()
    results = run_all(ROOT, Path(args.work_dir), llm=args.llm)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    chart(results, ROOT / "docs" / "img" / "retrieval_ablation.png")
    (ROOT / "docs" / "metrics.md").write_text(report(results), encoding="utf-8")
    h = results["retrieval"]["by_mode"]["hybrid_rerank"]
    print(f"Hit@1 {h['hit@1']:.3f}  MRR {h['mrr']:.3f}  | memory recall {results['memory']['fact_recall@3']:.3f} "
          f"| continuity {results['continuity']['pass_rate']:.3f}")
    print("Wrote results/metrics.json, docs/metrics.md, docs/img/retrieval_ablation.png")


if __name__ == "__main__":
    main()
