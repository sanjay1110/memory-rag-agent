"""Memory & retrieval evaluation.

Four metric families, matching requirement 7 of the lab:

1. **Retrieval accuracy** - labelled QA set (data/eval/retrieval_qa.json):
   Hit@1, Hit@3, Recall@5, MRR, nDCG@5, with an ablation over
   dense-only / BM25-only / hybrid-RRF / hybrid+rerank.
2. **Context relevance** - after the full optimiser: context precision
   (share of chunks in the final context that are gold-relevant), answer
   support (does the final context contain the answer keywords?), token
   reduction, and how often filtering / dedup / compaction fired.
3. **Memory quality** - after the 3-session scenario, probing long-term
   memory with fresh agents: fact recall@3, stale-value leakage after
   supersession, precision of stored memories, duplicate rate.
4. **Response consistency** - (a) the same knowledge questions asked in a
   new session by a new agent instance, plus paraphrases: answer-keyword
   agreement, cited-source Jaccard, answer embedding similarity;
   (b) cross-session continuity checks from the scripted scenario.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
from pathlib import Path

import numpy as np

from ..text import count_tokens
from .runner import fresh_agent, run_scenario
from .scenario import CONTINUITY_CHECKS, EXPECTED_FACT_SUBSTRINGS, MEMORY_PROBES, USER


# ----------------------------------------------------------------- helpers
def is_relevant(meta: dict, gold: list[list[str]]) -> bool:
    return any(meta.get("source") == src and sec.lower() in meta.get("section", "").lower() for src, sec in gold)


def rank_metrics(rels: list[bool], k: int = 5) -> dict:
    first = next((i for i, r in enumerate(rels) if r), None)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, max(1, sum(rels[:k])))))
    return {
        "hit@1": float(bool(rels[:1] and rels[0])),
        "hit@3": float(any(rels[:3])),
        "recall@5": float(any(rels[:5])),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        "ndcg@5": dcg / idcg if idcg else 0.0,
    }


def mean_dict(rows: list[dict]) -> dict:
    return {k: round(statistics.mean(r[k] for r in rows), 3) for k in rows[0]}


# ------------------------------------------------------ 1. retrieval accuracy
def eval_retrieval(agent, qa: list[dict]) -> dict:
    kb = agent.ltm.knowledge
    retr = agent.retriever
    cfg = agent.cfg
    modes: dict[str, list[dict]] = {"dense_only": [], "bm25_only": [], "hybrid_rrf": [], "hybrid_rerank": []}
    per_q = []
    for q in qa:
        qvec = agent.embedder.embed([q["question"]])[0]
        dense = [m for _, _, m, _ in kb.store.search(qvec, 5)]
        sparse = [kb.store.get(k)[2] for k, _ in kb.bm25.search(q["question"], 5)]
        fused = [c.meta for c in kb.search(q["question"], qvec, cfg.dense_k, cfg.sparse_k, cfg.rrf_k)[:5]]
        reranked_c = retr.retrieve(q["question"], ("knowledge",)).items
        reranked = [c.meta for c in reranked_c[:5]]
        for name, metas in [("dense_only", dense), ("bm25_only", sparse), ("hybrid_rrf", fused), ("hybrid_rerank", reranked)]:
            rels = [is_relevant(m, q["gold"]) for m in metas] + [False] * (5 - len(metas))
            modes[name].append(rank_metrics(rels))
        per_q.append({
            "id": q["id"], "question": q["question"],
            "top1": f"{reranked[0]['source']} § {reranked[0]['section']}" if reranked else None,
            "top1_score": round(reranked_c[0].rerank_score, 3) if reranked_c else None,
            "hit@1": modes["hybrid_rerank"][-1]["hit@1"],
        })
    return {"n": len(qa), "by_mode": {k: mean_dict(v) for k, v in modes.items()}, "per_question": per_q}


# ------------------------------------------------------- 2. context relevance
def eval_context(agent, qa: list[dict], budget: int = 700) -> dict:
    rows, reports = [], []
    for q in qa:
        res = agent.retriever.retrieve(q["question"], ("knowledge",))
        raw_tokens = sum(count_tokens(c.text) for c in res.items)
        items, rep = agent.optimizer.optimise(res.items, q["question"], agent.cfg.final_k, budget)
        ctx = " ".join(c.text for c in items)
        precision = sum(is_relevant(c.meta, q["gold"]) for c in items) / len(items) if items else 0.0
        support = float(all(k.lower() in ctx.lower() for k in q["answer_keywords"]))
        rows.append({"context_precision": precision, "answer_support": support,
                     "tokens_before": raw_tokens, "tokens_after": rep.output_tokens,
                     "items_after": rep.output_items})
        reports.append(rep)
    agg = mean_dict(rows)
    agg["token_reduction"] = round(1 - sum(r["tokens_after"] for r in rows) / sum(r["tokens_before"] for r in rows), 3)
    agg["filtered_total"] = sum(r.filtered for r in reports)
    agg["near_dup_removed_total"] = sum(r.near_dups + r.exact_dups for r in reports)
    agg["compacted_total"] = sum(r.compacted_items for r in reports)
    agg["queries_with_dedup"] = sum(1 for r in reports if r.near_dups + r.exact_dups)
    return agg


def eval_context_ablation(agent, qa: list[dict]) -> dict:
    """Turn the optimiser stages on one at a time to show what each contributes."""
    opt = agent.optimizer
    saved = (opt.min_relevance, opt.relative_floor, opt.near_dup_threshold, opt.containment_threshold)
    variants = {
        "raw top-k (no optimisation)": (0.0, 0.0, 9.0, 9.0),
        "+ absolute floor": (saved[0], 0.0, 9.0, 9.0),
        "+ relative floor": (saved[0], saved[1], 9.0, 9.0),
        "+ dedup (full pipeline)": saved,
    }
    out = {}
    for name, (mr, rf, nd, ct) in variants.items():
        opt.min_relevance, opt.relative_floor, opt.near_dup_threshold, opt.containment_threshold = mr, rf, nd, ct
        r = eval_context(agent, qa)
        out[name] = {k: r[k] for k in ("context_precision", "answer_support", "items_after", "tokens_after")}
    opt.min_relevance, opt.relative_floor, opt.near_dup_threshold, opt.containment_threshold = saved
    return out


def eval_filtering_calibration(agent, qa: list[dict]) -> dict:
    """How well does the relevance floor separate gold from non-gold candidates?"""
    gold_scores, other_scores = [], []
    for q in qa:
        for c in agent.retriever.retrieve(q["question"], ("knowledge",)).items:
            (gold_scores if is_relevant(c.meta, q["gold"]) else other_scores).append(c.rerank_score)
    floor = agent.cfg.min_relevance
    return {
        "relevance_floor": floor,
        "gold_kept_rate": round(sum(s >= floor for s in gold_scores) / max(1, len(gold_scores)), 3),
        "non_gold_filtered_rate": round(sum(s < floor for s in other_scores) / max(1, len(other_scores)), 3),
        "mean_gold_score": round(statistics.mean(gold_scores), 3) if gold_scores else None,
        "mean_non_gold_score": round(statistics.mean(other_scores), 3) if other_scores else None,
    }


# ---------------------------------------------------------- 3. memory quality
def eval_memory(data_dir: Path) -> dict:
    agent = fresh_agent(data_dir)  # brand-new instance: everything below comes from disk
    probes = []
    for query, must, must_not in MEMORY_PROBES:
        res = agent.retriever.retrieve(query, ("memories",), use_llm_rerank=False)
        top = [c for c in res.items if c.meta.get("user_id") == USER and c.meta.get("kind") != "episode"][:3]
        texts = " || ".join(c.text for c in top)
        ok = must.lower() in texts.lower()
        stale = bool(must_not) and any(must_not.lower() in c.text.lower() for c in top
                                       if c.meta.get("slot") and c.meta.get("slot") == (top[0].meta.get("slot") if top else None))
        probes.append({"probe": query, "expected": must, "found": ok, "stale_value_returned": stale, "top3": texts})

    mems = agent.ltm.user_memories(USER, kinds=("fact", "preference", "decision"))
    useful = [m for m in mems if any(s.lower() in m.text.lower() for s in EXPECTED_FACT_SUBSTRINGS)]
    vecs = agent.embedder.embed([m.text for m in mems]) if mems else np.zeros((0, 1))
    sims = vecs @ vecs.T if len(mems) else np.zeros((0, 0))
    dup_pairs = int(((sims >= agent.cfg.ltm_dedup_threshold).sum() - len(mems)) / 2) if len(mems) else 0
    superseded = [m for m in mems if m.meta.get("history")]
    return {
        "fact_recall@3": round(sum(p["found"] for p in probes) / len(probes), 3),
        "stale_value_rate": round(sum(p["stale_value_returned"] for p in probes) / len(probes), 3),
        "stored_memories": len(mems),
        "memory_precision": round(len(useful) / max(1, len(mems)), 3),
        "duplicate_pairs": dup_pairs,
        "superseded_with_history": len(superseded),
        "episodes": len(agent.ltm.user_memories(USER, kinds=("episode",))),
        "probes": probes,
    }


# ---------------------------------------------------- 4. response consistency
def _answer(agent, q: str):
    r = agent.chat(q)
    sources = {c["ref"] for c in r.citations if f"[{c['tag']}]" in r.text and c["kind"] != "memory"}
    return r.text, sources


def eval_consistency(data_dir: Path, knowledge_dir: Path, pairs: list[dict]) -> dict:
    a1 = fresh_agent(data_dir)
    if len(a1.ltm.knowledge) == 0:
        a1.ingest(knowledge_dir)
    a1.start_session("consistency-user")
    first = {p["id"]: _answer(a1, p["question"]) for p in pairs}
    a1.end_session()
    del a1

    a2 = fresh_agent(data_dir)  # new instance + new session: tests persistence too
    a2.start_session("consistency-user")
    rows = []
    for p in pairs:
        t1, s1 = first[p["id"]]
        for variant, q in (("repeat", p["question"]), ("paraphrase", p["paraphrase"])):
            t2, s2 = _answer(a2, q)
            kw = p["answer_keywords"]
            agree = float(all(k.lower() in t1.lower() for k in kw) == all(k.lower() in t2.lower() for k in kw))
            correct_both = float(all(k.lower() in t1.lower() for k in kw) and all(k.lower() in t2.lower() for k in kw))
            jac = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 1.0
            sim = float(a2.embedder.embed([t1])[0] @ a2.embedder.embed([t2])[0])
            rows.append({"variant": variant, "keyword_agreement": agree, "both_correct": correct_both,
                         "source_jaccard": jac, "answer_similarity": sim, "id": p["id"]})
    a2.end_session()
    out = {}
    for variant in ("repeat", "paraphrase"):
        sub = [{k: r[k] for k in ("keyword_agreement", "both_correct", "source_jaccard", "answer_similarity")}
               for r in rows if r["variant"] == variant]
        out[variant] = mean_dict(sub)
    out["n_pairs"] = len(pairs)
    return out


def eval_continuity(transcript: list[list[dict]]) -> dict:
    results = []
    for s_idx, t_idx, where, needle, label in CONTINUITY_CHECKS:
        user_turns = [r for r in transcript[s_idx] if r["kind"] == "user"]
        r = user_turns[t_idx]
        hay = r["response"] if where == "response" else r["prompt"]
        results.append({"check": label, "session": s_idx + 1, "message": r["message"], "passed": needle.lower() in hay.lower()})
    return {"pass_rate": round(sum(x["passed"] for x in results) / len(results), 3), "checks": results}


def eval_efficiency(transcript: list[list[dict]]) -> dict:
    turns = [r for s in transcript for r in s if r["kind"] == "user"]
    plans = [t["trace"]["plan"] for t in turns]
    ctx = [t["trace"]["context_total"] for t in turns]
    comp = [t["trace"]["session_compaction"] for t in turns if t["trace"].get("session_compaction")]
    starts = [r for s in transcript for r in s if r["kind"] == "session_start"]
    return {
        "turns": len(turns),
        "jit_knowledge_calls": sum(p["knowledge"] for p in plans),
        "served_from_upfront": sum(1 for p in plans if "pinned" in p["reason"]),
        "no_retrieval_needed": sum(1 for p in plans if not p["knowledge"] and "pinned" not in p["reason"]),
        "mean_context_tokens": round(statistics.mean(ctx), 1),
        "max_context_tokens": max(ctx),
        "context_budget": None,
        "upfront_tokens_per_session": [s["upfront_tokens"] for s in starts],
        "session_compactions": len(comp),
        "compaction_tokens_saved": sum(c["tokens_before"] - c["tokens_after"] for c in comp),
        "mean_turn_latency_ms": round(statistics.mean(t["trace"]["latency_ms"] for t in turns), 1),
    }


# ------------------------------------------------------------------- driver
def run_all(root: Path, work_dir: Path, llm: str = "offline") -> dict:
    knowledge = root / "data" / "knowledge"
    qa = json.loads((root / "data" / "eval" / "retrieval_qa.json").read_text())
    pairs = json.loads((root / "data" / "eval" / "consistency.json").read_text())
    if work_dir.exists():
        shutil.rmtree(work_dir)

    scen_dir = work_dir / "scenario"
    transcript = run_scenario(scen_dir, knowledge, llm=llm)
    agent = fresh_agent(scen_dir, llm)

    results = {
        "config": {"embedder": agent.embedder.name, "llm": agent.llm.name, "chunks": len(agent.ltm.knowledge),
                   "final_k": agent.cfg.final_k, "min_relevance": agent.cfg.min_relevance,
                   "relative_floor": agent.cfg.relative_floor,
                   "context_token_budget": agent.cfg.context_token_budget},
        "retrieval": eval_retrieval(agent, qa),
        "context": eval_context(agent, qa),
        "context_ablation": eval_context_ablation(agent, qa),
        "compaction_stress": {k: v for k, v in eval_context(agent, qa, budget=90).items()
                              if k in ("answer_support", "tokens_after", "compacted_total", "token_reduction")},
        "filter_calibration": eval_filtering_calibration(agent, qa),
        "memory": eval_memory(scen_dir),
        "consistency": eval_consistency(work_dir / "consistency", knowledge, pairs),
        "continuity": eval_continuity(transcript),
        "efficiency": eval_efficiency(transcript),
    }
    results["efficiency"]["context_budget"] = agent.cfg.context_token_budget
    return results
