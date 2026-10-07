"""MemoryAgent - ties the four memory tiers, the retrieval pipeline and the
context optimiser together into one conversational agent.

Lifecycle
---------
``start_session(user)``   UPFRONT LOAD: preferences, open tasks (+ restore the
                          last working-memory snapshot), previous-session
                          summaries, top long-term memories, and knowledge
                          chunks pinned to the open task.  Cached for the session.
``chat(message)``         per turn: extract memories -> track task progress ->
                          plan retrieval (JIT or served-from-upfront) -> retrieve
                          -> optimise -> build prompt -> generate -> log.
``end_session()``         CONSOLIDATE: summarise the session into an episode
                          memory, checkpoint working memory into the task,
                          close the session.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .config import AgentConfig
from .context.builder import ContextBuilder
from .context.optimizer import ContextOptimizer
from .embeddings import make_embedder
from .llm import make_llm
from .memory.long_term import LongTermMemory, extract_memories
from .memory.session import SessionMemory
from .memory.working import WorkingMemory
from .retrieval.chunking import chunk_directory
from .retrieval.index import Candidate
from .retrieval.retriever import Retriever
from .text import count_tokens, tokenize

SMALLTALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|cool|great|bye|good (morning|night))\b[\s!.]*$", re.I)
MEMORY_CUE = re.compile(
    r"\b(remember|recall|last (time|session|week)|previous(ly)?|earlier|decid(e|ed)|decision|where were we|"
    r"le(ft|ave) off|progress|my (name|preference|deadline|service|region)|what do you know about me|next step)\b",
    re.I,
)
STEP_DONE = re.compile(r"\b(?:i(?:'ve| have)? )?(?:finished|completed|done with)\s+(?:the\s+)?(.+?)(?:[.!?,]|$)", re.I)
QUESTION = re.compile(r"\?\s*$|^\s*(what|which|how|why|when|where|who|can|could|should|do|does|is|are)\b", re.I)
# Words that only express "look in memory", not a knowledge topic.
RECALL_WORDS = {"remember", "recall", "leave", "left", "off", "where", "were", "we", "next", "step", "progress",
                "know", "about", "me", "project", "status", "what", "whats", "s", "did", "last", "time", "session",
                "hey", "hi", "hello", "so", "ok", "okay", "please", "again", "remind", "tell", "this"}


@dataclass
class AgentResponse:
    text: str
    citations: list[dict]
    trace: dict = field(default_factory=dict)

    def render(self) -> str:
        used = [c for c in self.citations if f"[{c['tag']}]" in self.text]
        if not used:
            return self.text
        lines = [self.text, "", "Sources:"]
        for c in used:
            lines.append(f"  [{c['tag']}] {c['ref']}  (relevance {c['score']:.2f})")
        return "\n".join(lines)


class MemoryAgent:
    def __init__(self, cfg: AgentConfig | None = None, llm=None):
        self.cfg = cfg or AgentConfig()
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = make_embedder(self.cfg.embedder, self.cfg.embedding_dim)
        self.llm = llm or make_llm(self.cfg.llm, self.cfg.anthropic_model)
        self.ltm = LongTermMemory(
            self.cfg.data_dir / "ltm", self.embedder, self.cfg.ltm_dedup_threshold, self.cfg.ltm_recency_half_life_days
        )
        self.sessions = SessionMemory(
            self.cfg.data_dir / "session.sqlite", self.cfg.session_recent_turns, self.cfg.session_compact_after
        )
        self.working = WorkingMemory(self.cfg.working_max_items, self.cfg.working_token_budget)
        self.retriever = Retriever(
            {"knowledge": self.ltm.knowledge, "memories": self.ltm.memories}, self.embedder, self.cfg, self.llm
        )
        self.optimizer = ContextOptimizer(self.embedder, self.cfg.min_relevance, self.cfg.near_dup_threshold,
                                         self.cfg.relative_floor)
        self.builder = ContextBuilder(self.cfg.context_token_budget)
        self.user_id: str | None = None
        self.session_id: str | None = None
        self.upfront: dict = {}
        self.turn_log: list[dict] = []

    # ================================================================ ingest
    def ingest(self, directory: str | Path) -> int:
        chunks = chunk_directory(directory, target_tokens=self.cfg.chunk_target_tokens,
                                 overlap_tokens=self.cfg.chunk_overlap_tokens)
        return self.ltm.ingest_chunks(chunks)

    # ======================================================= session control
    def start_session(self, user_id: str) -> dict:
        self.user_id = user_id
        self.session_id = self.sessions.start_session(user_id)
        self.working = WorkingMemory(self.cfg.working_max_items, self.cfg.working_token_budget)
        self.turn_log = []
        self._load_upfront()
        return {"session_id": self.session_id, "upfront_tokens": self.upfront["tokens"],
                "restored_task": self.working.active_task}

    def _load_upfront(self) -> None:
        """Hybrid strategy, part 1: load stable, high-prior context once."""
        t0 = time.perf_counter()
        uid = self.user_id
        prefs = self.sessions.preferences(uid)
        tasks = self.sessions.open_tasks(uid)
        prev = self.sessions.previous_sessions(uid, exclude=self.session_id, limit=2)
        mems = self.ltm.top_memories(uid, 6, kinds=("fact", "preference", "decision"))

        if tasks and tasks[0].get("working_snapshot"):
            self.working.restore(tasks[0]["working_snapshot"])

        # Pin knowledge relevant to the open task so common follow-ups need no JIT call.
        pinned: list[Candidate] = []
        if tasks:
            t = tasks[0]
            nxt = next((s["name"] for s in t["steps"] if not s.get("done")), "")
            pin_query = f"{t['title']} {nxt}"
            res = self.retriever.retrieve(pin_query, ("knowledge",), use_llm_rerank=False)
            pinned, _ = self.optimizer.optimise(res.items, pin_query, self.cfg.upfront_knowledge_k, 450)

        self.upfront = {
            "prefs": prefs,
            "tasks": tasks,
            "previous": prev,
            "memories": mems,
            "pinned": pinned,
            "load_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        self.upfront["sections"] = self._upfront_sections()
        self.upfront["tokens"] = sum(count_tokens(v) for v in self.upfront["sections"].values())

    def _upfront_sections(self) -> dict[str, str]:
        u = self.upfront
        prefs = "\n".join(f"- {k}: {v}" for k, v in u["prefs"].items())
        tasks = "\n".join(SessionMemory.render_task(t) for t in u["tasks"][:2])
        prev = "\n".join(f"- {p['started_at'][:16]}: {p['summary']}" for p in u["previous"] if p.get("summary"))
        mems = "\n".join(f"- {m.text} ({m.kind})" for m in u["memories"])
        pinned = "\n".join(f"[P{i}] ({c.citation}) {c.text}" for i, c in enumerate(u["pinned"], 1))
        return {
            "User profile & preferences": prefs,
            "Task progress": tasks,
            "Previous sessions": prev,
            "Long-term memories about this user": mems,
            "Pinned knowledge": pinned,
        }

    def end_session(self) -> dict:
        """Consolidation: episode summary -> LTM, working snapshot -> task."""
        msgs = self.sessions.messages(self.session_id)
        live = self.sessions.messages(self.session_id, include_compacted=False)
        prior = self.sessions.summary(self.session_id)  # rolling summary of already-compacted turns
        transcript = (f"Earlier summary: {prior}\n" if prior else "") + "\n".join(
            f"{m['role']}: {m['content']}" for m in live)
        full = self.llm.summarise(transcript, max_words=70) if msgs else ""
        summary = full
        self.sessions.end_session(self.session_id, full)
        episode = None
        if summary:
            episode = self.ltm.remember(self.user_id, f"Session {self.session_id}: {summary}", "episode", 0.6,
                                        self.session_id)
        for t in self.sessions.open_tasks(self.user_id)[:1]:
            self.sessions.upsert_task(self.user_id, t["title"], t["steps"], t["status"], task_id=t["task_id"],
                                      session_id=self.session_id, working_snapshot=self.working.snapshot())
        report = {"session_id": self.session_id, "summary": full, "episode": episode,
                  "ltm_stats": dict(self.ltm.stats), "turns": len(msgs)}
        self.session_id = None
        return report

    # ================================================================= tasks
    def start_task(self, title: str, steps: list[str]) -> str:
        tid = self.sessions.upsert_task(self.user_id, title, [{"name": s, "done": False} for s in steps],
                                        session_id=self.session_id)
        self.working.set_task(title, steps)
        self.ltm.remember(self.user_id, f"User is working on task: {title}", "fact", 0.8, self.session_id,
                          slot="project:task")
        self._load_upfront()
        return tid

    def _track_progress(self, message: str) -> list[str]:
        done = []
        tasks = self.sessions.open_tasks(self.user_id)
        if not tasks:
            return done
        task = tasks[0]
        for m in STEP_DONE.finditer(message):
            said = m.group(1).lower()
            best, best_i = 0.0, None
            for i, s in enumerate(task["steps"]):
                if s.get("done"):
                    continue
                name = s["name"].lower()
                score = max(SequenceMatcher(None, said, name).ratio(),
                            len(set(tokenize(said, True)) & set(tokenize(name, True))) / max(1, len(set(tokenize(name, True)))))
                if score > best:
                    best, best_i = score, i
            if best_i is not None and best >= 0.5:
                self.sessions.complete_step(task["task_id"], best_i, note=f"done in {self.session_id}")
                self.working.advance(f"Step completed: {task['steps'][best_i]['name']}")
                done.append(task["steps"][best_i]["name"])
        if done:
            self._load_upfront()  # task progress section changed -> refresh the upfront cache
        return done

    # ================================================================== chat
    def plan_retrieval(self, message: str, events: dict | None = None) -> dict:
        """Hybrid strategy, part 2: decide what to fetch just-in-time.

        * small talk                         -> nothing
        * statement that only updated memory -> nothing (acknowledge)
        * pure recall ("where did we leave off?") -> memories only
        * question answerable from pinned upfront knowledge -> no knowledge call
        * everything else                    -> JIT knowledge (+ memories on memory cues)
        """
        events = events or {}
        if SMALLTALK.match(message):
            return {"knowledge": False, "memories": False, "intent": "smalltalk", "reason": "small talk - no retrieval"}
        is_question = bool(QUESTION.search(message))
        want_mem = bool(MEMORY_CUE.search(message))
        if not is_question and (events.get("memory_writes") or events.get("steps_completed")):
            return {"knowledge": False, "memories": False, "intent": "update",
                    "reason": "statement handled by memory/task update - no retrieval"}
        topic_terms = [t for t in tokenize(message, True) if t not in RECALL_WORDS]
        if want_mem and len(topic_terms) == 0:
            return {"knowledge": False, "memories": True, "intent": "recall",
                    "reason": "pure recall - answered from session + long-term memory"}
        pinned = self.upfront.get("pinned") or []
        if pinned and is_question:
            qv = self.embedder.embed([message])[0]
            probes = [Candidate(c.key, c.text, c.meta, c.collection,
                                dense_score=float(qv @ self.embedder.embed([c.meta.get("embed_text", c.text)])[0]))
                      for c in pinned]
            scores = self.retriever.cross.score(message, probes, use_sparse=False)
            best = max(scores)
            cov = probes[scores.index(best)].signals["coverage"]
            if best >= 0.55 and cov >= 0.6:
                return {"knowledge": False, "memories": want_mem, "intent": "question",
                        "reason": f"served from pinned upfront context (score {best:.2f}) - JIT skipped"}
        return {"knowledge": True, "memories": want_mem, "intent": "question" if is_question else "statement",
                "reason": "just-in-time retrieval"}

    def _turn_events(self, trace: dict) -> str:
        """Things the agent itself did this turn, so the model can acknowledge them."""
        lines = []
        for w in trace.get("memory_writes", []):
            extra = f" (replaces: {w['previous']})" if w.get("previous") else ""
            verb = {"written": "Saved", "merged": "Already known, reinforced", "superseded": "Updated"}[w["action"]]
            lines.append(f"- {verb} in long-term memory: {w['text']}{extra}")
        for step in trace.get("steps_completed", []):
            lines.append(f"- Marked task step done: {step}")
        if trace.get("steps_completed"):
            t = (self.sessions.open_tasks(self.user_id) or [None])[0]
            nxt = next((s["name"] for s in t["steps"] if not s.get("done")), None) if t else None
            lines.append(f"- Next step: {nxt}" if nxt else "- All task steps are complete")
        return "\n".join(lines)

    def chat(self, message: str) -> AgentResponse:
        assert self.session_id, "call start_session() first"
        t0 = time.perf_counter()
        trace: dict = {}
        self.sessions.add_message(self.session_id, self.user_id, "user", message)

        # 1. memory extraction (write-through to LTM with dedup/supersession)
        writes = []
        for mem in extract_memories(message):
            res = self.ltm.remember(self.user_id, mem["text"], mem["kind"], mem["importance"], self.session_id, mem["slot"])
            writes.append({**mem, **res})
            if mem["kind"] == "preference":
                key = (mem["slot"] or "pref:general").split(":")[1]
                self.sessions.set_preference(self.user_id, key, mem["text"].replace("User prefers ", ""))
        if writes:
            self._load_upfront()
        trace["memory_writes"] = writes

        # 2. task progress (statements only - "did we finish X?" is not a completion)
        trace["steps_completed"] = [] if QUESTION.search(message) else self._track_progress(message)

        # 3. hybrid retrieval plan + JIT retrieval
        plan = self.plan_retrieval(message, trace)
        trace["plan"] = plan
        sources: list[Candidate] = []
        mem_hits: list[Candidate] = []
        if plan["knowledge"]:
            res = self.retriever.retrieve(message, ("knowledge",))
            pinned_texts = [c.text for c in self.upfront.get("pinned", [])]
            sources, rep = self.optimizer.optimise(res.items, message, self.cfg.final_k, 700, pinned_texts)
            trace["knowledge_retrieval"] = res.trace
            trace["knowledge_optimiser"] = rep.__dict__ | {"reduction": round(rep.reduction, 3)}
            self.working.add_tool_output(
                "search_knowledge", f"'{message[:60]}' -> " + (", ".join(c.citation for c in sources) or "no relevant hits")
            )
        if plan["memories"]:
            res = self.retriever.retrieve(message, ("memories",), use_llm_rerank=False)
            for c in res.items:
                c.meta["_prior"] = self.ltm.prior(c.meta)
            own = [c for c in res.items if c.meta.get("user_id") == self.user_id]
            loaded = [m.text for m in self.upfront.get("memories", [])]
            mem_hits, rep = self.optimizer.optimise(own, message, 3, 250, loaded)
            self.ltm.mark_accessed([c.key for c in mem_hits])
            trace["memory_retrieval"] = res.trace | {"kept": len(mem_hits)}
            self.working.add_tool_output("recall_memory", ", ".join(c.text[:60] for c in mem_hits) or "nothing new")

        # 4. build prompt
        citations: list[dict] = []
        for i, c in enumerate(self.upfront.get("pinned", []), 1):
            citations.append({"tag": f"P{i}", "kind": "pinned", "ref": c.citation, "score": c.rerank_score})
        src_lines = []
        for i, c in enumerate(sources, 1):
            citations.append({"tag": f"S{i}", "kind": "retrieved", "ref": c.citation, "score": c.rerank_score,
                              "key": c.key})
            src_lines.append(f"[S{i}] ({c.citation}) {c.text}")
        mem_refs = [f"{m.meta.get('kind')}:{m.key} (session {m.meta.get('session_id', '?')})"
                    for m in self.upfront.get("memories", [])]
        mem_refs += [f"{c.meta.get('kind')}:{c.key} (session {c.meta.get('session_id', '?')}, JIT)" for c in mem_hits]
        if mem_refs:
            citations.append({"tag": "M", "kind": "memory", "ref": "long-term memory — " + "; ".join(mem_refs[:6]),
                              "score": max([c.rerank_score for c in mem_hits], default=1.0)})

        sections = {"Turn events": self._turn_events(trace)} | dict(self.upfront["sections"])
        if mem_hits:
            extra = "\n".join(f"- {' '.join(c.text.split()[:40])} ({c.meta.get('kind')})" for c in mem_hits)
            sections["Long-term memories about this user"] = (sections.get("Long-term memories about this user", "") + "\n" + extra).strip()
        sections["Working memory"] = self.working.render()
        sections["Conversation summary"] = self.sessions.summary(self.session_id)
        recent = self.sessions.recent(self.session_id)[:-1]  # exclude current message
        sections["Recent turns"] = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        sections["Retrieved sources"] = "\n".join(src_lines)
        ctx = self.builder.build(sections, message, citations)
        trace["context_tokens"] = ctx.tokens
        trace["context_total"] = ctx.total_tokens

        # 5. generate
        text = self.llm.complete(ctx.system, ctx.prompt, self.cfg.max_output_tokens).strip()
        self.sessions.add_message(self.session_id, self.user_id, "assistant", text,
                                  {"citations": [c["ref"] for c in citations if f"[{c['tag']}]" in text]})
        compaction = self.sessions.maybe_compact(self.session_id, self.llm)
        if compaction:
            trace["session_compaction"] = compaction
        trace["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        resp = AgentResponse(text, citations, trace)
        self.turn_log.append({"message": message, "response": text, "trace": trace})
        self._last_context = ctx
        return resp
