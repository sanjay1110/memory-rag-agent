"""Tests for each memory tier, the retrieval pipeline, context optimisation and
cross-session persistence.  Run with:  pytest -q"""

from pathlib import Path

import pytest

from memagent import AgentConfig, MemoryAgent
from memagent.context.optimizer import ContextOptimizer
from memagent.embeddings import HashingEmbedder
from memagent.memory.long_term import extract_memories
from memagent.memory.working import WorkingMemory
from memagent.retrieval.chunking import chunk_markdown
from memagent.retrieval.index import Candidate

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "data" / "knowledge"


def make_agent(tmp_path) -> MemoryAgent:
    agent = MemoryAgent(AgentConfig(data_dir=tmp_path, llm="offline"))
    if len(agent.ltm.knowledge) == 0:
        agent.ingest(KNOWLEDGE)
    return agent


# --------------------------------------------------------------- working memory
def test_working_memory_bounds_and_eviction():
    wm = WorkingMemory(max_items=3, token_budget=10_000)
    wm.set_task("ship it", ["a", "b"])
    wm.add("note", "pinned", priority=2)
    for i in range(5):
        wm.add_tool_output("search", f"result {i}")
    assert len(wm.items) == 3
    assert any(i.content == "pinned" for i in wm.items), "high-priority item must survive eviction"
    assert wm.evictions == 3


def test_working_memory_token_budget_and_clip():
    wm = WorkingMemory(max_items=50, token_budget=120, max_item_tokens=40)
    for _ in range(10):
        wm.add_tool_output("dump", "word " * 500)
    assert wm.tokens() <= 120
    assert all(i.tokens <= 45 for i in wm.items)


def test_working_memory_snapshot_roundtrip():
    wm = WorkingMemory()
    wm.set_task("migrate", ["x", "y"])
    wm.advance("x done")
    snap = wm.snapshot()
    wm2 = WorkingMemory()
    wm2.restore(snap)
    assert wm2.active_task == "migrate" and wm2.workflow_state["step"] == 1
    assert "x done" in wm2.render()


# --------------------------------------------------------------- session memory
def test_session_memory_compaction(tmp_path):
    agent = make_agent(tmp_path)
    agent.start_session("u")
    for i in range(8):
        agent.chat(f"How long are logs kept? ({i})")
    live = agent.sessions.messages(agent.session_id, include_compacted=False)
    assert len(live) <= agent.cfg.session_compact_after
    assert agent.sessions.summary(agent.session_id)
    assert len(agent.sessions.messages(agent.session_id)) == 16, "history is never deleted"


def test_task_progress_from_natural_language(tmp_path):
    agent = make_agent(tmp_path)
    agent.start_session("u")
    agent.start_task("Migrate svc", ["Containerise the service", "Write the service manifest"])
    agent.chat("Did we finish the manifest?")  # a question is not a completion
    assert not agent.sessions.open_tasks("u")[0]["steps"][1]["done"]
    r = agent.chat("Done with the service manifest.")
    assert r.trace["steps_completed"] == ["Write the service manifest"]


# ------------------------------------------------------------ long-term memory
def test_extraction_rules():
    found = {m["text"] for m in extract_memories(
        "My name is Ana. I prefer Go examples. We decided to use blue/green. The deadline is May 3.")}
    assert {"User's name is Ana", "User prefers Go examples", "Decision: use blue/green",
            "Project deadline is May 3"} <= found


def test_ltm_dedup_and_supersession(tmp_path):
    agent = make_agent(tmp_path)
    ltm = agent.ltm
    assert ltm.remember("u", "User prefers Python examples", "preference", slot="pref:language")["action"] == "written"
    assert ltm.remember("u", "User prefers Go examples", "preference", slot="pref:language")["action"] == "superseded"
    prefs = [m for m in ltm.user_memories("u") if m.meta.get("slot") == "pref:language"]
    assert len(prefs) == 1 and "Go" in prefs[0].text and prefs[0].meta["history"]
    assert ltm.remember("u", "The team uses Kafka for events", "fact")["action"] == "written"
    assert ltm.remember("u", "The team uses Kafka for events.", "fact")["action"] == "merged"
    assert len(ltm.user_memories("u")) == 2


def test_memories_are_user_scoped(tmp_path):
    agent = make_agent(tmp_path)
    agent.start_session("alice")
    agent.chat("Remember that our cluster is called aurora.")
    agent.end_session()
    agent.start_session("bob")
    r = agent.chat("What do you remember about me?")
    assert "aurora" not in r.text.lower()


# ------------------------------------------------------------------- retrieval
def test_chunking_keeps_sections_and_overlap():
    md = "# Doc\n\n## A\n" + " ".join(f"Sentence number {i} is here." for i in range(60)) + "\n\n## B\nShort."
    chunks = chunk_markdown(md, "doc.md", target_tokens=60, overlap_tokens=15)
    assert {c.section for c in chunks} == {"A", "B"}
    a = [c for c in chunks if c.section == "A"]
    assert len(a) > 1
    assert a[0].text.split(". ")[-1].rstrip(".") in a[1].text, "overlap carries the tail sentence forward"
    assert all(c.citation.startswith("doc.md § ") for c in chunks)


@pytest.mark.parametrize("query,source,section", [
    ("What is the maximum value for NIMBUS_DB_POOL?", "databases.md", "Connection pooling"),
    ("How fast must a sev1 be acknowledged?", "incidents.md", "Severity levels"),
    ("What does E429 mean?", "rate_limits.md", "Throttling behaviour"),
])
def test_retrieval_returns_relevant_with_attribution(tmp_path, query, source, section):
    agent = make_agent(tmp_path)
    top = agent.retriever.retrieve(query).items[0]
    assert top.meta["source"] == source and section in top.meta["section"]
    assert top.citation == f"{source} § {top.meta['section']}"


def test_answer_has_visible_sources(tmp_path):
    agent = make_agent(tmp_path)
    agent.start_session("u")
    r = agent.chat("How long are database backups retained?")
    rendered = r.render()
    assert "14 days" in r.text and "Sources:" in rendered and "databases.md § Backups and restore" in rendered


# --------------------------------------------------------- context optimisation
def _cand(key, text, score, coverage=1.0):
    c = Candidate(key, text, {"source": "x.md", "section": key}, "knowledge", rerank_score=score)
    c.signals["coverage"] = coverage
    return c


def test_optimizer_filter_dedup_compact():
    opt = ContextOptimizer(HashingEmbedder(), min_relevance=0.2, relative_floor=0.5)
    long_text = "Backups run nightly at 02:00 UTC. They are kept for 14 days. " + "Unrelated filler sentence. " * 40
    items = [
        _cand("a", long_text, 0.9),
        _cand("b", "Backups run nightly at 02:00 UTC. They are kept for 14 days.", 0.8),   # contained in a
        _cand("c", "Totally irrelevant cost text.", 0.15),                                 # below floor
        _cand("d", "Spot instances are for batch only.", 0.3),                             # below relative floor
        _cand("e", "Logs are retained 30 days.", 0.7, coverage=0.0),                       # zero coverage
    ]
    kept, rep = opt.optimise(items, "how long are backups kept", final_k=4, token_budget=40)
    assert [c.key for c in kept] == ["a"]
    assert rep.filtered == 3 and rep.near_dups == 1
    assert rep.compacted_items == 1 and "14 days" in kept[0].text
    assert rep.output_tokens <= 40


def test_prompt_stays_within_budget(tmp_path):
    agent = make_agent(tmp_path)
    agent.start_session("u")
    for q in ["How do I roll back?", "What is the canary percentage?", "How long are logs kept?"] * 3:
        r = agent.chat(q)
        assert r.trace["context_total"] <= agent.cfg.context_token_budget


# ----------------------------------------------- hybrid strategy & persistence
def test_hybrid_upfront_then_jit(tmp_path):
    agent = make_agent(tmp_path)
    agent.start_session("u")
    agent.start_task("Migrate orders-api to Nimbus", ["Containerise the service", "Write the service manifest"])
    assert agent.upfront["pinned"], "task-relevant knowledge is pinned upfront"
    r1 = agent.chat("Which base image should I use to containerise it?")
    assert "pinned" in r1.trace["plan"]["reason"] and "knowledge_retrieval" not in r1.trace
    assert "[P1]" in r1.text
    r2 = agent.chat("How long are logs kept?")
    assert r2.trace["plan"]["knowledge"] and "knowledge_retrieval" in r2.trace


def test_continuity_across_sessions_and_restarts(tmp_path):
    a1 = make_agent(tmp_path)
    a1.start_session("priya")
    a1.start_task("Migrate orders-api to Nimbus", ["Containerise the service", "Write the service manifest"])
    a1.chat("My name is Priya. We decided to use logical replication for the database move.")
    a1.chat("I've finished containerise the service.")
    a1.end_session()
    del a1

    a2 = make_agent(tmp_path)  # new instance - only disk is shared
    info = a2.start_session("priya")
    assert info["restored_task"] == "Migrate orders-api to Nimbus"
    r = a2.chat("Where did we leave off?")
    assert "Write the service manifest" in r.text
    r = a2.chat("What did we decide for the database move?")
    assert "logical replication" in r.text and "[M]" in r.text
    assert a2.upfront["previous"], "previous session summary is loaded upfront"
