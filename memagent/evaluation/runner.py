"""Shared helpers to run the multi-session scenario against fresh agent instances."""

from __future__ import annotations

from pathlib import Path

from ..agent import MemoryAgent
from ..config import AgentConfig
from .scenario import SESSIONS, USER


def fresh_agent(data_dir: Path, llm: str = "offline", **overrides) -> MemoryAgent:
    return MemoryAgent(AgentConfig(data_dir=Path(data_dir), llm=llm, **overrides))


def run_scenario(data_dir: Path, knowledge_dir: Path, llm: str = "offline", on_event=None) -> list[list[dict]]:
    """Run every session in SESSIONS with a NEW agent instance per session.

    Returns a list (per session) of turn records:
      {kind: user|action|session_start|session_end, ...}
    ``on_event`` is called with each record as it happens (used by the demo
    to stream a transcript).
    """
    emit = on_event or (lambda e: None)
    data_dir = Path(data_dir)
    first = fresh_agent(data_dir, llm)
    if len(first.ltm.knowledge) == 0:
        first.ingest(knowledge_dir)
    del first

    transcript: list[list[dict]] = []
    for s_idx, turns in enumerate(SESSIONS):
        agent = fresh_agent(data_dir, llm)  # new process-equivalent: state only via disk
        info = agent.start_session(USER)
        rec = {"kind": "session_start", "session": s_idx + 1, **info,
               "upfront_sections": {k: v for k, v in agent.upfront["sections"].items() if v}}
        emit(rec)
        records = [rec]
        for turn in turns:
            if isinstance(turn, tuple):
                action, *args = turn
                if action == "start_task":
                    tid = agent.start_task(*args)
                    r = {"kind": "action", "action": "start_task", "args": args, "task_id": tid}
            else:
                resp = agent.chat(turn)
                r = {
                    "kind": "user",
                    "message": turn,
                    "response": resp.text,
                    "rendered": resp.render(),
                    "citations": [c for c in resp.citations if f"[{c['tag']}]" in resp.text],
                    "trace": resp.trace,
                    "prompt": agent._last_context.prompt,
                }
            emit(r)
            records.append(r)
        end = agent.end_session()
        end_rec = {"kind": "session_end", "session": s_idx + 1, **end}
        emit(end_rec)
        records.append(end_rec)
        transcript.append(records)
    return transcript
