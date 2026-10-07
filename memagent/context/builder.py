"""Prompt assembly under a token budget.

The prompt is a fixed sequence of named sections.  Each section has a cap;
anything over its cap is truncated, and the builder returns a per-section token
breakdown so context usage is observable on every turn.

    ┌ System instructions (grounding + citation rules)       fixed
    ├ Turn events                             memory writes / task updates made this turn
    ├ User profile & preferences            ┐
    ├ Task progress                         │ UPFRONT  (loaded once at
    ├ Previous sessions                     │          session start, cached)
    ├ Long-term memories about this user    │  + JIT memory hits
    ├ Pinned knowledge                      ┘
    ├ Working memory                          active task, tool outputs
    ├ Conversation summary                    rolling summary of compacted turns
    ├ Recent turns                            last N verbatim turns
    ├ Retrieved sources                       JUST-IN-TIME, cited as [S1]…[Sn]
    └ Current user message
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..text import count_tokens, truncate_to_tokens

SYSTEM = """You are a long-running engineering assistant with a multi-tier memory.
Ground every factual claim about the platform in the Retrieved sources or Pinned knowledge and
cite them inline as [S1], [S2] ... (pinned knowledge is cited as [P1] ...). Facts that come from your
memory of the user are cited as [M]. If nothing in context supports an answer, say so instead of guessing.
Respect the user's stated preferences. Use Task progress to continue multi-session work: say what is
done and what comes next."""

SECTION_CAPS = {
    "Turn events": 120,
    "User profile & preferences": 120,
    "Task progress": 260,
    "Previous sessions": 220,
    "Long-term memories about this user": 220,
    "Pinned knowledge": 420,
    "Working memory": 300,
    "Conversation summary": 160,
    "Recent turns": 500,
    "Retrieved sources": 900,
}


@dataclass
class BuiltContext:
    prompt: str
    system: str
    sections: dict[str, str]
    tokens: dict[str, int]
    citations: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens.values())


class ContextBuilder:
    def __init__(self, budget: int = 2400, caps: dict | None = None):
        self.budget = budget
        self.caps = {**SECTION_CAPS, **(caps or {})}

    def build(self, sections: dict[str, str], user_message: str, citations: list[dict]) -> BuiltContext:
        out: dict[str, str] = {}
        tokens = {"System": count_tokens(SYSTEM), "Current user message": count_tokens(user_message)}
        remaining = self.budget - sum(tokens.values())
        # Order expresses priority for the global budget: recent turns and
        # retrieved sources are protected; summaries go first if we must cut.
        for name in self.caps:
            body = (sections.get(name) or "").strip()
            if not body:
                continue
            cap = min(self.caps[name], max(0, remaining))
            if cap <= 0:
                continue
            if count_tokens(body) > cap:
                body = truncate_to_tokens(body, cap)
            out[name] = body
            tokens[name] = count_tokens(body)
            remaining -= tokens[name]
        parts = [f"## {k}\n{v}" for k, v in out.items()]
        parts.append(f"## Current user message\n{user_message}")
        return BuiltContext("\n\n".join(parts), SYSTEM, out, tokens, citations)
