"""Tier 1 - Working memory.

Short-lived, in-process scratch space for the *current* task.  It holds:

* ``active_task``    - the goal the agent is working on right now
* ``workflow_state`` - structured state of the current workflow (plan, current
                       step, status, variables)
* ``items``          - recent tool outputs and observations (bounded FIFO with
                       priority-aware eviction and a token budget)

Working memory is deliberately small: it is always rendered into the prompt, so
every token here is paid on every turn.  Large tool outputs are clipped on
entry, and when either the item cap or the token budget is exceeded the
lowest-priority, oldest item is evicted first.  When a task completes the tier
is cleared; anything worth keeping must be promoted to session or long-term
memory by the agent.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field

from ..text import count_tokens, truncate_to_tokens


@dataclass
class WMItem:
    kind: str            # tool_output | observation | note
    content: str
    source: str = ""     # e.g. tool name
    priority: int = 1    # 0 = low, 1 = normal, 2 = pinned for this task
    ts: float = field(default_factory=time.time)

    @property
    def tokens(self) -> int:
        return count_tokens(self.content)


class WorkingMemory:
    def __init__(self, max_items: int = 12, token_budget: int = 900, max_item_tokens: int = 220):
        self.max_items = max_items
        self.token_budget = token_budget
        self.max_item_tokens = max_item_tokens
        self.active_task: str | None = None
        self.workflow_state: dict = {}
        self.items: deque[WMItem] = deque()
        self.evictions = 0

    # ----------------------------------------------------------- task state
    def set_task(self, goal: str, plan: list[str] | None = None) -> None:
        self.active_task = goal
        self.workflow_state = {"plan": plan or [], "step": 0, "status": "in_progress", "vars": {}}

    def advance(self, note: str | None = None) -> None:
        self.workflow_state["step"] = self.workflow_state.get("step", 0) + 1
        if note:
            self.add("observation", note, source="workflow")

    def set_var(self, key: str, value) -> None:
        self.workflow_state.setdefault("vars", {})[key] = value

    def complete_task(self) -> dict:
        snap = self.snapshot()
        self.active_task, self.workflow_state = None, {}
        self.items.clear()
        return snap

    # ---------------------------------------------------------------- items
    def add(self, kind: str, content: str, source: str = "", priority: int = 1) -> WMItem:
        content = truncate_to_tokens(content.strip(), self.max_item_tokens)
        item = WMItem(kind, content, source, priority)
        self.items.append(item)
        self._enforce_limits()
        return item

    def add_tool_output(self, tool: str, output: str) -> WMItem:
        return self.add("tool_output", output, source=tool)

    def tokens(self) -> int:
        return sum(i.tokens for i in self.items) + count_tokens(str(self.active_task or "")) + count_tokens(
            str(self.workflow_state)
        )

    def _enforce_limits(self) -> None:
        while len(self.items) > self.max_items or (self.tokens() > self.token_budget and len(self.items) > 1):
            victim = min(self.items, key=lambda i: (i.priority, i.ts))
            self.items.remove(victim)
            self.evictions += 1

    # ------------------------------------------------------------ rendering
    def render(self) -> str:
        if not self.active_task and not self.items:
            return ""
        lines = []
        if self.active_task:
            ws = self.workflow_state
            plan = ws.get("plan") or []
            step = ws.get("step", 0)
            lines.append(f"Active task: {self.active_task} (status: {ws.get('status')})")
            if plan:
                for i, p in enumerate(plan):
                    mark = "x" if i < step else (">" if i == step else " ")
                    lines.append(f"  [{mark}] {p}")
            if ws.get("vars"):
                lines.append(f"  vars: {ws['vars']}")
        for it in self.items:
            tag = f"{it.kind}:{it.source}" if it.source else it.kind
            lines.append(f"- ({tag}) {it.content}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {
            "active_task": self.active_task,
            "workflow_state": self.workflow_state,
            "items": [asdict(i) for i in self.items],
        }

    def restore(self, snap: dict) -> None:
        self.active_task = snap.get("active_task")
        self.workflow_state = snap.get("workflow_state") or {}
        self.items = deque(WMItem(**i) for i in snap.get("items", []))
        self._enforce_limits()
