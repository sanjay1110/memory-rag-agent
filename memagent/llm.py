"""LLM clients used for generation, reranking and memory consolidation.

* ``AnthropicLLM`` - Claude via the Anthropic Messages API (set ``ANTHROPIC_API_KEY``).
* ``OfflineLLM``  - a deterministic, extractive stand-in so the whole system
  (demo, tests, evaluation) runs with no network and no key, and so evaluation
  numbers are reproducible.  It answers by stitching together the most
  query-relevant sentences from the supplied context and keeps their citation
  markers, which is exactly the behaviour the grounded prompt asks the real
  model for.

``make_llm("auto")`` picks Anthropic when a key is present, otherwise offline.
"""

from __future__ import annotations

import json
import os
import re

from .text import jaccard, split_sentences, tokenize


class BaseLLM:
    name = "base"

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        raise NotImplementedError

    # Generic helpers built on ``complete`` -------------------------------
    def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        listing = "\n\n".join(f"[{i}] {p[:700]}" for i, p in enumerate(passages))
        prompt = (
            f"Query: {query}\n\nPassages:\n{listing}\n\n"
            "Score how useful each passage is for answering the query, 0 (irrelevant) to 10 "
            "(directly answers it). Reply with only a JSON list of numbers, one per passage, in order."
        )
        try:
            raw = self.complete("You are a precise relevance judge.", prompt, max_tokens=200)
            scores = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            if len(scores) != len(passages):
                return None
            return [max(0.0, min(1.0, float(s) / 10.0)) for s in scores]
        except Exception:
            return None

    def summarise(self, text: str, max_words: int = 80) -> str:
        return self.complete(
            "You compress conversation history for an AI agent's memory. Keep decisions, "
            "facts about the user, open tasks and numbers. Drop pleasantries.",
            f"Summarise in at most {max_words} words:\n\n{text}",
            max_tokens=max_words * 2,
        ).strip()


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OfflineLLM(BaseLLM):
    """Extractive, deterministic generator.  Not a language model - a baseline
    that lets the memory/retrieval machinery be exercised and measured offline
    with reproducible results.  It follows the same contract the system prompt
    gives a real model: acknowledge turn events, answer recall questions from
    task progress and memories, answer knowledge questions only from cited
    context, and refuse to guess when nothing supports an answer."""

    name = "offline-extractive"
    RECALL = re.compile(r"\b(remember|recall|le(ft|ave) off|where were we|next step|what'?s next|progress|know about me|decid)", re.I)

    def __init__(self):
        from .embeddings import HashingEmbedder

        self.emb = HashingEmbedder(512)

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        if prompt.startswith("Summarise"):
            body = prompt.split("\n\n", 1)[-1]
            m = re.search(r"at most (\d+) words", prompt)
            return self._summarise(body, int(m.group(1)) if m else 80)
        return self._answer(prompt)

    def rerank(self, query, passages):  # defer to the cross-feature reranker
        return None

    # ------------------------------------------------------------------
    def _summarise(self, text: str, max_words: int) -> str:
        """Keep the user's own statements (they carry facts, decisions and
        progress), clipped, oldest first."""
        out, words = [], 0
        prior = re.match(r"Earlier summary: (.*)", text)
        if prior:
            out.append(prior.group(1).strip())
            words += len(out[0].split())
        for line in text.splitlines():
            if not line.lower().startswith("user:"):
                continue
            said = " ".join(line[5:].split()[:16])
            if words + len(said.split()) > max_words:
                break
            out.append(said)
            words += len(said.split())
        return "; ".join(out)

    def _expand(self, terms: set[str]) -> set[str]:
        from .embeddings import SYNONYMS

        out = set(terms)
        for t in terms:
            out.update(SYNONYMS.get(t, ()))
            out.add(t[:5])
        return out

    def _overlap(self, qx: set[str], text: str) -> float:
        toks = set(tokenize(text, True))
        toks |= {t[:5] for t in toks}
        return len(qx & toks)

    def _answer(self, prompt: str) -> str:
        query = _section(prompt, "Current user message").strip() or prompt[-300:]
        qterms = set(tokenize(query, True))
        qx = self._expand(qterms)
        prefs = _section(prompt, "User profile & preferences").lower()
        concise = "concise" in prefs
        parts: list[str] = []

        is_question = bool(re.search(r"\?\s*$|^\s*(what|which|how|why|when|where|who|can|should|do|does|is)\b", query, re.I))

        # 1. acknowledge what the agent did this turn
        events = re.findall(r"^- (.*)$", _section(prompt, "Turn events"), re.M)
        for e in events:
            if e.startswith(("Saved", "Updated", "Already known")):
                parts.append(e.split(": ", 1)[-1].rstrip(".") + " - noted for future sessions.")
            elif e.startswith("Marked task step done"):
                parts.append(f"Marked '{e.split(': ', 1)[1]}' as done.")
            elif e.startswith("Next step"):
                parts.append(f"Next up: {e.split(': ', 1)[1]}.")

        # 2. recall from task progress + memories
        if is_question and self.RECALL.search(query):
            task = _section(prompt, "Task progress")
            head = re.search(r"^- (.*?) — (\d+)/(\d+) steps done", task, re.M)
            nxt = re.search(r"next: (.*)", task)
            if head and re.search(r"le(ft|ave) off|next|progress|remember|know about", query, re.I):
                parts.append(f"You're working on '{head.group(1)}': {head.group(2)} of {head.group(3)} steps done"
                             + (f", next step is '{nxt.group(1).strip()}'." if nxt else "."))
            mems = [re.sub(r" \((fact|preference|decision|episode)\)$", "", l)
                    for l in re.findall(r"^- (.*)$", _section(prompt, "Long-term memories about this user"), re.M)]
            mems = [m for m in mems if not m.startswith("Session ")]
            if re.search(r"remember|know about", query, re.I):
                picked = mems[:5]
            else:
                ranked = sorted(mems, key=lambda m: -self._overlap(qx, m))
                picked = [m for m in ranked if self._overlap(qx, m) > 0][:2]
            if picked:
                parts.append("From memory: " + "; ".join(picked) + " [M]")

        # 3. grounded knowledge sentences (skipped for pure "where are we?" questions
        #    that were already answered from task progress)
        pure_recall = bool(parts) and re.search(r"le(ft|ave) off|next step|what'?s next|remember|know about me", query, re.I)
        if is_question and not pure_recall:
            qv = self.emb.embed([query])[0]
            cands = []
            for m in re.finditer(r"\[([SP]\d+)\] \(([^)]*)\)\s*(.*?)(?=\n\[[SP]\d+\]|\n##|\Z)", prompt, re.S):
                tag, body = m.group(1), m.group(3)
                bonus = (0.12 if tag[0] == "S" else 0.08) / int(tag[1:])
                for sent in split_sentences(body):
                    ov = self._overlap(qx, sent) / max(1, len(qterms))
                    cos = float(self.emb.embed([sent])[0] @ qv)
                    score = 0.5 * min(ov, 1.0) + 0.5 * cos + bonus
                    if ov > 0 and score >= 0.22:
                        cands.append((score, f"{sent} [{tag}]"))
            cands.sort(key=lambda x: -x[0])
            picked = []
            for _, sent in cands:
                if all(jaccard(sent, p) < 0.5 for p in picked):
                    picked.append(sent)
                if len(picked) == (2 if concise else 3):
                    break
            parts.extend(picked)
        elif any(e.startswith("Next step") for e in events):
            pinned = re.search(r"\[P1\] \([^)]*\)\s*(.*?)(?=\n\[P\d+\]|\n##|\Z)", prompt, re.S)
            if pinned:
                first = split_sentences(pinned.group(1))
                if first:
                    parts.append(f"For that step: {first[0]} [P1]")

        if not parts:
            if is_question:
                return "I don't have grounded information on that in my knowledge base or memory."
            return "Noted."
        return " ".join(parts)


def _section(prompt: str, name: str) -> str:
    m = re.search(rf"## {re.escape(name)}\n(.*?)(?:\n## |\Z)", prompt, re.S)
    return m.group(1) if m else ""


def make_llm(kind: str = "auto", model: str = "claude-sonnet-5-5") -> BaseLLM:
    if kind == "anthropic" or (kind == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicLLM(model)
    return OfflineLLM()
