"""Structure-aware chunking.

Strategy (documented in docs/retrieval_pipeline.md):

1. Split a Markdown document on headings, so a chunk never straddles two
   sections and every chunk carries its *section path* ("Deployments > Rollback")
   for attribution.
2. Inside a section, pack whole sentences greedily until ``target_tokens`` is
   reached.  Sentences are never cut in half.
3. Start the next chunk with the trailing ``overlap_tokens`` worth of
   sentences from the previous one, so a fact that spans a boundary is still
   retrievable from either side.
4. Prefix every chunk's *embedded* text with its title and section path
   ("contextual chunk headers") which measurably improves retrieval for short
   chunks whose body alone is ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..text import content_hash, count_tokens, split_sentences

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    source: str            # file name / URI
    title: str             # document title
    section: str           # heading path within the document
    text: str              # body text shown to the model and the user
    position: int          # ordinal of the chunk within its document
    char_start: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        return f"{self.title} | {self.section}\n{self.text}"

    @property
    def citation(self) -> str:
        return f"{self.source} § {self.section}"


def _sections(markdown: str) -> list[tuple[list[str], str, int]]:
    """Return (heading_path, body, char_offset) for every section."""
    path: list[str] = []
    out: list[tuple[list[str], str, int]] = []
    buf: list[str] = []
    start = 0
    offset = 0
    for line in markdown.splitlines(keepends=True):
        m = _HEADING.match(line.strip())
        if m:
            if "".join(buf).strip():
                out.append((list(path), "".join(buf), start))
            level, name = len(m.group(1)), m.group(2).strip()
            path = path[: level - 1] + [name]
            buf, start = [], offset + len(line)
        else:
            buf.append(line)
        offset += len(line)
    if "".join(buf).strip():
        out.append((list(path), "".join(buf), start))
    return out


def chunk_markdown(
    markdown: str,
    source: str,
    target_tokens: int = 160,
    overlap_tokens: int = 30,
) -> list[Chunk]:
    sections = _sections(markdown)
    title = next((p[0] for p, _, _ in sections if p), Path(source).stem)
    chunks: list[Chunk] = []
    for path, body, start in sections:
        section = " > ".join(path[1:] or path) or "Introduction"
        # bullet lists are kept as sentence-like units
        units: list[str] = []
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para:
                continue
            if re.match(r"^[-*\d]", para):
                units.extend(l.strip() for l in para.splitlines() if l.strip())
            else:
                units.extend(split_sentences(para))
        cur: list[str] = []
        for unit in units:
            if cur and count_tokens(" ".join(cur + [unit])) > target_tokens:
                chunks.append(_make(cur, source, title, section, len(chunks), start))
                # overlap: carry trailing sentences forward
                carry: list[str] = []
                for s in reversed(cur):
                    if count_tokens(" ".join([s] + carry)) > overlap_tokens:
                        break
                    carry.insert(0, s)
                cur = carry
            cur.append(unit)
        if cur:
            chunks.append(_make(cur, source, title, section, len(chunks), start))
    return chunks


def _make(units, source, title, section, pos, start) -> Chunk:
    text = " ".join(units)
    cid = f"{Path(source).stem}:{pos}:{content_hash(text)[:6]}"
    return Chunk(cid, source, title, section, text, pos, start)


def chunk_directory(directory: str | Path, **kw) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name, **kw))
    return chunks
