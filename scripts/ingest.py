"""Chunk, embed and index a folder of Markdown files into long-term memory.

    python scripts/ingest.py --docs data/knowledge --data-dir .memagent
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memagent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.retrieval.chunking import chunk_directory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/knowledge")
    ap.add_argument("--data-dir", default=".memagent")
    ap.add_argument("--show", type=int, default=3, help="print the first N chunks")
    args = ap.parse_args()

    cfg = AgentConfig(data_dir=Path(args.data_dir), llm="offline")
    agent = MemoryAgent(cfg)
    n = agent.ingest(args.docs)
    print(f"Indexed {n} chunks from {args.docs} into {cfg.data_dir}/ltm (embedder: {agent.embedder.name})")
    for c in chunk_directory(args.docs, target_tokens=cfg.chunk_target_tokens,
                             overlap_tokens=cfg.chunk_overlap_tokens)[: args.show]:
        print(f"\n[{c.chunk_id}] {c.citation}\n  {c.text[:200]}…")


if __name__ == "__main__":
    main()
