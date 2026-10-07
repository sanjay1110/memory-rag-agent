"""Central configuration for the memory agent.

Every tunable number in the system lives here so the memory design document
can point at one place, and so experiments can override values without
touching code (``AgentConfig(**overrides)``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    # ---- storage -------------------------------------------------------
    data_dir: Path = Path(os.environ.get("MEMAGENT_DATA_DIR", ".memagent"))

    # ---- working memory (tier 1) --------------------------------------
    working_max_items: int = 12          # recent tool outputs / observations kept
    working_token_budget: int = 900      # hard cap on tokens working memory may occupy

    # ---- session memory (tier 2) --------------------------------------
    session_recent_turns: int = 6        # verbatim turns kept in the prompt
    session_compact_after: int = 10      # turns beyond this get folded into the rolling summary

    # ---- long-term memory (tier 3) ------------------------------------
    ltm_dedup_threshold: float = 0.90    # cosine above which a new memory merges into an old one
    ltm_recency_half_life_days: float = 30.0

    # ---- chunking ------------------------------------------------------
    chunk_target_tokens: int = 160
    chunk_overlap_tokens: int = 30

    # ---- retrieval -----------------------------------------------------
    embedder: str = os.environ.get("MEMAGENT_EMBEDDER", "hashing")  # hashing | sentence-transformers
    embedding_dim: int = 1024
    dense_k: int = 20                    # candidates from FAISS
    sparse_k: int = 20                   # candidates from BM25
    rrf_k: int = 60                      # reciprocal-rank-fusion constant
    rerank_top_n: int = 12               # candidates sent to the reranker
    final_k: int = 4                     # chunks that survive into the prompt
    min_relevance: float = 0.20          # absolute reranker score floor (context filtering)
    relative_floor: float = 0.5          # also drop items scoring < 50% of the best candidate

    # ---- context optimisation -----------------------------------------
    context_token_budget: int = 2400     # total prompt budget the builder must respect
    near_dup_threshold: float = 0.85     # cosine for near-duplicate removal
    upfront_knowledge_k: int = 2         # pinned knowledge chunks loaded at session start

    # ---- generation ----------------------------------------------------
    llm: str = os.environ.get("MEMAGENT_LLM", "auto")  # auto | anthropic | offline
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    llm_rerank: bool = True              # use the LLM as a reranker when one is available
    max_output_tokens: int = 600

    extra: dict = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        p = Path(self.data_dir, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
