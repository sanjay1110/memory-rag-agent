"""Interactive chat with the memory agent.

    python scripts/chat.py --user alice            # offline extractive generator
    ANTHROPIC_API_KEY=... python scripts/chat.py   # Claude for generation + reranking

Commands:  /task <title> | step one; step two; ...   start a multi-step task
           /trace      show the last turn's retrieval/memory trace
           /memory     list long-term memories for this user
           /context    print the exact prompt sent to the model last turn
           /end        end the session (consolidate) and start a new one
           /quit       end the session and exit
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memagent import AgentConfig, MemoryAgent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="demo-user")
    ap.add_argument("--data-dir", default=".memagent")
    ap.add_argument("--docs", default="data/knowledge")
    ap.add_argument("--llm", default="auto", choices=["auto", "anthropic", "offline"])
    args = ap.parse_args()

    agent = MemoryAgent(AgentConfig(data_dir=Path(args.data_dir), llm=args.llm))
    if len(agent.ltm.knowledge) == 0:
        print(f"Indexing {args.docs} …", agent.ingest(args.docs), "chunks")
    info = agent.start_session(args.user)
    print(f"Session {info['session_id']} for {args.user} | LLM: {agent.llm.name} | "
          f"upfront context: {info['upfront_tokens']} tokens"
          + (f" | resumed task: {info['restored_task']}" if info["restored_task"] else ""))
    last = None
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            msg = "/quit"
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            print(json.dumps(agent.end_session(), indent=2, default=str)[:800])
            break
        if msg == "/end":
            print(json.dumps(agent.end_session(), indent=2, default=str)[:800])
            info = agent.start_session(args.user)
            print(f"New session {info['session_id']} (upfront {info['upfront_tokens']} tokens)")
            continue
        if msg.startswith("/task "):
            title, _, steps = msg[6:].partition("|")
            agent.start_task(title.strip(), [s.strip() for s in steps.split(";") if s.strip()])
            print("Task started.")
            continue
        if msg == "/trace":
            print(json.dumps(last.trace if last else {}, indent=2, default=str))
            continue
        if msg == "/memory":
            for m in agent.ltm.user_memories(args.user):
                print(f"- [{m.kind}, imp {m.importance:.2f}] {m.text}")
            continue
        if msg == "/context":
            print(agent._last_context.prompt if last else "(no turn yet)")
            continue
        last = agent.chat(msg)
        print(f"\nagent> {last.render()}")
        t = last.trace
        print(f"  · {t['plan']['reason']} · context {t['context_total']} tok · {t['latency_ms']} ms")


if __name__ == "__main__":
    main()
