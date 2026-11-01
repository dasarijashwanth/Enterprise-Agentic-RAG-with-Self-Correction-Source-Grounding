"""FDE demo scenario 3: the feedback loop.

Runs one agent question end to end, then records a thumbs-down with a
comment through the real POST /api/v1/feedback handler, and reads the
appended line back from data/feedback.jsonl (gitignored). This is the
raw material the Day 7 eval loop consumes: low-rated answers become
regression cases.

Run: cd backend && python -m scripts.demo_scenario_3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.v1.feedback import FeedbackRequest, feedback
from app.graph import build_graph, run_question
from eval.ragas_eval import EvalRetriever, make_corpus


def main() -> int:
    graph = build_graph(EvalRetriever(make_corpus()))
    question = "how do bounding boxes work"
    final = run_question(graph, question)
    print(f"question: {question}")
    print(f"grounded: {final['grounded']}")
    print(f"answer: {final['answer'][:160]}...")

    resp = feedback(
        FeedbackRequest(
            query=question,
            answer=final["answer"],
            rating="down",
            comment="demo: answer cited the right passage but I wanted the click-to-zoom behavior explained",
        )
    )
    print(f"\nfeedback: {resp.status}")

    log = Path(__file__).resolve().parent.parent / "data" / "feedback.jsonl"
    last = log.read_text().strip().splitlines()[-1]
    entry = json.loads(last)
    print(f"appended to data/feedback.jsonl: rating={entry['rating']} "
          f"query={entry['query'][:40]}...")
    assert entry["rating"] == "down"
    print("\nOK: feedback recorded; low-rated answers feed the eval regression set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
