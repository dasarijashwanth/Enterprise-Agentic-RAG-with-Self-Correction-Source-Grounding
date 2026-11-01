# VeriRAG: 3 FDE demo scenarios

FDE = forward-deployed engineer. These are the three stories you tell
on a call, each runnable end to end with one command. All run fully
offline with the deterministic components; nothing phones home.

## Scenario 1: contract Q&A with visual source grounding

The classic enterprise ask: "answer from THIS document, and prove it."

```bash
cd backend
python -m scripts.make_sample_pdf   # builds data/sample_contract.pdf
python -m scripts.demo_scenario_1
```

What happens: the 2-page vendor agreement is parsed with PyMuPDF
(words + bounding boxes), chunked without crossing page boundaries,
and the agent answers "what is the termination notice period" with a
grounded answer and a `[1, 72, 72, 537.87, 314.58]` citation. In the
frontend, that citation renders as a page-1 badge; a viewer can open
the PDF and land on the exact clause.

Talking point: the citation is a coordinate, not a quote. The answer
is checkable by a human in one click.

## Scenario 2: self-correction when retrieval misses

The honest failure mode: the agent does not guess, it rewrites.

```bash
cd backend
python -m scripts.demo_scenario_2
```

What happens: round 1 retrieval returns nothing relevant for "how to
grow tomatoes in a home garden". Instead of answering from thin air,
the agent rewrites the query toward the gardening vocabulary, retries,
and returns a grounded, cited answer. The trace shows the full path:
`retrieve -> grade_documents -> rewrite_query -> retrieve ->
grade_documents -> generate -> check_hallucination`, with the rewrite
counter bounded at 2.

Talking point: bounded loops are a product decision. The agent tries
exactly twice, then says "I could not find relevant context... I will
not guess." That is the behavior you want in compliance and legal.

## Scenario 3: the feedback loop that feeds eval

The demo does not end at the answer; it ends at the thumbs-down.

```bash
cd backend
python -m scripts.demo_scenario_3
```

What happens: an answer is generated, a thumbs-down with a comment is
recorded through the real `POST /api/v1/feedback` handler, and the
appended line in `data/feedback.jsonl` is read back and shown.
Low-rated answers become the regression set for the Day 7 eval harness
(`backend/eval/ragas_eval.py`).

Talking point: evaluation is a loop, not a report. Every bad answer a
user flags becomes a test case the next eval run must handle.
