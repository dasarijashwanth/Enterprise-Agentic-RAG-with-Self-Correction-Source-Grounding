# Day 7 eval

`ragas_eval.py` is a Ragas-style harness for the LangGraph agent. Run it
with `cd backend && python -m eval.ragas_eval`. It writes
`eval/results.json` and prints a summary table.

## What it measures

- **faithfulness**: fraction of answer claims supported by the chunk
  they cite (deterministic offline checker)
- **citation precision / recall**: do the emitted `[Page, X1, Y1, X2, Y2]`
  citations point at the chunks that genuinely answer the question?
- **context recall**: did the expected chunks survive retrieve + grade?
- **answer rate**: fraction of answerable questions answered grounded
- **decline honesty**: fraction of adversarial questions declined
  honestly ("will not guess") instead of fabricated
- **rewrite bound**: the rewrite loop must terminate at <= MAX_REWRITES

## Honest caveats

1. This harness uses the deterministic offline components (overlap
   dense searcher, token-overlap grader, extractive generator,
   token-overlap hallucination checker). It measures **pipeline
   correctness**: the grounding guarantee, citation validity, and the
   rewrite-loop bound. It does **not** measure LLM reasoning quality,
   and nothing here is a claim about benchmark numbers. The spec's
   template benchmark figures are not repeated anywhere in this repo.
2. Faithfulness is 1.0 by construction with the extractive offline
   generator: every claim is taken from its cited chunk's first
   sentence. The metric's value is regression detection, e.g. it
   catches a generator that starts fabricating claims.
3. Citation precision dips below 1.0 when the grader keeps topically
   related chunks (e.g. other validation chunks for a cross-validation
   question). That is genuine pipeline behavior, not a scoring bug.
4. Latency numbers are local-VM figures for the offline pipeline
   (no model calls), useful only as a smoke baseline.
