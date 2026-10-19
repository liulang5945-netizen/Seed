# PLAN-N1-127 — full-context source-attention mask diagnostic

**Status: approved for one-shot execution on 2026-10-19 under the user's standing authorization for normal local N1 diagnostics.** PLAN-N1-126 completed 180 free-generation queries. Keeping only target-entity rows made position exact on all 45 decodable v13 items and all 45 v14 items, while cut reached 43/45 on each face. Because row pruning also shortened context and changed recurrent state, the result cannot isolate attention selection. This diagnostic keeps the complete sixteen-row observation history while masking non-target evidence only at decoder memory reads.

## Question

With the full observation sequence and its contextual states intact, does forcing decoder reads to the target entity's source rows recover the saved lookup and transition errors? Recovery would show that the current model can use the relevant content representation under an oracle address and would motivate a later learned attention-selection revision. It remains an oracle result, not autonomous retrieval or capability evidence.

## Frozen intervention and cohort

- Reuse the same confirm lookup/revision/transition items from v13 and v14 (15 items per face; exclude unknown).
- Observe all sixteen original rows in original order and leave the full tape and all question/context/relation encodings unchanged. During free generation only, intercept one-dimensional source-attention logits whose length equals the tape length; apply `-inf` to non-target source positions before softmax. This masks both the native first-hop content read and the relative reader's second-hop read at every autoregressive step, including the step that predicts the boundary. Leave teacher-forced scoring unmasked. Before relative transport, zero and renormalize the carried address outside the same target rows, then verify the transported address remains normalized and source-local. Select rows by the target entity and task relation, never by answer bytes or gold byte position; provide no answer prefix.
- Query the six N1-120 candidate position/cut endpoints (seeds 11–13) on both faces: 2 faces × 15 items × 6 endpoints = 180 free-generation queries. Compare each result with the saved native full-context row and PLAN-N1-126 source-pruned row for the same checkpoint and item.
- Before queries, verify both face/report chains, six candidate checkpoint hashes, the 36-endpoint readiness chain, scorer identity, and the PLAN-N1-126 result. Stop if any identity differs.

## One-shot execution

Complete a zero-query instrument preflight first. On synthetic logits, the all-allowed mask must reproduce native softmax bit-for-bit; a partial mask must assign exactly zero probability outside selected rows and normalize retained weights to one. A synthetic source-local transport must preserve the mask. At runtime require exactly two masked content-read softmaxes per generated step, normalized final attention with no mass outside selected rows, and native boundary handling. Verify full observation/question/context states and checkpoint bundles remain unchanged. Then run the 180 queries exactly once with cumulative CPU≤600 seconds including readiness, recovery, scoring, and final source hashes. Preserve partial rows and counts on failure; do not retry.

No training, optimizer updates, new sample, dev query, unknown query, gate change, or capability promotion. Gold target-source identity controls the mask, so gains are an oracle upper bound. This intervention preserves the full state history and directly tests decoder access under an oracle source choice; it still does not prove the model can select that source itself.

## Proposed outputs

- Instrument and evaluator: `scripts/training/eval_taiji_n1_v13_v14_full_context_source_mask.py`.
- Readiness, attempt, and result: `reports/taiji_n1_v13_v14_full_context_source_mask_readiness_20261019.json`, `reports/taiji_n1_v13_v14_full_context_source_mask_attempt_20261019.json`, and `reports/taiji_n1_v13_v14_full_context_source_mask_result_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-127_execution_closeout_20261019.md`.
