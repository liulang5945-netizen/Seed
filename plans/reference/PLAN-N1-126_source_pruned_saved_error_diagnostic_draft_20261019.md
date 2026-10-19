# PLAN-N1-126 — source-pruned saved-error diagnostic

**Status: draft awaiting approval for a new model-query intervention.** PLAN-N1-123 and PLAN-N1-124 both failed the preregistered position-over-cut margin: +5/60 and +1/60. A read-only output audit found zero free-versus-teacher exactness disagreements. Position errors were chiefly lookup false abstentions and valid-but-wrong transition directions; revision was 15/15 exact on both faces. The active reader already contains question-byte alignment, contextual question encoding, and question-conditioned key/value residuals, so another generic alignment module is not supported by current evidence.

## Question

When the position model gets a lookup or transition wrong, does it recover if the same item contains only its own task-relevant evidence rows? This intervention removes competing entities while leaving the question, retained evidence order, answer, byte writer, checkpoint, and scorer unchanged. Because it also shortens context and changes recurrent state, a gain is an upper bound for reduced interference/context burden; it does not isolate learned source selection, establish capability, or generalize to new items.

## Frozen intervention and cohort

- Reuse the twenty confirm items from each frozen v13 and v14 face; evaluate only lookup, revision, and transition, five items per class per face. Exclude unknown items because they have no task-relevant evidence rows to retain.
- Construct the intervention input by retaining only rows referring to the target entity: lookup keeps its single `uses` row; revision keeps both the stale and `now uses` rows in their original order; transition keeps the `faces` and `turns` rows in their original order. Do not reorder rows or edit question/answer bytes.
- Query the six N1-120 candidate position/cut endpoints (seeds 11–13) on both faces: 2 faces × 15 items × 6 endpoints = 180 queries. Compare each result with the already-saved native full-context row for the same checkpoint and item. Do not reload baseline models or query terminal endpoints.
- Before querying, verify both original confirmation reports, all 12 candidate checkpoint hashes, scorer identity, and prior restore/source receipts. If any identity differs, stop without attempting queries.

## One-shot handling

One zero-query readiness pass must bind this plan, both faces and reports, all source/prior exclusion hashes, checkpoint hashes, scorer/evaluator code, and the exact target-row extraction code. Then execute the 180 row queries exactly once, with an independent CPU ceiling of 600 seconds. Preserve partial rows and query counts on failure; do not retry.

No training, optimizer updates, new face generation, dev queries, unknown queries, threshold changes, or capability promotion are included. Gold-selected rows are an oracle diagnostic input; an improvement is an upper bound on source access under this intervention, not autonomous retrieval performance. A failure to recover directs the next review toward byte decoding or task composition; recovery across both lookup and transition would justify a later, separately approved source-selection mechanism proposal.

## Proposed outputs

- Readiness: `reports/taiji_n1_v13_v14_source_pruned_readiness_20261019.json`.
- Attempt and result: `reports/taiji_n1_v13_v14_source_pruned_attempt_20261019.json` and `reports/taiji_n1_v13_v14_source_pruned_result_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-126_execution_closeout_20261019.md`.
