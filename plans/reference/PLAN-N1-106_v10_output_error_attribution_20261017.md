# PLAN-N1-106 · v10 output error attribution · 2026-10-17

**Status: frozen for read-only analysis.** This step uses only the already-consumed PLAN-N1-105 diagnostic face and its complete result. It does not create another measurement face.

## Question

For the matched synthetic items where a frozen endpoint's free output was not exact, did it emit the current target, a stale revision value, another value observed in the same item's observations, the `unknown` abstention token, an empty output, or a value absent from the item's observations? How do these descriptive error classes vary by seed, arm, task kind, answer length, and target position?

## Fixed analysis

- Read only `plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json` (the 248 already-consumed diagnostic items) and `reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json` (the complete 2,232 output rows). Bind both exact file hashes, the PLAN-N1-105 SHA, and this plan SHA in one write-once report at `reports/taiji_n1_hard_neighbor_v10_output_error_attribution_20261017.json`.
- Verify the result is complete and has exactly one row for every item × seed (11/12/13) × arm (`position`/`terminal`/`cut`), that row identity matches the face, and that the result remains `can_promote=false`, has zero updates/optimizer steps, no gate recomputation, and no dev or existing confirmation queries.
- Classify each row into mutually exclusive categories: exact answer; stale prior material for the same revision target; another observed material for the target entity; a material observed for a distractor entity (or any observed material when the item has no target row); `unknown` emitted for a known-answer question; empty output; or other/unobserved output. Preserve the original result row and face item key in every classified record. A lexical match is only an output attribution, not evidence of the model's internal retrieval path.
- Report every denominator by seed, arm, kind, answer length, and target position where defined. For position errors, include the category distribution when cut or terminal is exact on the paired item. Do not recompute PLAN-N1-94 or PLAN-N1-102 gates, change any threshold, or infer a causal mechanism from these synthetic outcomes.
- Freeze the analyzer and its contract tests before running it. CPU ceiling: 30 seconds. The analyzer must read JSON only; it may not import model runtimes or training loaders, load checkpoints, make model queries, update parameters, train, access dev/holdout, or write to the input face/result. Write the report once and refuse overwrite.

## Interpretation boundary

The output labels can describe whether a wrong string matches an available target or distractor material. They cannot prove why the model produced it, establish generalization, repair the failed v9 confirmation, or support C3/C4, retention, three-organ, or capability-promotion claims. Any training change, gate change, or new measurement face remains a separate plan decision.

## Authorization record

The owner has granted standing authority to carry out the next suitable normal research-planning step without pausing for routine budget decisions. This plan stays within the active PLAN-N1-105 result, does not alter its frozen criteria, and performs no irreversible operation.
