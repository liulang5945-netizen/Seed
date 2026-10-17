# PLAN-N1-102 · 12-observation d=1 range probe · 2026-10-17

**Status: owner-approved and frozen on 2026-10-17; implementation and execution authorized.** This plan authorizes one new measurement face after v7 (four observations) and v8 (eight observations) each produced the same 4/9 usable endpoints. It fixes the new face, source boundary, one-shot execution order and unchanged gates.

## Question and rationale

Does increasing the in-distribution observation context from eight to twelve items expose enough errors in the low-range position endpoints to satisfy the existing dynamic-range gate? The evidence motivates this as a testable hypothesis; it does not predict a pass. The usable endpoint identities on v7 and v8 were the same four, and both confirmation subsets remain unqueried.

## Frozen v9 face and source boundary

- Build one new 40-item face, four kinds ×10, 20 sacrifice and 20 confirm (five per kind in each subset), fixed builder seed 109, twelve observations per item, and every distractor entity exactly Hamming distance 1 from the query subject.
- Preserve task semantics: lookup has one target row plus eleven distractors; revision preserves stale→current order plus ten distractors; transition keeps its two ordered facts plus ten distractor facts; unknown has no target observation plus twelve distractors. Balance target-row positions within each kind and subset; keep material values unique within each item.
- Query entities and material-answer values must be disjoint from v3 train, v5, v6, and only the already queried sacrifice items from v7 and v8. Stream both prior manifests; decode sacrifice episodes only and inspect only ID/split/subset for confirm episodes before skipping their remaining fields. Do not read or use v7/v8 confirm question, entity, observation or answer fields.
- Use only the same nine frozen PLAN-N1-100 final endpoints. Do not train, update parameters, change architecture, or access v3 dev/holdout inputs.

## Frozen execution and gates

Implement `scripts/training/build_taiji_n1_hard_neighbor_v9.py`, `scripts/training/eval_taiji_n1_hard_neighbor_v9.py`, and focused contract tests. The builder must produce an immutable manifest and construction report. The evaluator must have separate readiness, range and confirmation paths, with write-once attempt/report targets. Freeze and commit implementation and face artifacts before model queries.

First run zero-model-load readiness. Then query only the 20 sacrifice items across the nine new endpoints exactly once (180 rows; CPU≤180 seconds). Preserve the original range gate exactly: at least 6/9 endpoints must each have at least 6 errors among the 15 addressable lookup/revision/transition items; unknown errors remain disclosure-only. If range fails, do not query confirm, redraw the face or try again; close this plan at `range_not_met`.

Only if range passes, query the 20 confirm items once across the nine new endpoints and the same nine old descriptive controls (360 rows; CPU≤600 seconds). Preserve the four PLAN-N1-94 confirmation gates: each position seed ≥10/20; each position seed/kind ≥2/5; aggregate position advantage ≥12/60 over terminal; and ≥12/60 over cut. Old endpoints remain descriptive controls only. No promotion follows automatically.

Any interruption, source/checkpoint change, guard failure, or CPU overrun is retained and not retried, redrawn or overwritten. Confirmation eligibility is decided only from the completed sacrifice range report.

## Approval record

On 2026-10-17, the owner explicitly approved PLAN-N1-102 by answering “批准 PLAN-N1-102（推荐）”. This authorizes the new twelve-observation d=1 measurement face and one-shot execution. It does not authorize gate changes, v7/v8 confirm access, training, or promotion.
