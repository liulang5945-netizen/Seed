# PLAN-N1-102 draft · 12-observation d=1 range probe · 2026-10-17

**Status: proposal only; not approved, not frozen, and not executable.** This draft responds to two consecutive range failures on the same nine-endpoint family: v7 (four observations) and owner-approved v8 (eight observations) each had 4/9 usable endpoints, with the same four endpoint identities. It proposes one new measurement face; it does not authorize generation or querying.

## Proposed question

Does increasing the in-distribution observation context from eight to twelve items expose enough errors in the low-range position endpoints to satisfy the already-frozen dynamic-range gate? The evidence motivates this as a testable hypothesis; it does not predict a pass. The v8 confirm subset remains unqueried and is excluded from all source, analysis and deduplication inputs.

## Proposed frozen face and source boundary

- One new 40-item face, four kinds ×10, 20 sacrifice and 20 confirm (five per kind per subset), fixed builder seed 109, twelve observations per item, and every distractor entity exactly Hamming distance 1 from the query subject.
- Preserve task semantics: lookup has one target row plus eleven distractors; revision preserves stale→current order plus ten distractors; transition keeps its two ordered facts plus ten distractor facts; unknown has no target observation plus twelve distractors. Balance target-row positions within each kind and subset; keep material values unique within each item.
- Query entities and material-answer values must be disjoint from v3 train, v5, v6, and only the already queried sacrifice items from v7 and v8. Stream both prior manifests; decode sacrifice episodes only and inspect only ID/split/subset for confirm episodes before skipping their remaining fields. Do not read or use v7/v8 confirm question, entity, observation or answer fields.
- Use only the same nine frozen PLAN-N1-100 final endpoints. No training, parameter update, dev/holdout access, or architecture change.

## One-shot execution and unchanged gates

First implement and contract-test a one-shot builder and evaluator. Freeze and commit the code, manifest and construction evidence before model queries. Run zero-model-load readiness. Then query the 20 sacrifice items across the nine new endpoints exactly once (180 rows; CPU≤180 seconds).

Preserve the original range gate exactly: at least 6/9 endpoints must each have at least 6 errors among the 15 addressable lookup/revision/transition items. Unknown errors remain disclosure-only. If range fails, do not query confirm, redraw the face or try again.

Only if range passes, query the 20 confirm items once across the nine new endpoints and the same nine old descriptive controls (360 rows; CPU≤600 seconds). Preserve all four PLAN-N1-94 confirmation gates: each position seed ≥10/20; each position seed/kind ≥2/5; aggregate position advantage ≥12/60 over terminal; and ≥12/60 over cut. Old endpoints remain descriptive controls only. No promotion follows automatically.

## Required owner decision

This proposal creates a new measurement face and extends the observation count beyond the approved v8 scope. Do not implement, generate, or query it unless the owner explicitly approves PLAN-N1-102 or chooses another direction.
