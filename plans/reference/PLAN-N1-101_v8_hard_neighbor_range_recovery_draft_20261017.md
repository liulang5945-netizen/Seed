# PLAN-N1-101 draft · 8-observation d=1 v8 range recovery · 2026-10-17

**Status: proposal only; not approved, not frozen, and not executable.** This draft proposes a new measurement face after PLAN-N1-100's v7 range gate failed. It does not authorize face generation, endpoint queries, gate changes, or confirmation access.

## Evidence and reason for the proposal

The sealed v7 sacrifice report is `reports/taiji_n1_hard_neighbor_v7_range_check_20261017.json`, SHA-256 `c0bf9e588b3785240c61a0a90907f4849de64b1027546a749d00a8e2768c30eb`. All 180 queries completed in 38.171875 CPU seconds, but only 4/9 endpoints reached the frozen line of at least 6 errors among 15 addressable items. Three terminal controls and seed13/position supplied those four; position seeds11/12 and all cut endpoints were below the line. No confirm item was queried, no parameters changed, and `can_promote=false`.

The per-class mean errors were lookup 2.7778/5, revision 2.2222/5, transition 1.5556/5 and unknown 1.4444/5. The gate itself counts only addressable lookup/revision/transition errors and remains 6/9 endpoints with at least 6/15. The v7 confirm half must remain unqueried.

PLAN-N1-95's earlier d=1 range sweep reported 9/9 usable endpoints with eight observations, but it used the prior endpoint family. That is motivation for a new test, not a prediction that the hard-neighbor-trained models will pass.

## Proposed frozen v8 face

- New face: 40 items, four kinds ×10, sacrifice/confirm 20 each and five per kind in each subset. Fixed generator seed 108. Use eight observations per item; each distractor entity is exactly Hamming distance 1 from the query subject.
- Preserve task semantics and answer positions: lookup has one target row plus seven distractors; revision preserves stale→current order plus six distractors; transition keeps its two ordered facts plus six distractor facts; unknown has no target observation plus eight distractors. Balance target-row positions and preserve unique material values within each item.
- Generated query entities and material-answer values must be disjoint from v3 train, v5, v6, and the already queried v7 sacrifice items. The v7 confirm item payloads remain unread and unqueried; no exclusion values may be extracted from that subset. Fixed direction and `unknown` labels are task tokens, not material novelty words.
- No training, architecture change, or optimizer update. The nine PLAN-N1-100 final endpoints stay frozen; all v3 dev/holdout inputs remain out of scope.

## Proposed execution and unchanged gates

After approval, implement a one-shot builder, contract tests, range runner and confirm runner, then commit the frozen code and v8 manifest. First run readiness with zero model/checkpoint loads. Query only the 20 sacrifice items across the nine new endpoints (180 rows, CPU≤180 seconds). Keep `usable_endpoints≥6/9`, with at least 6/15 addressable errors, unchanged. If it fails, do not query confirm and do not regenerate v8.

If range passes, query the 20 confirm items once across the nine new endpoints and the same nine old descriptive controls (360 rows, CPU≤600 seconds). Preserve the four PLAN-N1-94 confirmation gates exactly: each position seed ≥10/20, each position seed/kind ≥2/5, aggregate position advantage ≥12/60 over terminal, and ≥12/60 over cut. Old endpoints remain descriptive controls only. No promotion follows automatically.

## Required owner decision

This draft changes the measurement face from four to eight observations and adds the already-consumed v7 sacrifice item names/materials to the exclusion sources. Those are scope and data-boundary changes. Do not implement or execute until the owner approves this draft or selects another direction.
