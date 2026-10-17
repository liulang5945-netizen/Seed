# PLAN-N1-101 · 8-observation d=1 v8 range recovery · 2026-10-17

**Status: owner-approved and frozen on 2026-10-17; implementation and execution authorized.** The owner approved the proposed measurement-face change. This plan fixes the v8 face, exclusion boundary, one-shot execution order and existing gates; implementation may proceed within those limits.

## Evidence and reason

The sealed v7 sacrifice report is `reports/taiji_n1_hard_neighbor_v7_range_check_20261017.json`, SHA-256 `c0bf9e588b3785240c61a0a90907f4849de64b1027546a749d00a8e2768c30eb`. All 180 queries completed in 38.171875 CPU seconds, but only 4/9 endpoints reached the frozen line of at least 6 errors among 15 addressable items. Three terminal controls and seed13/position supplied those four; position seeds11/12 and all cut endpoints were below the line. No v7 confirm item was queried, no parameters changed, and `can_promote=false`.

PLAN-N1-95's earlier d=1 range sweep reported 9/9 usable endpoints with eight observations, but it used the prior endpoint family. That result motivates this fresh test and does not predict that the hard-neighbor-trained models will pass.

## Frozen v8 face and data boundary

- Generate one new face with 40 items, four kinds ×10, 20 sacrifice and 20 confirm (five per kind in each subset), fixed generator seed 108, eight observations per item, and each distractor entity exactly Hamming distance 1 from its query subject.
- Preserve task semantics and balanced target positions: lookup has one target row plus seven distractors; revision preserves stale→current order plus six distractors; transition keeps its two ordered facts plus six distractor facts; unknown has no target observation plus eight distractors. Preserve unique material values within each item.
- Generated query entities and material-answer values must be disjoint from v3 train, v5, v6, and the already queried v7 sacrifice items. Do not use v7 confirm entity/material fields for exclusion, construction, deduplication, or analysis, and never query v7 confirm. The v8 builder must stream the v7 manifest: decode sacrifice episode objects only; for confirm objects inspect only the ID/split/subset prefix required to skip each object, without parsing question, entity, observation or answer fields. The prior v7 structural check established the 20/20 partition but did not score or use confirm rows. Fixed direction words and `unknown` labels are task tokens, not material novelty words.
- Use only the nine frozen PLAN-N1-100 final endpoints. Do not train, update parameters, change architecture, or access v3 dev/holdout inputs.

## Frozen execution and gates

Implement `scripts/training/build_taiji_n1_hard_neighbor_v8.py`, `scripts/training/eval_taiji_n1_hard_neighbor_v8.py`, and focused contract tests. The builder must produce a new immutable manifest and construction report. The evaluator must have separate readiness, range, and confirmation paths, with write-once attempt/report targets. Freeze and commit implementation and face artifacts before model queries.

First run readiness with zero model/checkpoint loads and verify all face, endpoint, source and code bindings. Then query only the 20 sacrifice items across the nine new endpoints: 180 rows, CPU≤180 seconds. Preserve the original range gate exactly: at least 6/9 endpoints must each have ≥6 errors among the 15 addressable lookup/revision/transition items; unknown errors remain disclosure-only. If this gate fails, do not query confirm, do not regenerate v8, and stop this plan.

Only if range passes, query the 20 v8 confirm items once across the nine new endpoints and the same nine old descriptive controls (360 rows, CPU≤600 seconds). Apply the four PLAN-N1-94 confirmation gates only to the nine new endpoints: each position seed ≥10/20; each position seed/kind ≥2/5; aggregate position advantage ≥12/60 over terminal; and ≥12/60 over cut. Old endpoints are descriptive controls only. No promotion follows automatically.

Any attempt is one-shot. An interruption, guard failure, changed source/checkpoint, or CPU overrun is retained as a failed attempt and is not retried, redrawn, or overwritten. If confirmation runs, its result adjudicates this plan; `can_promote` remains false and `TRAIN_SHA` remains None absent separate evidence and an authorized promotion decision.

## Approval record

On 2026-10-17, the owner explicitly approved PLAN-N1-101 by answering “批准 PLAN-N1-101（推荐）”. This approves the 8-observation d=1 measurement face and the specified use of v7 sacrifice names/materials as exclusions. It does not approve changing gates, accessing v7 confirm fields, rerunning v7, training, or promotion.
