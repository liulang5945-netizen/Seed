# PLAN-N1-105 draft · counterbalanced input-feature diagnostic · 2026-10-17

**Status: proposal only; not approved, not frozen, and not executable.** This proposal changes the measurement face and requires an owner decision before implementation or model queries. It makes no training, gate, capability, or promotion claim.

## Question

Does the position arm's v9 lookup/revision deficit persist when target-row position is crossed with material-answer length within matched item templates, rather than being tied to the construction order and individual item?

## Proposed one-shot diagnostic

- Reuse the same nine frozen PLAN-N1-100 final checkpoints used by v9 and the three arms (`position`, `cut`, `terminal`); do not train or update checkpoints.
- Generate one deterministic diagnostic face with four independent base templates per kind. For lookup, cross all 12 feasible first-target-row positions with answer lengths 6 and 7 on each template (96 items). For revision, cross all 11 feasible first-target-row positions with both answer lengths on each template (88 items). For transition, use each template at all 11 feasible first-row positions (44 items). Add 20 distinct unknown items with no target row, for 248 items total. Keep twelve observations per item and Hamming-distance-1 distractor entities.
- Within each base template, keep entity and question fixed across position/length variants. Reuse the same answer value for a given template/length across its position variants; keep values distinct across templates and lengths. Use distinct entities and distractors between templates, preserve target meaning and revision/transition row order, and keep each unknown item without a target observation.
- Exclude names/material values from v3 train, v5/v6, v7/v8 queried sacrifice items, and the entire consumed v9 face; do not read v7/v8 confirmation payloads. Construction checks must verify exact crossed-cell counts (four items per lookup/revision cell), row bounds, Hamming distance, source disjointness, and the intentional within-template pairing before any checkpoint is loaded.
- **A new builder and evaluator are required.** The v9 confirmation runner hardcodes twenty items and also queries prior-family controls, so do not call its `run_confirmation` path. Store the new face as `split=diagnostic` under schema `n1-hard-neighbor-v10-factorial-diagnostic-v1`; implement a dedicated stateless scorer rather than relabeling items as holdout. The scorer must isolate each query, remove per-query observation state, and prove the complete model/optimizer/RNG/history bundle is unchanged before and after each endpoint. Proposed artifacts: `scripts/training/build_taiji_n1_input_factorial_diagnostic.py`, `scripts/training/eval_taiji_n1_input_factorial_diagnostic.py`, `tests/taiji_native/test_n1_run105_input_factorial_diagnostic_contract.py`, `plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json`, `reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_attempt_20261017.json`, and `reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json`.
- Freeze builder, evaluator, tests, face, and endpoint identities before querying. After the construction and full-bundle recovery preflight passes, make exactly one diagnostic query per item and endpoint: 2,232 total queries. Independent CPU ceiling: 20 minutes including construction, preflight, recovery, and evaluation. Any source, identity, or recovery failure stops before queries; never retry a consumed face.
- Report per-seed/per-arm exact counts and within-template position/length contrasts with all denominators. No threshold is added or altered. Do not recompute PLAN-N1-94 gates, query dev or any existing confirmation face, or treat the diagnostic as a new unseen-capability confirmation. Preserve `can_promote=false`.

## Interpretation boundary

This diagnostic can separate the construction aliasing seen in v9 from within-template position and answer-length associations. It cannot establish a causal mechanism beyond the paired synthetic templates, change a frozen gate, demonstrate general language ability, or authorize promotion. Any subsequent training change or capability-confirmation face requires a separate decision.

## Decision required

Approve or decline this new diagnostic face before implementation. If approved, freeze the exact generator, exclusion registry, query order, checkpoint identities, recovery procedure, report schema, and one-shot execution guard before creating the face.
