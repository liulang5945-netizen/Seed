# PLAN-N1-133 — v15 confirmation of the N1-132 continuation package

**Status: revised draft for test-scope discussion; no face or query is authorized.** PLAN-N1-132 completed train-only calibration and all three position seeds passed the unchanged N1-120 training gate. No unseen evaluation was run. This proposal tests whether the resulting N1-132 continuation package generalizes; it does not isolate the causal contribution of the binding loss because the candidate also received 9,216 additional updates and no matched no-binding continuation arm was trained.

## Consolidated evidence from N1-126 through N1-132

- N1-126–128 reused the same consumed v13/v14 items and frozen N1-120 position/cut endpoints. Gold-target source pruning reached 45/45 position and 43/45 cut per face, but it also removed distractors and shortened the tape; this is an oracle upper bound. Full-context decoder/address masking improved native exactness from 52/90 to 82/90 position and 45/90 to 78/90 cut. Adding target-only question binding reached 90/90 position and 86/90 cut, matching source pruning, but the masks used gold target rows and do not establish autonomous selection. See [126](PLAN-N1-126_execution_closeout_20261019.md), [127](PLAN-N1-127_execution_closeout_20261019.md), and [128](PLAN-N1-128_execution_closeout_20261019.md).
- N1-129/130 found that native alignment already assigns relevant target bytes and relation rows measurable, modestly enriched mass, but aggregate and per-row mass did not distinguish correct from incorrect outputs. The remaining transition errors still carried both target relations, so the consumed-face evidence does not support a simple missing-row explanation. See [129](PLAN-N1-129_execution_closeout_20261019.md) and [130](PLAN-N1-130_execution_closeout_20261019.md).
- N1-131 classified all four residual N1-128 cut errors as seed13 inverse-turn outputs; the position arm was 30/30 under the same oracle binding intervention. This localizes those oracle-masked errors to turn composition, not to a demonstrated native retrieval cause. See [131](PLAN-N1-131_execution_closeout_20261019.md).
- N1-132 added `0.25 * L_bind` and 9,216 continued updates per position seed. All three seeds passed the unchanged train-only gate, but no dev, v13, v14, or new confirmation query occurred. The result validates training fit only; it provides no generalization or binding-loss-specific causal evidence. See the [132 closeout](PLAN-N1-132_execution_closeout_20261019.md).

Taken together, the prior work justifies testing the N1-132 package on a new family while keeping the interpretation bounded: success would support package-level generalization, not prove that `L_bind` alone caused a gain or that native transition composition is solved.

## Proposed face and frozen cohort

- Build one fresh seed-115 v15 face with 40 holdout items: ten each of lookup, revision, transition, and unknown; sixteen observations per item; entity-neighbor distance `d=1`; twenty sacrifice and twenty confirm items, balanced five per kind in each subset.
- Exclude all entities and material/answer values from v3 training, consumed v5/v6/v7-v14 faces, N1-95 scratch faces, and their derived source-exclusion manifests. Bind the full exclusion union and regenerate the face deterministically before any model query.
- Freeze twelve endpoints before face generation: N1-132 position candidates for seeds 11–13; their same-seed N1-120 position parents; and N1-120 terminal/cut controls for seeds 11–13. Verify all twelve full bundles, checkpoint hashes, and the N1-132 provenance reconciliation.

## Pre-authorization checkpoint integrity

A read-only audit verified the current final-checkpoint SHA-256 values for all twelve planned endpoints against their existing restore receipts: nine N1-120 endpoints and three N1-132 position candidates. The N1-120 receipt revalidation is complete with all 18 zero/final restore checks equal and the three position seeds meeting the position training gate; terminal/cut are comparison controls and are not required to meet that gate. The N1-120 training result itself remains `stopped` because its generic receipt validator requested finite Adam-proposal evidence; the separate bound revalidation closes that receipt issue without changing checkpoint bytes. The N1-132 reconciliation is complete and its three final bundles match their recorded hashes. See [pre-authorization endpoint audit](../../reports/taiji_n1_plan133_pre_authorization_endpoint_integrity_20261019.json).

This preflight only hashed checkpoint files and cross-checked existing reports. It loaded no checkpoints or models, made no model queries or updates, and generated no v15 face. The twelve-bundle restore/hash readiness will be rechecked within the approved execution before any face generation.

## One-shot evaluation

- First query the twenty sacrifice items on the nine main endpoints (three N1-132 position candidates plus three N1-120 terminal and three cut controls): 180 queries. Continue only if at least six of these nine endpoints have at least six errors among the fifteen decodable lookup/revision/transition items; the five unknown items are excluded from this range denominator. The three same-seed N1-120 position parents are paired descriptive anchors and do not count toward this range gate.
- If the range gate passes, query the twenty confirm items exactly once on all twelve endpoints: 240 queries. Maximum total: 420 queries. Do not redraw, retry, train, or query dev.
- Retain the N1-121 numeric confirmation thresholds for N1-132 candidates: each seed at least 10/20 exact, each kind at least 2/5, and aggregate position at least 12/60 above both terminal and cut. The endpoint cohort differs from N1-121, so this is a reused numeric gate rather than an exact cohort replication. Report same-seed N1-120 position-parent paired correctness gains/losses by seed and kind descriptively, without adding a new pass criterion.
- Cumulative CPU ceiling: 1,200 seconds, including readiness, construction, recovery, sacrifice/confirm queries, and final source hashes. `can_promote=false` regardless of this local face outcome; broader project gates remain outstanding.

## Interpretation and unresolved test question

- If the range gate fails, stop after 180 sacrifice queries; do not query confirm or redraw. If confirmation runs and any frozen gate fails, close the face as failed without retry. If all gates pass, conclude only that the N1-132 continuation package passed this v15 generalization gate; keep `can_promote=false`.
- The same-seed N1-120 position parent is a descriptive before/after anchor. Because it did not receive a matched 9,216-update continuation without `L_bind`, its contrast cannot isolate the auxiliary loss from additional training. A binding-loss causal claim would require a separate matched-control training plan and is outside this draft.
- Recommended scope for discussion: keep PLAN-N1-133 as a package-level generalization test with the frozen numeric gates above. Decide this estimand before any face builder, manifest, or evaluator is run; generating and consuming the v15 face remains a separate owner authorization.

## Proposed outputs after approval

- Face builder and readiness artifacts in `scripts/training/` and `plans/manifests/`.
- Sacrifice/confirmation evaluator and receipts in `scripts/training/` and `reports/`.
- Closeout: `plans/reference/PLAN-N1-133_execution_closeout_20261019.md`.
