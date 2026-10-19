# PLAN-N1-133 — v15 confirmation for supervised question binding

**Status: draft awaiting owner authorization for a new holdout measurement surface.** PLAN-N1-132 completed its train-only calibration and all three position seeds passed the unchanged N1-120 training gate. No unseen evaluation was run. A fresh paired confirmation is needed to test whether supervised question binding improves generalization.

## Proposed face and frozen cohort

- Build one fresh seed-115 v15 face with 40 holdout items: ten each of lookup, revision, transition, and unknown; sixteen observations per item; entity-neighbor distance `d=1`; twenty sacrifice and twenty confirm items, balanced five per kind in each subset.
- Exclude all entities and material/answer values from v3 training, consumed v5/v6/v7-v14 faces, N1-95 scratch faces, and their derived source-exclusion manifests. Bind the full exclusion union and regenerate the face deterministically before any model query.
- Freeze twelve endpoints before face generation: N1-132 position candidates for seeds 11–13; their same-seed N1-120 position parents; and N1-120 terminal/cut controls for seeds 11–13. Verify all twelve full bundles, checkpoint hashes, and the N1-132 provenance reconciliation.

## One-shot evaluation

- First query the twenty sacrifice items on the nine main endpoints (three N1-132 position candidates plus three N1-120 terminal and three cut controls): 180 queries. Continue only if at least six of these nine endpoints have at least six errors among the fifteen decodable sacrifice items. The three same-seed N1-120 position parents are paired descriptive anchors and do not count toward this unchanged range gate.
- If the range gate passes, query the twenty confirm items exactly once on all twelve endpoints: 240 queries. Maximum total: 420 queries. Do not redraw, retry, train, or query dev.
- Apply the N1-121 position confirmation gates unchanged to N1-132 candidates: each seed at least 10/20 exact, each kind at least 2/5, and aggregate position at least 12/60 above both terminal and cut. Report the same-seed N1-120 position-parent difference descriptively, without adding it as a pass criterion.
- Cumulative CPU ceiling: 1,200 seconds, including readiness, construction, recovery, sacrifice/confirm queries, and final source hashes. `can_promote=false` regardless of this local face outcome; broader project gates remain outstanding.

This plan creates and consumes a new holdout confirmation face, which is outside the current authorized diagnostic surface. No face builder, manifest, or evaluator will be run until the plan is approved.

## Proposed outputs after approval

- Face builder and readiness artifacts in `scripts/training/` and `plans/manifests/`.
- Sacrifice/confirmation evaluator and receipts in `scripts/training/` and `reports/`.
- Closeout: `plans/reference/PLAN-N1-133_execution_closeout_20261019.md`.
