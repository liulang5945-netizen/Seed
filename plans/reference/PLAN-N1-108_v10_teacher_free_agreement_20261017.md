# PLAN-N1-108 · v10 teacher-forced/free-output agreement · 2026-10-17

**Status: frozen for read-only aggregation.** This plan inspects existing per-item fields from the consumed PLAN-N1-105 result; it adds no model evaluation or measurement face.

## Question

Do the stored `teacher_exact` and `free_exact` outcomes agree on every v10 item and endpoint, or are there rows where gold-prefix scoring succeeds while free generation fails (or vice versa)? How is any disagreement distributed by seed, arm, task kind, answer length, and target position?

## Fixed analysis

- Read only the frozen 248-item face `plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json` and complete result `reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json`; require exact face/result/PLAN-N1-105 hashes. Bind both sources and this plan/analyzer/test hashes in one write-once report at `reports/taiji_n1_hard_neighbor_v10_teacher_free_agreement_20261017.json`.
- Validate one unique result row for every item × seed (11/12/13) × arm (`position`/`terminal`/`cut`) and re-derive `free_exact` from the decoded raw output and face answer. Keep `teacher_exact` as the already-recorded gold-prefix result; do not rerun teacher scoring.
- Report the four paired cells (both exact, free-only, teacher-only, both wrong) overall and by seed, arm, task kind, answer length, and target position where defined, preserving complete denominators. Do not compute or change a gate.
- CPU≤15 seconds. The analyzer reads JSON only; it must not import model runtimes/loaders, load a checkpoint, query a model, update parameters, train, access dev/holdout, or write to the face/result. Refuse to overwrite the output report.

## Interpretation boundary

Agreement or disagreement distinguishes observed gold-prefix exactness from free-output exactness only. It cannot establish causality, explain internal dynamics, establish generalization, repair the v9 four-gate failure, or support promotion. Any training or architecture change and any new confirmation face remain a separate plan decision.

## Provenance note

A preliminary in-memory tally of the stored rows found no `teacher_exact`/`free_exact` disagreements. That tally made no model query, loaded no checkpoint, and wrote no artifact. This frozen analyzer independently validates the exact inputs, full row identity, all strata, and the result before publication.
