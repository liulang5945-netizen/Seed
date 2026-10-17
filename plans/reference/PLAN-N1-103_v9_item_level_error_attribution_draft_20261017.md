# PLAN-N1-103 draft · v9 item-level error attribution · 2026-10-17

**Status: proposal only; not approved, not frozen, and not executable.** PLAN-N1-102's one-shot v9 confirmation is complete and its four-gate conjunction failed. This draft proposes a read-only analysis of already-consumed confirmation output to guide the next mechanism decision; it creates no new measurement face.

## Question

Are the position seed13 errors and the aggregate position-versus-cut deficit concentrated in particular items/classes, shared across arms, or spread across seeds? The current gate report identifies seed13 and two failed class cells, but does not provide the proposed item-level overlap summary.

## Proposed read-only analysis

- Read only the sealed `hard_neighbor_v9` rows from `reports/taiji_n1_hard_neighbor_v9_confirmation_20261017.json`; do not load checkpoints or revisit v9 source manifests. The analysis may use item IDs, kinds, expected/predicted strings and exact-match fields already present in those rows.
- For each seed and kind, count position-only correct, cut-only correct, both correct, both wrong, and exact-output agreement; separately show the same position-versus-terminal breakdown. Report denominators explicitly and preserve the evaluator's existing gate result without recomputing or changing it.
- CPU≤30 seconds; zero model/checkpoint loads, queries, training, optimizer updates, or new holdout access. Produce one write-once report bound to the confirmation report SHA.
- Label every result exploratory/descriptive. Do not infer mechanism causality, claim capability gain, alter any gate, authorize promotion, or use this analysis to query or regenerate a face.

## Required owner decision

This is a new diagnostic scope over a consumed confirm report, beyond PLAN-N1-102's frozen adjudication. Do not implement it unless the owner explicitly approves PLAN-N1-103 or chooses another direction.
