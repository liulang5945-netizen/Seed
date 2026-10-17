# PLAN-N1-107 · v10 output geometry and position stability · 2026-10-17

**Status: frozen for read-only analysis.** This plan extends only the offline inspection of the already-consumed PLAN-N1-105 diagnostic outputs. It creates no new face and does not revise any gate.

## Question

For each frozen endpoint, are wrong free outputs close to the expected answer or to a material/relation value already present in that item's observations, and do outputs stay constant as the target row moves through matched variants of the same template?

## Fixed analysis

- Read only the frozen 248-item face `plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json` and its complete 2,232-row result `reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json`. Require the exact face/result/PLAN-N1-105 hashes. Bind these inputs, this plan, analyzer, and contract-test hashes in one write-once report at `reports/taiji_n1_hard_neighbor_v10_output_geometry_20261017.json`.
- Validate the complete item × seed × arm design and all matched lookup/revision/transition template families before calculating any summaries. Within each family, the answer, question, entities, and observation multiset must remain constant; only the frozen row-order factor may vary. Unknown items are excluded from position-stability groups because they have no target row.
- For each of all 2,232 rows, record output character count, expected-answer character count, signed length difference, and Levenshtein distance from the exact answer. For lookup/revision, also record the nearest explicit `uses` material by role (current target, stale target, distractor); for transition, record the nearest explicit `faces`/`turns` value by role (target entity, other entity). Preserve ties and full denominators. These string distances describe the emitted bytes only; they do not prove an internal retrieval path.
- For each seed/arm/kind/template/answer-length family with a row-position factor, report the count of position variants, number of unique output strings, modal-output support, whether all outputs are identical, and exact-answer count by position. Also summarize output-length and edit-distance distributions by seed, arm, kind, and answer length.
- CPU≤30 seconds. The analyzer reads JSON only. It must not import the model runtime/training loader, load checkpoints, query any model, update parameters, train, access dev/holdout, write to either input, recompute a gate, or change capability status. It must refuse to overwrite the output report.

## Interpretation boundary

Near-string distance and within-template output stability can localize the shape of the observed output errors. They cannot establish causality, generalization, repair the v9 four-gate failure, or support C3/C4, retention, three-organ, or capability-promotion claims. Any training change, threshold change, or new measurement face remains a separate plan decision.

## Authorization record

The owner has granted standing authority for the next suitable normal research-planning step without pausing for routine CPU-budget decisions. This analysis remains read-only on the already-consumed diagnostic result and does not alter its frozen criteria.
