# PLAN-N1-103 · v9 item-level error attribution · 2026-10-17

**Status: owner-approved and frozen on 2026-10-17; read-only analysis authorized.** This plan examines the already-consumed v9 confirmation rows to determine how the position seed13 errors and position-versus-cut deficit distribute across items and kinds. It creates no new measurement face.

## Fixed analysis

- Read only the 180 `family=hard_neighbor_v9` rows from `reports/taiji_n1_hard_neighbor_v9_confirmation_20261017.json`. Do not load checkpoints, revisit manifests, read v7/v8 confirmation payloads, or use the 180 prior-family descriptive rows to adjudicate the new family.
- For each seed and kind, classify every item as position-only correct, cut-only correct, both correct, or both wrong; separately classify position versus terminal. Also count exact output-byte agreement using the existing `output_hex` values. Preserve denominators and report seed/kind cells without recomputing or changing the four frozen gates.
- Freeze the confirmation report SHA before analysis. Produce one write-once report at `reports/taiji_n1_hard_neighbor_v9_item_error_attribution_20261017.json`, bound to that SHA and this plan SHA.
- CPU≤30 seconds; zero model/checkpoint loads, model queries, training, optimizer updates, new holdout access, or writes to any face. Label all results exploratory and descriptive.

## Interpretation boundary

This analysis can localize observed error overlap; it cannot establish a causal mechanism, undo the failed four-gate conjunction, or support a capability, C3, C4, retention, three-organ, or promotion claim. Any new training objective, architecture, gate or measurement face requires a separate plan decision.

## Approval record

On 2026-10-17, the owner explicitly approved PLAN-N1-103 by answering “批准，这种正常的推进计划也给你权限”. This authorizes the read-only analysis specified above and no new face, training run or gate change.
