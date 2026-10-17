# PLAN-N1-104 · v9 confirmation input-feature audit · 2026-10-17

**Status: owner-authorized under delegated normal-plan authority and frozen on 2026-10-17; read-only analysis authorized.** This plan stratifies the already-consumed v9 confirmation outcomes by input features to guide a later mechanism decision. It creates no new measurement face and changes no frozen gate.

## Question

Do seed13 position failures, especially on lookup/revision, cluster by target-row position or material-answer length in the consumed v9 confirmation face, or are they spread across those input features? This is descriptive stratification only.

## Fixed read-only analysis

- Read the 20 `subset=confirm` episode objects from `plans/manifests/n1_hard_neighbor_v9_20261017.json` by streaming. Inspect only ID/split/subset for the 20 sacrifice objects and skip their remaining fields without decoding. Do not read v7/v8 confirmation fields or any v3/v5/v6 source.
- Join those 20 confirm item IDs to the 180 `family=hard_neighbor_v9` outcome rows in `reports/taiji_n1_hard_neighbor_v9_confirmation_20261017.json`; do not decode or use prior-family descriptive rows.
- Extract kind, target-row position (where a target row exists), material-answer length (lookup/revision only), and per-seed/per-arm `free_exact`. Publish grouped denominators and exact counts; do not recompute the PLAN-N1-94 gates.
- CPU≤30 seconds; zero model/checkpoint loads, queries, training, optimizer updates, or face writes. Produce one write-once report bound to both source SHAs and this plan SHA. Mark results exploratory and descriptive.

## Interpretation boundary

The audit may show associations between input features and the already-observed errors, but cannot identify a causal mechanism, alter the failed confirmation result, establish C3/C4/retention/three-organ capability, or support promotion. Any new mechanism, training objective, threshold, or measurement face requires its own decision.

## Authorization record

On 2026-10-17, the owner approved PLAN-N1-103 and added “批准，这种正常的推进计划也给你权限”. Under this delegated authority for normal plan progression, PLAN-N1-104 is frozen as a read-only analysis of already-consumed v9 confirmation data. It authorizes no new face, training run, gate change, or promotion.
