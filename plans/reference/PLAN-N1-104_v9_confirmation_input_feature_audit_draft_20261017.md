# PLAN-N1-104 draft · v9 confirmation input-feature audit · 2026-10-17

**Status: proposal only; not approved, not frozen, and not executable.** PLAN-N1-103 found that the position-versus-cut deficit is concentrated in seed13, especially on revision items, but its report-only analysis did not examine input features.

## Proposed question

Do seed13's position-only failures cluster by a measured input feature already present in the consumed v9 confirmation face, such as target-row position or answer length, or do they remain spread across those feature groups? This is an exploratory stratification, not a causal test.

## Proposed read-only analysis

- Read only the 20 v9 `confirm` episode objects needed to join IDs and input features to the already-consumed `hard_neighbor_v9` result rows. Read no `v9` sacrifice item fields for this analysis; do not read or use v7/v8 confirm fields or any v3/v5/v6 source.
- Compare position and cut item outcomes by kind, target-row position, and material-answer length where applicable. Keep the denominator and missing-feature counts explicit; report unknown and transition categories separately.
- CPU≤30 seconds; zero model/checkpoint loads, model queries, training, optimizer updates, new face generation, or gate changes. Bind the output to the v9 manifest and confirmation report SHAs. Label findings exploratory and descriptive.
- Do not infer causality, rejudge PLAN-N1-94, promote the model, or use this analysis to query or regenerate a face.

## Required owner decision

This proposal expands PLAN-N1-103's report-only input boundary to the already-consumed v9 confirmation manifest. Do not implement it unless the owner explicitly approves PLAN-N1-104 or chooses another direction.
