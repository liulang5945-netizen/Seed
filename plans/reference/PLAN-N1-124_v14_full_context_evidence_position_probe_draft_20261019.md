# PLAN-N1-124 — v14 full-context evidence-position probe

**Status: draft awaiting approval for a new measurement face.** PLAN-N1-123 completed its one-shot seed-113 sixteen-observation run. The range gate passed at 8/9 endpoints, but the candidate confirmation cohort reached only 41/60 position versus 36/60 cut, below the frozen +12/60 advantage; seed 12 lookup was 1/5 and seed 13 transition was 1/5. A read-only audit also found that confirm evidence began only at observation indices 0–4 (indices 0–5 for two-row tasks). This follow-up tests whether the position/cut gap persists when task evidence itself is distributed throughout the long context.

## Frozen question and sources

- Use the same nine PLAN-N1-120 candidate checkpoints for the sacrifice range stage and the same nine PLAN-N1-100 baselines plus nine N1-120 candidates for confirmation. Reuse the 18/18 full-bundle restore evidence only while all checkpoint and source hashes remain unchanged.
- Construct one fresh seed-114 face: 40 items, 10 each lookup, revision, transition, and unknown; 16 observations/item; 10-character entity names; all distractors at Hamming distance `d=1`; 20 sacrifice and 20 confirm, with five of each class in each subset.
- Preserve the PLAN-N1-121/122/123 grammar and answer semantics. Preserve the lookup/revision answer-length split: sacrifice has three six-character and two seven-character answers per kind; confirm has two six-character and three seven-character answers per kind.
- Before readiness, statically audit the entire N1 hard-neighbor face ledger for any prior use of seed 114. If the seed is already recorded, stop without generating a face; do not choose a replacement seed.
- Stratify target evidence positions within each five-item class/subset. For lookup's one-row evidence, use start indices `[0, 4, 8, 12, 15]`. For revision and transition's two-row evidence, use start indices `[0, 4, 7, 11, 14]`. Unknown items continue to have no target evidence and retain sixteen non-target observations. These positions cover early, middle, and late context while keeping each class/subset count at five.
- Exclude the complete N1-123 face and all prior source sets, including the full N1-121/N1-122 faces, v3 training values, v5/v6 official faces, all queried v7/v8/v9/v10 values, all 288 N1-110/N1-120 mapped values, and the 16-face N1-95 scratch inventory. The expected union after adding all N1-123 face values is 6,198 entity names and 4,968 answer/observation word values; the source builder must verify the exact set and bind the N1-123 plan, readiness, face, range, confirmation, attribution, and closeout hashes.

## One-shot protocol and unchanged gates

1. Run source/scorer/checkpoint readiness with zero model loads, queries, and updates. Stop if the 36 endpoint hashes, source exclusion counts, or 18 restore receipts fail validation.
2. Generate and freeze seed 114 exactly once. Verify face size, class/subset balance, sixteen observation rows, target-position schedule, answer-length split, `d=1`, and full source disjointness. Do not redraw if a contract fails.
3. Query the 20 sacrifice items once on the nine N1-120 candidates (180 queries). Require at least 6/9 endpoints with at least six errors among the 15 decodable lookup/revision/transition items. If the range gate fails, stop without confirm.
4. Only if range passes, query the 20 confirm items once on the nine PLAN-N1-100 baselines and nine N1-120 candidates (360 additional queries). Preserve the unchanged per-cohort gates: position at least 10/20 per seed, at least 2/5 per seed and kind, and position at least 12/60 above both terminal and cut.

Maximum total: 540 model queries and 600 CPU seconds including readiness, face construction, restore verification, scoring, and final source-hash verification. No training, optimizer updates, dev queries, retries, redraws, threshold changes, or capability promotion. A failed gate closes the plan with `can_promote=false`; do not automatically add another face.

This is a new confirm face and therefore needs explicit approval before implementation or query. It is a long-context positioning diagnostic, not an in-distribution test or a causal estimate against v13's different sampled items.

## Proposed outputs

- Face: `plans/manifests/n1_hard_neighbor_v14_confirmation_seed114_20261019.json`.
- Source exclusions: `plans/manifests/n1_v14_after_n123_source_exclusions_20261019.json`.
- Readiness and face checks: `reports/taiji_n1_v14_after_n123_confirmation_readiness_20261019.json` and `reports/taiji_n1_v14_after_n123_face_checks_20261019.json`.
- Range and attempt: `reports/taiji_n1_v14_after_n123_range_20261019.json` and `reports/taiji_n1_v14_after_n123_range_20261019_attempt.json`.
- Conditional confirmation and attempt: `reports/taiji_n1_v14_after_n123_confirmation_20261019.json` and `reports/taiji_n1_v14_after_n123_confirmation_20261019_attempt.json`.
