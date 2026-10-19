# PLAN-N1-123 — v13 long-context hard-neighbor range probe

**Status: approved for one-shot execution on 2026-10-19.** User approved the frozen proposal. PLAN-N1-122 stopped at 3/9 usable endpoints on its four-observation face. This plan tests one fresh sixteen-observation face on the same frozen model cohorts, with the same item classes, `d=1` name confusability, length balance, and gates. No N1-123 face or attempt existed at approval.

## Why sixteen observations

On the current N1-120 checkpoints, PLAN-N1-121's twelve-observation face reached 5/9 usable endpoints, while PLAN-N1-122's four-observation face reached 3/9. At four observations, position and cut each made 8/45 decodable errors; only the three terminal controls met the range floor. The current training distribution contains exactly two observations per item. A sixteen-observation face would test whether the same retrieval rule holds with eight times that context length and more near-name distractors.

This is an explicit long-context stress test, not an in-distribution claim. The two previous faces used different sampled items, so their result difference is not a causal observation-count effect. PLAN-N1-95's older endpoint sweep over 4–12 observations was non-monotonic and provides no guarantee for sixteen observations or for these checkpoints. The unchanged range gate will determine whether the new face has adequate dynamic range.

## Frozen sources and proposed face

- Candidate endpoints: the same nine frozen PLAN-N1-120 final checkpoints (position, terminal, cut; seeds 11–13).
- Comparison endpoints, only if range passes: the same nine frozen PLAN-N1-100 final checkpoints.
- New face: sampling seed `113`; 40 items (10 each lookup, revision, transition, unknown); sixteen observations per item; 10-character entity names; every distractor name at Hamming distance 1; answer lengths 6 and 7 balanced within lookup/revision. Preserve the N1-121 per-kind split: sacrifice has three six-character and two seven-character lookup/revision answers; confirm has two six-character and three seven-character answers. Use 20 sacrifice and 20 confirm items with five of each kind per subset.
- Keep N1-121/N1-122 task grammar and answer semantics. Only the face seed and observation count change; endpoints, class balance, name distance, answer-length split, and score gates remain fixed.
- Use ID prefix `v13-hard-` and schema `n1-hard-neighbor-v13-confirmation-v1`. Stop if this seed has already been used in the N1 hard-neighbor face family or any N1-123 output path exists; do not draw another seed.

## Input disjointness

Build zero-query source readiness and require no overlap with all entity names and observed material values from:

- the existing N1-121/N1-122 exclusion sets, including v3 train inputs, official v5/v6 faces, v7/v8 queried sacrifice inputs, v9/v10, all 288 N1-110/N1-120 mapped values, and the complete seed-111 N1-121 and seed-112 N1-122 faces;
- the 16 N1-95 scratch faces preserved in [the exclusion manifest](../manifests/n1_v6_preflight_consumed_face_exclusions_20261019.json).

After adding the full N1-122 face, the source union contains 6,048 entity names and 4,848 answer/observation word values. The source builder must keep target-answer vocabulary separate from observed-material overlap checks. Bind both prior faces and their reports, the exclusion manifest and its 16 face hashes, source reports, code identity, and all 36 endpoint checkpoint hashes. Do not decode dev payloads or old v7/v8 confirmation payloads.

## One-shot evaluation

1. Complete source, scorer, and checkpoint-hash readiness with zero model loads, queries, or updates. Reuse the 18/18 restore evidence only if all bound source and checkpoint hashes still match.
2. After approval, build and freeze seed 113 once. Verify 40 items, 10 per kind, sixteen observations per item, the exact answer-length split, `d=1` distractors, and source disjointness.
3. Query only the 20 sacrifice items on the nine N1-120 candidates: 180 model queries. Keep the unchanged range gate: at least 6/9 endpoints must have at least 6 errors among the 15 decodable lookup/revision/transition items. If it fails, stop without confirmation or redraw.
4. Only if range passes, query the 20 confirm items exactly once on the nine PLAN-N1-100 baselines and nine N1-120 candidates: 360 additional queries. Keep the four confirmation gates unchanged for each cohort: position ≥10/20 per seed, ≥2/5 per seed and kind, and aggregate position advantage ≥12/60 over both terminal and cut.

Maximum query count is 540. The cumulative CPU ceiling is 600 seconds for N1-123 readiness, face construction, restore verification, range, and conditional confirmation. Record cumulative CPU only after final source-hash verification. No training, optimizer updates, dev queries, threshold changes, retries, redraws, or capability promotion are permitted; `can_promote=false`.

## Output paths

- Face: `plans/manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019.json`.
- Face-generation attempt: `plans/manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019_attempt.json`.
- Source-exclusion manifest: `plans/manifests/n1_v13_after_n122_source_exclusions_20261019.json`.
- Readiness: `reports/taiji_n1_v13_after_n122_confirmation_readiness_20261019.json`.
- Face checks: `reports/taiji_n1_v13_after_n122_face_checks_20261019.json`.
- Range report and attempt: `reports/taiji_n1_v13_after_n122_range_20261019.json` and `reports/taiji_n1_v13_after_n122_range_20261019_attempt.json`.
- Confirmation, only if range passes: `reports/taiji_n1_v13_after_n122_confirmation_20261019.json` and its `_attempt.json` companion.
