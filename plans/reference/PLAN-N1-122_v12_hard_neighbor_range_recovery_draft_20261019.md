# PLAN-N1-122 — fresh four-observation hard-neighbor confirmation after N1-121

**Status: draft awaiting approval for a new measurement surface.** PLAN-N1-121 stopped at 5/9 usable endpoints on its 12-observation face. This proposal uses one fresh four-observation face with the same `d=1` name-neighbor distance, item classes, answer-length balance, and frozen gates. No N1-122 face or attempt currently exists.

## Why this face

On the N1-121 sacrifice subset, the candidate position arm was exact on 29/45 decodable items and cut on 26/45. Their emitted outputs matched on 37/60 paired rows and their correctness masks matched on 32/45 decodable items. This modest descriptive difference does not predict the confirm subset or establish capability.

The prior PLAN-N1-95 range preflight tested four independent four-observation, `d=1` scratch faces on an older endpoint cohort and recorded 7/9, 7/9, 7/9, and 8/9 usable endpoints. This is a face-construction feasibility result only; it does not predict the current N1-120 cohort. Four observations are proposed because the 12-observation N1-121 face did not reach range, while the four-observation `d=1` setting has repeated prior range evidence.

## Frozen sources and proposed face

- Candidate endpoints: the nine frozen PLAN-N1-120 final checkpoints (position, terminal, cut; seeds 11–13).
- Comparison endpoints, only if range passes: the nine frozen PLAN-N1-100 final checkpoints.
- New face: sampling seed `112`; 40 items (10 each lookup, revision, transition, unknown); four observations per item; 10-character entity names; distractor-name Hamming distance exactly 1; answer lengths 6 and 7 balanced within lookup/revision. Preserve the N1-121 per-kind split: sacrifice has three six-character and two seven-character lookup/revision answers; confirm has two six-character and three seven-character answers. Use 20 sacrifice and 20 confirm items with five items of each class per subset.
- Keep the N1-121 task grammar and answer semantics. This proposal changes observation count and face seed while retaining the class balance, name-distance rule, answer-length balance, endpoints, and all score gates.
- Use ID prefix `v12-hard-` and schema `n1-hard-neighbor-v12-confirmation-v1`. Stop if seed 112 or any N1-122 output path has already been used; do not draw another seed.

## Input disjointness

Before any face generation, build zero-query source readiness and require a zero intersection with all entity names and observed material values from:

- the existing N1-121 exclusion set, including the v3 train inputs, official v5/v6 faces, v7/v8 queried sacrifice inputs, v9/v10, and all 288 N1-110/N1-120 mapped material values;
- the complete N1-121 seed-111 face, including its unqueried confirm half;
- all 16 unique N1-95 scratch faces in [the exclusion manifest](../manifests/n1_v6_preflight_consumed_face_exclusions_20261019.json).

The full union currently contains 5,898 entity names and 4,728 answer/observation word values. The source-readiness implementation must separate target-answer vocabulary from observation-material disjointness; the five shared task tokens found by the N1-121 audit are fixed labels, while new observed material values must remain disjoint. Bind the source reports, all 16 scratch-face hashes, N1-121 report and face hashes, code identity, and all 36 endpoint checkpoint hashes. Do not decode dev payloads or old v7/v8 confirmation payloads.

## One-shot evaluation

1. Complete source, scorer, and checkpoint-hash readiness with zero model loads, queries, or updates. Reuse the existing independent full-bundle restore evidence only if all bound source and checkpoint hashes still match; otherwise stop and prepare a separately approved restore check.
2. After approval, build and freeze the seed-112 face once. Verify the class/length balance, four observations per item, exact `d=1` confusability, and source disjointness.
3. Query only the 20 sacrifice items on the nine N1-120 candidates: 180 model queries. The unchanged range gate passes only if at least 6/9 endpoints have at least 6 errors among the 15 decodable lookup/revision/transition items. If it fails, stop and do not query confirm or redraw.
4. Only if range passes, query the 20 confirm items exactly once on the nine N1-100 baselines and nine N1-120 candidates: 360 additional queries. Apply the unchanged four confirmation gates separately to each cohort:
   - Each position seed scores at least 10/20.
   - Each position seed scores at least 2/5 in every kind.
   - Aggregate position advantage over terminal is at least 12/60.
   - Aggregate position advantage over cut is at least 12/60.

Maximum query count is 540. The independent cumulative CPU ceiling is 600 seconds for N1-122 source readiness, face construction, restore verification, range, and conditional confirmation. No training, optimizer steps, dev queries, threshold changes, retries, redraws, or model promotion are permitted. `can_promote=false` regardless of the result.

## Output paths

- Face: `plans/manifests/n1_hard_neighbor_v12_confirmation_seed112_20261019.json`.
- Face-generation attempt: `plans/manifests/n1_hard_neighbor_v12_confirmation_seed112_20261019_attempt.json`.
- Source-exclusion manifest: `plans/manifests/n1_v12_after_n121_source_exclusions_20261019.json`.
- Readiness: `reports/taiji_n1_v12_after_n121_confirmation_readiness_20261019.json`.
- Face checks: `reports/taiji_n1_v12_after_n121_face_checks_20261019.json`.
- Range: `reports/taiji_n1_v12_after_n121_range_20261019.json`.
- Range attempt: `reports/taiji_n1_v12_after_n121_range_20261019_attempt.json`.
- Confirmation, only if range passes: `reports/taiji_n1_v12_after_n121_confirmation_20261019.json`.
- Confirmation attempt, only if range passes: `reports/taiji_n1_v12_after_n121_confirmation_20261019_attempt.json`.
