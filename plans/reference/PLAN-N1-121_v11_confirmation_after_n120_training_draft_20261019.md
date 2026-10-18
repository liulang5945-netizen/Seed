# PLAN-N1-121：conditional v11 confirmation after N1-120

**Status: awaiting owner approval for the new confirmation-surface generation and one-shot queries.** The N1-120 training gate passed, but its plan explicitly requires a separate plan before confirmation. No face has been generated and no query is authorized by this draft alone.

## Decision question

Does the N1-120 candidate cohort pass the existing one-shot v11 range and confirmation gates against the frozen PLAN-N1-100 baselines, using the v11 face specification already frozen but never consumed under PLAN-N1-110?

## Frozen sources and face

- PLAN-N1-100 baseline report: `reports/taiji_n1_hard_neighbor_training_20261017.json`, SHA-256 `dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909`.
- PLAN-N1-120 official training report: `reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019.json`, SHA-256 `873c237623fa7b97de0f4d8d95edaf4a5d8f9bf944c8af301468bd1d0f19ab2f`.
- PLAN-N1-120 corrected receipt revalidation: `reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019_receipt_revalidation.json`, SHA-256 `22055b74784e7c35891098595928af238561405a52cbfb4b94eb87491ba7e03d`.
- Reuse the unchanged v11 specification from PLAN-N1-110: seed `111`, 40 new items (10 per kind), 12 observations, Hamming distance 1, 20 sacrifice and 20 confirm. Lookup/revision retain the frozen five/six-character split for each kind and the frozen 2/3 versus 3/2 sacrifice/confirm assignment.
- Before constructing the face, verify seed 111 and the v11 output path remain unused. If any v11 face or attempt already exists, stop; do not redraw with another seed.
- Require fresh entities/materials disjoint from v3 train, all six/fresh training material values, v5/v6, v7/v8 queried sacrifice, v9, and v10 sources. Do not read or decode dev/holdout payloads or old v7/v8 confirmation payloads.

## One-shot evaluation

1. Build and freeze the 40-item v11 face only after approval. Run builder, lineage, checkpoint-restore, scorer, and source-binding readiness with zero model queries.
2. Query only the 20 sacrifice items on the nine N1-120 candidate endpoints (180 model queries). Apply the unchanged range gate of at least 6/9 usable endpoints. If it fails, do not query confirm and do not redraw the face.
3. If range passes, query the 20 confirm items exactly once on both the nine frozen PLAN-N1-100 baselines and nine N1-120 candidates (360 additional model queries). Report the four confirmation gates separately for each cohort:
   - Each position seed scores at least 10/20.
   - Each position seed scores at least 2/5 in every kind.
   - Aggregate position advantage over terminal is at least 12/60.
   - Aggregate position advantage over cut is at least 12/60.

The maximum is 540 model queries if both phases run; no parameter updates or optimizer steps are allowed. Use a cumulative CPU ceiling of 600 seconds for construction, restore, range, and optional confirmation. The original thresholds remain fixed.

## Readiness and limits

- Verify the nine PLAN-N1-100 and nine PLAN-N1-120 full bundles restore independently before any query; bind all 18 checkpoint hashes, report hashes, code identity, and the exact v11 face SHA.
- N1-120's main report remains `stopped` because its run-time generic receipt was incompatible with the normalized policy. Use its completed corrected receipt revalidation as the source of training-gate validity; preserve both reports unchanged.
- Keep `can_promote=false`. Even a four-gate confirmation pass does not itself promote a model; any promotion review remains separate.
- Preserve the consumed face and all outputs. No retry, second face draw, threshold change, or checkpoint overwrite.

## Output paths

- Face manifest: `plans/manifests/n1_hard_neighbor_v11_confirmation_seed111_20261019.json`.
- Build/readiness report: `reports/taiji_n1_v11_after_n120_confirmation_readiness_20261019.json`.
- Sacrifice range report: `reports/taiji_n1_v11_after_n120_range_20261019.json`.
- One-shot confirmation report, only if range passes: `reports/taiji_n1_v11_after_n120_confirmation_20261019.json`.
