# PLAN-N1-127 execution closeout — full-context decoder source mask

**Status: complete; oracle diagnostic only, no capability promotion.** The one-shot reused the same 15 decodable confirm items on each consumed v13/v14 face and the six frozen N1-120 position/cut endpoints. It kept every sixteen-row input and its contextual encodings intact, then masked both decoder content reads and the carried relative address to the target entity's task-relevant source rows during free generation.

## Bound execution and integrity

- Frozen plan SHA-256: `a3f75cc4a8c524eed31643aeb12e47d797075f0569df8886257f49e475149ac8`.
- Zero-query readiness: [report](../../reports/taiji_n1_v13_v14_full_context_source_mask_readiness_20261019.json), SHA-256 `cd4d401adb544635ca56f95fafda880811c526d23a5140c5ffbb34ab23b3c35f`; CPU 2.75 seconds, 141 source identities and six candidate checkpoint hashes, with zero checkpoint/model loads, queries, or updates.
- Synthetic preflight passed: all-allowed softmax reproduced the native output bit-for-bit; masked attention and source-local transport had exactly zero off-target mass and each normalized to one.
- One-shot result: [report](../../reports/taiji_n1_v13_v14_full_context_source_mask_result_20261019.json), SHA-256 `0efbd0cfc58de67fe155595d8f56a77565c9058bae70111b1278097752f3bbbf`; 180/180 rows, 12 model loads, 50.140625 query CPU seconds, 52.890625 cumulative readiness/query CPU seconds, unchanged sources/checkpoints, zero updates, dev queries, or unknown queries. The [attempt receipt](../../reports/taiji_n1_v13_v14_full_context_source_mask_attempt_20261019.json) is SHA-256 `5307b0df7f8c4f27cd272e0c5809ad0c3d19020c3fdda82f07e2e0956e19519c`.
- Runtime checks covered 1,228 generated steps, 2,456 masked content reads, and 1,048 carried-address transports. Every final attention and transported address was normalized with zero off-target mass; all 180 generations stopped at the native boundary. Teacher-forced rows matched the saved native rows 180/180. Teacher scoring remained unmasked, so free/teacher exactness differences under this intervention are not same-path comparisons.

## Results

| Face | Arm | Native full context | Full-context decoder mask | Source-pruned N1-126 | Mask gains / losses vs native |
|---|---|---:|---:|---:|---:|
| v13 | position | 27/45 | 41/45 | 45/45 | 14 / 0 |
| v13 | cut | 22/45 | 39/45 | 43/45 | 17 / 0 |
| v14 | position | 25/45 | 41/45 | 45/45 | 16 / 0 |
| v14 | cut | 23/45 | 39/45 | 43/45 | 16 / 0 |

Across all endpoints and both faces, decoder masking raised position from 52/90 to 82/90 and cut from 45/90 to 78/90, with no regressions against native free generation. Lookup and revision were 15/15 for every face/arm. Remaining errors were all transition direction words: position was 11/15 and cut 9/15 transitions per face. Position seed 13 was 11/15 (5/5 lookup, 5/5 revision, 1/5 transition); cut seeds 11/12/13 were 11/15, 15/15, and 13/15 respectively on both faces.

Compared with N1-126 source pruning, the full-context mask lost four transition rows per face/arm and gained none. Those transition answers were correct under row pruning but remained wrong under the decoder mask, showing that decoder access alone does not reproduce the pruning result. The mask supplies the target entity and relation as an oracle, so these gains do not establish autonomous retrieval or generalization. No training gate or capability claim is changed; `can_promote=false`.

The masked free outputs differed from unmasked teacher exactness on 63/180 rows; teacher symbols themselves matched the saved native teacher outputs on all 180 rows. This is expected because only free-generation attention was intervened on. It is not evidence that the checkpoint changed or that a frozen gate was met.

## Next step

The single continuation is [PLAN-N1-128](PLAN-N1-128_full_context_question_binding_mask_diagnostic_draft_20261019.md): keep the full tape, apply the same target-row decoder mask, and additionally mask only the question-to-observation local alignment and its contextual confidence before question binding. The same consumed items and endpoints isolate whether the remaining transition gap comes from upstream question binding or from retaining the distractor history. It remains a zero-training oracle diagnostic with no gate change or capability claim.
