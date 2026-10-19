# PLAN-N1-128 execution closeout — full-context question-binding source mask

**Status: complete; oracle diagnostic only, no capability promotion.** The one-shot reused the N1-126/N1-127 consumed v13/v14 confirm items and the six frozen N1-120 position/cut checkpoints. It retained all sixteen observations, masked non-target bytes in local question alignment and its max-confidence feature during free generation, and retained N1-127's decoder-content and relative-address masks.

## Bound execution and integrity

- Frozen plan SHA-256: `26884577737afc5b46dcff061329705fbd6adb96d942629c1d53de5cbe9a4b74`.
- Zero-query readiness: [report](../../reports/taiji_n1_v13_v14_full_context_question_binding_mask_readiness_20261019.json), SHA-256 `0141c12b7a45a2e48b54b6965b86a97dcbdc5b246a4a7f129ebfd590b5fac90c`; CPU 3.09375 seconds, 146 source identities and six candidate checkpoint hashes, zero checkpoint/model loads, queries, or updates.
- Synthetic preflight passed: all-allowed binding matched native local/contextual binding tensors bit-for-bit; target-only alignment and confidence excluded every non-target byte; row weights normalized; synthetic model parameters and full observation tape stayed unchanged.
- One-shot result: [report](../../reports/taiji_n1_v13_v14_full_context_question_binding_mask_result_20261019.json), SHA-256 `a9278debaf3c2c57f4a6678a692c8b4afa6e9a5e867745d3d7c5ba5526d6b28e`; 180/180 rows, 12 model loads, 51.4375 query CPU seconds and 54.53125 cumulative readiness/query CPU seconds. All hashes remained unchanged; parameter updates, dev queries, and unknown queries were zero. The [attempt receipt](../../reports/taiji_n1_v13_v14_full_context_question_binding_mask_attempt_20261019.json) is SHA-256 `761e1a6a9b583d795aec8b2548671100d69a89475db42d23aa2a743a1cdfa8d4`.
- Runtime assertions covered one target-masked bind per item, 1,236 decoder steps, and 2,472 masked content reads, with zero off-target alignment, attention, or address mass. All 180 full-context inputs retained sixteen rows and used the native boundary stop. Teacher-forced symbols matched the saved native rows 180/180; the teacher path was intentionally unmasked.

## Results

| Face | Arm | Native full context | N1-127 decoder mask | N1-128 question + decoder masks | N1-126 source-pruned |
|---|---|---:|---:|---:|---:|
| v13 | position | 27/45 | 41/45 | 45/45 | 45/45 |
| v13 | cut | 22/45 | 39/45 | 43/45 | 43/45 |
| v14 | position | 25/45 | 41/45 | 45/45 | 45/45 |
| v14 | cut | 23/45 | 39/45 | 43/45 | 43/45 |

Question-binding masking rescued four more rows per face/arm than decoder-only masking, without a regression, and matched the earlier source-pruned exact totals. All position rows were exact on both faces. Cut was exact on 43/45 per face; the only remaining failures were seed13 transition items `030` (gold `west`, output `east`) and `034` (gold `north`, output `south`) on both faces. Every lookup and revision row was exact. Aggregate exactness rose from native 52/90 to 90/90 position and from 45/90 to 86/90 cut.

The delta is consistent with cross-source contamination in local question binding and its contextual confidence feature being part of the remaining transition gap. It does not prove that the learned model can select the correct rows without the gold identity; all N1-126/127/128 masks are oracle interventions on consumed items. The 79/180 free/teacher exactness disagreements are expected because the free path was masked while teacher scoring remained native and unmasked. Frozen gates are unchanged and `can_promote=false`.

## Next step

The single continuation is [PLAN-N1-129](PLAN-N1-129_native_question_binding_mass_audit_draft_20261019.md): measure the unmasked model's native question-alignment mass on target versus distractor source bytes, without generating answers or updating parameters. This will determine whether the learned binding already identifies the target but later writing/composition fails, or whether it still distributes mass across distractors. It uses only the same consumed items and endpoints and makes no capability claim.
