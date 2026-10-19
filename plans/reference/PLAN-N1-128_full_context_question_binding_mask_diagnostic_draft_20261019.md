# PLAN-N1-128 — full-context question-binding source mask diagnostic

**Status: authorized for one-shot execution on 2026-10-19 under the user's standing authority for normal local N1 diagnostics.** PLAN-N1-127 completed 180 queries while retaining full-context state and masking decoder reads. Position rose from 52/90 native rows to 82/90; cut rose from 45/90 to 78/90. Both arms reached 15/15 lookup and revision per face but left transition errors. The earlier N1-126 row-pruning intervention was higher by four transition rows per face/arm. This follow-up isolates the model's question-binding path while retaining all sixteen observations.

## Question

Do the remaining transition errors disappear when the local question-to-observation alignment and contextual confidence are restricted to the target entity's task-relevant rows, while the full input tape remains present? This is an oracle binding intervention; it does not test whether the model can select the target rows itself.

## Frozen intervention and cohort

- Reuse the same 15 decodable confirm items from each consumed v13 and v14 face, with five lookup, five revision, and five transition items per face. Reuse the same six N1-120 candidate position/cut endpoints. This is 2 faces × 15 items × 6 endpoints = 180 queries; query unknown items zero times.
- Observe all sixteen original rows in their original order. Preserve the full writer tape and all source states. During free generation only, mask non-target byte columns before both the local question-alignment softmax and its max-based confidence feature; the contextual question GRU therefore receives target-row context and confidence while retaining its full native question sequence.
- Keep N1-127's decoder content mask and carried-address mask unchanged. Keep source encoding, relation projection, answer writer, byte output, and relative transport mechanism unchanged. Do not provide gold answer bytes, byte positions, or answer prefixes to the mask.
- Compare with the saved native full-context rows, N1-127 decoder-only mask rows, and N1-126 source-pruned rows for the same checkpoint/item. No new face, split, item, label, or threshold is introduced.

## One-shot handling

Run a zero-query instrument preflight first. On a synthetic model and tape, an all-allowed binding mask must reproduce the native local/contextual bind outputs exactly; target-only alignment must have zero mass outside target source bytes, each query row must normalize to one, and the confidence maximum must be computed only over allowed bytes. Verify the full tape and synthetic model weights remain unchanged. The preflight must load zero checkpoints and issue zero model queries.

Then perform the 180 free-generation queries exactly once, with cumulative CPU≤600 seconds including readiness, loading, scoring, and final source hashes. Runtime assertions require one target-masked question bind per item, the N1-127 decoder and address contracts at every generated step, native boundary handling, and exact restoration of each frozen checkpoint bundle. Preserve partial rows and counters on any failure; do not retry.

No training, optimizer updates, new samples, dev/unknown queries, threshold changes, or capability promotion are included. Report oracle outcomes separately from native model capability; `can_promote=false` regardless of accuracy.

## Proposed outputs

- Evaluator: `scripts/training/eval_taiji_n1_v13_v14_full_context_question_binding_mask.py`.
- Readiness, attempt, and result: `reports/taiji_n1_v13_v14_full_context_question_binding_mask_readiness_20261019.json`, `reports/taiji_n1_v13_v14_full_context_question_binding_mask_attempt_20261019.json`, and `reports/taiji_n1_v13_v14_full_context_question_binding_mask_result_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-128_execution_closeout_20261019.md`.
