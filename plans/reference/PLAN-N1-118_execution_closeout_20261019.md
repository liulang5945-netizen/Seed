# PLAN-N1-118 execution closeout — normalized coupled RWL trust region

**Date:** 2026-10-19  
**Frozen plan SHA-256:** `4e7624c385256b41df2ae62de4a632292ea753c4412dea9f53ff712084e435d4`

## Training result

The fixed nine-endpoint batch reached all 24,576 updates per endpoint: 221,184 updates total, with 9,416.53125 CPU seconds charged against the 18,000-second ceiling. The run-time source and code guards remained unchanged, the length-balanced schedule checks passed, all nine lossless model shards were written, and all 18 zero/final independent restore checks matched. Dev, holdout, and confirmation queries were zero.

The original training report is intentionally preserved with `status=stopped` and `complete=false`. At its final receipt-validation stage, the pre-existing generic validator required each reader/writer/local group to remain within its own individual relative cap. PLAN-N1-118 instead caps their joint normalized radius at √3, which permits a single group to exceed one cap while keeping the combined radius bounded. This was a receipt-validator mismatch after all training endpoints completed, not a training, checkpoint, or source-guard failure.

The [read-only receipt revalidation](../../reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018_receipt_revalidation.json) binds to the unchanged original report SHA-256 `5ceabffadc7d62e2aa7be3996847c92713fe9cce604c0f10e463ad97fbe8217e`. It validates all nine lossless shards, 221,184 updates, 18/18 restore checks, 663,552 normalized RWL group measurements, and 442,368 independent context/relation measurements in 6.5625 CPU seconds. It loads no model or checkpoint and performs no query or update. The largest recorded joint actual radius is 1.7320512535 against √3 = 1.7320508076, a 4.46e-7 float-rounding difference within the projection's recorded tolerance. The corrected run-specific validator is isolated in [`_taiji_n1_plan118_lossless_receipt.py`](../../scripts/training/_taiji_n1_plan118_lossless_receipt.py) with SHA-256 `2afe60b83816412c4967525b427e2ece725cfd0039b72daf171b7ac16bd8c80b`; the shared validator remains at its run-time SHA-256 `b235dd722de90bab69994e854b98e1628740c2dcff4c2fac7c92d9d0f31049a9`, preserving the frozen source guards.

## Frozen training gate

Position seeds 11 and 12 passed all original gates: each scored 48/48 joint-four in both original and renamed views, with 48/48 in every condition. Seed 13 failed: joint-four was 24/48 original and 23/48 renamed; its four condition counts were 39/33/38/33 original and 41/30/41/29 renamed. The additional position score was 129/192 original with 13/96 paired, and 128/192 renamed with 12/96 renamed-paired. Therefore `training_sufficient=false`.

No new confirmation face was generated or queried. The run did not access dev, holdout, or confirmation; `can_promote=false`. This closes N1-118 without changing its frozen gates.

## Next step

Perform a read-only, item-level and cycle-level comparison of the already-consumed N1-117 and N1-118 position training shards. Focus on seed 13's unchanged joint-four outcome and changes among the four transition conditions. Use only the sealed reports and shards; load no model or checkpoint, make no new query or update, and do not change gates or generate a confirmation face. See [PLAN-N1-119](PLAN-N1-119_seed13_normalized_rwl_itemwise_attribution_draft_20261019.md).
