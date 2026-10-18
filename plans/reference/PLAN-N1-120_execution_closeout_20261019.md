# PLAN-N1-120 execution closeout — expanded normalized RWL radius

**Date:** 2026-10-19  
**Frozen plan SHA-256:** `5b22795d84b9ac589b0de5cd73a03d5019d5ac0b2c9a6efd8ec2b84a0ecabae2`

## Training and receipt result

The nine-endpoint batch reached 24,576 updates per model: 221,184 total updates and 9,465.765625 CPU seconds of the 18,000-second ceiling. All nine lossless shards were written, all 18 zero/final restore checks matched, and source/code guards remained unchanged. The 54-source runner binding and nine-zero preflight passed before training. Dev, holdout, and confirmation queries were zero.

The original training report is preserved with `status=stopped` and `complete=false` because the dynamically isolated controller still held the generic per-group receipt validator. That controller copy was not replaced by the N1-120 normalized-radius validator, so the last receipt audit rejected records that match the shared `2√3` policy. The corrected [read-only receipt revalidation](../../reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019_receipt_revalidation.json), SHA-256 `22055b74784e7c35891098595928af238561405a52cbfb4b94eb87491ba7e03d`, binds to the original report SHA-256 `873c237623fa7b97de0f4d8d95edaf4a5d8f9bf944c8af301468bd1d0f19ab2f`. It validates all nine shards, all 221,184 updates, 18/18 restore checks, 663,552 RWL measurements, and 442,368 independent context/relation measurements in 6.609375 CPU seconds. It loads no model or checkpoint and performs no query or update.

## Frozen training gate

All three position seeds passed. Each scored 48/48 joint-four in both original and renamed views, with 48/48 on every one of the four transition conditions. Each also scored 192/192 full-train exactness in both views and 96/96 paired and renamed-paired exactness. The corrected receipt audit records `training_sufficient=true`.

This clears only the training prerequisite. No confirmation face has been generated or queried, and `can_promote=false`.

## Next step

PLAN-N1-110's seed-111 v11 face remains unused. Prepare PLAN-N1-121 to apply that unchanged face design to the nine frozen PLAN-N1-100 baselines and nine PLAN-N1-120 endpoints: query the 20-item sacrifice subset on N1-120 first (180 queries); only if at least 6/9 endpoints pass range, query the 20-item confirmation subset once on both cohorts (360 more queries). Preserve the original four confirmation gates and stop without retry if range fails. This consumes a new confirmation surface, so wait for owner approval after the plan is reviewable. See [PLAN-N1-121](PLAN-N1-121_v11_confirmation_after_n120_training_draft_20261019.md).
