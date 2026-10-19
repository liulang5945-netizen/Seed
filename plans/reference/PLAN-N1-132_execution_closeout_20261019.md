# PLAN-N1-132 execution closeout — trainable question-binding supervision

**Status: complete; train-only calibration passed the unchanged training gate, no capability promotion.** The calibration continued the three frozen N1-120 position checkpoints with the original QA-plus-observation objective and an added 0.25 target-row alignment loss. It used the same 192 train parents and existing raw/hard length-balanced schedule; no new evaluation face was generated or queried.

## Bound execution and integrity

- Frozen plan SHA-256: `4177e1fa9da00ef1911a3c23f8598a67efd2e5fd5e54b2d4bd96b02fd60dd10f`.
- Zero-query source readiness: [report](../../reports/taiji_n1_plan132_binding_supervision_calibration_readiness_20261019.json), SHA-256 `4395057564bdf31cc84c0b1e9f9f6734ae64a976e50a1212a792fba9b59b7537`. Two code-only source revalidations followed a return-value handling defect; [revalidation v2](../../reports/taiji_n1_plan132_binding_supervision_calibration_readiness_revalidation_v2_20261019.json) binds the final runner. Its independent [source binding](../../reports/taiji_n1_plan132_binding_supervision_calibration_source_binding_revalidation_v2_20261019.json) preserves the original readiness and failed-preflight receipts.
- Two early checkpoint-preflight attempts stopped before model/checkpoint loads or updates. Their attempt records and the stopped report remain preserved. The repaired [checkpoint preflight](../../reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_revalidation_20261019.json) independently restored three complete zero-copy parent bundles (3/3), loaded three models/six checkpoints, and performed zero updates in 24.21875 CPU seconds. All parent parameters, Adam, RNG, and 43,008-session histories matched.
- One-shot result: [report](../../reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json), SHA-256 `9c9b6a89aa1e6b006469bda8a310fa9cc57cbb3478956482bebf1828000f7fcc`; 27,648 updates, 2,880 train-only score queries, 980.40625 execution CPU seconds and 1,006.421875 cumulative reported CPU seconds. The [per-step progress log](../../reports/taiji_n1_plan132_binding_supervision_calibration_progress_20261019.jsonl) contains all 27,648 QA, observation, and binding-loss rows. The [training attempt](../../reports/taiji_n1_plan132_binding_supervision_calibration_attempt_20261019.json) is SHA-256 `9bab5c64d1a637d7ac8ccb735482c0d9ce740975cd2e6a0a3cb732f5f9aace03`.
- All three final bundles independently restored. Each model retained 41 Adam states at 33,792 total updates and 52,224 history sessions; all three zero and all three final checkpoint copies remain in `data/checkpoints/n1_plan132_binding_supervision_calibration_20261019/`. Sources and parent checkpoints were unchanged; dev, v13, v14, and confirmation queries were zero; `can_promote=false`.

## Training-gate result

All three seeds passed the unchanged N1-120 position training-sufficiency gate. Each scored 192/192 on original train items, 96/96 paired, 192/192 renamed, and 96/96 renamed-paired. Original and renamed transition factorials were 48/48 joint-four for every seed. These are train-only scores and do not establish generalization.

Mean QA / observation / binding losses were seed11 `0.00561 / 0.000114 / 0.3975`, seed12 `0.00349 / 0.000117 / 0.3405`, and seed13 `0.00525 / 0.000097 / 0.3461`. The N1-120 QA-plus-observation objective and trust region stayed intact; only the supervised binding term was added.

## Provenance reconciliation

The result's top-level readiness and checkpoint-preflight fields point to the initial readiness and the stopped preflight report. The successful execution used the complete v2 readiness revalidation and repaired preflight; the [read-only provenance reconciliation](../../reports/taiji_n1_plan132_provenance_reconciliation_20261019.json), SHA-256 `895c3ede4c84e7d57ced3c7bd62727330eb2cbc6b51a9ab5cd0aa4e93bc0e87e`, binds those active receipts to the completed result, confirms all 27,648 progress rows and six checkpoint hashes, and verifies that both failed preflights stopped before any checkpoint load or update. The original result and attempt files remain unchanged.

## Next step

The single next proposal is [PLAN-N1-133](PLAN-N1-133_v15_binding_supervision_confirmation_draft_20261019.md): evaluate the N1-132 candidates on a fresh seed-115 v15 face, paired with same-seed N1-120 position parents and terminal/cut controls. This creates and consumes a new holdout measurement surface, so owner authorization is required before face generation or queries.
