# PLAN-N1-132 — trainable question-binding supervision calibration

**Status: draft requiring owner decision before training.** PLAN-N1-128 shows that full-context position reaches 90/90 when its question-binding path receives an oracle target-row mask. PLAN-N1-129/130 show native alignment gives target facts only modest mass enrichment and does not separate correct from incorrect transition outputs. PLAN-N1-131 further attributes every remaining oracle-masked cut error to inverse-turn outputs. A train-only calibration can test whether the existing aligner learns the oracle selection without adding inference-time labels or changing the model architecture.

## Proposed calibration

- Continue the three frozen N1-120 position endpoints from their complete final parameter, Adam, RNG, and observation-history bundles. Run exactly 9,216 additional updates per seed on the same 192 v3 train episodes; no cut or terminal arm is trained.
- Keep the model architecture and QA objective unchanged. Add one supervised local-alignment term. For each question byte position `i`, let `A[i,t]` be the existing native question-to-observation alignment and let `M[t]` mark bytes in the target entity's task-relevant fact row(s). Define `L_bind = mean_i(-log(sum_t(M[t] * A[i,t]) + 1e-8))`, then train with `L_total = L_QA + 0.25 * L_bind`. The target mask is used only during training from the known training item; normal inference remains unmasked and receives no gold entity-row tensor.
- First verify loss calculation, gradient path, zero/final complete-bundle save/restore, and history/RNG preservation. Then execute the fixed three-seed continuation once, with cumulative CPU≤2,400 seconds. Preserve all training loss and sufficiency records.
- Apply the existing N1-120 position training-sufficiency gate unchanged. Do not query dev, v13/v14, or any new confirmation face in this calibration. `can_promote=false` regardless of the train-only outcome; formal multi-seed confirmation remains a separate later step.

## Decision boundary

This is a new optimizer objective and continuation recipe. The 0.25 auxiliary-loss weight, 9,216-update schedule, three-seed continuation, and train-only gate are proposed; they have not been run. If approved, the next implementation will bind all source/checkpoint hashes and prove complete checkpoint recovery before training. If not approved, no training will start and the completed N1-126–131 diagnostic record remains the current evidence base.

## Proposed outputs after approval

- Runner: `scripts/training/run_taiji_n1_plan132_binding_supervision_calibration.py`.
- Readiness, attempt, and result: `reports/taiji_n1_plan132_binding_supervision_calibration_readiness_20261019.json`, `reports/taiji_n1_plan132_binding_supervision_calibration_attempt_20261019.json`, and `reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-132_execution_closeout_20261019.md`.
