# PLAN-N1-132 — trainable question-binding supervision calibration

**Status: authorized for one-shot train-only calibration under the user's standing authority for normal local N1 plans.** PLAN-N1-128 shows that full-context position reaches 90/90 when its question-binding path receives an oracle target-row mask. PLAN-N1-129/130 show native alignment gives target facts only modest mass enrichment and does not separate correct from incorrect transition outputs. PLAN-N1-131 further attributes every remaining oracle-masked cut error to inverse-turn outputs. A train-only calibration can test whether the existing aligner learns the oracle selection without adding inference-time labels or changing the model architecture.

## Proposed calibration

- Continue the three frozen N1-120 position endpoints from their complete final parameter, Adam, RNG, and observation-history bundles. Run exactly 9,216 additional updates per seed on the same 192 v3 train episodes; no cut or terminal arm is trained.
- Keep the model architecture and the complete N1-120 objective unchanged: `L_base = L_QA + AUXILIARY_WEIGHT * L_observation`, using the frozen `AUXILIARY_WEIGHT`. Add one supervised local-alignment term. For each answerable question byte position `i`, let `A[i,t]` be the existing native question-to-observation alignment and let `M[t]` mark bytes in the target entity's task-relevant fact row(s). Define `L_bind = mean_i(-log(sum_t(M[t] * A[i,t]) + 1e-8))`, average it across answerable questions in an update, and train with `L_total = L_base + 0.25 * L_bind`. An `unknown` parent question has no positive row and is excluded from `L_bind`; its paired known lookup question is supervised normally. The target mask is used only during training from that train item; inference remains unmasked and receives no gold entity-row tensor.
- Reuse the existing length-balanced 6,144-step schedule from its beginning: 9,216 updates per endpoint equals one full schedule cycle plus its first half-cycle. Preserve its raw/hard proportions, N1-120 expanded normalized Adam trust region, and original learning-rate schedule. Since the parent checkpoints are already at update 24,576, all continuation updates use the frozen final-cycle learning rate 0.00025.
- First verify the loss calculation and gradient path synthetically, then save and independently restore a complete zero-copy of each frozen parent bundle. Check parameters, Adam, RNG, and observation history remain identical. If approved, execute the fixed three-seed continuation once, with cumulative CPU≤2,400 seconds; then save and independently restore all three final bundles. Score only existing train views (original, paired, renamed, renamed-paired), preserve per-step QA, observation, and binding losses, and apply the original unchanged training-sufficiency gate.
- Apply the existing N1-120 position training-sufficiency gate unchanged. Do not query dev, v13/v14, or any new confirmation face in this calibration. `can_promote=false` regardless of the train-only outcome; formal multi-seed confirmation remains a separate later step.

## Execution boundary

This is a one-time local optimizer-objective calibration, authorized under the user's standing instruction to continue normal local N1 plans without asking. The 0.25 binding-loss weight and 9,216-update schedule are fixed in this plan; the original training gate is unchanged. It introduces no inference-time labels, new data, dev/holdout queries, or capability claim. The run will start only after full source binding and complete-bundle zero/final restore preflight pass.

## Proposed outputs after approval

- Runner: `scripts/training/run_taiji_n1_plan132_binding_supervision_calibration.py`.
- Readiness, attempt, and result: `reports/taiji_n1_plan132_binding_supervision_calibration_readiness_20261019.json`, `reports/taiji_n1_plan132_binding_supervision_calibration_attempt_20261019.json`, and `reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-132_execution_closeout_20261019.md`.
