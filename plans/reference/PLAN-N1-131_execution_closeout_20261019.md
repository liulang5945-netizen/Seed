# PLAN-N1-131 execution closeout — saved transition-operator error attribution

**Status: complete; offline descriptive audit only, no capability promotion.** The analyzer joined the ten already-consumed v13/v14 transition items with saved outputs from native full context, N1-126 source pruning, N1-127 decoder masking, and N1-128 question-binding masking. It verified each manifest answer against the stated cardinal rotation before classifying outputs.

## Bound execution and integrity

- Frozen plan SHA-256: `f239c2a4b052ac03c05060a1ee8060d4c50d6fbaf38efce8c0d87c3be1970cba`.
- Report: [transition attribution](../../reports/taiji_n1_transition_operator_error_attribution_20261019.json), SHA-256 `90bf723459b085753a58fb78ade0914d78f4be10e422f6b934a53cdbf98edffc`.
- The report covers 240 saved output rows (10 inputs × 3 seeds × 2 arms × 4 conditions), read and classified in 3.015625 CPU seconds. All source hashes stayed unchanged; model/checkpoint loads, queries, and parameter updates were zero. The [offline analyzer](../../scripts/training/analyze_taiji_n1_transition_operator_error_attribution.py) and report are bound in the source map.

## Results

| Saved condition | Arm | Correct | Inverse turn | Unchanged base | Other cardinal/output |
|---|---|---:|---:|---:|---:|
| Native full context | position | 8/30 | 10 | 6 | 6 |
| Native full context | cut | 6/30 | 8 | 7 | 9 |
| N1-126 source pruned | position | 30/30 | 0 | 0 | 0 |
| N1-126 source pruned | cut | 26/30 | 4 | 0 | 0 |
| N1-127 decoder mask | position | 22/30 | 4 | 1 | 3 |
| N1-127 decoder mask | cut | 18/30 | 8 | 2 | 2 |
| N1-128 question-binding mask | position | 30/30 | 0 | 0 | 0 |
| N1-128 question-binding mask | cut | 26/30 | 4 | 0 | 0 |

All four N1-128 cut errors are seed13 inverse-turn outputs: for both faces, item `030` applies right when the observed turn is left, and item `034` applies left when the observed turn is right. The position arm gets all thirty transition outputs correct under the same oracle binding mask. This localizes the observed residual to turn composition in the cut arm, while still providing no evidence that the model can autonomously select sources or generalize. Original capability gates are unchanged and `can_promote=false`.

## Next step

The single next proposal is [PLAN-N1-132](PLAN-N1-132_trainable_question_binding_supervision_calibration_draft_20261019.md): train a local auxiliary loss that teaches the existing question-byte alignment to concentrate on its task-relevant fact rows, then evaluate only on the existing 192 training episodes. It keeps the architecture, data, and original training-sufficiency gate, but changes the optimizer objective and continuation recipe. That change is outside the current diagnostic plan and requires an owner decision before any training starts.
