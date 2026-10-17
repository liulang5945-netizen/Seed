# EXP-N1-108 · v10 teacher-forced/free-output agreement · 2026-10-17

## Result

The frozen read-only analyzer verified all 2,232 item/endpoint rows in 0.015625 CPU seconds. Report: `reports/taiji_n1_hard_neighbor_v10_teacher_free_agreement_20261017.json`, SHA-256 `5807948dd23cf21b1ecdade74bafe14ffce72f229b10b318dc9985fb3a948661`. It binds the exact v10 face, result, PLAN-N1-105 and PLAN-N1-108 hashes, plus analyzer and contract-test hashes; source hashes match and inputs are unchanged.

| Joint exactness | Rows / 2,232 |
|---|---:|
| Both teacher-forced and free exact | 981 |
| Free exact only | 0 |
| Teacher-forced exact only | 0 |
| Both wrong | 1,251 |

The equality holds in every seed/arm/kind/answer-length/position stratum: all 981 exact items are exact in both stored measures, and all 1,251 incorrect free outputs are also teacher-forced inexact. The analyzer did not rerun teacher scoring, load a model/checkpoint, query, train, update, or recompute a gate.

## Interpretation boundary

This rules out a measured free-only decoding discrepancy on this face: each recorded free error is already present when the target answer prefix is supplied. It points toward an answer-prediction/content mismatch in these stored outcomes rather than an error introduced only by free generation. It does not identify the internal mechanism or prove generalization; the v9 failure and `can_promote=false` remain unchanged.
