# EXP-N1-105 · counterbalanced input-feature diagnostic · 2026-10-17

## Result

The one-shot PLAN-N1-105 diagnostic completed all 2,232 item-endpoint queries in 116.921875 cumulative CPU seconds, within the 1,200-second ceiling. The result is `reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json`, SHA-256 `287cc5c4a38693d8e055f1290d801b1ec6494768386ff3ad1963e3059852a822`. It is bound to the frozen 248-item face SHA-256 `2d912941baecaa4811266ace64b56fa9bb2ace65d657691834911db16ed369d8` and PLAN-N1-105 SHA-256 `af6eebc3e3fb01fc8c409f762176aefa8b91f246e9ea7325f42c5fa3267537ca`.

All nine frozen endpoints were loaded and restored. Checkpoints, source guards, and code identity remained unchanged. The evaluator made 2,232 planned queries, zero updates, zero optimizer steps, no dev or existing confirmation-face queries, and did not recompute a gate. The construction and recovery receipts confirm that v7/v8 confirmation payloads were not decoded. `can_promote=false` remains in force.

## Free exact counts

| Seed | Position /248 | Terminal /248 | Cut /248 |
|---:|---:|---:|---:|
| 11 | 236 | 10 | 177 |
| 12 | 179 | 23 | 166 |
| 13 | 29 | 19 | 142 |
| Total | 444/744 | 52/744 | 485/744 |

Across all three seeds, position scored 168/288 lookup, 165/264 revision, 55/132 transition, and 56/60 unknown. Cut scored 108/288, 264/264, 55/132, and 58/60 respectively. Terminal scored 0/288, 0/264, 11/132, and 41/60. These are diagnostic-face counts only; they do not replace or modify any frozen gate.

The matched position-versus-cut item outcomes expose strong seed heterogeneity:

| Seed | Both exact | Position only | Cut only | Both wrong |
|---:|---:|---:|---:|---:|
| 11 | 165/248 | 71/248 | 12/248 | 0/248 |
| 12 | 142/248 | 37/248 | 24/248 | 45/248 |
| 13 | 18/248 | 11/248 | 124/248 | 95/248 |

For lookup in the position arm, seeds 11 and 12 each scored 48/48 on six-character answers and 36/48 on seven-character answers; seed 13 scored 0/48 at either length. For revision, seed 11 scored 44/44 at both lengths, seed 12 scored 44/44 at length six and 33/44 at length seven, and seed 13 scored 0/44 at both lengths. In seed 13, position scored 0/96 lookup and 0/88 revision while cut scored 12/96 lookup and 88/88 revision.

## Interpretation boundary

Crossing answer length and target-row position within each matched template removes the v9 construction's position/length/item-order alias from this diagnostic. The observed position-arm seven-character lookup deficit in seeds 11 and 12 and the seed-13 collapse are descriptive patterns on these synthetic templates. They do not identify a causal mechanism, establish generalization, change the v9 four-gate failure, or support capability promotion. PLAN-N1-105 is complete as a diagnostic; no training, gate change, or new confirmation was authorized.

## Provenance

The result records 78.671875 CPU seconds for evaluation, 28.09375 for the successful readiness pass, 0.15625 for face construction, and the conservative 10-second upper bound for the failed preflight attempt. Total cumulative CPU is 116.921875 seconds. The failed preflight made no model query or update; its correction and source binding are recorded in `reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_preflight_repair_20261017.json`.
