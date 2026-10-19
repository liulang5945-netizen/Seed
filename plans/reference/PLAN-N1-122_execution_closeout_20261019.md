# PLAN-N1-122 execution closeout — v12 range gate

**Status: stopped at the preregistered range gate.** The approved seed-112, four-observation, `d=1` face passed construction, and its 20-item sacrifice subset was queried once. It produced 3/9 usable endpoints, below the frozen 6/9 threshold. The confirm subset was not queried; there was no redraw or retry.

The approved draft remains byte-identical because its hash is bound into the readiness and face receipts. This closeout records the operative post-execution status.

## Bound execution

- Face: `plans/manifests/n1_hard_neighbor_v12_confirmation_seed112_20261019.json`, SHA-256 `dc8daeecc190a40b308fc8f7f79b432a021970649de782c7799dd995ce0c6a11`.
- Zero-query readiness: SHA-256 `fa659e3ab59685bfc3ec3273698ca58d9537f3fbf2ef5f205af1b33634733717`; 36 checkpoint hashes matched and the 18/18 independent full-bundle restore evidence remained equal.
- Range report: `reports/taiji_n1_v12_after_n121_range_20261019.json`, SHA-256 `3f3e68cdddd692daac3d3ba51e2573ce71195469b693388d42b9c215f696853b`.
- Range attempt: `reports/taiji_n1_v12_after_n121_range_20261019_attempt.json`, SHA-256 `59874cd5410b57aeef0979f6e1feadbc6b59a93421a9ab0b456e8714aa6594f2a`.

The zero-query readiness took 3.203125 CPU seconds, reused 18/18 equal full-bundle restore receipts, and recorded zero model/checkpoint loads or updates. The face contract and the six focused PLAN-N1-122 tests passed; Ruff, Black, and `py_compile` passed. The 180 queries completed in 35.46875 CPU seconds; construction used 0.015625 CPU seconds. The report's `cumulative_cpu_seconds` field recorded 38.5 because it was sampled before the final source-hash check; the [CPU reconciliation](../../reports/taiji_n1_v12_after_n121_cpu_reconciliation_20261019.json) recomputes the total as **38.6875/600 seconds**. This 0.1875-second undercount does not affect the budget or range result. Source hashes remained unchanged; there were nine model/checkpoint loads, zero parameter updates, zero dev queries, zero confirm queries, and `can_promote=false`.

## Result and offline attribution

Only the three terminal endpoints reached the error floor. Decodable endpoint errors were:

| Seed | Position | Cut | Terminal |
|---|---:|---:|---:|
| 11 | 3 | 3 | 14 |
| 12 | 1 | 1 | 13 |
| 13 | 4 | 4 | 13 |

The [offline attribution report](../../reports/taiji_n1_v12_after_n121_range_attribution_20261019.json) reuses the same 180 saved rows. Position and cut each scored 37/45 decodable items; terminal scored 5/45. Position and cut outputs were identical on 47/60 paired rows, and their correctness masks matched on 39/45 decodable items. Each arm had zero free-versus-teacher-forced exactness disagreements. These are descriptive sacrifice-set results; the failed range gate bars any confirm-set or capability claim.

The N1-120 training set has exactly two observations per item. N1-121 used twelve observations and reached 5/9 usable endpoints; N1-122 used four and reached 3/9. Those faces contain different sampled items, so the contrast is not a causal estimate of observation count. N1-95 also found that 4–12 observations were non-monotonic on older endpoints. The next proposal therefore treats sixteen observations as a single, untested long-context challenge, eight times the training context length, with no predicted range or capability outcome.

## Next step

The single proposed continuation is [PLAN-N1-123](PLAN-N1-123_v13_long_context_range_probe_draft_20261019.md): one fresh seed-113, sixteen-observation, `d=1` face, keeping the same class balance and gates and excluding the full N1-121/N1-122 faces plus all prior sources. It is a new measurement surface; no N1-123 face or query is authorized by this closeout.
