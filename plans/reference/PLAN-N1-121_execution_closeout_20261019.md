# PLAN-N1-121 execution closeout — v11 range gate

**Status: stopped at the preregistered range gate.** The v11 face was generated once and its 20-item sacrifice subset was queried once. It produced 5/9 usable endpoints, below the frozen 6/9 threshold. The confirm subset was not queried, and the face will not be redrawn under PLAN-N1-121.

The approved plan and its source-exclusion correction remain byte-identical because their hashes are bound into the face and readiness receipts. This closeout records the operative post-execution status.

## Bound execution

- Face: `plans/manifests/n1_hard_neighbor_v11_confirmation_seed111_20261019.json`, SHA-256 `1d9236c5a258493ace574af74ae6b032632cf53011b009ad47660f862ea87f5e`.
- Zero-query source readiness: v3, SHA-256 `a5ea5246123444df8ec71640857482dd7a17b8330f476ebc8d16f0679e418e41`; it excludes 1,918 entity names and 1,633 word values, including all 288 N1-110/N1-120 mapped values.
- Scorer readiness: v6, SHA-256 `742cf914f2dc8eee7a439fdb0576fac141bc3cb1b5fb5203c981298027477653`; 36 checkpoint hashes were bound and independently rechecked.
- Range report: `reports/taiji_n1_v11_after_n120_range_20261019.json`, SHA-256 `f1ce49385c1ee6001385a93a5bf65a7de08546c094afac697f3e6fee46fc8cf2`.
- Range attempt: `reports/taiji_n1_v11_after_n120_range_20261019_attempt.json`, SHA-256 `03986b9cc823f31998f45e83623690bfd14c330d5f82e032a39df8b96041e233`.

The face checks passed; the 180 sacrifice queries completed in 38.390625 CPU seconds, bringing PLAN-N1-121 cumulative CPU to 83.1875/600 seconds. Source hashes stayed unchanged. There were zero parameter updates, dev queries, or confirmation queries; `can_promote=false`.

## Range result and offline attribution

An endpoint is usable when it makes at least 6 errors among the 15 decodable items. The errors were:

| Seed | Position | Cut | Terminal |
|---|---:|---:|---:|
| 11 | 6 | 9 | 14 |
| 12 | 5 | 5 | 14 |
| 13 | 5 | 5 | 14 |

Only five endpoints reached the error floor: position/seed11, cut/seed11, and all three terminal endpoints. The gate therefore failed without ambiguity; no confirm query followed.

The zero-query attribution report [taiji_n1_v11_after_n120_range_attribution_20261019.json](../../reports/taiji_n1_v11_after_n120_range_attribution_20261019.json) reuses the 180 saved rows. Across position's 45 decodable items, exact counts were lookup 10, revision 15, transition 4; cut scored 7, 15, and 4. Position and cut emitted identical output on 37/60 paired rows; their correctness masks matched on 32/45 decodable items. Position alone solved 8 items and cut alone solved 5, a net difference of 3/45 on this sacrifice subset. There were no free-versus-teacher-forced exactness disagreements. These are descriptive sacrifice-set results, not confirmation evidence or a capability claim.

## Historical input-boundary audit

The [source-overlap audit](../../reports/taiji_n1_v11_after_n120_source_overlap_audit_20261019.json) compared this face with the 16 unique scratch faces referenced by the consumed PLAN-N1-95 preflight reports. It found zero shared entity names and zero shared observation-material values. The five shared strings (`east`, `north`, `south`, `west`, `unknown`) are fixed answer tokens required by the task schema, not repeated observation inputs. The audit therefore found no repeated model input values in the v11 face.

Those prior scratch faces are now preserved as an exclusion-only inventory in [n1_v6_preflight_consumed_face_exclusions_20261019.json](../manifests/n1_v6_preflight_consumed_face_exclusions_20261019.json): 3,660 entity names, 2,855 observed material values, and 324 answer values across 16 unique faces. The inventory includes all 17 run references from the range and confusability reports and their source hashes.

## Next step

Close PLAN-N1-121 at `range_not_met`. The single proposed continuation is PLAN-N1-122: one fresh seed-112 face with four observations per item and Hamming distance 1, retaining the same class balance and all range/confirmation gates. The earlier PLAN-N1-95 preflight reached 7/9, 7/9, 7/9, and 8/9 usable endpoints on four independent four-observation/distance-1 faces, but those were older endpoints and do not predict N1-120 performance. A fresh N1-122 face would exclude the entire v11 face and all 16 prior scratch faces. Since this changes the measurement surface, generate or query nothing for N1-122 until its approval is received; see [the draft](PLAN-N1-122_v12_hard_neighbor_range_recovery_draft_20261019.md).
