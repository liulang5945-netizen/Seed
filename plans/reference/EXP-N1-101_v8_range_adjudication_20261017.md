# EXP-N1-101 · v8 sacrifice range adjudication · 2026-10-17

## Conclusion

The fixed-seed108, eight-observation d=1 v8 face completed its single sacrifice range run, but did not pass the frozen gate: 4/9 endpoints reached at least 6 errors among 15 addressable items, below the required 6/9. The run was complete and within budget; v8 confirm remains untouched. PLAN-N1-101 is closed at `range_not_met`.

Range report: `reports/taiji_n1_hard_neighbor_v8_range_check_20261017.json`, SHA-256 `0e983c31338c00536dfdbe1250c1ea57e5489536d7015391288b80517adbd406`. One-shot attempt: `reports/taiji_n1_hard_neighbor_v8_range_check_20261017_attempt.json`, SHA-256 `c63ca8997a6263e32635d1b8a944b0047521c1a1026c1601ba71e25d26e61e81`. The runner recorded 180 sacrifice queries, CPU 36.671875/180 seconds, zero parameter/optimizer updates, unchanged sources and code, and `can_promote=false`.

## Endpoint readings

The gate counts only lookup, revision and transition errors (15 items). Unknown errors are shown separately and do not enter the threshold.

| Seed | Arm | Lookup | Revision | Transition | Unknown | Addressable errors / 15 | Usable |
|---:|---|---:|---:|---:|---:|---:|---|
| 11 | position | 0 | 0 | 0 | 0 | 0 | No |
| 11 | terminal | 5 | 5 | 5 | 3 | 15 | Yes |
| 11 | cut | 1 | 0 | 3 | 0 | 4 | No |
| 12 | position | 0 | 0 | 2 | 1 | 2 | No |
| 12 | terminal | 5 | 5 | 4 | 4 | 14 | Yes |
| 12 | cut | 0 | 1 | 3 | 0 | 4 | No |
| 13 | position | 5 | 5 | 3 | 2 | 13 | Yes |
| 13 | terminal | 5 | 5 | 3 | 2 | 13 | Yes |
| 13 | cut | 1 | 0 | 3 | 0 | 4 | No |

The four usable endpoints are seed11/terminal, seed12/terminal, seed13/position and seed13/terminal. Thus the position arm contributes one usable endpoint; all three terminal controls pass, and no cut endpoint passes. The executor's gate result is `range_check_passed=false`, `usable_endpoints=4`, `minimum_usable_endpoints=6`.

## Comparison with v7 and evidence boundary

The v7 four-observation d=1 sacrifice report also had 4/9 usable endpoints, and the usable endpoint identities are the same four. This is a descriptive comparison of two different faces, not a causal estimate. Increasing observations from four to eight did not add a usable endpoint for this model family.

No v7 confirm payload was decoded or queried while building v8. The v8 run queried only sacrifice; `confirm_subset_queried=false`, `v8_confirm_model_queries=0`, and neither a v8 confirmation attempt nor report exists. The run does not establish C3, C4, retention, three-organ evidence or promotion.

The frozen protocol prohibits redrawing or rerunning v8 after this failure. Any further range experiment requires a separate approved measurement face. A single proposal, PLAN-N1-102, tests whether twelve observations open the ruler while keeping d=1, endpoint family and all existing gates fixed; it remains unapproved and unexecuted.
