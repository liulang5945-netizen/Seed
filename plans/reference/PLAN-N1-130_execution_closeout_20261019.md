# PLAN-N1-130 execution closeout — native relation-row binding mass

**Status: complete; descriptive activation audit only, no capability promotion.** The one-shot reused the same 15 decodable confirm items on each v13/v14 face and the six frozen position/cut endpoints. It observed all sixteen rows and split the native question alignment by individual target fact row.

## Bound execution and integrity

- Frozen plan SHA-256: `9cc65dde0e99667e6039be9e11a7844a7d4ca15ec038141c1d59aae753aeff03`.
- Zero-query readiness: [report](../../reports/taiji_n1_v13_v14_native_relation_row_binding_mass_readiness_20261019.json), SHA-256 `24e256f5cb9d2c1443bb29921d418c9cd7167208a1ecf8912dd35f343e1d1979`; CPU 2.875 seconds, 158 source identities and six checkpoint hashes, with zero checkpoint/model loads or binding evaluations.
- Synthetic fact-row selection and disjoint-mask preflight passed for lookup (`uses`), revision (`uses` plus `now uses`), and transition (`faces` plus `turns`).
- One-shot result: [report](../../reports/taiji_n1_v13_v14_native_relation_row_binding_mass_result_20261019.json), SHA-256 `edd426464ec01dd961d8d27be0b829f5db2b2d1c5d72223e9b6f2039dece94a1`; 180/180 binding evaluations, 12 model loads, CPU 45.453125 seconds and 48.328125 cumulative seconds. There were zero generated-answer or teacher-forced queries, output tokens, parameter updates, dev queries, or unknown queries. All source/checkpoint hashes remained unchanged; `can_promote=false`.
- The [attempt receipt](../../reports/taiji_n1_v13_v14_native_relation_row_binding_mass_attempt_20261019.json) is SHA-256 `7fc0c25aa2e0119cda0579fd7d9583eccdd81a1733e162f2692c918b6bbc5fb7`.

## Results

Native attention gave each required fact row more mass than its byte share, but the row-level pattern still did not explain whether the model composed the relations correctly. Across faces and arms, `faces` rows averaged about 7.3% of alignment mass on 6.3% of tape bytes; `turns` averaged 11.3% on 6.3%. For cut transitions, `turns` averaged 11.8% on natively incorrect rows versus 9.4% on correct rows. The model was attending to the turn fact; more turn-row mass did not imply the correct direction operation.

Under N1-128's target-only binding mask, the only errors remained seed13/cut's two transition items on each face. Those rows still assigned measurable mass to both target relations; the task facts were present in the native alignment distribution. The row-mass audits therefore do not support a simple “missing fact row” explanation. Their small per-condition cohorts are descriptive and not a causal test.

## Next step

The single continuation is [PLAN-N1-131](PLAN-N1-131_transition_operator_error_attribution_draft_20261019.md): classify saved transition outputs against the observed base direction and turn relation to determine whether errors use the inverse turn, omit the turn, or emit another direction. This is an offline report over already-consumed inputs and outputs; it requires no model/checkpoint load, query, or training.
