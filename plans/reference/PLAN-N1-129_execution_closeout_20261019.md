# PLAN-N1-129 execution closeout — native question-binding mass audit

**Status: complete; descriptive activation audit only, no capability promotion.** The audit reused the 15 decodable confirm items on each consumed v13/v14 face and the six frozen position/cut endpoints. It observed the complete sixteen-row tape and measured the native local question-to-observation alignment and question-byte salience without teacher scoring or answer generation.

## Bound execution and integrity

- Frozen plan SHA-256: `7e35f01322ea6ecf88f1f53f8124d1b21378d3c8ae7f085c0c5ca52e2cce12ae`.
- Zero-query readiness: [report](../../reports/taiji_n1_v13_v14_native_question_binding_mass_readiness_20261019.json), SHA-256 `efed73791e26d97a6b3d91e65db1a758def484b686ce1e254ca4c956f63b743e`; CPU 2.890625 seconds, 152 source identities and six checkpoint hashes, zero model/checkpoint loads or activation reads.
- One-shot result: [report](../../reports/taiji_n1_v13_v14_native_question_binding_mass_result_20261019.json), SHA-256 `a533803ccf39d2f2f023a2e977374a9da4fc4737202fa74fe5b26605aa4bfd9d`; 180/180 binding evaluations, 12 model loads, CPU 40.578125 seconds and 43.46875 cumulative seconds. Generated-answer and teacher-forced queries, output tokens, parameter updates, dev queries, and unknown queries were all zero. Source and checkpoint hashes remained unchanged; `can_promote=false`.
- The [attempt receipt](../../reports/taiji_n1_v13_v14_native_question_binding_mass_attempt_20261019.json) is SHA-256 `070c02572e0efd0ad568ae7153a9c9cb9402b1381cbbe676caede481aa7f64cd`. It records the frozen six-checkpoint cohort and zero generated-answer queries.

## Results

The local alignment puts more mass on the target entity's task-relevant bytes than their share of the tape, but the enrichment is modest: roughly 1.15–1.50× across the three task types and two arms. The target row receives the largest share for only about 12–32% of question-byte positions in aggregate task/arm cells. The learned alignment is therefore not simply ignoring the target, but the total target-row mass is not a reliable discriminator of output correctness.

| Arm | Kind | Target byte share | Mean target alignment mass | Enrichment | Mean question-byte argmax on target |
|---|---|---:|---:|---:|---:|
| position | lookup | 6.4% | 7.6% | 1.20× | 17.7% |
| position | revision | 13.7% | 19.8% | 1.44× | 31.6% |
| position | transition | 12.5% | 15.4% | 1.23× | 28.3% |
| cut | lookup | 6.4% | 7.5% | 1.16× | 15.2% |
| cut | revision | 13.7% | 16.8% | 1.22× | 25.4% |
| cut | transition | 12.5% | 18.8% | 1.50× | 27.4% |

For position transitions, native exact rows had 15.1% mean target mass and native errors 15.3%. For cut transitions, native exact rows had 16.4% and errors 19.1%. Thus the aggregate mass did not separate correct from incorrect answers in the expected direction. N1-128's gold-selected binding mask still raised position to 90/90 and cut to 86/90, so restricting the binding path matters, but summing both relevant rows hides which relation fact each question byte uses. This audit does not identify autonomous source selection or prove a causal explanation.

## Next step

The single continuation is [PLAN-N1-130](PLAN-N1-130_native_relation_row_binding_mass_audit_draft_20261019.md): split target alignment mass by individual fact row—especially `faces` versus `turns` for transition, and stale versus `now uses` for revision—on the same inputs and endpoints. This will show whether correct and incorrect outputs differ in which required relation row is bound, rather than in total target mass. It remains an activation-only, zero-output audit.
