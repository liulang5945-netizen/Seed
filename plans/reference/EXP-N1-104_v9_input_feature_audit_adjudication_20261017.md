# EXP-N1-104 · v9 confirmation input-feature audit · 2026-10-17

## Disposition

PLAN-N1-104 completed as a descriptive, read-only analysis with a disclosed setup-time protocol deviation. The one-shot report is `reports/taiji_n1_hard_neighbor_v9_input_feature_audit_20261017.json`, SHA-256 `c0cf0ab7163ec62be7f2ffc76f78e573917bbbbf785d6da7a2923400c6ce1c5e`. It analyzed 20 v9 confirmation inputs and 180 v9 outcome rows in 0.5625 CPU seconds. The analysis reader skipped all 20 sacrifice payloads and all 180 prior-family payloads; it loaded no model/checkpoint, made no query or update, generated no face, and did not recompute a gate.

During setup, a whole-line preview of the compact v9 manifest decoded fields from its 20 sacrifice episodes. This violated PLAN-N1-104's no-read boundary. The analysis code was then implemented to skip those payloads, and none of the exposed sacrifice fields were used in the calculation. The report marks `complete_with_protocol_deviation`; these results must not be described as a protocol-clean audit.

## Findings

The 20 confirmation items produced 16 joint feature cells. Lookup and revision each occupy five singleton cells at first target-row positions 1–5. Their answer lengths are 7, 6, 7, 6, and 7 characters in that same order. Transition has five singleton position cells; unknown has one five-item cell with no target row or material-answer length.

The v9 builder source confirms this alignment is deterministic: each kind's confirmation ordinals are 5–9; answer length is chosen by `ordinal % len(ANSWER_LENGTHS)`, while target start is chosen from `ordinal % 5`. Thus position, answer length, and item identity are aliased in the observed lookup/revision rows. The builder's balance check only limits the difference between position counts to one; it does not provide repeated items within position-by-length cells.

For seed13, position is correct on 0/5 lookup items and 0/5 revision items at both answer lengths. Cut is correct on lookup length-6 items 1/2 and length-7 items 0/3; on revision, cut is correct on length-6 items 2/2 and length-7 items 3/3. Each stratum has only two or three items and remains confounded with item identity. This does not identify whether position, answer length, or item difficulty explains the arm difference. The prior PLAN-N1-102 four-gate result remains failed and unchanged (`position` vs `terminal` +29/60; vs `cut` −5/60; `can_promote=false`).

## Provenance

- v9 manifest SHA-256: `d93dd6c6e0d98125ef1c91ed7ad8aec07a46f6621fb597dbffa6e889f29ac60f`
- v9 confirmation report SHA-256: `7b9b7e7cccf27a5e9ba6f9e002b1b4df09347fb7264b3827f9eb6cdab44a1802`
- Frozen PLAN-N1-104 SHA-256: `5943bd3f0f167ca8287a14dd4c7ff56196fda2b70e8d4bc3157c7f9045bacc0a`
- Analysis code SHA-256: `fa658ab9a8de4519cb7e072d8039d0b6fae48d138be80ee32ac494b8dfb3da9d`
- v9 builder SHA-256 (static code inspection only): `90c1c88a7f5cca931611bfbd9a5b348f474b8dae0333baba7ec32d9e9c001293`

This is an exploratory description of already-consumed confirmation results. It does not establish causality, unseen capability, C3/C4, retention, three-organ behavior, or promotion readiness.

## Next step

The single proposed next step is the draft [PLAN-N1-105](PLAN-N1-105_counterbalanced_input_feature_diagnostic_draft_20261017.md): one new, counterbalanced diagnostic face using the same nine frozen endpoints, with target position and material-answer length independently crossed. This changes the measurement face, so the plan remains unapproved and no face has been generated or queried.
