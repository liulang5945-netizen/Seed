# PLAN-N1-130 — native relation-row binding mass audit

**Status: authorized for one-shot execution on 2026-10-19 under the user's standing authority for normal local N1 diagnostics.** PLAN-N1-129 found modest native target-row alignment enrichment (about 1.15–1.50× tape byte share), but total target mass did not track correctness: position transition mass was essentially equal on correct and wrong rows, and cut transition errors carried more target mass than correct rows. Since each multi-fact task requires two distinct relations, total mass can hide which fact the question encoder binds.

## Question

Does native question alignment allocate the required mass to each individual target fact row, and do row-specific allocations differ between correct and incorrect native outputs or between the source-pruned and full-context oracle outcomes?

## Frozen cohort and measurements

- Reuse the same 15 decodable confirm items per consumed v13/v14 face and the same six N1-120 position/cut endpoints: 180 full-context binding evaluations.
- Load each frozen endpoint, observe the original sixteen rows, and run only native `question_state` plus `bind_question`. Do not read answer bytes, call teacher logits, call the decoder, generate outputs, or update parameters.
- For every question byte, record alignment mass separately for each target fact row and for all distractor rows. Identify rows by the task relation and exact target entity: lookup `uses`; revision stale `uses` versus `now uses`; transition `faces` versus `turns`. Record row byte shares, mass enrichment, argmax-row fraction, and question salience weighting.
- Join the saved native, N1-127 decoder-mask, N1-128 binding-mask, and N1-126 source-pruned exactness flags after measuring activations. No new samples, splits, faces, thresholds, or gates are introduced.

## One-shot handling

Perform a zero-query source/readiness check, then exactly 180 activation evaluations with cumulative CPU≤600 seconds. Preserve partial rows on failure; do not retry. No answer generation, teacher-forced scoring, dev/unknown access, parameter update, or capability promotion is allowed.

This is a descriptive activation audit; row-specific alignment is not autonomous fact retrieval or capability evidence. `can_promote=false`.

## Proposed outputs

- Evaluator: `scripts/training/eval_taiji_n1_v13_v14_native_relation_row_binding_mass.py`.
- Readiness, attempt, and result: `reports/taiji_n1_v13_v14_native_relation_row_binding_mass_readiness_20261019.json`, `reports/taiji_n1_v13_v14_native_relation_row_binding_mass_attempt_20261019.json`, and `reports/taiji_n1_v13_v14_native_relation_row_binding_mass_result_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-130_execution_closeout_20261019.md`.
