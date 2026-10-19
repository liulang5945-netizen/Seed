# PLAN-N1-129 — native question-binding mass audit

**Status: authorized for one-shot execution on 2026-10-19 under the user's standing authority for normal local N1 diagnostics.** PLAN-N1-128 recovered source-pruned exactness while retaining all sixteen input rows, by restricting local question alignment and contextual confidence to gold-selected target rows. Position reached 90/90; cut reached 86/90. This audit measures what the frozen model's native, unmasked question binder already attends to on those same inputs.

## Question

On correct and incorrect native outputs, how much of each question byte's learned local alignment mass falls on the target entity's task-relevant rows versus distractor rows? This is a descriptive internal-activation audit; it does not intervene on generation or measure generalization.

## Frozen cohort and measurements

- Reuse the same 15 decodable confirm items from each consumed v13/v14 face and the same six N1-120 position/cut endpoints: 180 full-context binding evaluations. Do not read `answer` bytes into the model path.
- Load each frozen endpoint, observe all sixteen original rows, build the normal evidence tape, and call the native unmasked `question_state` and `bind_question` path once per item. Do not call teacher logits, `generate_trace`, or the output head.
- For every item and endpoint, record the fraction of local alignment mass on target-row bytes by question byte, target-byte share of the tape, normalized mass enrichment over that share, and the fraction of question bytes whose maximum alignment is on a target row. Join saved native, N1-127, and N1-128 correctness flags only after the model activation is recorded.
- Verify exact tape identity, model/optimizer/history fingerprint, all six checkpoint hashes, both face/report chains, and prior N1-126/127/128 source receipts. No new inputs, faces, splits, thresholds, or capability gates are introduced.

## One-shot handling

Run a zero-query source and code readiness pass first. Then perform the 180 activation evaluations once, with cumulative CPU≤600 seconds including readiness, checkpoint loads, measurement, and final source hashes. Preserve partial rows on failure; do not retry. Zero generated-answer queries, parameter updates, optimizer steps, dev queries, unknown queries, or new data are allowed.

No training, criterion change, or capability promotion is included. Report alignment mass as descriptive evidence only; `can_promote=false`.

## Proposed outputs

- Evaluator: `scripts/training/eval_taiji_n1_v13_v14_native_question_binding_mass.py`.
- Readiness, attempt, and result: `reports/taiji_n1_v13_v14_native_question_binding_mass_readiness_20261019.json`, `reports/taiji_n1_v13_v14_native_question_binding_mass_attempt_20261019.json`, and `reports/taiji_n1_v13_v14_native_question_binding_mass_result_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-129_execution_closeout_20261019.md`.
