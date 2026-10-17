# PLAN-N1-109 · train/diagnostic answer-length coverage audit · 2026-10-17

**Status: frozen for read-only schedule audit.** This checks a train-derived schedule against the already-consumed v10 face to determine whether known-answer length is represented in the existing schedule. It does not modify the schedule or authorize training.

## Question

Do the lookup/revision examples in PLAN-N1-100's frozen train-derived hard schedule expose the model to both six- and seven-character known answers, as used in v10's counterbalanced diagnostic?

## Fixed analysis

- Read only `plans/manifests/n1_hard_neighbor_training_schedule_v1_20261017.json` (3,072 train-derived entries) and the already-consumed v10 face `plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json`. Require their frozen hashes and this plan hash. Do not read the original v3 manifest, any dev/holdout data, model, checkpoint, or output-score report.
- Validate that every schedule `item` and every present `second_question` is marked `split=train`, all 3,072 schedule entries match the frozen summary counts, and all 248 face items match the v10 schema and frozen kind/length design.
- Count answer character lengths by task kind, schedule condition, and source field (`item.answer` versus `second_question.answer`). Separately count diagnostic-face target answer lengths by task kind. Keep unknown's literal `unknown` label separate from known material answers; do not call it material-length exposure.
- Report the train-schedule versus diagnostic-face lookup/revision length support and all denominators. This is a static exposure comparison only; it does not infer that answer length caused any v10 error.
- CPU≤15 seconds. JSON-only analysis; zero model/checkpoint loads, queries, training, parameter updates, dev/holdout access, gate recomputation, or source writes. Refuse to overwrite `reports/taiji_n1_train_v10_answer_length_coverage_20261017.json`.

## Interpretation boundary

This audit can show whether the frozen schedule contains known-answer examples of the diagnostic's lengths. It cannot establish a causal length effect, justify changing a training objective, change a gate, or support capability promotion. Any training-plan change or new confirmation face requires a separate decision.

## Provenance note

A preliminary read-only tally of the train-derived schedule and v10 face found six-character lookup/revision answers in the schedule and both six/seven-character answers in the diagnostic. No data was changed. The frozen analyzer independently verifies the complete allowed input files and reports the full distribution.
