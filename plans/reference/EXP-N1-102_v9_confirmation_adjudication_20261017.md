# EXP-N1-102 · v9 confirmation adjudication · 2026-10-17

## Conclusion

The v9 sacrifice range gate passed at exactly 6/9 usable endpoints, so the one authorized confirmation query ran. The run completed 360 comparisons (180 v9 endpoints and 180 prior-family descriptive controls) in 83.0625/600 CPU seconds. All 18 model bundles passed the full-bundle restore checks; source guards remained intact; parameter updates and optimizer steps were zero. The four confirmation gates did not pass as a conjunction. `can_promote=false` and `TRAIN_SHA=None`.

Confirmation report: `reports/taiji_n1_hard_neighbor_v9_confirmation_20261017.json`, SHA-256 `7b9b7e7cccf27a5e9ba6f9e002b1b4df09347fb7264b3827f9eb6cdab44a1802`. One-shot attempt: `reports/taiji_n1_hard_neighbor_v9_confirmation_20261017_attempt.json`, SHA-256 `78e8d9d13e90ba235688a4731385255682ee4075afb08ab94c13b172eda149a3`. The passed range report is SHA-256 `ee229efcbe5c309eaab76705bc38ef5adb0f1d1e47571a459ea217f9af98576c`.

## Frozen gates

| Gate | Frozen line | Instrument result | Outcome |
|---|---:|---|---|
| `each_position_seed` | Every position seed ≥10/20 | seed11 20/20; seed12 14/20; seed13 5/20 | Fail |
| `each_position_kind` | Every position seed/kind ≥2/5 | seed11: lookup 5/5, revision 5/5, transition 5/5, unknown 5/5; seed12: 4/5, 5/5, 1/5, 4/5; seed13: 0/5, 0/5, 1/5, 4/5 | Fail |
| `advantage_terminal` | ≥12/60 | +29/60 | Pass |
| `advantage_cut` | ≥12/60 | −5/60 | Fail |

The gate values above come from the evaluator's `evaluate_gates` result. The prior-family endpoints are descriptive controls only and do not enter the four-gate decision.

## Execution integrity and limits

- v9 confirmation was queried once on the 20-item confirm subset. `v9_confirm_model_queries=360` includes both the nine new endpoints and the nine descriptive controls. The new-family gate denominator is 60 for each aggregate advantage and 20 per position seed.
- `parameter_updates=0`, `optimizer_steps=0`, `sources_unchanged=true`, `can_promote=false`, and `TRAIN_SHA=null`. The v7 and v8 confirmation faces remain unqueried; no v3 dev/holdout data was used.
- The first `--confirm` invocation stopped before writing a confirmation attempt, loading checkpoint weights, or querying a model because the old-family report-hash map indexed a receipt by the wrong key. The receipt lookup was corrected and covered by a regression test. The pre-attempt failure is recorded in `reports/taiji_n1_hard_neighbor_v9_confirmation_preflight_repair_20261017.json`, SHA-256 `e807c144d2ac9e9c21c4af5b98926175d83a785423aa5cce85e06e7b07f64936`. No range rerun occurred.
- The confirmation runner SHA differs from the runner used by the completed range check because of that pre-attempt repair. Both hashes and the repair receipt are bound in the confirmation attempt/report. The repair changed only control receipt-hash collection and provenance binding; score computation and frozen gates were unchanged.

## Interpretation boundary

PLAN-N1-102 is complete and the C3 confirmation conjunction remains unmet. Position beat terminal by 29/60 but trailed cut by 5/60; position seed13 scored 5/20, including zero on lookup and revision. These are measured outcomes, not causal explanations. No further face, training, threshold change or promotion is authorized by this adjudication.

The next proposed work is a read-only, item-level attribution of the already-consumed v9 confirmation rows to determine how seed13 and the position-versus-cut deficit are distributed across items and classes. It would not load models, query a face, train or alter gates. The separate draft PLAN-N1-103 remains unapproved and unexecuted.
