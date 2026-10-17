# EXP-N1-109 · train/diagnostic answer-length coverage · 2026-10-17

## Result

The read-only audit validated all 3,072 entries in the frozen PLAN-N1-100 train-derived hard schedule and all 248 items in the consumed v10 diagnostic face in 0.015625 CPU seconds. Report: `reports/taiji_n1_train_v10_answer_length_coverage_20261017.json`, SHA-256 `43d2e75989d42571a36ff7038b137c3d03c7ff764aa95a2990f54344564d4d9b`. Both inputs matched their frozen hashes before and after. Every nested schedule `item` and `second_question` was `split=train`; the schedule records zero dev/holdout queries and zero parameter updates. The audit loaded no model/checkpoint, made no query, and recomputed no gate.

| Source | Lookup answer lengths | Revision answer lengths |
|---|---|---|
| PLAN-N1-100 schedule `item.answer` | 768/768 at length 6 | 768/768 at length 6 |
| v10 diagnostic face | 48 at length 6; 48 at length 7 | 44 at length 6; 44 at length 7 |

The schedule's lookup `second_question.answer` values were also 768/768 length 6. Unknown examples' literal `unknown` answer has seven characters, but it is an abstention label rather than a known material answer; their paired `second_question.answer` values were length 6. Transition direction answers have lengths 4 or 5 and are outside the material-answer comparison.

Therefore, seven-character known lookup/revision answers are absent from the run100 hard schedule and present in the v10 diagnostic. This is a concrete exposure gap that could explain the isolated seed12 one-character near misses or seed11 abstention on seven-character lookup. It cannot explain seed13's failures on both lengths (edit distances 2–7), so length exposure alone is not an adequate account.

## Interpretation boundary

This is a train-schedule/diagnostic-face coverage comparison. It does not establish that length caused any error, validate a new training objective, change any frozen gate, or support capability promotion. The v9 four-gate failure and `can_promote=false` remain unchanged.

## Provenance note

The write-once report file was already present when the final CLI execution check began. Its plan, schedule, face, analyzer, and contract-test hashes match the frozen values; the report records complete 3,072/248 input coverage, equal before/after hashes, and zero model/checkpoint/query/update operations. I did not overwrite or rerun it. The report SHA and contents were independently inspected before recording this adjudication.
