# EXP-N1-106 · v10 output error attribution · 2026-10-17

## Result

The read-only analyzer classified every one of the 2,232 rows in the consumed PLAN-N1-105 result in 0.03125 CPU seconds. It made zero model/checkpoint loads, queries, updates, or optimizer steps. The write-once report is `reports/taiji_n1_hard_neighbor_v10_output_error_attribution_20261017.json`, SHA-256 `3823b7e6add0d415b3e42413635fe4ab693b2ebfd8718298186c2f9beac5f796`. It binds the PLAN-N1-105 face and result hashes, PLAN-N1-106 hash, analyzer hash, and contract-test hash; input hashes matched before and after. No gate was recomputed and `can_promote=false` remains in force.

| Output class | Count / 2,232 |
|---|---:|
| Exact answer | 981 |
| `unknown` on a known-answer question | 576 |
| Output absent from the item's explicit `uses` material facts | 672 |
| Material copied from a distractor entity | 3 |
| Stale revision material | 0 |
| Other target-entity material | 0 |
| Empty or non-text output | 0 |

The 3 distractor-material copies all occurred on `unknown` items. No lookup or revision error copied a stale revision value or another material from the target/distractor entities. For position seed13, all 96 lookup answers were wrong (36 `unknown`, 60 other strings absent from material facts); all 88 revision answers were wrong (11 `unknown`, 77 other strings absent from material facts). Cut answered all 88 seed13 revisions exactly. Thus the seed13 revision deficit is not explained by emitting the previous observed material on this face.

The seven-character lookup misses also split by seed: position seed11's 12 errors were all `unknown`, while seed12's 12 were all unobserved strings. Seed12's 11 seven-character revision errors were also unobserved strings. Seed13 failed at both answer lengths: lookup length6 had 36 unobserved strings and 12 `unknown` outputs; length7 had 24 and 24. Revision length6 had 33 and 11; length7 had 44 unobserved strings. These are output-level descriptions of the matched templates, not causal evidence.

## Scope limitation

The material-copy taxonomy matches explicit `uses` facts. For `transition` items, the class “absent from the item's explicit `uses` material facts” does not mean absent from every `faces`/`turns` relation token in that item's observations. The report's classification is suitable for lookup/revision material attribution; it does not claim a semantic-token audit of transition outputs. This limitation does not affect the lookup/revision counts above.

## Interpretation boundary

Most errors were abstentions or strings not found among the item's explicit material facts; only three were observed distractor materials, all on unknown items, and no stale revision material was emitted. This can help distinguish copying from failure to produce a supported answer, but it does not establish the model's internal mechanism or generalization. It does not change the v9 four-gate failure, any frozen threshold, or capability status. Any training change or new confirmation face remains a separate decision.
