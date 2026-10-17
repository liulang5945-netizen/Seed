# EXP-N1-103 · v9 item-level error attribution · 2026-10-17

## Result

The read-only attribution completed in 0.5 CPU seconds from the sealed v9 confirmation report. It decoded the 180 `hard_neighbor_v9` rows, skipped all 180 `prior_family_descriptive` row payloads after reading their `family` labels, and made zero model/checkpoint loads, queries or updates. It did not recompute the four gates. Report: `reports/taiji_n1_hard_neighbor_v9_item_error_attribution_20261017.json`, SHA-256 `293b8fbb73f71cd592014396d38365f0229443dfd51d39c7fe18199084870ead`.

## Position versus cut

Across 60 item pairs, both arms were exact on 34, position alone was exact on 5, cut alone was exact on 10, and both were wrong on 11. Their output bytes were identical on 35/60. The seed split shows the aggregate deficit is concentrated in seed13:

| Seed | Both correct | Position only | Cut only | Both wrong | Output bytes identical |
|---:|---:|---:|---:|---:|---:|
| 11 | 17/20 | 3/20 | 0/20 | 0/20 | 17/20 |
| 12 | 12/20 | 2/20 | 2/20 | 4/20 | 13/20 |
| 13 | 5/20 | 0/20 | 8/20 | 7/20 | 5/20 |

The class split is also uneven. Lookup ties at one position-only and one cut-only item. Revision has zero position-only and five cut-only items. Transition favors position four to two. Unknown has zero position-only and two cut-only items. These descriptive cells localize the observed −5/60 position-minus-cut difference; they do not identify its cause.

## Position versus terminal

Across 60 item pairs, both arms were exact on 8, position alone was exact on 31, terminal alone was exact on 2, and both were wrong on 19. Their output bytes were identical on 13/60. Position's terminal advantage appears across all three seeds, but this pairwise description does not override the failed four-gate conjunction.

## Interpretation boundary

PLAN-N1-102 remains negative for C3: the four-gate conjunction failed because `each_position_seed`, `each_position_kind`, and `advantage_cut` were false. The attribution is exploratory and descriptive. It is not causal, is not a capability gain, and does not support C3, C4, retention, three-organ, or promotion claims.

The dominant observed pattern is seed13: on this confirmation subset, position had no item that cut answered correctly while cut had eight such items; five of the eight were revision items. The result motivates a proposed read-only input-feature audit of the already-consumed v9 confirmation items. That proposal would inspect question/observation features from the v9 manifest, which PLAN-N1-103 did not use; it remains a separate scope decision in PLAN-N1-104.
