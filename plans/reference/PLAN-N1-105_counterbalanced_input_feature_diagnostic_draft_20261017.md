# PLAN-N1-105 draft · counterbalanced input-feature diagnostic · 2026-10-17

**Status: proposal only; not approved, not frozen, and not executable.** This proposal changes the measurement face and requires an owner decision before implementation or model queries. It makes no training, gate, capability, or promotion claim.

## Question

Does the position arm's v9 lookup/revision deficit persist when target-row position is crossed with material-answer length within matched item templates, rather than being tied to the construction order and individual item?

## Proposed one-shot diagnostic

- Reuse the same nine frozen v9 endpoints and three arms (`position`, `cut`, `terminal`); do not train or update checkpoints.
- Generate one deterministic diagnostic face with two matched base templates per kind. For lookup, cross all 12 feasible first-target-row positions with answer lengths 6 and 7 on each template (48 items). For revision, cross all 11 feasible first-target-row positions with both answer lengths on each template (44 items). For transition, use both templates at each of the 11 feasible first-row positions (22 items). Add 10 distinct unknown items with no target row, for 124 items total.
- Keep each matched template's entity and question fixed across its position/length variants, use isolated stateless queries, and preserve the target meaning. Construction checks must verify exact cell counts, row bounds, fresh names/material values against the existing exclusion registry, and no cross-cell leakage before any checkpoint is loaded.
- After construction and checkpoint-identity/recovery preflight, make exactly one diagnostic query per item and endpoint: 1,116 total queries. Independent CPU ceiling: 10 minutes including construction, preflight, recovery, and evaluation.
- Report per-seed/per-arm exact counts and paired position/length contrasts. No threshold is added or altered. Do not recompute PLAN-N1-94 gates, query dev or any existing confirmation face, or treat the diagnostic as a new unseen-capability confirmation. Preserve `can_promote=false`.

## Interpretation boundary

This diagnostic can separate the construction aliasing seen in v9 from within-template position and answer-length associations. It cannot establish a causal mechanism beyond the paired synthetic templates, change a frozen gate, demonstrate general language ability, or authorize promotion. Any subsequent training change or capability-confirmation face requires a separate decision.

## Decision required

Approve or decline this new diagnostic face before implementation. If approved, freeze the exact generator, exclusion registry, query order, checkpoint identities, recovery procedure, report schema, and one-shot execution guard before creating the face.
