# PLAN-N1-123 execution closeout — v13 long-context range and confirmation

**Status: complete; the frozen confirmation gates failed, so the candidate is not promoted.** The approved seed-113, sixteen-observation, `d=1` face was built once. Its 180-query sacrifice stage passed the unchanged range gate at 8/9 endpoints, releasing the one permitted 360-query confirmation stage. The full run used 540 queries and 118.046875/600 CPU seconds. No training, dev query, retry, redraw, or promotion occurred.

## Bound execution

- Approved plan: [PLAN-N1-123](PLAN-N1-123_v13_long_context_range_probe_draft_20261019.md), SHA-256 `421daf945e87af1c85d1a0b135d124b83a5f3c404f619c537f289200d34962eb`.
- Source exclusions: [manifest](../manifests/n1_v13_after_n122_source_exclusions_20261019.json), SHA-256 `71335c2d8185d645df6c85de5077aab2191de82be1ea7a93be9d0d4cae337776`; 6,048 entity names and 4,848 answer/observation word values, including the full v11/v12 faces and the 16 N1-95 scratch faces.
- Zero-query readiness: [report](../../reports/taiji_n1_v13_after_n122_confirmation_readiness_20261019.json), SHA-256 `427882d582f3fc963ce4f82ba2f638d2f92729882772cc28ca389fa1c0ad1d14`; all 36 frozen checkpoint hashes matched, the 18/18 independent full-bundle restores were reused, and model/checkpoint loads, queries, and updates were zero. CPU was 3.25 seconds.
- Frozen face: [seed-113 manifest](../manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019.json), SHA-256 `b2fdc3f4877c08921571d2cb8965c24f653c1e0b9984e78e35695f13f34c657d`; 40 items, 10 per class, 16 observations per item, 20 sacrifice plus 20 confirm, and `d=1`. [Face checks](../../reports/taiji_n1_v13_after_n122_face_checks_20261019.json) passed; construction used 0.015625 CPU seconds with zero model queries or updates.
- Face attempt: [one-shot receipt](../manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019_attempt.json), SHA-256 `0b067954c4cdc774aea0d890b3b85b5ca753ab591c49599302fa5ef2587cb1ef`.
- Range: [report](../../reports/taiji_n1_v13_after_n122_range_20261019.json), SHA-256 `e6cce51b74ca1ae43a2dcd49a1a2201d7508a83474da2c9fef2eb732be9bd617`; all nine candidate endpoints completed the 180 sacrifice queries, 8/9 met the minimum six errors, and the stage used 39.40625 CPU seconds. The preregistered range gate passed. Its [one-shot receipt](../../reports/taiji_n1_v13_after_n122_range_20261019_attempt.json) is SHA-256 `3519aa142dd888bfc55b3b29962421653139c7b30fea03781d9cb7738754d0ef`.
- Confirmation: [report](../../reports/taiji_n1_v13_after_n122_confirmation_20261019.json), SHA-256 `cc6d34dfb95b08eee0444fae21e9aa7730eff4443e4a5c4c07c8d0915d72e2f2`; all 360 queries across the nine baselines and nine candidates completed in 75.375 CPU seconds. Source hashes remained unchanged. Its [one-shot receipt](../../reports/taiji_n1_v13_after_n122_confirmation_20261019_attempt.json) is SHA-256 `23d524f662fb840a8351d99b5b2c1a13c31d5f7bce882c15ca29e1afc8d928e5`.
- Read-only itemwise analysis: [attribution report](../../reports/taiji_n1_v13_after_n122_itemwise_attribution_20261019.json), SHA-256 `e813416cfb6b320ada608351ce0cdd70749c1e019ee7ce6b9c71a49c33c8b8d7`; 0 model loads, queries, or updates and 0.0 CPU seconds at the process clock's recorded resolution.

The targeted eight PLAN-N1-123 contract tests passed. Ruff, Black, and `py_compile` passed for the v13 builder/evaluator and the offline attribution script. Pytest used a dedicated workspace-local temporary directory because the sandbox's default Windows temp path rejected directory creation.

## Result against the unchanged gates

The candidate position cohort scored 41/60 exact, versus 36/60 for cut and 18/60 for terminal. Its advantage was +23 over terminal but only +5 over cut, below the required +12/60. Each seed reached the 10/20 position floor (15, 13, 13), but the per-seed/per-kind gate failed: seed 12 lookup scored 1/5 and seed 13 transition scored 1/5. Therefore the candidate's four-gate conjunction failed. The PLAN-N1-100 baseline also failed the full conjunction (position 36/60; cut 34/60; terminal 12/60). `can_promote=false`.

Across the candidate cohort, position/cut exact counts by class were lookup 7/15 versus 4/15, revision 15/15 versus 15/15, transition 5/15 versus 3/15, and unknown 14/15 versus 14/15. These are descriptive results on the consumed face; they do not establish an observation-count causal effect or generalize to new items.

The offline feature audit found that confirm target rows start only at indices 0–4; the two-row revision/transition evidence spans indices 0–5, while unknown items have no target row. Thus the sixteen-row face stressed distractor interference after early evidence but did not place target evidence throughout the sixteen-position context. This limits claims about late-position retrieval and is the main design issue to resolve before another long-context query.

## Next step

The single proposed continuation is [PLAN-N1-124](PLAN-N1-124_v14_full_context_evidence_position_probe_draft_20261019.md): use the same frozen cohorts, tasks, `d=1` names, answer-length balance, and gates on a new seed-114 sixteen-row face, while deliberately placing lookup evidence from index 0 through 15 and two-row evidence across early, middle, and late positions. It excludes all N1-123 inputs and preserves the same 180-query range / conditional 360-query confirm structure. This is a new measurement face and remains **awaiting approval**; no N1-124 data or queries are authorized by this closeout.
