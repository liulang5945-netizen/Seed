# PLAN-N1-131 — saved transition-operator error attribution

**Status: authorized for one-shot execution on 2026-10-19 under the user's standing authority for normal local N1 diagnostics.** N1-128 retained the same two transition errors in seed13/cut after oracle target-row binding. N1-130 found that the native binder assigned substantial mass to both `faces` and `turns` facts, including on wrong outputs. This offline audit classifies existing outputs by the transition operation they express.

## Question

Across native, N1-126 source-pruned, N1-127 decoder-masked, and N1-128 question-binding-masked outputs, do transition errors apply the opposite turn, omit the turn, or produce another direction?

## Frozen inputs and calculation

- Use only the five transition confirm items per face from the already-consumed v13/v14 manifests and saved output rows for seeds 11–13, position/cut, across the four existing conditions.
- Offline, extract the target entity's `faces` direction and `turns` relation, compute the expected direction under the stated cardinal rotation, and classify each saved output as expected, inverse turn, unchanged base direction, another cardinal direction, malformed/non-direction, or expected text with a non-boundary stop.
- Keep exact item/seed/arm/condition rows and source hashes. Do not load models/checkpoints, query, train, or access dev/unknown/new data.

## One-shot handling

Run a read-only source binding check, calculate the attribution report once, and verify all source hashes remain unchanged. This is a descriptive audit only; it does not alter model results, gates, or capability status. `can_promote=false`.

## Proposed outputs

- Analyzer: `scripts/training/analyze_taiji_n1_transition_operator_error_attribution.py`.
- Report: `reports/taiji_n1_transition_operator_error_attribution_20261019.json`.
- Closeout: `plans/reference/PLAN-N1-131_execution_closeout_20261019.md`.
