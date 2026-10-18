# PLAN-N1-120：expanded normalized RWL radius training

**Status:** authorized under the user's standing instruction to continue the mainline. This plan changes one training-policy constant, retains the measurement surface and gates, and does not authorize confirmation queries.

## Evidence and hypothesis

- N1-118 completed 221,184 updates with 18/18 zero/final restores. Its normalized shared radius met the frozen projection contract, but seed13 remained 24/48 joint-four original and 23/48 renamed.
- PLAN-N1-119 found no overlap between the 24 original-view joint-four parents solved by N1-117 and the 24 solved by N1-118; only one renamed-view parent overlapped. N1-118 raised direction-opposite and turn-flip counts while reducing the `both` condition.
- N1-118 seed13's normalized shared scale averaged 0.502/0.732/0.912/0.999 across cycles and clipped 6,078/5,154/2,080/21 of 6,144 updates per cycle. QA loss exceeded N1-117 in each cycle. This supports, but does not prove, the hypothesis that the current normalized radius restrains early adaptation.

## Single recipe change

- Set the normalized shared RWL radius to `2√3` for all four cycles instead of `√3`.
- Keep per-group cap normalization, one shared scale across reader/writer/local, and the unchanged independent context/relation caps.
- Keep model architecture, parameter count, initialization, Adam moments, RNG, loss, paired-question objective, data, material values, raw/hard schedule, learning-rate schedule, updates, and scoring unchanged.

## Fixed protocol

- Fresh run83 zero initialization for seeds 11/12/13 × position/terminal/cut; 24,576 updates per endpoint and 221,184 total.
- New write-once checkpoint root: `data/checkpoints/n1_expanded_normalized_rwl_length_balanced_20261019/`.
- New report root: `reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019`.
- First run radius-contract tests, zero-query schedule readiness, full source binding, and nine complete zero-checkpoint bundle saves with independent restore. Stop if any guard fails.
- Zero-preflight CPU ≤300 seconds; batch CPU ≤18,000 seconds. Require all 18 zero/final restore checks to match.

## Frozen gates and isolation

- Preserve the PLAN-N1-110 position training predicate exactly: each seed's original and renamed joint-four score ≥44/48, and renamed transition free-exact ≥40/48.
- No dev, holdout, confirmation, or sacrifice query; do not generate a new confirmation face in this run. If the training conjunction fails, close the candidate without confirmation. If it passes, any new confirmation requires a separate plan.
- `can_promote=false` throughout. Report the training result separately from engineering validity.
- Bind N1-118's original report and corrected receipt revalidation, PLAN-N1-119's attribution report, its six position shards, the schedule, the new code, tests, and the zero-checkpoint hashes.

## Next step

Implement the radius-`2√3` projection variant and isolated runner, validate the schedule and source binding, complete nine zero-checkpoint save/restore checks, then run the fixed batch once. Do not adjust the radius or training budget mid-run.
