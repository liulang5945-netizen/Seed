# PLAN-N1-119 execution closeout — N1-117/N1-118 position attribution

**Date:** 2026-10-19

The read-only comparison completed in 2.59375 CPU seconds. It read six frozen position shards, recomputed 11,520 four-condition rows across all milestones, and made zero model or checkpoint loads, queries, or updates. The [attribution report](../../reports/taiji_n1_n117_n118_position_itemwise_attribution_20261019.json) is SHA-256 `8564e9ca548010a966a770201bc689d7c1508805e8f8169707824cf910b731e8` and binds both source reports, the N1-118 receipt revalidation, and all six shard hashes.

At the final milestone, seed11 and seed12 remain 48/48 joint-four in both runs and both views. Seed13 remains 24/48 original and 23/48 renamed in both runs, but the correct-parent sets almost completely changed. In the original view, none of N1-117's 24 joint-four parents overlap N1-118's 24. In the renamed view, only one parent overlaps; each run has 22 parents the other misses, and three remain wrong in both.

The per-condition results show a tradeoff rather than a uniform gain. For original names, N1-118 changed the four counts from 39/33/38/33 to 37/37/38/39. For renamed names, they changed from 41/30/41/29 to 37/40/40/37. N1-118 improved direction-opposite and turn-flip counts while reducing the `both` condition; the joint-four gate did not move.

N1-118's seed13 mean QA loss stayed above N1-117 in every cycle: 2.287/1.436/0.977/0.596 versus 2.009/1.309/0.872/0.529. Its normalized shared scale averaged 0.502/0.732/0.912/0.999, with 6,078/5,154/2,080/21 clipped updates per cycle. These measurements suggest early clipping is a useful next target, but they do not establish causality; the scale definitions differ between the two recipes.

No gate, training, or confirmation policy changed. `can_promote=false`.

## Next step

Run a fresh matched nine-endpoint training batch with a constant normalized RWL radius of `2√3`, leaving the model, objectives, data, schedule, update count, and frozen gates unchanged. The larger radius directly tests whether N1-118's frequent early clipping limited seed13 adaptation. Save and independently restore all zero checkpoints before training, keep the 18,000-second CPU ceiling, and query no dev, holdout, or confirmation data. See [PLAN-N1-120](PLAN-N1-120_expanded_normalized_rwl_radius_training_draft_20261019.md).
