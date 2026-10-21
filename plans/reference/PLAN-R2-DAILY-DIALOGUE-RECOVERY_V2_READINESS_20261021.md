# R2 dialogue recovery v2 readiness and execution handoff

Date: 2026-10-21. This addendum records results from the durable recovery contract in `PLAN-R2-DAILY-DIALOGUE-RECOVERY_20261021.md` without changing its source-bound content.

## Confirmed readiness

All three isolated v2 preparations completed and their zero full-turn envelope and assistant-calibration canary restored successfully in fresh Python processes. Each preserved its parent zero SHA, and all probes wrote zero formal training updates, zero dev/reference queries, and `can_promote=false`.

| Arm | Prepare CPU seconds | Episode reserve | Available memory | Two CPU samples | Fresh process restores |
|---|---:|---:|---:|---|---|
| `full_turn_1m` | 125.90625 | 1320 seconds | 28.167 GiB | 3.7%, 2.0% | 2/2 |
| `warm_start` | 45.828125 | 240 seconds | 28.078 GiB | 9.4%, 8.7% | 2/2 |
| `fresh_same_capacity` | 44.859375 | 240 seconds | 28.203 GiB | 6.2%, 4.7% | 2/2 |

The measured longest full-turn target contains 14,536 bytes; the assistant calibration maximum contains 489 bytes. Each v2 readiness binds the corresponding probe report hashes and the zero checkpoint hashes. The earlier v1 readiness also passed zero save/restore, but it used an inadequate fixed 120-second episode reserve and is superseded; do not train from v1.

## Budget and concurrency gate

Three CPUs samples were low, and memory remained above 28 GiB during preparation. The user requested matched comparisons be run concurrently when resources permit. The new training runs may therefore start in parallel. Each checks two fresh system CPU samples and at least 8 GiB available memory immediately before training, checks the same memory floor while training, and uses its arm-specific reserve before each episode and checkpoint. The 1M arm stops with at least 1,320 CPU seconds reserved; both 300K arms stop with at least 240 CPU seconds reserved.

Original per-arm hard caps remain 240,000 CPU seconds for `full_turn_1m` and 120,000 for each 300K comparison. The previous interrupted CPU usage remains unknown and is not set to zero; this execution uses the separately authorized v2 restart allocation. If a run is interrupted, use only `--resume` on its own v2 arm directory. Do not resume a v1 attempt.

## Evidence paths

- Readiness: `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_{full_turn_1m,warm_start,fresh_same_capacity}_readiness.json`.
- Zero envelopes and receipts: `data/checkpoints/r2_daily_dialogue_recovery_v2_20261021/{full_turn_1m,warm_start,fresh_same_capacity}/`.
- Longest-episode probes: `reports/taiji_r2_daily_dialogue_recovery_20261021_*_worst_*_episode_probe.json`.
- Code review and delivery record: `plans/reference/delivery-runs/20261021-n-mainline/artifacts/07-recovery-review.md` and `board.md`.

## Next step

The three `--execute` arms started concurrently at 2026-10-21 19:06 Asia/Shanghai. Latest durable checkpoint receipts are full_turn_1m=704/30,096 episodes and 316,821 target bytes at 2,358.671875 CPU seconds; warm_start=992/30,096 and 458,368 bytes at 2,146.515625 CPU seconds; fresh_same_capacity=992/30,096 and 458,368 bytes at 2,145.046875 CPU seconds. The first full_turn_1m checkpoint and the initial 32-episode checkpoints for both comparison arms passed independent restore. All three arms remain in training, with no dev/reference queries. Continue monitoring their immutable progress/checkpoint receipts until each completes 30,096 full-turn episodes and 525 assistant calibration episodes, then independently restore final bundles. Keep `can_promote=false`; do not access the reserved 39-family evaluation set until all three final restore gates pass and the evaluation contract is reviewed.
