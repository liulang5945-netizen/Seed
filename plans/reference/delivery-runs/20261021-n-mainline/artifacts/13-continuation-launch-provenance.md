# Continuation Launch Provenance and Recovery Update

Updated: 2026-10-23 09:43 Asia/Shanghai. This is a dated execution note; live
progress, checkpoint receipts, process handles, and the aggregate ledger take
precedence over this snapshot.

## Decision

Resume each arm from its latest durable receipt. Start recovery commands one at
a time because concurrent recovery attempts contended on the shared aggregate
CPU ledger. Once each arm has recorded a durable `continuation_resumed` event,
the three training arms may run concurrently under their existing CPU, memory,
and projection guards.

The failed `full_turn_1m` recovery attempt exited while reconciling the shared
ledger, before it wrote a new training event. Its checkpoint and previous
progress remain intact. A conservative 252-second reservation for
`fresh_same_capacity` remains included in the ledger and must be settled by the
existing resume flow; it is not discarded or credited back.
At the 09:43 snapshot, the `warm_start` and `fresh_same_capacity` recovery
commands had not returned and had not yet written a new durable resume event;
their startup must be verified from progress and ledger evidence.

## Budget and training evidence

The last durable pretraining progress was 17,019 / 24,607 / 24,827 episodes for
`full_turn_1m` / `warm_start` / `fresh_same_capacity`, respectively, at about
09:29 local. There are no final result or SFT endpoint files yet. The old
process handles are unavailable.

The latest ledger snapshot records 4,141,968.470 CPU seconds in charges and a
252-second active reservation, leaving 157,779.530 seconds under the approved
4,300,000-second cumulative ceiling. A read-only target-update projection
estimated 137,057.657 CPU seconds remaining, with all three individual arm
ceilings feasible. The restart runner will repeat its live budget and resource
gates before training work begins.

## Provenance check added

`validate_taiji_r2_continuation_launches.py` adds a metadata-only gate to
`prepare_taiji_r2_continuation_endpoints.py`. For each arm, it verifies every
target-update resume manifest against:

- the v1.3 source revision, frozen extension contract, boot audit, and reviewed
  architecture report;
- the exact bound continuation, boot-accounting, target-update, and checkpoint
  recovery source files and tests;
- the arm's extension attempt, approved per-arm and aggregate ceilings, and
  internally consistent feasible target-update forecast;
- a durable progress receipt matching the launch-time checkpoint, plus the
  checkpoint bundle and receipt files on disk.

The live read-only check found two manifests per arm (the original launch and
the latest resume attempt) and verified 22 unique source and audit files,
including the endpoint and preflight validators with their contract tests. This
records declared launch provenance and saved starting checkpoints; it does not
prove final completion, model capability, or independent source review.

## Verification and remaining gates

The focused suite passed 20 tests: seven launch-provenance contract cases, eight
endpoint-validator cases, and five endpoint-preflight cases. Ruff passed for
the changed Python files. The provenance check read only manifests, progress,
receipts, bundles, and bound source files; it issued no generation, dev, or
reference queries.

Still required: complete all three training endpoints; reconcile any active
reservation and verify unique completion events; publish and validate the
endpoint manifest; finish the extension evaluation adapter and its final
source review; then run the frozen 156-generation W5b comparison and fixed
human review. Engineering validation does not establish dialogue ability or
promotion readiness.

## Next action

Confirm the outstanding recovery launches' outcomes, then restart only the
first arm that remains stopped; record its durable resume event before starting
the next recovery command.
