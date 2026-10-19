# PLAN-N1-126 execution closeout — oracle source-pruned saved-error diagnostic

**Status: complete; diagnostic only, no capability promotion.** The approved intervention queried the same 15 decodable confirm items on each consumed face with only each target entity's evidence rows retained. It covered the six N1-120 position/cut endpoints and completed all 180 free-generation queries in 50.515625 CPU seconds; cumulative readiness plus query CPU was 53.484375/600 seconds. There were 12 model loads, zero parameter updates, zero dev/unknown queries, and no new face.

## Bound sources and execution

- Approved contract identity at execution: SHA-256 `24740d08e0d61fa0ccf77530f309d47be42b84af07349bd4d4217e9009b3b614`.
- Initial zero-query readiness: [report](../../reports/taiji_n1_v13_v14_source_pruned_readiness_20261019.json), SHA-256 `d7b4c267c052aad09120891ec4983183ff01fdd6d8208dfee6347e66e50387ba`; CPU 2.96875 seconds, 36 checkpoint hashes, 18 independent bundle restores, and zero loads/queries/updates.
- One-shot result: [report](../../reports/taiji_n1_v13_v14_source_pruned_result_20261019.json), SHA-256 `12c4c582c682b71a6a52185dfc855bdc7b165d12fcb5082a193f304f8a980102`; status complete, 180/180 queries, source hashes unchanged at execution, `can_promote=false`. The attempt receipt is SHA-256 `0f31b3eb76a921af00c32186dfedff3c37016ff566df423ada73368099578270`.

## Result

For v13 position, exact free outputs rose from 27/45 on the saved full-context rows to 45/45 after target-row pruning; v13 cut rose from 22/45 to 43/45. For v14 position, exact outputs rose from 25/45 to 45/45; v14 cut rose from 23/45 to 43/45. The pruned position condition was exact on all 15 lookup, all 15 revision, and all 15 transition items in both faces. Cut missed two transitions on each face. There were no source-pruned position/cut losses versus their saved native outputs and no free-versus-teacher exactness disagreements.

The intervention selects target rows with gold entity identity and shortens the input tape, so these gains are an oracle upper bound that combines reduced distractor load with changed recurrent context. They do not establish autonomous retrieval, isolate a source-attention cause, or support capability promotion.

## Source identity note

At execution, readiness and result bind plan SHA-256 `24740d08e0d61fa0ccf77530f309d47be42b84af07349bd4d4217e9009b3b614` and runner SHA-256 `6e0192d6e69da0a53a5a9b6da6b58dc9b7d97508db2236af8898c3e63e28f982`; the result reports source hashes unchanged across the one-shot run. After the result was written, the local plan and runner were amended with clarifying context and a source-revalidation path. The current plan hash is `ed843b5af3cf23b5034905c84296aca77f6b24cc7050c08760a8432322b14eef`, and the current runner hash is `8baebe3c826ff433b2bad55cce0a721356ee9e37d5e4ceaa57038e270d8e5d09`. The saved rows and execution identity are preserved, but the execution-time runner bytes are not present in the current worktree for byte-for-byte replay. This limits replayability; it does not change the stored outputs or their 180-query accounting.

## Next step

The single continuation is [PLAN-N1-127](PLAN-N1-127_full_context_source_attention_mask_diagnostic_draft_20261019.md): repeat the same 180 consumed-item/checkpoint comparisons while keeping every observation in the recurrent context and masking only decoder attention to non-target source rows. This tests whether N1-126 gains came from source access or the shorter context; it remains a zero-training oracle diagnostic and makes no capability claim.
