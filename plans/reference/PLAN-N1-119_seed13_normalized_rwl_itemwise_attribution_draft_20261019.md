# PLAN-N1-119：N1-117/N1-118 position itemwise attribution

**Status:** authorized read-only follow-up under the user's standing continuation instruction. This plan analyzes only already-consumed training evidence. It does not open a new evaluation surface or change any gate.

## Question

PLAN-N1-118 changed the coupled reader/writer/local update rule from an absolute shared radius to a radius normalized by each group's relative cap. The training batch completed, but seed13's frozen joint-four result remained 24/48 original and 23/48 renamed, the same as N1-117. Identify which transition parents and conditions changed between those two fixed endpoints, and how those changes progressed across the existing milestones.

## Frozen sources

- N1-117 training report: `reports/taiji_n1_coupled_rwl_length_balanced_r1_20261018.json`, SHA-256 `1d4f43519790a1e36cb7dfbc580506d697815dccbdecb95fd576ec3a7bc2e391`.
- N1-118 training report: `reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018.json`, SHA-256 `5ceabffadc7d62e2aa7be3996847c92713fe9cce604c0f10e463ad97fbe8217e`.
- N1-118 corrected read-only receipt validation: `reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018_receipt_revalidation.json`, SHA-256 `7fa1007bda4d9099b0778789057fb59eaf5d72621787a5f9d182db30650cbd24`.
- The six position shards are bound by their compressed SHA-256 values:

| Seed | N1-117 shard SHA-256 | N1-118 shard SHA-256 |
|---|---|---|
| 11 | `9aae12b5d707b664c6776d3fabc182b15599e99ca6506a9fc9ef1d615041a2ff` | `231370ccfaeac1b4753692c19f293344173d3a780dd3b6f908f0d1dfb440932f` |
| 12 | `5ce2fd970ba0da2f26d77235f197e6c68d6b9f3fc537d3789c0fdfa3e5a4fecb` | `b7e7504ecc71c580018276ce0b122b275e713bcfcefa9f0e399a6f30fbda3cf3` |
| 13 | `58481413c35da22b6815269e05eca51a68c36a8159a3dccac406550cc9ab8cfa` | `c45c3ccbec502d88c026ddea3fabcec8032a273d12ec8f1f6fee6cd5210481be` |

## Analysis

- For each seed, original/renamed view, and existing milestone (0, 6,144, 12,288, 18,432, 24,576), recompute each of the four transition-condition exact counts and the per-parent joint-four count from the stored rows.
- At the final milestone, classify each parent/view as correct in both runs, N1-117 only, N1-118 only, or neither. Preserve each condition's separate rescue and regression counts; do not infer causality from the paired comparison.
- Summarize the existing recorded QA loss and RWL update scales by seed and cycle as context for the itemwise changes. Report only statistics already present in the sealed shards.
- Reconcile recomputed summaries with both source receipts and stop if any shard, milestone, parent identity, or anchor differs.

## Limits and output

- CPU ceiling: 60 seconds for local parsing and comparison.
- Read existing N1-117/N1-118 position shards only. Load no model or checkpoint; perform no model query, parameter/optimizer update, training, or data generation.
- Do not read or query dev, holdout, confirmation, or sacrifice data. Do not alter training gates or make capability/promotion claims.
- Write a new report once at `reports/taiji_n1_n117_n118_position_itemwise_attribution_20261019.json`; retain all source reports and shards unchanged.

## Decision use

This attribution is descriptive. It can identify a specific training behavior for a later proposal, but it does not authorize a new recipe, checkpoint change, or confirmation query.
