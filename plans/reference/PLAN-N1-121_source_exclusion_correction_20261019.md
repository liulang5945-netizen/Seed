# PLAN-N1-121 source-exclusion correction — N1-120 material map

**Status:** correction to the pending PLAN-N1-121 approval request. It strengthens input disjointness only; the v11 face size, seed, range/confirmation gates, query count, and one-shot rule remain unchanged.

## Finding

The first N1-121 source-readiness report excluded the v3 train values and consumed v5/v6, v7/v8 sacrifice, v9, and v10 values, but it did not include the 288 seven-character outputs created by the N1-110/N1-120 length map. Those values are part of the candidate training inputs and must also be excluded from the confirmation face.

## Required correction

- Bind `plans/manifests/n1_hard_neighbor_length_balanced_schedule_v1_20261018.json`, SHA-256 `b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c`.
- Recompute and require the `value_map_sha256` `dba1d9eb8fec6d60e6b1ceca496bce3b3386b05d1539ad88696177db0b9e9ba8`; require exactly 288 injective seven-character outputs and verify every source key and mapped output is excluded from the v11 face.
- Include the manifest and value-map digest in the builder source binding and the corrected source-readiness report.
- Stop if the manifest or mapping differs, if any mapped output overlaps a generated answer or observation value, or if seed111 face artifacts already exist.

The original source-readiness report remains preserved as superseded evidence. This correction neither generates a face nor authorizes range or confirmation queries. A revised approval request must link this addendum and the corrected readiness report.
