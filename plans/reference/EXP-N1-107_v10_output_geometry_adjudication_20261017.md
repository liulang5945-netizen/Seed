# EXP-N1-107 · v10 output geometry and position stability · 2026-10-17

## Result

The read-only geometry analysis covered all 2,232 consumed v10 outputs and all 180 matched position/length template families in 0.15625 CPU seconds. It loaded no models or checkpoints and made no queries or updates. Report: `reports/taiji_n1_hard_neighbor_v10_output_geometry_20261017.json`, SHA-256 `402d58d663c2359a7c23f69cad635cfe335675572f742be50488382014185675`. The report binds the frozen PLAN-N1-105 face/result and PLAN-N1-107 hashes, plus analyzer and contract-test hashes; all input hashes matched before and after.

Every one of the 180 matched template families produced exactly one unique output across every target-row position. Within a fixed seed, arm, template, task kind, and answer length, moving the target row never changed the emitted string on this face. This is an output-stability observation; it does not prove whether an endpoint did or did not use position internally.

The position-arm error geometry separates into distinct patterns:

- Seed 11's seven-character lookup miss is one full 12-position template that emits `unknown` at every position. The other three seven-character lookup templates are exact at all positions.
- Seed 12's seven-character lookup miss is one full 12-position template whose output is one character edit from the answer at every position. Its seven-character revision miss is one full 11-position template, also one character edit from the answer; those outputs are six characters long. The other corresponding templates are exact.
- Seed 13 misses all eight lookup/revision template-length families. Each family emits one fixed string across positions, with character edit distance 2–7 from the answer; none is an exact or one-edit near miss. For the position arm, lookup is 0/96 and revision 0/88, while cut revision is 88/88.

These results sharpen the PLAN-N1-106 attribution: seed 12 has isolated template-level near-copy errors on seven-character outputs, seed 11 has an isolated template-level abstention, and seed 13 has a broad seed-specific mismatch across every lookup/revision template. The observed outcomes do not vary by target-row position within any matched family.

## Interpretation boundary

The all-position string invariance and edit distances describe only deterministic outputs on the matched synthetic face. They do not prove a causal position mechanism, distinguish initialization from optimization or representation effects, establish generalization, or alter the failed v9 gates. `can_promote=false`; no training, gate recomputation, or promotion is supported by this report.
