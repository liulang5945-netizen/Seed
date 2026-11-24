"""Lifecycle-training consistency regression test (D-20261124-443/444 lesson).

Checks:
1. A ``retired`` cohort has its drive masked in the graph forward pass (states are all zero).
2. The trainable-range guard rejects training over non-mature units and accepts mature ones.
3. After restoring ``mature`` the cohort drive returns (non-zero states).

Run: python scripts/tests/test_lifecycle_training_consistency_20261124.py
Exit code 0 => PASS.
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from scripts.training.diag_p0ci_20261123 import mask_fn, run_states
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import SELECTIVE_LIFECYCLE

SEED = 20261120
MATURE_ID = SELECTIVE_LIFECYCLE.index("mature")
IMMATURE_ID = SELECTIVE_LIFECYCLE.index("immature")
RETIRED_ID = SELECTIVE_LIFECYCLE.index("retired")
# the forward pass keeps only these two phases active (drive masked otherwise)
ACTIVE_LIFECYCLE = {MATURE_ID, IMMATURE_ID}


def trainable_range_ok(units, lo: int, hi: int) -> bool:
    """Training is only valid over units whose drive is active (immature or mature)."""
    tags = units.unit_lifecycle[lo:hi].tolist()
    return all(tag in ACTIVE_LIFECYCLE for tag in tags)


def main() -> int:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    subj = Subject72(SEED)
    subj.set_masks(True)
    subj.graph.grow_units_neutral_(2)
    subj.n_units = 96
    subj.graph.selective_units.attach_extra_(8, seed=SEED)
    ticks = torch.rand(8, 14, 48)
    extra = torch.rand(8, 14, 8)
    mask = mask_fn(96, 72, 96)

    units = subj.graph.selective_units
    subj.train_mode()
    with torch.no_grad():
        z_mature, a_mature = run_states(subj, ticks, mask, extra=extra)
    if not trainable_range_ok(units, 72, 96):
        print("FAIL: freshly grown cohort is not trainable (immature should be active)")
        return 1

    units.set_unit_lifecycle_(range(72, 96), "retired")
    if trainable_range_ok(units, 72, 96):
        print("FAIL: guard accepted a retired cohort")
        return 1
    with torch.no_grad():
        z_ret, a_ret = run_states(subj, ticks, mask, extra=extra)
    if float(z_ret[:, 72:96].abs().max()) > 1e-6 or float(a_ret[:, 72:96].abs().max()) > 1e-6:
        print("FAIL: retired cohort still has non-zero drive (training would see features)")
        return 1

    units.set_unit_lifecycle_(range(72, 96), "mature")
    with torch.no_grad():
        z_back, a_back = run_states(subj, ticks, mask, extra=extra)
    if float(z_back[:, 72:96].abs().max()) < 1e-6 and float(a_back[:, 72:96].abs().max()) < 1e-6:
        print("FAIL: drive did not return after restoring mature")
        return 1
    if not trainable_range_ok(units, 72, 96):
        print("FAIL: guard rejected a restored mature cohort")
        return 1

    print(
        "PASS: retired drive masked (|z|=0), guard rejects retired / accepts mature, "
        f"drive restored (|z|max={float(z_back[:, 72:96].abs().max()):.4f}, lifecycle ids {MATURE_ID}/{RETIRED_ID})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
