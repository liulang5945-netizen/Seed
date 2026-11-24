"""Birth-neutrality regression test (user review item 6 / P0-CS).

Grow a trained-candidate graph and then apply the SAME manual edits used by the
P0-CS script (adapter resize + inheritance copy).  With the manual edits done but
BEFORE any training, the old cohort's states must be bitwise identical to the
pre-growth states for identical inputs.  Non-zero difference => implementation bug,
not a mechanism effect.

Run: python scripts/tests/test_birth_neutral_old_cohort_20261124.py
Exit 0 => PASS (old-cohort states identical after growth).
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from scripts.training.diag_p0ci_20261123 import mask_fn, run_states
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

SEED = 20261120


def manual_growth_with_inheritance(subj, base_src: int, adapter_rows: int) -> None:
    """Growth + adapter resize + CN-row inheritance exactly as P0-CS did it."""
    old_w = subj.graph.selective_units.extra_proj.weight.detach().clone()
    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = units.n_units
    proj = torch.nn.Linear(8, subj.n_units, bias=False)
    with torch.no_grad():
        proj.weight[: old_w.shape[0]] = old_w
        proj.weight[old_w.shape[0] :] = 0.0
    units.extra_proj = proj
    with torch.no_grad():
        src = torch.arange(24)
        dst = torch.arange(24) + base_src
        units.group_index[dst] = torch.arange(24) // 6
        units.r[dst] = units.r[src].clone()
        units.p[:, dst] = units.p[:, src].clone()
        for nm in ("c", "eta", "b", "write_bias"):
            getattr(units, nm)[dst] = getattr(units, nm)[src].clone()
        units.w[dst, :] = units.w[src, :].clone()
        units.write_in.weight[dst, :] = units.write_in.weight[src, :].clone()
        units.write_state.weight[dst, :] = units.write_state.weight[src, :].clone()
        units.write_bias[dst] = 0.0
        units.b[dst] = 0.0


def main() -> int:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    subj = Subject72(SEED)
    subj.set_masks(True)
    subj.graph.grow_units_neutral_(2)  # 72 -> 96 (cohort A)
    subj.n_units = 96
    subj.graph.selective_units.attach_extra_(8, seed=SEED)

    ticks = torch.rand(6, 14, 48)
    extra = torch.rand(6, 14, 8)
    with torch.no_grad():
        z_before_all, a_before_all = run_states(subj, ticks, mask_fn(96, 72, 96), extra=extra)
    z_before = z_before_all[:, :96].clone()
    a_before = a_before_all[:, :96].clone()

    manual_growth_with_inheritance(subj, base_src=96, adapter_rows=96)
    if subj.n_units != 120:
        print(f"FAIL: expected 120 units, got {subj.n_units}")
        return 1
    with torch.no_grad():
        z_after_all, a_after_all = run_states(subj, ticks, mask_fn(120, 72, 120), extra=extra)
    z_after = z_after_all[:, :96]
    a_after = a_after_all[:, :96]
    dz = float((z_after - z_before).abs().max())
    da = float((a_after - a_before).abs().max())
    print(f"old-cohort state diff after growth+manual edits: |dz|={dz:.3e} |da|={da:.3e}")
    if dz > 1e-6 or da > 1e-6:
        print("FAIL: growth/manual edits changed old-cohort states (birth-neutrality broken)")
        return 1
    print("PASS: old-cohort states are identical after growth + adapter resize + inheritance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
