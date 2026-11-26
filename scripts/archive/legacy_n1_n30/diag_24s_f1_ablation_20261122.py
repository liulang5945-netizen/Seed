"""Diagnostic (zero check): where does the 24-S.F1 correction live?

Post-hoc, non-gated ablation of the F1 representation arm (seed 20261120):
  1. normal held-out revision accuracy after 8 correction interactions;
  2. pre-registered mask-zero lesion (as in F1; known to leave writes intact);
  3. restore base selective_units parameters (corrected head kept);
  4. restore base p24 head parameters (corrected units kept);
  5. zero the p24 slice of the readout state (z/a) before the head;
  6. full revert (units + head) sanity.

Zero check/final; standing authorization. This diagnostic changes no verdict.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_24s_f1_20261122 import (
    BLOCKS,
    FIT,
    FIT_OFFSET,
    SEED,
    base_step,
    eval_base,
    eval_rev,
    make_revision_stream,
)
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

OUT = Path(r"E:\Seed\reports\diag_24s_f1_ablation_20261122.json")


def build_base() -> Subject72:
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in BLOCKS.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(True)
    subj.train_mode()
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=1e-4)
    for k in range(1200):
        b = _make_p24_stream(48, SEED * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        base_step(opt, loss, subj)
    return subj


def train_arm_a(subj, interactions: int = 8):
    for p in subj.graph.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(False)
    for p in subj.phase_heads["p24"].parameters():
        p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    for p in subj.graph.read_gate.parameters():
        p.requires_grad_(True)
    head_opt = torch.optim.Adam(subj.phase_heads["p24"].parameters(), lr=2e-2)
    graph_opt = torch.optim.Adam([p for p in subj.graph.parameters() if p.requires_grad], lr=1e-2)
    for i in range(interactions):
        b = make_revision_stream(96, SEED * 77000 + i)
        subj.graph.eval()
        subj.phase_heads["p24"].train()
        logits = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(logits[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            logits[:, 4], torch.zeros_like(logits[:, 4])
        )
        head_opt.zero_grad(set_to_none=True)
        graph_opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "p24")
        torch.nn.utils.clip_grad_norm_(
            list(subj.phase_heads["p24"].parameters())
            + [p for g in graph_opt.param_groups for p in g["params"]],
            1.0,
        )
        head_opt.step()
        graph_opt.step()


def logits_zero_p24_readout(subj, ticks) -> torch.Tensor:
    mask = subj.masks["p24"].view(1, 1, subj.n_units)
    gates = mask.expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    with torch.no_grad():
        out = subj.graph(
            _channels_seq(ticks),
            gates,
            subj.graph.initial_state(batch_size=ticks.shape[0]),
        )
    z, a = out["next_state"]["z"].clone(), out["next_state"]["a"].clone()
    z[:, 24:48] = 0.0
    a[:, 24:48] = 0.0
    return subj.phase_heads["p24"](torch.cat([z, a], dim=-1))


def acc_from_logits(logits, data) -> float:
    return float((logits[:, :4].argmax(-1) == data["val_labels"]).float().mean())


def main():
    started = time.process_time()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    base = build_base()
    base_eval = eval_base(base)
    probe = make_revision_stream(FIT, SEED + 444 + FIT_OFFSET)
    base_rev = eval_rev(base, probe)

    arm = copy.deepcopy(base)
    train_arm_a(arm, 8)
    normal = eval_rev(arm, probe)

    # 1. pre-registered mask-zero lesion (restored immediately)
    saved = arm.masks["p24"].clone()
    arm.masks["p24"].zero_()
    mask0 = eval_rev(arm, probe)["acc_new"]
    arm.masks["p24"].copy_(saved)

    # 2. zero the p24 slice of the readout state before the head (non-parametric)
    readout_zero = acc_from_logits(logits_zero_p24_readout(arm, probe["ticks"]), probe)

    # 3. restore base head, keep corrected units
    arm.phase_heads["p24"].load_state_dict(copy.deepcopy(base.phase_heads["p24"].state_dict()))
    head_restored = eval_rev(arm, probe)["acc_new"]

    # 4. additionally restore base units => full revert sanity
    arm.graph.selective_units.load_state_dict(
        copy.deepcopy(base.graph.selective_units.state_dict())
    )
    full_revert = eval_rev(arm, probe)["acc_new"]

    # 5. fresh corrected candidate, restore base units but keep corrected head
    arm2 = copy.deepcopy(base)
    train_arm_a(arm2, 8)
    arm2.graph.selective_units.load_state_dict(
        copy.deepcopy(base.graph.selective_units.state_dict())
    )
    units_restored = eval_rev(arm2, probe)["acc_new"]

    payload = {
        "schema": "taiji-u2-diag-24s-f1-ablation-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "base": base_eval,
        "base_revision": base_rev,
        "arm_normal_acc_new": normal["acc_new"],
        "ablation": {
            "mask_zero_lesion": mask0,
            "restore_base_units_keep_head": units_restored,
            "restore_base_head_keep_units": head_restored,
            "zero_p24_readout_state": readout_zero,
            "full_revert_units_and_head": full_revert,
        },
        "interpretation_boundary": "post-hoc diagnostic; F1 G3 gate unchanged; no verdict change",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "authority": "owner:standing-authorization:2026-11-20#recommendation-default",
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
