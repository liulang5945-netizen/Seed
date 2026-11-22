"""ACT-1b: multimodal perception + action + real-result feedback (action-outcome prediction
and rule revision).

World (deterministic, verifiable): corridor positions 0..4, goal in {0,4}, a door between
1 and 2 with state open/closed/locked.  Actions L/R/PUSH return the true simulator outcome
(MOVE/REACH/BLOCK/STAY/TOGGLE).  Modalities: vision channels carry position/goal one-hots
only; the language channel carries the door-state character embedding only; the action is
a third input.  Predicting BLOCK therefore requires integrating vision + language + action.

Protocol: base 800 steps on simulator rollouts; held-out outcome accuracy; counterfactual
action sensitivity; modality zeroing ablations; then a new word "locked" regime with 8
interactions of real receipts (plus no-feedback and permuted-receipt controls), checking
rule revision and base-rule retention.

Single seed, <=700 CPU, zero check/final. Standing authorization; owner may revise.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import LEGAL_INPUT_CHANNELS

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_act1b_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_act1b_20261123.json"
SEED = 20261120
FIT = 256
FIT_OFFSET = 10_000_000
N_ACTIONS = 3
N_OUTCOMES = 5
TEXT_LO, TEXT_HI = 16, 24
VIS_LO, VIS_HI = 0, 10
ACT_LO = 12
QUERY = 15
_PROJ_GEN = torch.Generator().manual_seed(2026112301)
_PROJ_RAW = torch.randn(3, 8, generator=_PROJ_GEN)
TEXT_PROJ = _PROJ_RAW / _PROJ_RAW.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def outcome(pos: int, goal: int, door: int, action: int) -> tuple[int, int]:
    """Return (outcome_class, new_door) for the deterministic world."""
    if action == 2:  # PUSH
        if pos in (1, 2):
            if door == 2:  # locked: push has no effect
                return 2, door  # BLOCK
            return 4, 0 if door == 1 else 1  # TOGGLE open<->closed
        return 3, door  # STAY
    target = pos - 1 if action == 0 else pos + 1
    crossing = (pos == 1 and action == 1) or (pos == 2 and action == 0)
    if crossing and door != 0:
        return 2, door  # BLOCK
    if target < 0 or target > 4:
        return 3, door  # STAY
    if target == goal:
        return 1, door  # REACH
    return 0, door  # MOVE


def encode(pos: int, goal: int, door: int, action: int) -> torch.Tensor:
    ticks = torch.zeros(14, 48)
    ticks[0, pos] = 1.0
    ticks[0, 5 + goal] = 1.0
    ticks[0, TEXT_LO:TEXT_HI] = TEXT_PROJ[door]
    ticks[1, ACT_LO + action] = 1.0
    ticks[13, QUERY] = 1.0
    return ticks


def make_batch(n: int, seed: int, door_pool: list[int]):
    g = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(n, 14, 48)
    labels = torch.zeros(n, dtype=torch.long)
    meta = {"pos": [], "goal": [], "door": [], "action": []}
    for i in range(n):
        pos = int(torch.randint(0, 5, (1,), generator=g))
        goal = int(torch.randint(0, 2, (1,), generator=g)) * 4
        door = door_pool[int(torch.randint(0, len(door_pool), (1,), generator=g))]
        action = int(torch.randint(0, N_ACTIONS, (1,), generator=g))
        label, _ = outcome(pos, goal, door, action)
        ticks[i] = encode(pos, goal, door, action)
        labels[i] = label
        meta["pos"].append(pos)
        meta["goal"].append(goal)
        meta["door"].append(door)
        meta["action"].append(action)
    return ticks, labels, meta


DOOR_DEPENDENT = ((1, 1), (2, 0), (1, 2), (2, 2))


def make_dependent_locked_batch(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(n, 14, 48)
    labels = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        pos, action = DOOR_DEPENDENT[int(torch.randint(0, len(DOOR_DEPENDENT), (1,), generator=g))]
        goal = int(torch.randint(0, 2, (1,), generator=g)) * 4
        door = 2
        label, _ = outcome(pos, goal, door, action)
        ticks[i] = encode(pos, goal, door, action)
        labels[i] = label
    return ticks, labels


def act_logits(subj, ticks: torch.Tensor) -> torch.Tensor:
    mask = subj.masks["cn"].view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(
        _channels(ticks),
        mask,
        subj.graph.initial_state(batch_size=ticks.shape[0]),
    )
    z, a = out["next_state"]["z"][:, 0:24], out["next_state"]["a"][:, 0:24]
    return subj.phase_heads["act"](torch.cat([z, a], dim=-1))


def _channels(ticks: torch.Tensor) -> dict[str, torch.Tensor]:
    parts = ticks.split(8, dim=-1)
    return {name: parts[idx] for idx, name in enumerate(LEGAL_INPUT_CHANNELS)}


def evaluate(subj, ticks, labels, *, zero_text=False, zero_vis=False) -> dict:
    subj.eval_mode()
    t = ticks.clone()
    if zero_text:
        t[:, :, TEXT_LO:TEXT_HI] = 0.0
    if zero_vis:
        t[:, :, VIS_LO:VIS_HI] = 0.0
    with torch.no_grad():
        pred = act_logits(subj, t).argmax(-1)
    correct = pred == labels
    return {
        "acc": float(correct.float().mean()),
        "per_outcome": {
            int(k): float(correct[labels == k].float().mean())
            for k in range(N_OUTCOMES)
            if int((labels == k).sum()) > 0
        },
        "block_acc": float(correct[labels == 2].float().mean()),
        "n": int(labels.numel()),
    }


def build_subject() -> Subject72:
    torch.manual_seed(SEED)
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 501)
        subj.phase_heads["act"] = PhaseHead(48, N_OUTCOMES, 64)
    for p in subj.phase_heads["act"].parameters():
        p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    subj.train_mode()
    return subj


def main() -> int:
    started = time.process_time()
    crit = json.loads(CRITERIA.read_text(encoding="utf-8"))
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    digest = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == crit["criteria_sha256"], "criteria_sha256_mismatch"
    torch.set_num_threads(1)
    p = crit["protocol"]

    subj = build_subject()
    # base training data stream: 30k samples from simulator (door open/closed only)
    train_ticks, train_labels, _ = make_batch(
        2000, SEED + p["train_seed_offset"] + FIT_OFFSET, [0, 1]
    )
    opt = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    for _k in range(p["base_steps"]):
        idx = torch.randint(0, train_ticks.shape[0], (p["batch"],))
        loss = F.cross_entropy(act_logits(subj, train_ticks[idx]), train_labels[idx])
        subj.phase_heads["act"].train()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "cn")
        torch.nn.utils.clip_grad_norm_(
            list(subj.phase_heads["act"].parameters())
            + [p for g in opt.param_groups for p in g["params"]],
            1.0,
        )
        opt.step()
    base_ticks, base_labels, base_meta = make_batch(
        p["test_count"], SEED + p["test_seed_offset_base"] + FIT_OFFSET, [0, 1]
    )
    base = evaluate(subj, base_ticks, base_labels)
    text_zero = evaluate(subj, base_ticks, base_labels, zero_text=True)
    vis_zero = evaluate(subj, base_ticks, base_labels, zero_vis=True)
    # counterfactual action sensitivity: same state, all three actions
    cf_correct = 0
    cf_total = 0
    cf_ticks, cf_labels = [], []
    for i in range(p["counterfactual_states"]):
        pos = int(base_meta["pos"][i])
        goal = int(base_meta["goal"][i])
        door = int(base_meta["door"][i])
        for action in range(N_ACTIONS):
            cf_ticks.append(encode(pos, goal, door, action))
            cf_labels.append(outcome(pos, goal, door, action)[0])
    cf_ticks = torch.stack(cf_ticks)
    cf_labels = torch.tensor(cf_labels, dtype=torch.long)
    subj.eval_mode()
    with torch.no_grad():
        cf_pred = act_logits(subj, cf_ticks).argmax(-1)
    cf_acc = float((cf_pred == cf_labels).float().mean())
    cf_correct = int((cf_pred == cf_labels).sum())
    cf_total = int(cf_labels.numel())

    # revision: new word "locked" with real receipts
    locked_test_ticks, locked_test_labels, _ = make_batch(
        p["test_count"], SEED + p["test_seed_offset_locked"] + FIT_OFFSET, [2]
    )
    no_feedback = evaluate(subj, locked_test_ticks, locked_test_labels)
    dep_ticks, dep_labels = make_dependent_locked_batch(
        p["test_count"], SEED + p["test_seed_offset_locked_dependent"] + FIT_OFFSET
    )
    no_feedback_dep = evaluate(subj, dep_ticks, dep_labels)
    lock_train_ticks, lock_train_labels, _ = make_batch(
        1000, SEED + p["train_seed_offset"] + 777 + FIT_OFFSET, [2]
    )

    def revision_arm(permute: bool):
        arm = copy.deepcopy(subj)
        arm_opt = torch.optim.Adam(
            [x for x in arm.params() if x.requires_grad], lr=p["revision_lr"], weight_decay=0.0
        )
        curve = []
        for _ in range(p["revision_interactions"]):
            idx = torch.randint(0, lock_train_ticks.shape[0], (p["revision_batch"],))
            labels = lock_train_labels[idx]
            if permute:
                labels = (labels + 1) % N_OUTCOMES
            loss = F.cross_entropy(act_logits(arm, lock_train_ticks[idx]), labels)
            arm.phase_heads["act"].train()
            arm_opt.zero_grad(set_to_none=True)
            loss.backward()
            mask_grads(arm, "cn")
            torch.nn.utils.clip_grad_norm_(
                list(arm.phase_heads["act"].parameters())
                + [x for g in arm_opt.param_groups for x in g["params"]],
                1.0,
            )
            arm_opt.step()
            arm.eval_mode()
            with torch.no_grad():
                pred = act_logits(arm, locked_test_ticks).argmax(-1)
            curve.append(float((pred == locked_test_labels).float().mean()))
        arm.eval_mode()
        dep_after = evaluate(arm, dep_ticks, dep_labels)
        return arm, curve, dep_after

    revised, curve, revised_dep = revision_arm(permute=False)
    revised_locked = evaluate(revised, locked_test_ticks, locked_test_labels)
    revised_base = evaluate(revised, base_ticks, base_labels)
    permuted, perm_curve, permuted_dep = revision_arm(permute=True)
    permuted_locked = evaluate(permuted, locked_test_ticks, locked_test_labels)

    gates = {
        "G0": base["acc"] >= crit["gates"]["G0_base_acc_min"]
        and base["block_acc"] >= crit["gates"]["G0_block_acc_min"],
        "G1": cf_acc >= crit["gates"]["G1_counterfactual_acc_min"],
        "G2": (base["block_acc"] - text_zero["block_acc"])
        >= crit["gates"]["G2_text_zero_block_drop_min"]
        and (base["acc"] - vis_zero["acc"]) >= crit["gates"]["G2_visual_zero_acc_drop_min"],
        "G3": revised_dep["acc"] >= crit["gates"]["G3_dependent_revision_acc_min"]
        and revised_base["acc"] >= crit["gates"]["G3_retention_min"]
        and no_feedback_dep["acc"] <= crit["gates"]["G3_no_feedback_max"]
        and permuted_dep["acc"] <= crit["gates"]["G3_permuted_max"],
    }
    if not gates["G0"]:
        verdict = "act1b_baseline_insufficient"
    elif not gates["G3"]:
        verdict = "act1b_no_revision"
    elif not gates["G2"]:
        verdict = "act1b_unimodal"
    elif not gates["G1"]:
        verdict = "act1b_action_insensitive"
    elif all(gates.values()):
        verdict = "act1b_verified"
    else:
        verdict = "act1b_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-act1-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "base": base,
        "text_zero": text_zero,
        "visual_zero": vis_zero,
        "counterfactual": {"acc": cf_acc, "correct": cf_correct, "total": cf_total},
        "revision": {
            "no_feedback_locked_acc": no_feedback["acc"],
            "no_feedback_dependent_acc": no_feedback_dep["acc"],
            "revised_dependent_acc": revised_dep["acc"],
            "permuted_dependent_acc": permuted_dep["acc"],
            "revised_locked_acc": revised_locked["acc"],
            "revised_base_acc": revised_base["acc"],
            "curve": curve,
            "permuted_locked_acc": permuted_locked["acc"],
            "permuted_curve": perm_curve,
        },
        "gates": gates,
        "verdict": verdict,
        "run_conditions": {"threads": torch.get_num_threads()},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": cost},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "base": {
                    "acc": round(base["acc"], 4),
                    "block": round(base["block_acc"], 4),
                    "per_outcome": {k: round(v, 4) for k, v in base["per_outcome"].items()},
                },
                "text_zero_block": round(text_zero["block_acc"], 4),
                "visual_zero_acc": round(vis_zero["acc"], 4),
                "counterfactual_acc": round(cf_acc, 4),
                "revision": {
                    "no_feedback": round(no_feedback["acc"], 4),
                    "no_feedback_dependent": round(no_feedback_dep["acc"], 4),
                    "revised_dependent": round(revised_dep["acc"], 4),
                    "permuted_dependent": round(permuted_dep["acc"], 4),
                    "revised_locked": round(revised_locked["acc"], 4),
                    "revised_base": round(revised_base["acc"], 4),
                    "curve": [round(x, 4) for x in curve],
                    "permuted_locked": round(permuted_locked["acc"], 4),
                },
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
