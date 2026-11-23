"""SYN-1d: two-block synergy (joint task contribution + unit-level causal influence).

Blocks: cn (0:24) trained on CN seen pairs, p24 (24:48) 500 steps, p27 (48:72) 600 steps
(UQ-1 recipe).  Joint task label = ((cn_class // 2) + p27_class) % 4, which requires both
blocks' information.  All block parameters and single-block heads are frozen; only a
96-dim joint readout (cn state 48 + p27 state 48) is trained 300 steps.  Evaluation: joint
test accuracy, block dormancy ablations (cn/p27/both), unit-level dormancy (12 units per
block), single-block arms, capsule identity.

Single seed, <=300 CPU, zero check/final. Standing authorization; owner may revise.
"""

from __future__ import annotations

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
from scripts.training.verify_taiji_23s2a_gain_20261122 import cn_split
from scripts.training.verify_taiji_24s_f1c_20261122 import capsule_state_diff
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p27_physics_batch
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_uq1b_20261123 import build_subject, train

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_syn1d_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_syn1d_20261123.json"
SEED = 20261120
FIT = 256
FIT_OFFSET = 10_000_000
N_JOINT = 4


def channels(ticks: torch.Tensor) -> dict[str, torch.Tensor]:
    from taiji.u2_selective_units import LEGAL_INPUT_CHANNELS

    parts = ticks.split(8, dim=-1)
    return {name: parts[idx] for idx, name in enumerate(LEGAL_INPUT_CHANNELS)}


def block_states(subj, task: str, ticks: torch.Tensor) -> torch.Tensor:
    lo, hi = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}[task]
    mask = torch.zeros(subj.n_units)
    mask[lo:hi] = 1.0
    m = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(channels(ticks), m, subj.graph.initial_state(batch_size=ticks.shape[0]))
    z = out["next_state"]["z"][:, lo:hi]
    a = out["next_state"]["a"][:, lo:hi]
    return torch.cat([z, a], dim=-1)


def joint_batch(n: int, seed: int) -> dict:
    cn = cn_split(n, seed, "seen")
    p27 = _make_p27_physics_batch(n, seed + 7777)
    labels = ((cn["labels"] // 2) + p27["labels"]) % N_JOINT
    return {"cn": cn["ticks"], "p27": p27["ticks"], "labels": labels}


def joint_logits(subj, head, batch: dict, norm: dict) -> torch.Tensor:
    z_cn = block_states(subj, "cn", batch["cn"])
    z_p27 = block_states(subj, "p27", batch["p27"])
    z_cn = (z_cn - norm["mu_cn"]) / norm["sd_cn"]
    z_p27 = (z_p27 - norm["mu_p27"]) / norm["sd_p27"]
    return head(torch.cat([z_cn, z_p27], dim=-1))


def acc(subj, head, batch: dict, norm: dict) -> float:
    subj.eval_mode()
    with torch.no_grad():
        pred = joint_logits(subj, head, batch, norm).argmax(-1)
    return float((pred == batch["labels"]).float().mean())


def arm_acc(subj, head, batch: dict, task: str, norm: dict) -> float:
    subj.eval_mode()
    with torch.no_grad():
        s = block_states(subj, task, batch[task])
        s = (s - norm["mu_" + task]) / norm["sd_" + task]
        pred = head(s).argmax(-1)
    return float((pred == batch["labels"]).float().mean())


def train_head(subj, head, batch: dict, steps: int, batch_size: int, lr: float, norm: dict) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=0.0)
    n = batch["labels"].numel()
    for _k in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        b = {key: val[idx] for key, val in batch.items()}
        loss = F.cross_entropy(joint_logits(subj, head, b, norm), b["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
        opt.step()


def train_arm(
    subj, head, batch: dict, task: str, steps: int, batch_size: int, lr: float, norm: dict
) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=0.0)
    n = batch["labels"].numel()
    for _k in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        b = {key: val[idx] for key, val in batch.items()}
        states = block_states(subj, task, b[task])
        states = (states - norm["mu_" + task]) / norm["sd_" + task]
        loss = F.cross_entropy(head(states), b["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


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
    oc = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    for k in range(p["cn_steps"]):
        b = cn_split(p["joint_batch"], SEED * 6000 + k, "seen")
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        loss = F.cross_entropy(subj.logits("cn", b["ticks"], sm), b["labels"])
        oc.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "cn")
        torch.nn.utils.clip_grad_norm_([x for g in oc.param_groups for x in g["params"]], 1.0)
        oc.step()
    for x in subj.phase_heads["cn"].parameters():
        x.requires_grad_(False)
    o24 = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    train(subj, o24, "p24", p["p24_steps"])
    for x in subj.graph.cross_channel_mix.parameters():
        x.requires_grad_(False)
    for x in subj.phase_heads["p24u"].parameters():
        x.requires_grad_(False)
    for x in subj.phase_heads["p27u"].parameters():
        x.requires_grad_(True)
    o27 = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    train(subj, o27, "p27", p["p27_steps"])
    for param in subj.params():
        param.requires_grad_(False)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 801)
        head_joint = PhaseHead(96, N_JOINT, 64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 802)
        head_cn = PhaseHead(48, N_JOINT, 64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 803)
        head_p27 = PhaseHead(48, N_JOINT, 64)

    train_batch = joint_batch(p["joint_train_count"], SEED + p["train_seed_offset"] + FIT_OFFSET)
    test_batch = joint_batch(FIT, SEED + p["test_seed_offset"] + FIT_OFFSET)
    with torch.no_grad():
        s_cn_tr = block_states(subj, "cn", train_batch["cn"])
        s_p27_tr = block_states(subj, "p27", train_batch["p27"])
    norm = {
        "mu_cn": s_cn_tr.mean(0),
        "sd_cn": s_cn_tr.std(0) + 1e-6,
        "mu_p27": s_p27_tr.mean(0),
        "sd_p27": s_p27_tr.std(0) + 1e-6,
    }
    with torch.no_grad():
        s_cn = (block_states(subj, "cn", train_batch["cn"]) - norm["mu_cn"]) / norm["sd_cn"]
        s_p27 = (block_states(subj, "p27", train_batch["p27"]) - norm["mu_p27"]) / norm["sd_p27"]
        X = torch.cat([s_cn, s_p27], dim=-1)
        Y = torch.nn.functional.one_hot(train_batch["labels"], N_JOINT).float()
        sol = torch.linalg.lstsq(X, Y).solution
        probe_acc = float((X @ sol).argmax(-1).eq(train_batch["labels"]).float().mean())
        print(
            "PROBE linear train acc",
            probe_acc,
            "label counts",
            torch.bincount(train_batch["labels"]).tolist(),
            "cn std",
            float(s_cn.std()),
            "p27 std",
            float(s_p27.std()),
        )
    train_head(
        subj, head_joint, train_batch, p["joint_steps"], p["joint_batch"], p["joint_lr"], norm
    )
    joint_train_acc = acc(subj, head_joint, train_batch, norm)
    joint_acc = acc(subj, head_joint, test_batch, norm)
    # diagnostics: single-task accuracies
    cn_diag = cn_split(FIT, SEED + 900 + FIT_OFFSET, "seen")
    sm = (cn_diag["ticks"].abs().sum(-1) > 0).float()
    subj.eval_mode()
    with torch.no_grad():
        cn_step_acc = float(
            (subj.logits("cn", cn_diag["ticks"], sm).argmax(-1) == cn_diag["labels"]).float().mean()
        )
        p27_diag = _make_p27_physics_batch(FIT, SEED + 901 + FIT_OFFSET)
        p27_step_acc = float(
            (
                subj.phase_heads["p27u"](block_states(subj, "p27", p27_diag["ticks"])).argmax(-1)
                == p27_diag["labels"]
            )
            .float()
            .mean()
        )
    print(
        "DIAG cn_step_acc",
        cn_step_acc,
        "p27_step_acc",
        p27_step_acc,
        "joint_train_acc",
        joint_train_acc,
    )

    train_arm(subj, head_cn, train_batch, "cn", p["arm_steps"], p["joint_batch"], p["lr"], norm)
    train_arm(subj, head_p27, train_batch, "p27", p["arm_steps"], p["joint_batch"], p["lr"], norm)
    cn_arm = arm_acc(subj, head_cn, test_batch, "cn", norm)
    p27_arm = arm_acc(subj, head_p27, test_batch, "p27", norm)

    units = subj.graph.selective_units
    dormant = units.set_unit_lifecycle_

    dormant(range(0, 24), "dormant")
    cn_lesion = acc(subj, head_joint, test_batch, norm)
    dormant(range(0, 24), "mature")
    dormant(range(48, 72), "dormant")
    p27_lesion = acc(subj, head_joint, test_batch, norm)
    dormant(range(48, 72), "mature")
    dormant(range(0, 24), "dormant")
    dormant(range(48, 72), "dormant")
    both_lesion = acc(subj, head_joint, test_batch, norm)
    dormant(range(0, 24), "mature")
    dormant(range(48, 72), "mature")

    n_units = p["unit_lesion_count"]
    dormant(range(0, n_units), "dormant")
    cn_unit_lesion = acc(subj, head_joint, test_batch, norm)
    dormant(range(0, n_units), "mature")
    dormant(range(48, 48 + n_units), "dormant")
    p27_unit_lesion = acc(subj, head_joint, test_batch, norm)
    dormant(range(48, 48 + n_units), "mature")

    capsule_diff = capsule_state_diff(subj, test_batch["cn"][:32], torch.ones(subj.n_units))
    gates = {
        "G0": joint_acc >= crit["gates"]["G0_joint_acc_min"],
        "G1": joint_acc - cn_lesion >= crit["gates"]["G1_cn_drop_min"],
        "G2": joint_acc - p27_lesion >= crit["gates"]["G2_p27_drop_min"],
        "G3": max(cn_arm, p27_arm) <= crit["gates"]["G3_single_block_acc_max"],
        "G4": (joint_acc - cn_unit_lesion) >= crit["gates"]["G4_unit_drop_min"]
        and (joint_acc - p27_unit_lesion) >= crit["gates"]["G4_unit_drop_min"],
        "G5": capsule_diff <= crit["gates"]["G5_capsule_diff_max"],
    }
    if not gates["G0"]:
        verdict = "syn1d_low_joint"
    elif not (gates["G1"] and gates["G2"]):
        verdict = "syn1d_not_synergistic"
    elif not gates["G3"]:
        verdict = "syn1d_single_block_sufficient"
    elif not gates["G4"]:
        verdict = "syn1d_no_unit_causality"
    elif all(gates.values()):
        verdict = "syn1d_verified"
    else:
        verdict = "syn1d_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-syn1-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "joint_acc": joint_acc,
        "arms": {"cn_only": cn_arm, "p27_only": p27_arm},
        "lesions": {
            "cn_block_dormant": cn_lesion,
            "p27_block_dormant": p27_lesion,
            "both_blocks_dormant": both_lesion,
            "cn_units_12_dormant": cn_unit_lesion,
            "p27_units_12_dormant": p27_unit_lesion,
        },
        "capsule_state_diff": capsule_diff,
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
                "joint_acc": round(joint_acc, 4),
                "arms": {k: round(v, 4) for k, v in payload["arms"].items()},
                "lesions": {k: round(v, 4) for k, v in payload["lesions"].items()},
                "capsule_diff": capsule_diff,
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
