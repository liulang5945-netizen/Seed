"""R2-X: exploratory pilot for S1-R2 candidate arms (disposable, 1 seed, <=300 CPU).

Arms (trained sequentially on CN -> P24 -> P27, 1 seed):
  C0  RT reference: per-task block masks + per-task heads, task ID at evaluation.
  C1  all-shared single union head (16-way: CN 0..7, P24 8..11, P27 12..15), no masks.
  R2A shared block (cn 0:24 always active) + learned input-conditioned router over the
      two remaining groups (24:48, 48:72), L2 anchor on all unit parameters.
  R2B HAT-style per-task gates over all 72 units with cumulative masking + sparsity.

Task-agnostic inference for C1/R2A/R2B (no task name); C0 uses the external task ID as
reference.  Read-out only exploratory: no capability claim; zero check/final.
"""

from __future__ import annotations

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
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import LEGAL_INPUT_CHANNELS

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_u2_r2x_20261123.json"
SEED = 20261120
N_UNITS = 72
N_UNION = 16
OFFSET = {"cn": 0, "p24": 8, "p27": 12}
N_CLASS = {"cn": 8, "p24": 4, "p27": 4}
TASK_ID = {"cn": 0, "p24": 1, "p27": 2}
STEPS = {"cn": 300, "p24": 300, "p27": 400}
BATCH = 48
LR = 7e-3
TRAIN_BASE = 5000
EVAL_BASE = 9000


def channels(ticks: torch.Tensor) -> dict[str, torch.Tensor]:
    parts = ticks.split(8, dim=-1)
    return {name: parts[idx] for idx, name in enumerate(LEGAL_INPUT_CHANNELS)}


def make_batch(task: str, n: int, seed: int):
    if task == "cn":
        b = cn_split(n, seed, "seen")
        return b["ticks"], b["labels"]
    if task == "p24":
        b = _make_p24_stream(n, seed)
        return b["ticks"], b["val_labels"]
    b = _make_p27_physics_batch(n, seed)
    return b["ticks"], b["labels"]


def built_subject() -> Subject72:
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
    return subj


def union_logits(subj, head, ticks, mask_vec):
    m = mask_vec.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], N_UNITS)
    out = subj.graph(channels(ticks), m, subj.graph.initial_state(batch_size=ticks.shape[0]))
    z = out["next_state"]["z"]
    a = out["next_state"]["a"]
    return head(torch.cat([z, a], dim=-1))


def add_union_head() -> PhaseHead:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 1201)
        head = PhaseHead(144, N_UNION, 64)
    for p in head.parameters():
        p.requires_grad_(True)
    return head


def c0_reference() -> dict:
    subj = built_subject()
    opt = torch.optim.Adam(subj.params(), lr=LR, weight_decay=0.0)
    for task in ("cn", "p24", "p27"):
        head_name = f"{task}u" if task != "cn" else None
        if head_name is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED + (401 if task == "p24" else 402))
                subj.phase_heads[head_name] = PhaseHead(48, N_CLASS[task], 64)
            for p in subj.phase_heads[head_name].parameters():
                p.requires_grad_(True)
        lo, hi = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}[task]
        for k in range(STEPS[task]):
            ticks, labels = make_batch(task, BATCH, SEED * 7000 + TASK_ID[task] * 1000 + k)
            if task == "cn":
                sm = (ticks.abs().sum(-1) > 0).float()
                logits = subj.logits("cn", ticks, sm)
                loss = F.cross_entropy(logits, labels)
            else:
                m = torch.zeros(N_UNITS)
                m[lo:hi] = 1.0
                mm = m.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], N_UNITS)
                out = subj.graph(
                    channels(ticks), mm, subj.graph.initial_state(batch_size=ticks.shape[0])
                )
                z = out["next_state"]["z"][:, lo:hi]
                a = out["next_state"]["a"][:, lo:hi]
                logits = subj.phase_heads[head_name](torch.cat([z, a], dim=-1))
                loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            mask_grads(subj, task)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
    res = {}
    for task in ("cn", "p24", "p27"):
        ticks, labels = make_batch(task, 256, SEED + EVAL_BASE + TASK_ID[task])
        subj.eval_mode()
        with torch.no_grad():
            if task == "cn":
                sm = (ticks.abs().sum(-1) > 0).float()
                pred = subj.logits("cn", ticks, sm).argmax(-1)
            else:
                lo, hi = (24, 48) if task == "p24" else (48, 72)
                m = torch.zeros(N_UNITS)
                m[lo:hi] = 1.0
                mm = m.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], N_UNITS)
                out = subj.graph(
                    channels(ticks), mm, subj.graph.initial_state(batch_size=ticks.shape[0])
                )
                z = out["next_state"]["z"][:, lo:hi]
                a = out["next_state"]["a"][:, lo:hi]
                pred = subj.phase_heads[f"{task}u"](torch.cat([z, a], dim=-1)).argmax(-1)
        res[task] = float((pred == labels).float().mean())
    return res


def eval_union(subj, head, mask_fn, n=256) -> dict:
    res = {}
    for task in ("cn", "p24", "p27"):
        ticks, labels = make_batch(task, n, SEED + EVAL_BASE + TASK_ID[task])
        subj.eval_mode()
        with torch.no_grad():
            pred = union_logits(subj, head, ticks, mask_fn(ticks)).argmax(-1)
        res[task] = float((pred == labels + OFFSET[task]).float().mean())
    return res


def train_union_arm(subj, head, mask_fn, *, anchor_lambda=0.0) -> None:
    params = list(subj.graph.selective_units.parameters()) + list(head.parameters())
    router = getattr(subj, "_r2x_router", None)
    if router is not None:
        params += list(router.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=0.0)
    anchor = {name: p.detach().clone() for name, p in subj.graph.selective_units.named_parameters()}
    for task in ("cn", "p24", "p27"):
        for k in range(STEPS[task]):
            ticks, labels = make_batch(task, BATCH, SEED * 7000 + TASK_ID[task] * 1000 + k)
            logits = union_logits(subj, head, ticks, mask_fn(ticks))
            loss = F.cross_entropy(logits, labels + OFFSET[task])
            if anchor_lambda:
                for name, p in subj.graph.selective_units.named_parameters():
                    loss = loss + anchor_lambda * ((p - anchor[name]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()


def c1_arm() -> dict:
    subj = built_subject()
    head = add_union_head()
    ones = torch.ones(N_UNITS)
    train_union_arm(subj, head, lambda t: ones)
    return {"eval": eval_union(subj, head, lambda t: ones)}


def r2a_arm() -> dict:
    subj = built_subject()
    head = add_union_head()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 1301)
        router = torch.nn.Linear(48, 2)
    subj._r2x_router = router

    def mask_fn(ticks):
        pooled = ticks.mean(dim=(0, 1))
        w = torch.softmax(router(pooled), dim=-1)
        base = torch.ones(N_UNITS)
        base[24:48] = w[0]
        base[48:72] = w[1]
        return base

    train_union_arm(subj, head, mask_fn, anchor_lambda=0.01)
    res = eval_union(subj, head, mask_fn)

    def control_route(ticks):
        base = torch.ones(N_UNITS)
        base[24:48] = 0.5
        base[48:72] = 0.5
        return base

    control = eval_union(subj, head, control_route)
    return {"eval": res, "control_random_route": control}


def r2b_arm() -> dict:
    subj = built_subject()
    head = add_union_head()
    mask_logits = torch.nn.Parameter(torch.full((3, N_UNITS), -1.0))
    frozen = torch.zeros(3, N_UNITS)
    params = list(subj.graph.selective_units.parameters()) + list(head.parameters()) + [mask_logits]
    opt = torch.optim.Adam(params, lr=LR, weight_decay=0.0)

    def gate(task_idx: int) -> torch.Tensor:
        prev = torch.zeros(N_UNITS)
        for j in range(task_idx):
            prev = 1.0 - (1.0 - prev) * (1.0 - frozen[j])
        return torch.sigmoid(mask_logits[task_idx]) * (1.0 - prev)

    for task_idx, task in enumerate(("cn", "p24", "p27")):
        for k in range(STEPS[task]):
            ticks, labels = make_batch(task, BATCH, SEED * 7000 + TASK_ID[task] * 1000 + k)
            g = gate(task_idx)
            logits = union_logits(subj, head, ticks, g)
            loss = F.cross_entropy(logits, labels + OFFSET[task]) + 1e-3 * g.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        with torch.no_grad():
            frozen[task_idx] = torch.sigmoid(mask_logits[task_idx]).detach()
    union = 1.0 - torch.prod(1.0 - frozen, dim=0)
    res = eval_union(subj, head, lambda t: union)
    control = eval_union(subj, head, lambda t: torch.ones(N_UNITS))
    return {
        "eval": res,
        "control_all_ones": control,
        "gates": {
            "per_task_active": [float((frozen[i] > 0.5).float().mean()) for i in range(3)],
            "union_active": float((union > 0.5).float().mean()),
        },
    }


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    arms = {"C0": {"eval": c0_reference()}, "C1": c1_arm(), "R2A": r2a_arm(), "R2B": r2b_arm()}
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-u2-r2x-pilot-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "steps": STEPS,
        "arms": arms,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["eval"] for k, v in arms.items()}, ensure_ascii=False))
    detail = {}
    for k, v in arms.items():
        extra = {kk: vv for kk, vv in v.items() if kk != "eval"}
        detail[k] = extra if extra else {}
    print(json.dumps(detail, ensure_ascii=False))
    print("cpu", cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
