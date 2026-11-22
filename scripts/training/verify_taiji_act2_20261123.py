"""ACT-2: multi-step corridor navigation (12-step closed loop) with multimodal state.

Same world/encoding as ACT-1b but deterministic transitions and a 3-action head: the
Subject72 (cn block + 3-action head) is behaviour-cloned from a value-iteration planner
(planner provides labels/upper bound only) and then evaluated closed-loop on 300 held-out
episodes: success rate, steps, utility vs random / always-R / always-PUSH / greedy / DP
baselines and modality-channel ablations.  Single seed, <=400 CPU, zero check/final.
Standing authorization; owner may revise.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_24s_f1c_20261122 import capsule_state_diff
from scripts.training.verify_taiji_act1_20261123 import TEXT_PROJ, outcome
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import LEGAL_INPUT_CHANNELS

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_act2_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_act2_20261123.json"
SEED = 20261120
H = 12
N_ACTIONS = 3
CH_POS = (0, 5)
CH_GOAL = (5, 10)
CH_ACTION = (12, 15)
CH_QUERY = 15
CH_TEXT = (16, 24)


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------
def step_state(pos: int, goal: int, door: int, action: int) -> tuple[int, int]:
    o, nd = outcome(pos, goal, door, action)
    if o == 1:
        return goal, nd
    if o == 0:
        return (pos - 1 if action == 0 else pos + 1), nd
    return pos, nd


def build_planner() -> dict:
    states = [(pos, goal, door) for pos in range(5) for goal in (0, 4) for door in (0, 1)]
    V = {s: 0.0 if s[0] == s[1] else float("inf") for s in states}
    for _ in range(60):
        delta = 0.0
        for s in states:
            pos, goal, door = s
            if pos == goal:
                continue
            best = float("inf")
            for a in range(N_ACTIONS):
                sp = step_state(pos, goal, door, a)
                cand = 1.0 + V[(sp[0], goal, sp[1])]
                if cand < best:
                    best = cand
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < 1e-9:
            break
    plan = {}
    for s in states:
        pos, goal, door = s
        if pos == goal:
            plan[s] = 0
            continue
        plan[s] = min(
            range(N_ACTIONS),
            key=lambda a: 1.0
            + V[
                (step_state(pos, goal, door, a))[0],
                goal,
                (step_state(pos, goal, door, a))[1],
            ],
        )
    return {"V": V, "plan": plan}


# ---------------------------------------------------------------------------
# encoding / model / env
# ---------------------------------------------------------------------------
def encode_state(pos: int, goal: int, door: int) -> torch.Tensor:
    ticks = torch.zeros(14, 48)
    ticks[0, CH_POS[0] + pos] = 1.0
    ticks[0, CH_GOAL[0] + goal] = 1.0
    ticks[0, CH_TEXT[0] : CH_TEXT[1]] = TEXT_PROJ[door]
    ticks[13, CH_QUERY] = 1.0
    return ticks


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
        torch.manual_seed(SEED + 701)
        subj.phase_heads["act"] = PhaseHead(48, N_ACTIONS, 64)
    for p in subj.phase_heads["act"].parameters():
        p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    subj.train_mode()
    return subj


def model_logits(subj, ticks: torch.Tensor, zero_groups: tuple[str, ...] = ()) -> torch.Tensor:
    t = ticks.clone()
    for group in zero_groups:
        lo, hi = {"text": CH_TEXT, "visual": (0, 10), "goal": CH_GOAL}[group]
        t[:, :, lo:hi] = 0.0
    mask = subj.masks["cn"].view(1, 1, -1).expand(t.shape[0], t.shape[1], subj.n_units)
    out = subj.graph(_channels(t), mask, subj.graph.initial_state(batch_size=t.shape[0]))
    z, a = out["next_state"]["z"][:, 0:24], out["next_state"]["a"][:, 0:24]
    return subj.phase_heads["act"](torch.cat([z, a], dim=-1))


def _channels(ticks: torch.Tensor) -> dict[str, torch.Tensor]:
    parts = ticks.split(8, dim=-1)
    return {name: parts[idx] for idx, name in enumerate(LEGAL_INPUT_CHANNELS)}


def init_episodes(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    eps = []
    for _ in range(n):
        eps.append(
            {
                "pos": rng.randint(0, 4),
                "goal": 0 if rng.random() < 0.5 else 4,
                "door": 0 if rng.random() < 0.5 else 1,
                "steps": 0,
                "done": False,
                "success": False,
            }
        )
    return eps


def run_episodes(subj, policy, n: int, seed: int, zero_groups=()) -> dict:
    eps = init_episodes(n, seed)
    rng = random.Random(seed + 1)
    for _ in range(H):
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        if policy == "learned":
            ticks = torch.stack([encode_state(e["pos"], e["goal"], e["door"]) for e in active])
            subj.eval_mode()
            with torch.no_grad():
                actions = model_logits(subj, ticks, zero_groups).argmax(-1).tolist()
        elif policy == "random":
            actions = [rng.randrange(N_ACTIONS) for _ in active]
        else:
            actions = [policy(e) for e in active]
        for e, a in zip(active, actions, strict=True):
            e["steps"] += 1
            o, nd = outcome(e["pos"], e["goal"], e["door"], a)
            e["door"] = nd
            if o == 1:
                e["pos"] = e["goal"]
                e["done"] = True
                e["success"] = True
            elif o == 0:
                e["pos"] = e["pos"] - 1 if a == 0 else e["pos"] + 1
    success = sum(1 for e in eps if e["success"]) / n
    steps = sum(e["steps"] for e in eps) / n
    utility = sum((1.0 if e["success"] else 0.0) - 0.02 * e["steps"] for e in eps) / n
    return {"success": success, "mean_steps": steps, "utility": utility}


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

    planner = build_planner()
    plan = planner["plan"]
    V = planner["V"]

    # ---------- behaviour cloning from planner labels ----------
    subj = build_subject()
    opt = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    rng = random.Random(SEED + 11)
    samples = []
    for _ in range(p["bc_samples"]):
        pos = rng.randint(0, 4)
        goal = 0 if rng.random() < 0.5 else 4
        door = 0 if rng.random() < 0.5 else 1
        state = (pos, goal, door)
        samples.append((state, plan[state]))
    for _k in range(p["bc_steps"]):
        idx = [rng.randrange(len(samples)) for _ in range(p["bc_batch"])]
        ticks = torch.stack([encode_state(*samples[i][0]) for i in idx])
        labels = torch.tensor([samples[i][1] for i in idx])
        loss = torch.nn.functional.cross_entropy(model_logits(subj, ticks), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "cn")
        torch.nn.utils.clip_grad_norm_(
            list(subj.phase_heads["act"].parameters())
            + [x for g in opt.param_groups for x in g["params"]],
            1.0,
        )
        opt.step()

    n_eval = p["eval_episodes"]
    seed_eval = SEED + p["eval_seed_offset"]

    def greedy(e):
        pos, goal, door = e["pos"], e["goal"], e["door"]
        if door == 1 and pos in (1, 2) and (goal - pos) * (1 if pos == 1 else -1) > 0:
            return 2
        return 1 if goal > pos else 0

    def always_r(e):
        return 1

    def always_push(e):
        return 2

    def dp(e):
        return plan[(e["pos"], e["goal"], e["door"])]

    learned = run_episodes(subj, "learned", n_eval, seed_eval)
    baselines = {
        "random": run_episodes(subj, "random", n_eval, seed_eval),
        "always_R": run_episodes(subj, always_r, n_eval, seed_eval),
        "always_PUSH": run_episodes(subj, always_push, n_eval, seed_eval),
        "greedy": run_episodes(subj, greedy, n_eval, seed_eval),
        "dp_optimal": run_episodes(subj, dp, n_eval, seed_eval),
    }
    ablations = {
        "zero_text": run_episodes(subj, "learned", n_eval, seed_eval, ("text",)),
        "zero_visual": run_episodes(subj, "learned", n_eval, seed_eval, ("visual",)),
        "zero_goal": run_episodes(subj, "learned", n_eval, seed_eval, ("goal",)),
    }
    best_fixed = max(
        baselines[k]["success"] for k in ("random", "always_R", "always_PUSH", "greedy")
    )
    capsule_ticks = torch.stack([encode_state(0, 0, 0) for _ in range(32)])
    capsule_diff = capsule_state_diff(subj, capsule_ticks, torch.ones(subj.n_units))
    gates = {
        "G0": learned["success"] >= crit["gates"]["G0_success_min"],
        "G1": learned["success"] >= best_fixed + crit["gates"]["G1_gain_over_fixed_min"],
        "G2": learned["success"]
        >= baselines["dp_optimal"]["success"] - crit["gates"]["G2_dp_success_gap_max"]
        and learned["mean_steps"]
        <= baselines["dp_optimal"]["mean_steps"] + crit["gates"]["G2_dp_steps_gap_max"],
        "G3": (learned["success"] - ablations["zero_text"]["success"])
        >= crit["gates"]["G3_text_drop_min"]
        and (learned["success"] - ablations["zero_visual"]["success"])
        >= crit["gates"]["G3_visual_drop_min"],
        "G4": capsule_diff <= crit["gates"]["G4_capsule_diff_max"],
    }
    if not gates["G0"]:
        verdict = "act2_low_success"
    elif not gates["G1"]:
        verdict = "act2_no_gain"
    elif not gates["G2"]:
        verdict = "act2_suboptimal"
    elif not gates["G3"]:
        verdict = "act2_modality_blind"
    elif all(gates.values()):
        verdict = "act2_verified"
    else:
        verdict = "act2_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-act2-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "planner": {"v_initial_high": V[(2, 0, 1)], "v_initial_low": V[(2, 4, 1)]},
        "learned": learned,
        "baselines": baselines,
        "ablations": ablations,
        "best_fixed": best_fixed,
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
                "learned": {k: round(v, 4) for k, v in learned.items()},
                "baselines": {k: round(v["success"], 4) for k, v in baselines.items()},
                "ablations": {k: round(v["success"], 4) for k, v in ablations.items()},
                "best_fixed": round(best_fixed, 4),
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
