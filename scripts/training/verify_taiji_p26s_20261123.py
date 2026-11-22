"""P26-S: autonomous choice without new prompts in a persistent-goal session world.

World: 12-step session, 3 items, energy 0..2, pending verification flag.  Actions:
CONTINUE / VERIFY / HELP / REST / WAIT(with wake condition = pending).  Terminal utility
= solved items + 0.15 * energy.  A finite-horizon DP computes Q*/optimal actions; the
Subject72 (cn block + 5-action head) is behaviour-cloned from DP labels (planner provides
training labels only) and then evaluated on 300 held-out sessions against fixed baselines,
state-channel ablations and context-sensitivity statistics.  No prompt channel exists:
decisions are made every step from persistent state alone.

Single seed, <=400 CPU, zero check/final. Standing authorization; owner may revise.
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
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import LEGAL_INPUT_CHANNELS

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_p26s_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_p26s_20261123.json"
SEED = 20261120
H = 12
N_ITEMS = 3
EMAX = 2
P_LOW, P_HIGH, P_HELP = 0.55, 0.9, 0.95
ENERGY_BONUS = 0.15
ACTIONS = ("CONTINUE", "VERIFY", "HELP", "REST", "WAIT")
CH_SOLVED = (0, 4)
CH_CONF = (4, 6)
CH_ENERGY = (6, 9)
CH_PENDING = (9, 11)
CH_TBUCKET = (11, 14)
CH_QUERY = 15


# ---------------------------------------------------------------------------
# finite-horizon DP
# ---------------------------------------------------------------------------
def build_dp() -> dict:
    states = []
    for solved in range(N_ITEMS + 1):
        for conf in (0, 1):
            for energy in range(EMAX + 1):
                for pending in (0, 1):
                    states.append((solved, conf, energy, pending))
    V: list[dict] = []
    Q: list[dict] = []
    V.append({s: s[0] + ENERGY_BONUS * s[2] for s in states})
    Q.append({s: [V[0][s]] * len(ACTIONS) for s in states})
    for t in range(1, H + 1):
        q_t: dict = {}
        for s in states:
            solved, conf, energy, pending = s
            q = [0.0] * len(ACTIONS)
            prev = V[t - 1]

            def next_conf_value(
                solved_next: int, energy_next: int, pending_next: int = 0, prev=prev, conf=conf
            ) -> float:
                if solved_next >= N_ITEMS:
                    return prev[(solved_next, conf, energy_next, pending_next)]
                return (
                    0.5 * prev[(solved_next, 1, energy_next, pending_next)]
                    + 0.5 * prev[(solved_next, 0, energy_next, pending_next)]
                )

            if solved >= N_ITEMS:
                q = [prev[s]] * len(ACTIONS)
            else:
                p = P_HIGH if conf == 1 else P_LOW
                # CONTINUE
                if energy >= 1:
                    succ = next_conf_value(solved + 1, energy - 1)
                    fail = prev[(solved, conf, energy - 1, 0)]
                    q[0] = p * succ + (1 - p) * fail
                else:
                    q[0] = prev[s]
                # VERIFY
                if energy >= 1 and conf == 0 and pending == 0:
                    q[1] = prev[(solved, conf, energy - 1, 1)]
                else:
                    q[1] = prev[s]
                # HELP
                if energy >= 1:
                    succ = next_conf_value(solved + 1, energy - 1)
                    fail = prev[(solved, conf, energy - 1, 0)]
                    q[2] = P_HELP * succ + (1 - P_HELP) * fail
                else:
                    q[2] = prev[s]
                # REST
                q[3] = prev[(solved, conf, min(EMAX, energy + 1), pending)]
                # WAIT
                if pending == 1:
                    q[4] = prev[(solved, 1, energy, 0)]
                else:
                    q[4] = prev[s]
            q_t[s] = q
        Q.append(q_t)
        V.append({s: max(q_t[s]) for s in states})
    return {"Q": Q, "V": V, "states": states}


def dp_action(dp: dict, t_remaining: int, state: tuple) -> int:
    q = dp["Q"][t_remaining][state]
    return max(range(len(ACTIONS)), key=lambda a: q[a])


def dp_policy(dp: dict, states: list[tuple], t_remaining: int) -> list[int]:
    return [dp_action(dp, t_remaining, s) for s in states]


# ---------------------------------------------------------------------------
# encoding / env
# ---------------------------------------------------------------------------
def encode_states(states: list[tuple], t_remaining: int) -> torch.Tensor:
    ticks = torch.zeros(len(states), 14, 48)
    bucket = 0 if t_remaining <= 4 else (1 if t_remaining <= 8 else 2)
    for i, (solved, conf, energy, pending) in enumerate(states):
        ticks[i, 0, CH_SOLVED[0] + min(solved, N_ITEMS)] = 1.0
        ticks[i, 0, CH_CONF[0] + conf] = 1.0
        ticks[i, 0, CH_ENERGY[0] + energy] = 1.0
        ticks[i, 0, CH_PENDING[0] + pending] = 1.0
        ticks[i, 0, CH_TBUCKET[0] + bucket] = 1.0
        ticks[i, 13, CH_QUERY] = 1.0
    return ticks


class SessionEnv:
    def __init__(self, n: int, seed: int) -> None:
        self.n = n
        self.rngs = [random.Random(seed * 7919 + i) for i in range(n)]
        self.solved = [0] * n
        self.conf = [1 if self.rngs[i].random() < 0.5 else 0 for i in range(n)]
        self.energy = [EMAX] * n
        self.pending = [0] * n
        self.t = 0
        self.stats = {"wait_pending": 0, "pending_seen": 0, "rest_zero": 0, "zero_seen": 0}
        self.action_counts = [0] * len(ACTIONS)
        self.steps_taken = 0

    def states(self) -> list[tuple]:
        return [
            (self.solved[i], self.conf[i], self.energy[i], self.pending[i]) for i in range(self.n)
        ]

    def step(self, actions: list[int]) -> None:
        for i, a in enumerate(actions):
            rng = self.rngs[i]
            solved, conf, energy, pending = (
                self.solved[i],
                self.conf[i],
                self.energy[i],
                self.pending[i],
            )
            self.action_counts[a] += 1
            self.steps_taken += 1
            if pending == 1:
                self.stats["pending_seen"] += 1
                if a == 4:
                    self.stats["wait_pending"] += 1
            if energy == 0:
                self.stats["zero_seen"] += 1
                if a == 3:
                    self.stats["rest_zero"] += 1
            if solved >= N_ITEMS:
                continue
            if a == 0:  # CONTINUE
                if energy >= 1:
                    energy -= 1
                    p = P_HIGH if conf == 1 else P_LOW
                    if rng.random() < p:
                        solved += 1
                        pending = 0
                        if solved < N_ITEMS:
                            conf = 1 if rng.random() < 0.5 else 0
            elif a == 1:  # VERIFY
                if energy >= 1 and conf == 0 and pending == 0:
                    energy -= 1
                    pending = 1
            elif a == 2:  # HELP
                if energy >= 1:
                    energy -= 1
                    if rng.random() < P_HELP:
                        solved += 1
                        pending = 0
                        if solved < N_ITEMS:
                            conf = 1 if rng.random() < 0.5 else 0
            elif a == 3:  # REST
                energy = min(EMAX, energy + 1)
            elif a == 4 and pending == 1:  # WAIT
                pending = 0
                conf = 1
            self.solved[i], self.conf[i], self.energy[i], self.pending[i] = (
                solved,
                conf,
                energy,
                pending,
            )
        self.t += 1

    def utilities(self) -> list[float]:
        return [self.solved[i] + ENERGY_BONUS * self.energy[i] for i in range(self.n)]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
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
        torch.manual_seed(SEED + 601)
        subj.phase_heads["act"] = PhaseHead(48, len(ACTIONS), 64)
    for p in subj.phase_heads["act"].parameters():
        p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    subj.train_mode()
    return subj


def model_logits(subj, ticks: torch.Tensor, zero_groups: tuple[str, ...] = ()) -> torch.Tensor:
    t = ticks.clone()
    for group in zero_groups:
        lo, hi = {
            "solved": CH_SOLVED,
            "conf": CH_CONF,
            "energy": CH_ENERGY,
            "pending": CH_PENDING,
            "t": CH_TBUCKET,
        }[group]
        t[:, :, lo:hi] = 0.0
    mask = subj.masks["cn"].view(1, 1, -1).expand(t.shape[0], t.shape[1], subj.n_units)
    out = subj.graph(_channels(t), mask, subj.graph.initial_state(batch_size=t.shape[0]))
    z, a = out["next_state"]["z"][:, 0:24], out["next_state"]["a"][:, 0:24]
    return subj.phase_heads["act"](torch.cat([z, a], dim=-1))


def _channels(ticks: torch.Tensor) -> dict[str, torch.Tensor]:
    parts = ticks.split(8, dim=-1)
    return {name: parts[idx] for idx, name in enumerate(LEGAL_INPUT_CHANNELS)}


def rollout(subj, policy, n_episodes: int, seed: int, zero_groups=()) -> dict:
    env = SessionEnv(n_episodes, seed)
    for _ in range(H):
        states = env.states()
        if policy == "learned":
            ticks = encode_states(states, H - env.t)
            subj.eval_mode()
            with torch.no_grad():
                actions = model_logits(subj, ticks, zero_groups).argmax(-1).tolist()
        else:
            actions = policy(states, H - env.t)
        env.step(actions)
    util = env.utilities()
    n = len(util)
    wait_rate = (
        env.stats["wait_pending"] / env.stats["pending_seen"]
        if env.stats["pending_seen"] > 0
        else float("nan")
    )
    rest_rate = (
        env.stats["rest_zero"] / env.stats["zero_seen"]
        if env.stats["zero_seen"] > 0
        else float("nan")
    )
    return {
        "mean_utility": sum(util) / n,
        "mean_solved": sum(env.solved) / n,
        "action_counts": {ACTIONS[i]: env.action_counts[i] for i in range(len(ACTIONS))},
        "wait_given_pending": wait_rate,
        "rest_given_zero_energy": rest_rate,
        "pending_seen": env.stats["pending_seen"],
        "zero_seen": env.stats["zero_seen"],
    }


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

    dp = build_dp()

    # ---------- behaviour cloning from DP labels ----------
    subj = build_subject()
    opt = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    rng = random.Random(SEED + 5)
    samples = []
    for _ in range(p["bc_samples"]):
        t_rem = rng.randint(1, H)
        solved = rng.randint(0, N_ITEMS)
        conf = rng.randint(0, 1)
        energy = rng.randint(0, EMAX)
        pending = rng.randint(0, 1)
        state = (solved, conf, energy, pending)
        samples.append((t_rem, state, dp_action(dp, t_rem, state)))
    for _k in range(p["bc_steps"]):
        idx = [rng.randrange(len(samples)) for _ in range(p["bc_batch"])]
        ticks = torch.cat([encode_states([samples[i][1]], samples[i][0]) for i in idx], dim=0)
        labels = torch.tensor([samples[i][2] for i in idx])
        loss = torch.nn.functional.cross_entropy(model_logits(subj, ticks), labels)
        subj.phase_heads["act"].train()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "cn")
        torch.nn.utils.clip_grad_norm_(
            list(subj.phase_heads["act"].parameters())
            + [x for g in opt.param_groups for x in g["params"]],
            1.0,
        )
        opt.step()

    # ---------- evaluation ----------
    n_eval = p["eval_episodes"]
    seed_eval = SEED + p["eval_seed_offset"]
    learned = rollout(subj, "learned", n_eval, seed_eval)

    def always_continue(states, t_rem):
        return [0] * len(states)

    def always_help(states, t_rem):
        return [2] * len(states)

    def random_policy(states, t_rem):
        return [rng.randrange(len(ACTIONS)) for _ in states]

    def heuristic(states, t_rem):
        out = []
        for solved, conf, energy, pending in states:
            if solved >= N_ITEMS:
                out.append(3)
            elif pending == 1:
                out.append(4)
            elif conf == 0 and energy >= 1:
                out.append(1)
            elif energy == 0:
                out.append(3)
            else:
                out.append(0)
        return out

    def dp_policy_fn(states, t_rem):
        return [dp_action(dp, max(1, t_rem), s) for s in states]

    baselines = {
        "always_continue": rollout(subj, always_continue, n_eval, seed_eval),
        "always_help": rollout(subj, always_help, n_eval, seed_eval),
        "random": rollout(subj, random_policy, n_eval, seed_eval),
        "heuristic": rollout(subj, heuristic, n_eval, seed_eval),
        "dp_optimal": rollout(subj, dp_policy_fn, n_eval, seed_eval),
    }
    ablations = {
        "zero_energy": rollout(subj, "learned", n_eval, seed_eval, ("energy",)),
        "zero_conf": rollout(subj, "learned", n_eval, seed_eval, ("conf",)),
        "zero_solved": rollout(subj, "learned", n_eval, seed_eval, ("solved",)),
    }
    best_fixed = max(
        baselines["always_continue"]["mean_utility"],
        baselines["always_help"]["mean_utility"],
        baselines["random"]["mean_utility"],
        baselines["heuristic"]["mean_utility"],
    )
    dp_util = baselines["dp_optimal"]["mean_utility"]
    capsule_ticks = encode_states([(0, 0, EMAX, 0)] * 32, 12)
    capsule_diff = capsule_state_diff(subj, capsule_ticks, torch.ones(subj.n_units))
    gates = {
        "G0": learned["mean_utility"] >= best_fixed + crit["gates"]["G0_gain_over_fixed_min"],
        "G1": learned["mean_utility"] >= dp_util - crit["gates"]["G1_dp_gap_max"],
        "G2": (
            learned["mean_utility"] - ablations["zero_energy"]["mean_utility"]
            >= crit["gates"]["G2_energy_drop_min"]
            and learned["mean_utility"] - ablations["zero_conf"]["mean_utility"]
            >= crit["gates"]["G2_conf_drop_min"]
            and learned["mean_utility"] - ablations["zero_solved"]["mean_utility"]
            >= crit["gates"]["G2_solved_drop_min"]
        ),
        "G3": learned["wait_given_pending"] >= crit["gates"]["G3_wait_given_pending_min"]
        and learned["rest_given_zero_energy"] >= crit["gates"]["G3_rest_given_zero_energy_min"],
        "G4": capsule_diff <= crit["gates"]["G4_capsule_diff_max"],
    }
    if not gates["G0"]:
        verdict = "p26s_no_gain"
    elif not gates["G1"]:
        verdict = "p26s_suboptimal"
    elif not gates["G2"]:
        verdict = "p26s_state_blind"
    elif not gates["G3"]:
        verdict = "p26s_context_weak"
    elif all(gates.values()):
        verdict = "p26s_verified"
    else:
        verdict = "p26s_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-p26s-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "dp": {
            "value_at_initial": dp["V"][H][(0, 1, EMAX, 0)],
            "value_at_low_conf": dp["V"][H][(0, 0, EMAX, 0)],
        },
        "learned": learned,
        "baselines": baselines,
        "ablations": ablations,
        "best_fixed": best_fixed,
        "dp_utility": dp_util,
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
                "learned": {
                    "utility": round(learned["mean_utility"], 4),
                    "solved": round(learned["mean_solved"], 4),
                    "wait_pending": round(learned["wait_given_pending"], 4),
                    "rest_zero": round(learned["rest_given_zero_energy"], 4),
                },
                "baselines": {k: round(v["mean_utility"], 4) for k, v in baselines.items()},
                "ablations": {k: round(v["mean_utility"], 4) for k, v in ablations.items()},
                "best_fixed": round(best_fixed, 4),
                "dp_utility": round(dp_util, 4),
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
