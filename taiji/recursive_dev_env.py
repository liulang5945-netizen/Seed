"""Recursive self-development benchmark environment for N33.4 (`taiji-recursive-dev-v1`).

Models a 4-cycle sequence of dependent architectural interventions with genuine skill/fragility/
budget coupling and no ``task_id``:
  * Interventions (`boost_s0/1/2`, `stabilize_fragile`, `consolidate_all`, `noop_hold`) modify
    3 skill capabilities, 3 fragility levels, and remaining development budget.
  * Boosting a high-fragility neighbor inflicts collateral damage; stabilizing reduces fragility
    before subsequent boosts; consolidating prevents budget overrun on later cycles.
  * Evaluators run through real `SessionStore` lineage persistence (`store_roundtrip_exact`).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from taiji.choice_simulator import seed_from  # noqa: F401
from taiji.session_store import SessionRecord, SessionStore

ENV_VERSION = "taiji-recursive-dev-v1"
ACTIONS = (
    "boost_s0",
    "boost_s1",
    "boost_s2",
    "stabilize_fragile",
    "consolidate_all",
    "noop_hold",
)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
STATE_DIM = 10
OUTCOME_DIM = 13  # delta_state (10) + [damage, cost, delta_u] (3)
HORIZON = 4


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class RecursiveState:
    cap: list[float]
    frag: list[float]
    budget: float
    t: int = 0
    last_gain: float = 0.0
    last_damage: float = 0.0

    def vector(self) -> torch.Tensor:
        return torch.tensor(
            [
                *self.cap,
                *self.frag,
                self.budget,
                self.t / HORIZON,
                self.last_gain,
                self.last_damage,
            ],
            dtype=torch.float32,
        )

    def utility(self, cum_damage: float, cum_cost: float) -> float:
        return (
            0.65 * min(self.cap)
            + 0.35 * (sum(self.cap) / 3.0)
            - 0.35 * cum_damage
            - 0.10 * cum_cost
        )


class RecursiveDevEnv:
    """4-cycle deterministic recursive development environment."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.gen = torch.Generator().manual_seed(self.seed % (2**63))
        self.state = self._sample_initial()
        self.cum_damage = 0.0
        self.cum_cost = 0.0
        self.init_utility = self.state.utility(0.0, 0.0)

    def _sample_initial(self) -> RecursiveState:
        perm_c = torch.randperm(3, generator=self.gen).tolist()
        raw_c = [
            0.18 + 0.14 * float(torch.rand((), generator=self.gen)),
            0.38 + 0.18 * float(torch.rand((), generator=self.gen)),
            0.55 + 0.20 * float(torch.rand((), generator=self.gen)),
        ]
        cap = [0.0, 0.0, 0.0]
        for dst, val in zip(perm_c, raw_c, strict=True):
            cap[dst] = val

        perm_f = torch.randperm(3, generator=self.gen).tolist()
        n_high = 1 + int(torch.randint(0, 2, (), generator=self.gen))
        frag = [0.0, 0.0, 0.0]
        for k, dst in enumerate(perm_f):
            frag[dst] = (
                0.34 + 0.16 * float(torch.rand((), generator=self.gen))
                if k < n_high
                else 0.06 + 0.10 * float(torch.rand((), generator=self.gen))
            )

        budget = 0.65 + 0.35 * float(torch.rand((), generator=self.gen))
        return RecursiveState(cap=cap, frag=frag, budget=budget)

    def reset(self) -> torch.Tensor:
        self.gen = torch.Generator().manual_seed(self.seed % (2**63))
        self.state = self._sample_initial()
        self.cum_damage = 0.0
        self.cum_cost = 0.0
        self.init_utility = self.state.utility(0.0, 0.0)
        return self.state.vector()

    def step(self, action: str) -> tuple[torch.Tensor, dict[str, float], bool]:
        if action not in ACTION_INDEX:
            raise ValueError(f"unknown action {action!r}")

        s = self.state
        u_before = s.utility(self.cum_damage, self.cum_cost)
        dmg = 0.0
        cost = 0.0

        if action.startswith("boost_s"):
            idx = int(action[-1])
            nbr = (idx + 1) % 3
            cost = 0.18
            if s.budget >= 0.18:
                s.cap[idx] = _clip(s.cap[idx] + 0.42 * (1.0 - s.cap[idx]), 0.05, 1.0)
                dmg = 0.32 * s.frag[nbr]
                s.cap[nbr] = _clip(s.cap[nbr] - 0.50 * dmg, 0.05, 1.0)
                s.budget = _clip(s.budget - 0.18, 0.0, 1.0)
            else:
                dmg = 0.10
        elif action == "stabilize_fragile":
            cost = 0.12
            if s.budget >= 0.12 and max(s.frag) > 0.22:
                s.frag = [f * 0.25 for f in s.frag]
                weak = min(range(3), key=lambda k: s.cap[k])
                s.cap[weak] = _clip(s.cap[weak] + 0.14 * (1.0 - s.cap[weak]), 0.05, 1.0)
                s.budget = _clip(s.budget - 0.12, 0.0, 1.0)
            elif s.budget < 0.12:
                dmg = 0.05
        elif action == "consolidate_all":
            cost = 0.08
            if s.budget < 0.35:
                s.cap = [_clip(c + 0.08 * (1.0 - c), 0.05, 1.0) for c in s.cap]
                s.budget = _clip(s.budget + 0.36, 0.0, 1.0)
        elif action == "noop_hold":
            pass

        self.cum_damage += dmg
        self.cum_cost += cost
        u_after = s.utility(self.cum_damage, self.cum_cost)
        delta_u = u_after - u_before

        s.last_gain = max(0.0, delta_u)
        s.last_damage = dmg
        s.t += 1
        info = {
            "delta_u": delta_u,
            "damage": dmg,
            "cost": cost,
            "net_gain_so_far": u_after - self.init_utility,
        }
        return s.vector(), info, s.t >= HORIZON


def scripted_oracle_policy(obs: torch.Tensor) -> str:
    """Reference upper-bound heuristic (C0 evaluation contrast only; never used in training)."""
    cap = obs[0:3]
    frag = obs[3:6]
    budget = float(obs[6])
    if budget < 0.20:
        return "consolidate_all"
    weak = int(cap.argmin().item())
    nbr = (weak + 1) % 3
    if float(frag[nbr]) > 0.24:
        return "stabilize_fragile"
    return f"boost_s{weak}"


def random_policy(_obs: torch.Tensor, gen: torch.Generator | None = None) -> str:
    g = gen if gen is not None else torch.Generator().manual_seed(0)
    return ACTIONS[int(torch.randint(0, len(ACTIONS), (), generator=g))]


def collect_logged_experience(seeds: list[int], gen_seed: int) -> list[dict[str, Any]]:
    """Collect exploratory random-intervention transition logs (no scripted oracle)."""
    gen = torch.Generator().manual_seed(gen_seed)
    transitions: list[dict[str, Any]] = []
    for seed in seeds:
        env = RecursiveDevEnv(seed)
        s = env.reset()
        for _ in range(HORIZON):
            act = random_policy(s, gen)
            a_idx = ACTION_INDEX[act]
            s2, info, done = env.step(act)
            transitions.append(
                {
                    "s": s.clone(),
                    "a_idx": a_idx,
                    "s2": s2.clone(),
                    "delta_s": (s2 - s).clone(),
                    "damage": float(info["damage"]),
                    "cost": float(info["cost"]),
                    "delta_u": float(info["delta_u"]),
                }
            )
            s = s2
            if done:
                break
    return transitions


def evaluate_recursive_with_store(
    policy: Callable[[torch.Tensor], str],
    seeds: list[int],
    *,
    save_lineage: bool = True,
) -> dict[str, Any]:
    """Evaluate a 4-cycle intervention policy and verify `SessionStore` lineage persistence."""
    net_gains: list[float] = []
    improving_steps_list: list[int] = []
    damages: list[float] = []
    hist = {a: 0 for a in ACTIONS}
    store_exact = True

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(tmpdir)
        for seed in seeds:
            env = RecursiveDevEnv(seed)
            obs = env.reset()
            init_obs = obs.clone()
            rec = SessionRecord(subject_id=f"rec-{seed}")
            rec.capsule = {"init": init_obs}
            imp = 0
            ep_dmg = 0.0
            final_gain = 0.0
            for step_i in range(HORIZON):
                act = policy(obs)
                hist[act] += 1
                obs, info, done = env.step(act)
                ep_dmg += info["damage"]
                if info["delta_u"] > 0.005:
                    imp += 1
                final_gain = info["net_gain_so_far"]
                rec.append_experience(
                    "intervention",
                    {"step": step_i, "action": act, "delta_u": round(info["delta_u"], 6)},
                    source="recursive_dev",
                )
                if done:
                    break
            net_gains.append(final_gain)
            improving_steps_list.append(imp)
            damages.append(ep_dmg)
            if save_lineage:
                rec.capsule["final"] = obs.clone()
                expected_digest = rec.digest()
                store.save(rec)
                loaded = store.load(f"rec-{seed}")
                if (
                    loaded.digest() != expected_digest
                    or not torch.equal(loaded.capsule["init"], init_obs)
                    or not torch.equal(loaded.capsule["final"], obs)
                ):
                    store_exact = False

    n = max(1, len(seeds))
    ge3 = sum(int(k >= 3) for k in improving_steps_list)
    return {
        "n": len(seeds),
        "mean_net_gain": round(sum(net_gains) / n, 6),
        "min_net_gain": round(min(net_gains), 6),
        "max_net_gain": round(max(net_gains), 6),
        "mean_improving_steps": round(sum(improving_steps_list) / n, 4),
        "frac_ge3_improving": round(ge3 / n, 4),
        "mean_damage": round(sum(damages) / n, 6),
        "store_roundtrip_exact": bool(store_exact),
        "action_histogram": hist,
    }
