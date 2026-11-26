"""P26-R2: reward-driven autonomous choice (AWR/REINFORCE) from real episode receipts.

Same session world as P26-S2 (12 steps, 3 items, energy 0..2, pending verification,
5 actions, no prompt channel) but the Subject72 (cn block + 5-action head) starts from a
random policy and learns ONLY from episode returns (solved items + 0.15*energy).  Data is
collected with epsilon-greedy; updates use advantage-weighted regression with normalised
batch advantages.  Final 300 greedy episodes are compared to random / always-continue /
always-help / heuristic / DP-optimal baselines and to state-channel ablations.

Single seed, <=500 CPU, zero check/final. Standing authorization; owner may revise.
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
from scripts.training.verify_taiji_p26s2_20261123 import (
    ACTIONS,
    N_ITEMS,
    SEED,
    H,
    SessionEnv,
    build_dp,
    build_subject,
    dp_action,
    encode_states,
    model_logits,
)

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_p26r2_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_p26r2_20261123.json"


def collect_episodes(subj, n: int, seed: int, epsilon: float) -> list[dict]:
    rng = random.Random(seed)
    env = SessionEnv(n, seed)
    traj = [[] for _ in range(n)]
    for _ in range(H):
        states = env.states()
        ticks = encode_states(states, H - env.t)
        subj.eval_mode()
        with torch.no_grad():
            probs = torch.softmax(model_logits(subj, ticks), dim=-1)
        actions = []
        for i in range(n):
            if rng.random() < epsilon:
                actions.append(rng.randrange(len(ACTIONS)))
            else:
                actions.append(int(torch.multinomial(probs[i], 1)))
        for i in range(n):
            traj[i].append((ticks[i], actions[i]))
        env.step(actions)
    utils = env.utilities()
    return [{"traj": traj[i], "return": utils[i], "solved": env.solved[i]} for i in range(n)]


def greedy_rollout(subj, n: int, seed: int, zero_groups=()) -> dict:
    env = SessionEnv(n, seed)
    for _ in range(H):
        states = env.states()
        ticks = encode_states(states, H - env.t)
        subj.eval_mode()
        with torch.no_grad():
            actions = model_logits(subj, ticks, zero_groups).argmax(-1).tolist()
        env.step(actions)
    return {
        "mean_utility": sum(env.utilities()) / n,
        "mean_solved": sum(env.solved) / n,
        "action_counts": {ACTIONS[i]: env.action_counts[i] for i in range(len(ACTIONS))},
        "rest_given_zero": (
            env.stats["rest_zero"] / env.stats["zero_seen"] if env.stats["zero_seen"] else 0.0
        ),
        "wait_given_pending": (
            env.stats["wait_pending"] / env.stats["pending_seen"]
            if env.stats["pending_seen"]
            else 0.0
        ),
    }


def policy_rollout(policy_fn, n: int, seed: int) -> float:
    env = SessionEnv(n, seed)
    for _ in range(H):
        actions = [policy_fn(s) for s in env.states()]
        env.step(actions)
    return sum(env.utilities()) / n


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
    rng = random.Random(SEED + 13)

    dp = build_dp()
    subj = build_subject()
    opt = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    n_batches = p["train_batches"]
    curve = []
    for batch in range(n_batches):
        epsilon = p["epsilon_start"] + (p["epsilon_end"] - p["epsilon_start"]) * batch / max(
            1, n_batches - 1
        )
        eps = collect_episodes(subj, p["episodes_per_batch"], SEED * 31 + batch, epsilon)
        returns = torch.tensor([e["return"] for e in eps])
        adv = (returns - returns.mean()) / (returns.std() + 1e-6)
        weights = torch.exp(adv).clamp(0.0, p["awr_clip"])
        samples = []
        for i, e in enumerate(eps):
            w = float(weights[i])
            for tick, action in e["traj"]:
                samples.append((tick, action, w))
        for _epoch in range(p["epochs_per_batch"]):
            rng.shuffle(samples)
            for start in range(0, len(samples), p["minibatch"]):
                chunk = samples[start : start + p["minibatch"]]
                ticks = torch.stack([c[0] for c in chunk])
                actions = torch.tensor([c[1] for c in chunk])
                w = torch.tensor([c[2] for c in chunk])
                logp = torch.log_softmax(model_logits(subj, ticks), dim=-1)
                loss = -(w * logp[torch.arange(len(chunk)), actions]).sum() / w.sum().clamp(
                    min=1e-6
                )
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
        if (batch + 1) % p["eval_every"] == 0:
            valid = greedy_rollout(subj, p["valid_episodes"], SEED * 41 + batch)
            curve.append({"batch": batch + 1, "utility": valid["mean_utility"]})

    n_eval = p["eval_episodes"]
    seed_eval = SEED + p["eval_seed_offset"]
    learned = greedy_rollout(subj, n_eval, seed_eval)

    def heuristic(state):
        solved, conf, energy, pending = state
        if solved >= N_ITEMS:
            return 3
        if pending == 1:
            return 4
        if conf == 0 and energy >= 1:
            return 1
        if energy == 0:
            return 3
        return 0

    baselines = {
        "always_continue": policy_rollout(lambda s: 0, n_eval, seed_eval),
        "always_help": policy_rollout(lambda s: 2, n_eval, seed_eval),
        "random": policy_rollout(lambda s: rng.randrange(len(ACTIONS)), n_eval, seed_eval),
        "heuristic": policy_rollout(heuristic, n_eval, seed_eval),
        "dp_optimal": policy_rollout(lambda s: dp_action(dp, max(1, H), s), n_eval, seed_eval),
    }
    ablations = {
        "zero_energy": greedy_rollout(subj, n_eval, seed_eval, ("energy",)),
        "zero_conf": greedy_rollout(subj, n_eval, seed_eval, ("conf",)),
        "zero_solved": greedy_rollout(subj, n_eval, seed_eval, ("solved",)),
    }
    capsule_diff = capsule_state_diff(
        subj,
        torch.stack([encode_states([(0, 0, 2, 0)], 12)[0] for _ in range(32)]),
        torch.ones(subj.n_units),
    )
    best_fixed = max(
        baselines[k] for k in ("always_continue", "always_help", "random", "heuristic")
    )
    total_actions = sum(learned["action_counts"].values())
    shares = {a: c / total_actions for a, c in learned["action_counts"].items()}
    used_actions = sum(1 for c in shares.values() if c >= crit["gates"]["G3_min_action_share"])
    gates = {
        "G0": learned["mean_utility"] >= best_fixed + crit["gates"]["G0_gain_over_fixed_min"],
        "G1": learned["mean_utility"] >= baselines["dp_optimal"] - crit["gates"]["G1_dp_gap_max"],
        "G2": (
            learned["mean_utility"] - ablations["zero_energy"]["mean_utility"]
            >= crit["gates"]["G2_energy_drop_min"]
            and learned["mean_utility"] - ablations["zero_conf"]["mean_utility"]
            >= crit["gates"]["G2_conf_drop_min"]
        ),
        "G3": used_actions >= 3
        and learned["rest_given_zero"] >= crit["gates"]["G3_rest_given_zero_min"],
        "G4": capsule_diff <= crit["gates"]["G4_capsule_diff_max"],
    }
    if not gates["G0"]:
        verdict = "p26r2_no_gain"
    elif not gates["G1"]:
        verdict = "p26r2_suboptimal"
    elif not gates["G2"]:
        verdict = "p26r2_state_blind"
    elif not gates["G3"]:
        verdict = "p26r2_behaviour_weak"
    elif all(gates.values()):
        verdict = "p26r2_verified"
    else:
        verdict = "p26r2_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-p26r-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "curve": curve,
        "learned": learned,
        "action_shares": shares,
        "used_actions": used_actions,
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
                "learned_utility": round(learned["mean_utility"], 4),
                "learned_solved": round(learned["mean_solved"], 4),
                "action_shares": {a: round(v, 3) for a, v in shares.items()},
                "used_actions": used_actions,
                "baselines": {k: round(v, 4) for k, v in baselines.items()},
                "ablations": {k: round(v["mean_utility"], 4) for k, v in ablations.items()},
                "best_fixed": round(best_fixed, 4),
                "curve": [(c["batch"], round(c["utility"], 3)) for c in curve],
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
