"""Fixed probe-state set for N33.1 (measurement-only, D-20261125-567).

The N33.1.06 review found that policy-conditioned probe trajectories collapse (a policy that
rests early almost never reaches low-energy states), so G2 denominators become tiny and
unstable. N33.1.07 freezes a neutral probe set instead:

* sampled from neutral random-policy rollouts (fixture v4), never from a learned/scripted policy;
* stratified so each of the three G2 domains has >= 40 states (K = 150 total);
* deterministic: manifest JSON with sha256, no confirmation seed involved.

The G2 thresholds are unchanged (share >= 0.50 per domain); this module only fixes *where*
the share is measured.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from taiji.choice_simulator import CHOICES, HORIZON, ChoiceSessionEnv, State

PROBE_PATH = ROOT / "plans" / "manifests" / "n33_1_probe_states_20261125.json"
PROBE_SEED = 424242
K_TOTAL = 150
MIN_PER_DOMAIN = 40
DOMAINS = ("low_energy_pending", "high_uncertainty_pending", "pending_work")


def domain_of(state: State) -> str | None:
    if state.energy < 0.35 and state.pending > 0 and state.uncertainty <= 0.55:
        return "low_energy_pending"
    if state.uncertainty > 0.55 and state.pending > 0 and state.energy >= 0.35:
        return "high_uncertainty_pending"
    if (
        state.pending > 0
        and state.progress < 0.3
        and state.energy >= 0.35
        and state.uncertainty <= 0.55
    ):
        return "pending_work"
    return None


def _state_from_vec(vec: list[float]) -> State:
    return State(*[float(x) for x in vec])


def build_probe_set() -> dict:
    """Neutral random-policy states, stratified per domain; deterministic via PROBE_SEED."""
    gen = torch.Generator().manual_seed(PROBE_SEED)
    pool: dict[str, list[list[float]]] = {d: [] for d in DOMAINS}
    for seed in range(9000, 9400):
        env = ChoiceSessionEnv(seed)
        state = env.reset()
        for _ in range(HORIZON):
            d = domain_of(state)
            if d is not None:
                pool[d].append([round(float(x), 6) for x in state.vector().tolist()])
            # neutral random action, but consume our own generator for full determinism
            a = int(torch.randint(0, len(CHOICES), (), generator=gen))
            state, _, done = env.step(CHOICES[a])
            if done:
                break
    # deterministic stratified selection; top up from uniform prior if a domain is short
    taken: dict[str, list[list[float]]] = {}
    for d in DOMAINS:
        uniq = sorted({tuple(v) for v in pool[d]})
        per = K_TOTAL // len(DOMAINS)
        chosen = [list(v) for v in uniq[:per]]
        if len(chosen) < MIN_PER_DOMAIN:
            tag = int(hashlib.sha256(d.encode()).hexdigest()[:8], 16) % 1000
            u = torch.Generator().manual_seed(PROBE_SEED + tag)
            while len(chosen) < MIN_PER_DOMAIN:
                vec = _top_up_state(d, u)
                chosen.append([round(float(x), 6) for x in vec.tolist()])
        taken[d] = chosen
    states: list[list[float]] = []
    domains: list[str] = []
    for d in DOMAINS:
        for v in taken[d]:
            states.append(v)
            domains.append(d)
    payload = {
        "schema": "taiji-n33-1-probes-v1",
        "fixture": "taiji-choice-session-v4",
        "probe_seed": PROBE_SEED,
        "k_total": len(states),
        "domains": DOMAINS,
        "states": states,
        "state_domains": domains,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["sha256"] = digest
    return payload


def _top_up_state(domain: str, gen: torch.Generator) -> torch.Tensor:
    def u() -> float:
        return float(torch.rand((), generator=gen))

    if domain == "low_energy_pending":
        return torch.tensor([1.0, u(), 0.05 + 0.25 * u(), 0.5 * u(), 0.0, 0.6 * u(), 0.5 * u()])
    if domain == "high_uncertainty_pending":
        return torch.tensor(
            [1.0, 0.2 + 0.8 * u(), 0.35 + 0.65 * u(), 0.6 + 0.4 * u(), 0.0, 0.6 * u(), 0.5 * u()]
        )
    return torch.tensor(
        [1.0, 0.2 + 0.8 * u(), 0.35 + 0.65 * u(), 0.55 * u(), 0.0, 0.29 * u(), 0.5 * u()]
    )


def save_probe_set(payload: dict | None = None) -> dict:
    payload = payload or build_probe_set()
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROBE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return payload


def load_probe_set() -> dict:
    if not PROBE_PATH.exists():
        save_probe_set()
    return json.loads(PROBE_PATH.read_text(encoding="utf-8"))


def probe_shares(action_fn: Callable[[State], str]) -> dict[str, float | int]:
    """G2 shares on the frozen probe states (thresholds unchanged: >= 0.50)."""
    payload = load_probe_set()
    hits = {d: 0 for d in DOMAINS}
    total = {d: 0 for d in DOMAINS}
    for vec, d in zip(payload["states"], payload["state_domains"], strict=True):
        state = _state_from_vec(vec)
        action = action_fn(state)
        total[d] += 1
        if d == "low_energy_pending":
            hits[d] += int(action == "rest")
        elif d == "high_uncertainty_pending":
            hits[d] += int(action in ("verify", "ask_help"))
        else:
            hits[d] += int(action in ("continue_task", "recall_learn"))
    return {
        "n": dict(total),
        "rest_share": round(hits["low_energy_pending"] / max(1, total["low_energy_pending"]), 4),
        "verify_or_help_share": round(
            hits["high_uncertainty_pending"] / max(1, total["high_uncertainty_pending"]), 4
        ),
        "continue_or_recall_share": round(hits["pending_work"] / max(1, total["pending_work"]), 4),
    }
