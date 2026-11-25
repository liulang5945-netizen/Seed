"""N33.2 cross-session task environment built atop SessionStore (taiji-cross-session-v1).

Tests genuine cross-session *use*: Session 1 establishes a persistent subject state (capsule,
pending item, handoff experience) and saves it via SessionStore; Session 2 runs under a fresh
environment and must read the restored cross-session record to choose which slot to work on,
whether to verify first, when to rest, or whether nothing is pending (`finish_idle`).

No continual plasticity / growth is mixed into this environment so cross-session utility is
isolated without confounding.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from taiji.choice_simulator import seed_from  # noqa: F401
from taiji.session_store import SessionRecord, SessionStore

ENV_VERSION = "taiji-cross-session-v1"
ACTIONS = ("work_0", "work_1", "work_2", "verify_pending", "rest", "finish_idle")
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
OBS_S2_DIM = 6
CROSS_DIM = 9
STATE_DIM = OBS_S2_DIM + CROSS_DIM  # 15
HORIZON = 8


@dataclass
class Session2Hidden:
    has_pending: int
    active_slot: int
    needs_verify: int
    remaining: int
    s2_energy_init: float


def run_session1(seed: int) -> tuple[SessionRecord, Session2Hidden]:
    """Deterministically generate Session 1 outcome and handoff record."""
    gen = torch.Generator().manual_seed(int(seed) % (2**63))
    has_pending = 1 if float(torch.rand((), generator=gen)) < 0.75 else 0
    if has_pending:
        active_slot = int(torch.randint(0, 3, (), generator=gen))
        needs_verify = int(torch.randint(0, 2, (), generator=gen))
        remaining = int(torch.randint(2, 4, (), generator=gen))
    else:
        active_slot = -1
        needs_verify = 0
        remaining = 0
    s2_energy_init = 0.25 + 0.65 * float(torch.rand((), generator=gen))

    rec = SessionRecord(subject_id=f"ep-{seed}")
    if has_pending:
        rec.add_pending(
            {
                "id": f"task-{active_slot}",
                "slot": active_slot,
                "remaining": remaining,
                "needs_verify": bool(needs_verify),
            }
        )
        rec.append_experience(
            "s1_handoff",
            {"slot": active_slot, "needs_verify": bool(needs_verify), "remaining": remaining},
            source="session1",
        )
    else:
        rec.append_experience("s1_handoff", {"status": "all_done"}, source="session1")

    rec.capsule = {
        "s1_state": torch.tensor(
            [
                float(has_pending),
                float(active_slot == 0),
                float(active_slot == 1),
                float(active_slot == 2),
                float(needs_verify),
                remaining / 3.0,
            ],
            dtype=torch.float32,
        )
    }
    hidden = Session2Hidden(
        has_pending=has_pending,
        active_slot=active_slot,
        needs_verify=needs_verify,
        remaining=remaining,
        s2_energy_init=s2_energy_init,
    )
    return rec, hidden


def encode_s1_record(record: SessionRecord) -> torch.Tensor:
    """Encode a loaded SessionRecord into a 9-dim float32 cross-session observation vector."""
    cap = record.capsule["s1_state"].detach().to(torch.float32).reshape(6)
    has_pend = 1.0 if record.pending else 0.0
    exp_verify = (
        1.0
        if any(
            e.get("kind") == "s1_handoff" and bool(e.get("payload", {}).get("needs_verify"))
            for e in record.experience
        )
        else 0.0
    )
    exp_density = min(1.0, len(record.experience) / 3.0)
    tail = torch.tensor([has_pend, exp_verify, exp_density], dtype=torch.float32)
    return torch.cat([cap, tail], dim=0)


class CrossSessionEnv:
    """Session 2 environment whose ground-truth task depends on Session 1's handoff."""

    def __init__(
        self,
        seed: int,
        loaded_record: SessionRecord | None = None,
        *,
        zero_cross: bool = False,
    ) -> None:
        s1_rec, hidden = run_session1(seed)
        active_record = loaded_record if loaded_record is not None else s1_rec
        self.cross_vec = (
            torch.zeros(CROSS_DIM, dtype=torch.float32)
            if zero_cross
            else encode_s1_record(active_record)
        )
        self.t = 0
        self.energy = hidden.s2_energy_init
        self.has_pending = hidden.has_pending
        self.active_slot = hidden.active_slot
        self.needs_verify = hidden.needs_verify
        self.remaining = hidden.remaining
        self.worked = 0
        self.s2_verified = 0
        self.resolved = 0
        self.last_ok = 0.0

    def obs(self) -> torch.Tensor:
        s2 = torch.tensor(
            [
                self.t / HORIZON,
                self.energy,
                self.last_ok,
                float(self.s2_verified),
                self.worked / 3.0,
                float(self.resolved),
            ],
            dtype=torch.float32,
        )
        return torch.cat([s2, self.cross_vec], dim=0)

    def reset(self) -> torch.Tensor:
        return self.obs()

    def step(self, action: str) -> tuple[torch.Tensor, float, bool]:
        if action not in ACTION_INDEX:
            raise ValueError(f"unknown action {action!r}")
        reward = -0.05
        ok = 0.0

        if action == "finish_idle":
            if self.has_pending == 0:
                reward += 0.40
                ok = 1.0
            else:
                reward -= 0.60
        elif action == "rest":
            if self.energy < 0.40 and self.has_pending == 1:
                self.energy = min(1.0, self.energy + 0.45)
                reward += 0.35
                ok = 1.0
            else:
                reward -= 0.50
        elif action == "verify_pending":
            if self.has_pending == 1 and self.needs_verify == 1 and self.energy >= 0.35:
                self.needs_verify = 0
                self.s2_verified = 1
                self.energy = max(0.0, self.energy - 0.10)
                reward += 1.20
                ok = 1.0
            else:
                reward -= 0.60
        else:
            slot = ACTION_INDEX[action]  # 0, 1, 2
            if (
                self.has_pending == 1
                and slot == self.active_slot
                and self.needs_verify == 0
                and self.energy >= 0.35
            ):
                self.remaining -= 1
                self.worked += 1
                self.energy = max(0.0, self.energy - 0.15)
                reward += 1.00
                ok = 1.0
                if self.remaining == 0:
                    self.has_pending = 0
                    self.resolved = 1
                    reward += 2.50
            elif self.has_pending == 1 and slot == self.active_slot and self.needs_verify == 1:
                reward -= 0.70
            else:
                reward -= 0.60

        self.last_ok = ok
        self.t += 1
        return self.obs(), reward, self.t >= HORIZON


def scripted_persistent_policy(obs: torch.Tensor) -> str:
    """Reference upper-bound heuristic using cross-session state (evaluation contrast only)."""
    energy = float(obs[1])
    s2_verified = float(obs[3])
    resolved = float(obs[5])
    s1_has_pending = float(obs[6])
    s1_needs_verify = float(obs[10])
    if s1_has_pending < 0.5 or resolved > 0.5:
        return "finish_idle"
    if energy < 0.35:
        return "rest"
    if s1_needs_verify > 0.5 and s2_verified < 0.5:
        return "verify_pending"
    slot = int(obs[7:10].argmax().item())
    return f"work_{slot}"


def random_policy(_obs: torch.Tensor, gen: torch.Generator | None = None) -> str:
    g = gen if gen is not None else torch.Generator().manual_seed(0)
    return ACTIONS[int(torch.randint(0, len(ACTIONS), (), generator=g))]


def evaluate_policy_with_store(
    policy: Callable[[torch.Tensor], str],
    seeds: list[int],
    *,
    mode: str = "normal",
) -> dict[str, Any]:
    """Run Session 1 -> SessionStore save -> load -> Session 2 -> post-S2 resolution save."""
    if mode not in ("normal", "zero_cross", "swap_cross"):
        raise ValueError(f"invalid mode {mode!r}")

    returns: list[float] = []
    hist = {a: 0 for a in ACTIONS}
    pending_episodes = 0
    resolved_episodes = 0
    roundtrip_exact = True
    post_s2_persist_exact = True

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(tmpdir)
        digests: dict[int, str] = {}
        capsules: dict[int, torch.Tensor] = {}
        for s in seeds:
            rec, _ = run_session1(s)
            digests[s] = rec.digest()
            capsules[s] = rec.capsule["s1_state"].clone()
            store.save(rec)

        for i, s in enumerate(seeds):
            loaded = store.load(f"ep-{s}")
            if loaded.digest() != digests[s] or not torch.equal(
                loaded.capsule["s1_state"], capsules[s]
            ):
                roundtrip_exact = False

            if mode == "normal":
                env = CrossSessionEnv(s, loaded_record=loaded, zero_cross=False)
            elif mode == "zero_cross":
                env = CrossSessionEnv(s, loaded_record=loaded, zero_cross=True)
            else:
                other = store.load(f"ep-{seeds[(i + 1) % len(seeds)]}")
                env = CrossSessionEnv(s, loaded_record=other, zero_cross=False)

            had_pending = env.has_pending == 1
            pending_episodes += int(had_pending)
            obs = env.reset()
            ep_ret = 0.0
            for _ in range(HORIZON):
                act = policy(obs)
                hist[act] += 1
                obs, rew, done = env.step(act)
                ep_ret += rew
                if done:
                    break
            returns.append(ep_ret)

            if had_pending and env.resolved == 1:
                resolved_episodes += 1
                if mode == "normal" and loaded.pending:
                    exp_before = len(loaded.experience)
                    loaded.resolve_pending(loaded.pending[0]["id"], "done")
                    store.save(loaded)
                    reloaded = store.load(f"ep-{s}")
                    if reloaded.pending != [] or len(reloaded.experience) <= exp_before:
                        post_s2_persist_exact = False

    return {
        "n": len(seeds),
        "mean_return": round(sum(returns) / max(1, len(returns)), 6),
        "min_return": round(min(returns), 6),
        "max_return": round(max(returns), 6),
        "pending_episodes": pending_episodes,
        "resolved_episodes": resolved_episodes,
        "resolve_rate": round(resolved_episodes / max(1, pending_episodes), 4),
        "store_roundtrip_exact": bool(roundtrip_exact),
        "post_s2_persist_exact": bool(post_s2_persist_exact),
        "action_histogram": hist,
    }
