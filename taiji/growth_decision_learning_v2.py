"""Learned state-dependent growth decisions v2 (`taiji-growth-decision-v2`).

Fixes the defects in v1 (`N34.3.01/02`):
1. Snapshot-fork lifecycle: `prepare_warmup()` runs Phase 1 + Phase 2 warm-up once and snapshots
   weights/stage, eliminating the 2x wasted probe episode during training & evaluation.
2. Proper actor-critic (`GrowthActorCritic`) with learned value baseline and batch-standardised
   advantages + entropy regularisation (replaces the buggy `baseline = 0.0` constant-lock loop).
3. Applies confirmed `N33.3.04` selective plasticity (`mature_()` + `lr = 1e-2` post-growth) so
   growing new capacity actually achieves high Phase 2 accuracy on hard parity within budget,
   making `no_grow` optimal on easy tasks (`-cost` penalty) and growing optimal on hard tasks.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from taiji.continual_growth_task import IN_DIM, NUM_CLASSES
from taiji.growth_decision_learning import make_task
from taiji.plastic_growth import PlasticGrowthNet

SCHEMA = "taiji-growth-decision-v2"
ACTIONS = ("no_grow", "grow8", "grow16", "grow24")
GROW_K = {"no_grow": 0, "grow8": 8, "grow16": 16, "grow24": 24}
COST_PER_UNIT = 0.002
P1_STEPS = 180
P2_WARMUP = 80
P2_STEPS = 360
BATCH = 64
LR_BASE = 5e-3
LR_POST_GROWTH = 1e-2


@dataclass
class WarmupSnapshot:
    seed: int
    hard: bool
    state_dict: dict[str, torch.Tensor]
    stage: torch.Tensor
    unit_ids: torch.Tensor
    next_unit_id: int
    data: dict[str, Any]
    p1_acc: float
    val_err: float
    val_loss: float


def _batches(
    gen: torch.Generator, x: torch.Tensor, y: torch.Tensor, n: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for _ in range(n):
        idx = torch.randint(0, x.shape[0], (BATCH,), generator=gen)
        out.append((x[idx], y[idx]))
    return out


def _eval_acc_loss(net: PlasticGrowthNet, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    net.eval()
    with torch.no_grad():
        logits = net(x)
        acc = float((logits.argmax(dim=-1) == y).float().mean())
        loss = float(F.cross_entropy(logits, y))
    net.train()
    return acc, loss


def prepare_warmup(seed: int, *, hard: bool, deadline: float) -> WarmupSnapshot:
    """Run Phase 1 and Phase 2 warm-up once and snapshot state prior to growth decision."""
    data = make_task(seed, hard=hard)
    p1_tr, p1_va = data["phase1"]["train"], data["phase1"]["val"]
    p2_tr, p2_va = data["phase2"]["train"], data["phase2"]["val"]

    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR_BASE)
    gen1 = torch.Generator().manual_seed(seed * 7919 + 1)
    gen2 = torch.Generator().manual_seed(seed * 7919 + 2)

    net.train()
    for xb, yb in _batches(gen1, p1_tr["x"], p1_tr["y"], P1_STEPS):
        if time.process_time() > deadline:
            raise TimeoutError("growth-decision-v2 p1")
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    p1_acc, _ = _eval_acc_loss(net, p1_va["x"], p1_va["y"])
    half = BATCH // 2
    for xb2, yb2 in _batches(gen2, p2_tr["x"], p2_tr["y"], P2_WARMUP):
        if time.process_time() > deadline:
            raise TimeoutError("growth-decision-v2 warmup")
        idx = torch.randint(0, p1_tr["x"].shape[0], (half,), generator=gen2)
        xb = torch.cat([xb2[:half], p1_tr["x"][idx]], dim=0)
        yb = torch.cat([yb2[:half], p1_tr["y"][idx]], dim=0)
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    p2_acc, p2_loss = _eval_acc_loss(net, p2_va["x"], p2_va["y"])
    return WarmupSnapshot(
        seed=seed,
        hard=hard,
        state_dict=copy.deepcopy(net.state_dict()),
        stage=net.stage.clone(),
        unit_ids=net.unit_ids.clone(),
        next_unit_id=net._next_unit_id,
        data=data,
        p1_acc=p1_acc,
        val_err=1.0 - p2_acc,
        val_loss=p2_loss,
    )


def finish_from_warmup(snap: WarmupSnapshot, action: str, deadline: float) -> dict[str, Any]:
    """Fork from warmup snapshot, apply selected growth action, and finish Phase 2 training."""
    if action not in GROW_K:
        raise ValueError(f"unknown action {action!r}")
    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=snap.seed)
    net.load_state_dict(copy.deepcopy(snap.state_dict))
    net.stage.copy_(snap.stage)
    net.unit_ids.copy_(snap.unit_ids)
    net._next_unit_id = snap.next_unit_id

    k = GROW_K[action]
    if k > 0:
        net.mature_()
        net.grow_neutral_(k, seed=snap.seed * 13 + k)
        lr = LR_POST_GROWTH
    else:
        lr = LR_BASE

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen_fin = torch.Generator().manual_seed(snap.seed * 7919 + 303 + k)
    p1_tr, p1_va = snap.data["phase1"]["train"], snap.data["phase1"]["val"]
    p2_tr, p2_va = snap.data["phase2"]["train"], snap.data["phase2"]["val"]
    half = BATCH // 2

    net.train()
    for xb2, yb2 in _batches(gen_fin, p2_tr["x"], p2_tr["y"], P2_STEPS):
        if time.process_time() > deadline:
            raise TimeoutError("growth-decision-v2 p2")
        idx = torch.randint(0, p1_tr["x"].shape[0], (half,), generator=gen_fin)
        xb = torch.cat([xb2[:half], p1_tr["x"][idx]], dim=0)
        yb = torch.cat([yb2[:half], p1_tr["y"][idx]], dim=0)
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if k > 0:
            net.apply_selective_plasticity_grad_mask(freeze_mature=True)
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()

    p1_after, _ = _eval_acc_loss(net, p1_va["x"], p1_va["y"])
    p2_acc, _ = _eval_acc_loss(net, p2_va["x"], p2_va["y"])
    combined = 0.5 * (p1_after + p2_acc)
    reward = combined - COST_PER_UNIT * k
    return {
        "seed": snap.seed,
        "hard": snap.hard,
        "action": action,
        "p1_acc": round(snap.p1_acc, 6),
        "val_err": round(snap.val_err, 6),
        "p1_after": round(p1_after, 6),
        "p2_acc": round(p2_acc, 6),
        "combined": round(combined, 6),
        "reward": round(reward, 6),
        "units": k,
    }


def snapshot_features(snap: WarmupSnapshot) -> torch.Tensor:
    return torch.tensor(
        [
            snap.val_err,
            snap.p1_acc,
            min(1.0, snap.val_loss / 2.0),
            1.0 - snap.val_err,
        ],
        dtype=torch.float32,
    )


class GrowthActorCritic(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.trunk = nn.Sequential(
                nn.Linear(4, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
            )
            self.actor = nn.Linear(32, len(ACTIONS))
            self.critic = nn.Linear(32, 1)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = feats.dim() == 1
        if single:
            feats = feats.unsqueeze(0)
        h = self.trunk(feats)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


def greedy_action_v2(policy: GrowthActorCritic, snap: WarmupSnapshot) -> int:
    with torch.no_grad():
        logits, _ = policy(snapshot_features(snap))
    return int(torch.argmax(logits, dim=-1).item())


def train_actor_critic(
    snapshots: list[WarmupSnapshot],
    seed: int,
    deadline: float,
    *,
    n_updates: int = 60,
    batch_ep: int = 8,
    ppo_epochs: int = 4,
    lr: float = 8e-3,
    entropy_coef: float = 0.03,
) -> GrowthActorCritic:
    """Train contextual-bandit actor-critic over pre-computed warmup snapshots."""
    policy = GrowthActorCritic(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 999 + 17)
    easy_snaps = [s for s in snapshots if not s.hard]
    hard_snaps = [s for s in snapshots if s.hard]
    assert easy_snaps and hard_snaps

    cache: dict[tuple[int, bool, str], dict[str, Any]] = {}
    for _upd in range(n_updates):
        batch_snaps: list[WarmupSnapshot] = []
        half_b = batch_ep // 2
        for _ in range(half_b):
            ei = int(torch.randint(0, len(easy_snaps), (), generator=gen))
            hi = int(torch.randint(0, len(hard_snaps), (), generator=gen))
            batch_snaps.extend([easy_snaps[ei], hard_snaps[hi]])

        feats_list: list[torch.Tensor] = []
        acts: list[int] = []
        old_logps: list[torch.Tensor] = []
        rews: list[float] = []
        vals: list[torch.Tensor] = []
        for snap in batch_snaps:
            f = snapshot_features(snap)
            with torch.no_grad():
                logits, val = policy(f)
                probs = F.softmax(logits, dim=-1)
                a = int(torch.multinomial(probs, 1, generator=gen).item())
                logp = torch.log(probs[a].clamp(min=1e-8))
            key = (snap.seed, snap.hard, ACTIONS[a])
            if key not in cache:
                cache[key] = finish_from_warmup(snap, ACTIONS[a], deadline)
            feats_list.append(f)
            acts.append(a)
            old_logps.append(logp)
            vals.append(val)
            rews.append(float(cache[key]["reward"]))

        feats_t = torch.stack(feats_list)
        acts_t = torch.tensor(acts, dtype=torch.long)
        old_logps_t = torch.stack(old_logps)
        rews_t = torch.tensor(rews, dtype=torch.float32)
        vals_t = torch.stack(vals)
        adv = rews_t - vals_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        for _ in range(ppo_epochs):
            logits, values = policy(feats_t)
            probs = F.softmax(logits, dim=-1)
            new_logps = torch.log(probs.gather(1, acts_t.unsqueeze(1)).squeeze(1).clamp(min=1e-8))
            ratio = torch.exp(new_logps - old_logps_t)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
            pi_loss = -surr.mean()
            v_loss = 2.0 * F.mse_loss(values, rews_t)
            ent = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1).mean()
            loss = pi_loss + v_loss - entropy_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()

    policy.eval()
    return policy
