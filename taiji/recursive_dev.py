"""Consequence model & model-based recursive intervention policy for N33.4 (`taiji-recursive-dev-v1`).

Learns an internal forward consequence model `(state, action) -> (delta_state, damage, cost, delta_u)`
from random exploratory intervention logs (no scripted oracle), then trains an intervention
policy inside the learned consequence model (`T=4` imagined rollouts) across 3 arms:
  * `full_recursive`: real outcome targets + consequence predictions fed to policy.
  * `masked_outcome`: `[damage, cost, delta_u]` masked to 0 (结果屏蔽对照).
  * `permuted_outcome`: transition targets randomly permuted (结果置换对照).
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from taiji.recursive_dev_env import ACTIONS, HORIZON, OUTCOME_DIM, STATE_DIM, seed_from

ARCH_VERSION = "taiji-recursive-dev-v1"
NUM_ACTIONS = len(ACTIONS)
POLICY_IN_DIM = STATE_DIM + 2 * NUM_ACTIONS  # 10 + 12 = 22


class ConsequenceModel(nn.Module):
    """Forward consequence model predicting delta_state (10) and [damage, cost, delta_u] (3)."""

    def __init__(self, seed: int, hidden: int = 64) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM + NUM_ACTIONS, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, OUTCOME_DIM),
            )

    def forward(self, state_and_action: torch.Tensor) -> torch.Tensor:
        return self.net(state_and_action)


class InterventionPolicy(nn.Module):
    """Actor-critic policy conditioned on current state + consequence-model action predictions."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.body = nn.Sequential(nn.Linear(POLICY_IN_DIM, 64), nn.Tanh())
            self.pi = nn.Linear(64, NUM_ACTIONS)
            self.v = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def _transitions_to_tensors(
    transitions: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.stack([tr["s"] for tr in transitions])
    actions = torch.tensor([tr["a_idx"] for tr in transitions], dtype=torch.long)
    onehot = F.one_hot(actions, num_classes=NUM_ACTIONS).to(torch.float32)
    x = torch.cat([states, onehot], dim=1)
    y = torch.stack(
        [
            torch.cat([tr["delta_s"], torch.tensor([tr["damage"], tr["cost"], tr["delta_u"]])])
            for tr in transitions
        ]
    )
    return x, y


def build_policy_input(
    state: torch.Tensor,
    cmodel: ConsequenceModel,
    *,
    mask_preds: bool = False,
) -> torch.Tensor:
    """Attach predicted `[delta_u, damage]` across all 6 candidate actions to `state`."""
    if mask_preds:
        return torch.cat([state, torch.zeros(2 * NUM_ACTIONS, dtype=torch.float32)], dim=0)
    eye = torch.eye(NUM_ACTIONS, dtype=torch.float32)
    s_rep = state.unsqueeze(0).expand(NUM_ACTIONS, -1)
    with torch.no_grad():
        preds = cmodel(torch.cat([s_rep, eye], dim=1))
    # Columns 12 (delta_u) and 10 (damage), scaled cleanly into O(1) range
    du = preds[:, 12] * 5.0
    dmg = preds[:, 10] * 5.0
    feats = torch.stack([du, dmg], dim=1).reshape(-1)
    return torch.cat([state, feats], dim=0)


def train_consequence_model(
    train_transitions: list[dict[str, Any]],
    val_transitions: list[dict[str, Any]],
    seed: int,
    *,
    mode: str = "full",
    steps: int = 600,
    batch_size: int = 128,
    lr: float = 5e-3,
    outcome_weight_scale: bool = False,
    hidden: int = 64,
    deadline: float,
) -> tuple[ConsequenceModel, dict[str, float]]:
    """Train forward consequence model under `full`, `masked`, or `permuted` mode."""
    assert mode in ("full", "masked", "permuted")
    x_tr, y_tr = _transitions_to_tensors(train_transitions)
    x_va, y_va = _transitions_to_tensors(val_transitions)

    if mode == "masked":
        y_tr = y_tr.clone()
        y_tr[:, 10:13] = 0.0
    elif mode == "permuted":
        perm_gen = torch.Generator().manual_seed(seed_from("N33.4", seed, "permute"))
        perm = torch.randperm(y_tr.shape[0], generator=perm_gen)
        y_tr = y_tr[perm]

    outcome_weights: torch.Tensor | None = None
    if outcome_weight_scale:
        # Standardised outcome-end weights: scale [damage, cost, delta_u] columns by the ratio
        # of the state-delta variance to their own variance, so small-scale delta_u is not
        # drowned by the 10 state-delta columns. Weights are clamped to keep training stable.
        state_var = y_tr[:, :10].var(dim=0).mean().clamp(min=1e-6)
        out_var = y_tr[:, 10:13].var(dim=0).clamp(min=1e-6)
        outcome_weights = torch.ones(y_tr.shape[1])
        outcome_weights[10:13] = (state_var / out_var).clamp(max=50.0)

    cmodel = ConsequenceModel(seed_from("N33.4", seed, mode, "cmodel"), hidden=hidden)
    opt = torch.optim.Adam(cmodel.parameters(), lr=lr)
    bgen = torch.Generator().manual_seed(seed_from("N33.4", seed, mode, "cm-batches"))

    cmodel.train()
    for step in range(1, steps + 1):
        if time.process_time() > deadline:
            raise TimeoutError(f"cmodel {mode} seed={seed} step={step}")
        idx = torch.randint(0, x_tr.shape[0], (batch_size,), generator=bgen)
        if outcome_weights is None:
            loss = F.mse_loss(cmodel(x_tr[idx]), y_tr[idx])
        else:
            loss = (((cmodel(x_tr[idx]) - y_tr[idx]) ** 2) * outcome_weights).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite cmodel {mode} seed={seed}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    cmodel.eval()
    with torch.no_grad():
        pred_va = cmodel(x_va)
        true_du = y_va[:, 12]
        pred_du = pred_va[:, 12]
        ss_res = float(((true_du - pred_du) ** 2).sum())
        ss_tot = float(((true_du - true_du.mean()) ** 2).sum()) + 1e-8
        r2_du = 1.0 - ss_res / ss_tot
        val_mse = float(F.mse_loss(pred_va, y_va))
    return cmodel, {"delta_u_r2": round(r2_du, 4), "val_mse": round(val_mse, 6)}


def train_policy_in_model(
    cmodel: ConsequenceModel,
    init_states: torch.Tensor,
    seed: int,
    *,
    mode: str = "full",
    episodes: int = 1600,
    deadline: float,
) -> InterventionPolicy:
    """Train intervention policy inside the learned consequence model via deterministic PPO."""
    assert mode in ("full", "masked", "permuted")
    cmodel.eval()
    mask_preds = mode == "masked"
    policy = InterventionPolicy(seed_from("N33.4", seed, mode, "policy"))
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    sample_gen = torch.Generator().manual_seed(seed_from("N33.4", seed, mode, "sample"))
    init_gen = torch.Generator().manual_seed(seed_from("N33.4", seed, mode, "init-pick"))

    rollout_eps = 16
    total_updates = max(1, episodes // rollout_eps)
    gamma = 0.98
    lam = 0.95
    clip = 0.2

    for update in range(total_updates):
        if time.process_time() > deadline:
            raise TimeoutError(f"policy {mode} seed={seed} update={update}")
        states_in: list[torch.Tensor] = []
        actions: list[int] = []
        logps: list[torch.Tensor] = []
        rewards: list[float] = []
        values: list[torch.Tensor] = []
        dones: list[float] = []

        idx_batch = torch.randint(0, init_states.shape[0], (rollout_eps,), generator=init_gen)
        for idx in idx_batch.tolist():
            s = init_states[idx].clone()
            for step_i in range(HORIZON):
                pin = build_policy_input(s, cmodel, mask_preds=mask_preds)
                logits, value = policy(pin)
                dist = F.softmax(logits, dim=-1)
                a = int(torch.multinomial(dist, 1, generator=sample_gen).item())
                states_in.append(pin)
                actions.append(a)
                logps.append(torch.log(dist[a].clamp(min=1e-8)))
                values.append(value)

                onehot = F.one_hot(torch.tensor(a), num_classes=NUM_ACTIONS).to(torch.float32)
                with torch.no_grad():
                    out = cmodel(torch.cat([s, onehot], dim=0))
                s = torch.clamp(s + out[:STATE_DIM], 0.0, 1.0)
                rew = 0.0 if mode == "masked" else float(out[12])
                rewards.append(rew)
                dones.append(float(step_i == HORIZON - 1))

        adv = torch.zeros(len(rewards))
        gae = 0.0
        next_v = 0.0
        for t in reversed(range(len(rewards))):
            vt = float(values[t].detach())
            nt = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_v * nt - vt
            gae = delta + gamma * lam * nt * gae
            adv[t] = gae
            next_v = vt
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = (adv + torch.stack(values).detach()).detach()
        st_t = torch.stack(states_in)
        act_t = torch.tensor(actions, dtype=torch.long)
        old_lp = torch.stack(logps).detach()
        ent_coef = 0.05 + (0.005 - 0.05) * (update / total_updates)

        for _ in range(3):
            logits, vals = policy(st_t)
            dist = F.softmax(logits, dim=-1)
            lp = torch.log(dist.gather(1, act_t.unsqueeze(1)).squeeze(1).clamp(min=1e-8))
            ratio = torch.exp(lp - old_lp)
            clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
            pi_loss = -(torch.min(ratio * adv_norm, clipped * adv_norm)).mean()
            v_loss = 0.5 * F.mse_loss(vals, returns)
            ent = -(dist * torch.log(dist.clamp(min=1e-8))).sum(-1).mean()
            loss = pi_loss + v_loss - ent_coef * ent
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite policy {mode} seed={seed}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    return policy


def greedy_recursive_policy(
    policy: InterventionPolicy,
    cmodel: ConsequenceModel,
    *,
    mask_preds: bool = False,
):
    """Build deterministic greedy evaluation policy wrapping `policy` + `cmodel`."""

    def act(state: torch.Tensor) -> str:
        pin = build_policy_input(state, cmodel, mask_preds=mask_preds)
        with torch.no_grad():
            logits, _ = policy(pin)
        return ACTIONS[int(logits.argmax().item())]

    return act
