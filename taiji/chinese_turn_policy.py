"""Shared recurrent Chinese turn policy and eval helpers for NCH / N34–N37."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from taiji.chinese_task_env_v2 import (
    ACTIONS,
    PAD_ID,
    VOCAB,
    ChineseTaskEnvV2,
)

EVAL_SEEDS = list(range(5100, 5140))
HELDOUT_SEEDS = list(range(5140, 5180))
ROLLOUT_EPISODES = 16
EPOCHS = 3
LR = 5e-3


class ChineseTurnPolicy(nn.Module):
    """Per-turn GRU encoder + optional episode-level recurrent memory GRU."""

    def __init__(self, seed: int, *, recurrent: bool) -> None:
        super().__init__()
        self.recurrent = bool(recurrent)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.emb = nn.Embedding(len(VOCAB), 24, padding_idx=PAD_ID)
            self.turn_gru = nn.GRU(24, 48, batch_first=True)
            if self.recurrent:
                self.ep_core: nn.Module = nn.GRU(48, 64, batch_first=True)
            else:
                self.ep_core = nn.Sequential(
                    nn.Linear(48, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
                )
            self.actor = nn.Linear(64, len(ACTIONS))
            self.critic = nn.Linear(64, 1)

    def _encode_turns(self, tokens: torch.Tensor) -> torch.Tensor:
        out, _ = self.turn_gru(self.emb(tokens))
        lengths = (tokens != PAD_ID).sum(dim=1).clamp(min=1) - 1
        return cast(torch.Tensor, out[torch.arange(tokens.shape[0]), lengths])

    def forward_sequence(self, tokens_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, t_steps, seq_len = tokens_seq.shape
        turn_vecs = self._encode_turns(tokens_seq.reshape(bsz * t_steps, seq_len)).reshape(
            bsz, t_steps, 48
        )
        h = self.ep_core(turn_vecs)[0] if self.recurrent else self.ep_core(turn_vecs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def step_hidden(
        self, tokens: torch.Tensor, h: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        vec = self._encode_turns(tokens.unsqueeze(0))
        if self.recurrent:
            out, nxt = self.ep_core(vec.unsqueeze(1), h)
            feat = out.squeeze(0).squeeze(0)
        else:
            feat = self.ep_core(vec).squeeze(0)
            nxt = None
        return self.actor(feat), self.critic(feat).squeeze(-1), nxt


def eval_chinese_turn_policy(
    policy: ChineseTurnPolicy, seeds: list[int], *, heldout: bool
) -> dict[str, Any]:
    """Evaluate `ChineseTurnPolicy` across a seed list without gradient updates."""
    policy.eval()
    correct = total = coref_ok = coref_tot = ep_ok = 0
    with torch.no_grad():
        for s in seeds:
            env = ChineseTaskEnvV2(s, heldout=heldout)
            env.reset()
            h = None
            ep_all = True
            while True:
                is_coref = env.current_is_coref
                tok = torch.tensor(env.current_tokens(), dtype=torch.long)
                logits, _, h = policy.step_hidden(tok, h)
                act = int(torch.argmax(logits, dim=-1).item())
                _, reward, done = env.step(act)
                hit = int(reward > 0.0)
                total += 1
                correct += hit
                if is_coref:
                    coref_tot += 1
                    coref_ok += hit
                if not hit:
                    ep_all = False
                if done:
                    break
            ep_ok += int(ep_all)
    policy.train()
    return {
        "mean_step_acc": round(correct / max(1, total), 4),
        "coref_acc": round(coref_ok / max(1, coref_tot), 4),
        "episode_success": round(ep_ok / max(1, len(seeds)), 4),
    }


_eval_arm = eval_chinese_turn_policy
