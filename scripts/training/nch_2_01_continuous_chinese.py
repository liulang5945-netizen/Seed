"""NCH.2.01 (exploration): closed-vocab Chinese continuous task with cross-turn coreference.

Design: plans/reference/DESIGN-NCH-2-01_20261126.md (D-20261126-653).
Compares `gru_recurrent` (carrying cross-turn hidden state h_t across all turns in an episode)
against `stateless_mlp` (same per-turn GRU encoder, zero cross-turn state) on `ChineseTaskEnvV2`.
Exploration seeds 6701-6703; zero confirmation seed / check / final reads.

Usage: python scripts/training/nch_2_01_continuous_chinese.py [--smoke]
Output: reports/nch_2_01_continuous_chinese.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from taiji.chinese_task_env_v2 import (  # noqa: E402
    ACTIONS,
    MAX_TOKENS,
    PAD_ID,
    PLACE_TOKEN_IDS,
    VOCAB,
    ChineseTaskEnvV2,
    eval_policy_v2,
    oracle_policy_v2,
    random_policy_v2,
)

CARD = "NCH.2.01"
DESIGN = "DESIGN-NCH-2-01_20261126.md"
TRAIN_SEEDS = (6701, 6702, 6703)
SPOT_CHECK_SEED = 6701
EVAL_SEEDS = list(range(5100, 5140))
HELDOUT_SEEDS = list(range(5140, 5180))
UPDATES = 220
ROLLOUT_EPISODES = 16
EPOCHS = 3
LR = 5e-3
ENTROPY = 0.02
VALUE_COEF = 0.5
SMOKE_UPDATES = 12
CAP_CPU = 120.0
OUT = ROOT / "reports" / "nch_2_01_continuous_chinese.json"


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
        # tokens: [N, L]
        out, _ = self.turn_gru(self.emb(tokens))
        lengths = (tokens != PAD_ID).sum(dim=1).clamp(min=1) - 1
        return out[torch.arange(tokens.shape[0]), lengths]

    def forward_sequence(self, tokens_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens_seq: [B, T, L]
        bsz, t_steps, seq_len = tokens_seq.shape
        turn_vecs = self._encode_turns(tokens_seq.reshape(bsz * t_steps, seq_len)).reshape(
            bsz, t_steps, 48
        )
        if self.recurrent:
            h, _ = self.ep_core(turn_vecs)
        else:
            h = self.ep_core(turn_vecs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def step_hidden(
        self, tokens: torch.Tensor, h: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        vec = self._encode_turns(tokens.unsqueeze(0))  # [1, 48]
        if self.recurrent:
            out, nxt = self.ep_core(vec.unsqueeze(1), h)
            feat = out.squeeze(0).squeeze(0)
        else:
            feat = self.ep_core(vec).squeeze(0)
            nxt = None
        return self.actor(feat), self.critic(feat).squeeze(-1), nxt


def _eval_arm(policy: ChineseTurnPolicy, seeds: list[int], *, heldout: bool) -> dict[str, Any]:
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


def train_arm(seed: int, *, recurrent: bool, updates: int, deadline: float) -> ChineseTurnPolicy:
    policy = ChineseTurnPolicy(seed, recurrent=recurrent)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(int(seed) * 7919 + (13 if recurrent else 37))
    ep_counter = 0

    for upd in range(updates):
        if time.process_time() > deadline:
            raise TimeoutError(f"nch_2_01 rec={recurrent} seed={seed} upd={upd}")
        max_t = 5
        tok_batch = torch.zeros((ROLLOUT_EPISODES, max_t, MAX_TOKENS), dtype=torch.long)
        act_batch = torch.zeros((ROLLOUT_EPISODES, max_t), dtype=torch.long)
        rew_batch = torch.zeros((ROLLOUT_EPISODES, max_t), dtype=torch.float32)
        old_logp = torch.zeros((ROLLOUT_EPISODES, max_t), dtype=torch.float32)
        old_val = torch.zeros((ROLLOUT_EPISODES, max_t), dtype=torch.float32)
        mask = torch.zeros((ROLLOUT_EPISODES, max_t), dtype=torch.float32)

        with torch.no_grad():
            for b in range(ROLLOUT_EPISODES):
                env = ChineseTaskEnvV2(seed * 1_000_000 + ep_counter)
                ep_counter += 1
                env.reset()
                h = None
                t = 0
                while True:
                    tok = torch.tensor(env.current_tokens(), dtype=torch.long)
                    logits, val, h = policy.step_hidden(tok, h)
                    probs = F.softmax(logits, dim=-1)
                    act = int(torch.multinomial(probs, 1, generator=gen).item())
                    _, reward, done = env.step(act)
                    tok_batch[b, t] = tok
                    act_batch[b, t] = act
                    rew_batch[b, t] = reward
                    old_logp[b, t] = torch.log(probs[act].clamp(min=1e-8))
                    old_val[b, t] = val
                    mask[b, t] = 1.0
                    t += 1
                    if done:
                        break

        active = mask > 0.5
        adv = torch.zeros_like(rew_batch)
        raw_adv = rew_batch[active] - old_val[active]
        adv[active] = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)

        for _ in range(EPOCHS):
            logits, values = policy.forward_sequence(tok_batch)
            probs = F.softmax(logits, dim=-1)
            logps = torch.log(probs.gather(2, act_batch.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8))
            ratio = torch.exp(logps[active] - old_logp[active])
            surr = torch.min(ratio * adv[active], torch.clamp(ratio, 0.8, 1.2) * adv[active])
            pi_loss = -surr.mean()
            v_loss = VALUE_COEF * F.mse_loss(values[active], rew_batch[active])
            ent = -(probs[active] * torch.log(probs[active].clamp(min=1e-8))).sum(dim=-1).mean()
            loss = pi_loss + v_loss - ENTROPY * ent
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss nch_2_01 seed={seed}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()

    policy.eval()
    return policy


def _calibrate() -> dict[str, Any]:
    ora = eval_policy_v2(oracle_policy_v2, EVAL_SEEDS)
    ora_h = eval_policy_v2(oracle_policy_v2, HELDOUT_SEEDS, heldout=True)
    rnd = eval_policy_v2(random_policy_v2, EVAL_SEEDS, gen=torch.Generator().manual_seed(0))
    const_best = max(
        eval_policy_v2(lambda _e, aid=i: aid, EVAL_SEEDS)["mean_step_acc"]
        for i in range(len(ACTIONS))
    )
    coref_clean = True
    for s in EVAL_SEEDS[:10]:
        env = ChineseTaskEnvV2(s)
        env.reset()
        while True:
            if env.current_is_coref and (set(env.current_tokens()) & PLACE_TOKEN_IDS):
                coref_clean = False
            if env.step(env.intended_action)[2]:
                break
    c0_pass = (
        ora["mean_step_acc"] == 1.0
        and ora_h["mean_step_acc"] == 1.0
        and rnd["mean_step_acc"] <= 0.20
        and const_best <= 0.25
        and coref_clean
    )
    return {
        "c0_pass": c0_pass,
        "oracle_train": ora["mean_step_acc"],
        "oracle_heldout": ora_h["mean_step_acc"],
        "random_acc": rnd["mean_step_acc"],
        "best_constant_acc": const_best,
        "coref_has_no_place_token": coref_clean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    updates = SMOKE_UPDATES if args.smoke else UPDATES

    calib = _calibrate()
    per_seed: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    if not calib["c0_pass"]:
        exit_ = "engineering_failure"
    else:
        try:
            for seed in TRAIN_SEEDS:
                t0 = time.process_time()
                pol_rec = train_arm(seed, recurrent=True, updates=updates, deadline=deadline)
                pol_stat = train_arm(seed, recurrent=False, updates=updates, deadline=deadline)
                rec_tr = _eval_arm(pol_rec, EVAL_SEEDS, heldout=False)
                rec_he = _eval_arm(pol_rec, HELDOUT_SEEDS, heldout=True)
                stat_tr = _eval_arm(pol_stat, EVAL_SEEDS, heldout=False)
                stat_he = _eval_arm(pol_stat, HELDOUT_SEEDS, heldout=True)

                exact = None
                if seed == SPOT_CHECK_SEED and not args.smoke:
                    again = train_arm(seed, recurrent=True, updates=updates, deadline=deadline)
                    exact = all(
                        torch.equal(a, b)
                        for a, b in zip(pol_rec.parameters(), again.parameters(), strict=True)
                    )
                per_seed[str(seed)] = {
                    "gru_recurrent": {"train": rec_tr, "heldout": rec_he},
                    "stateless_mlp": {"train": stat_tr, "heldout": stat_he},
                    "weights_exact": exact,
                    "cpu": round(time.process_time() - t0, 2),
                }

            checks = {
                "c0_calibration": bool(calib["c0_pass"]),
                "g1_train_recurrent": all(
                    v["gru_recurrent"]["train"]["mean_step_acc"] >= 0.85
                    and v["gru_recurrent"]["train"]["coref_acc"] >= 0.80
                    for v in per_seed.values()
                ),
                "g2_heldout_recurrent": all(
                    v["gru_recurrent"]["heldout"]["mean_step_acc"] >= 0.80
                    and v["gru_recurrent"]["heldout"]["coref_acc"] >= 0.75
                    for v in per_seed.values()
                ),
                "g3_memory_necessity": all(
                    v["gru_recurrent"]["train"]["mean_step_acc"]
                    >= v["stateless_mlp"]["train"]["mean_step_acc"] + 0.15
                    and v["gru_recurrent"]["train"]["coref_acc"]
                    >= v["stateless_mlp"]["train"]["coref_acc"] + 0.35
                    and v["stateless_mlp"]["train"]["coref_acc"] <= 0.45
                    for v in per_seed.values()
                ),
                "g4_deterministic": all(
                    v["weights_exact"] is True
                    for v in per_seed.values()
                    if v["weights_exact"] is not None
                ),
            }
            if args.smoke:
                exit_ = "smoke"
            elif all(checks.values()):
                exit_ = "chinese_continuous_v2_positive"
            else:
                exit_ = "chinese_continuous_v2_partial"
        except (TimeoutError, FloatingPointError) as exc:
            exit_ = "engineering_failure"
            per_seed["error"] = str(exc)

    payload = {
        "schema": "taiji-nch-2-01-v1",
        "card": CARD,
        "design": DESIGN,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "updates": updates,
        "rollout_episodes": ROLLOUT_EPISODES,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "heldout_seeds": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1], len(HELDOUT_SEEDS)],
        "calibration": calib,
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "oracle_used_in_training": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "checks": checks,
                "calibration": calib,
                "per_seed": per_seed,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("chinese_continuous_v2_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
