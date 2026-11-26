"""Closed-vocabulary Chinese continuous-task v2 (`taiji-chinese-continuous-v2`).

Fixes all v1 defects identified in review:
1. `VOCAB[0] == "<PAD>"` (`PAD_ID = 0`, separate from `"请"`); strict lookup with zero silent OOVs.
2. No action token exists in `VOCAB`: actions are 9 `(verb, place)` pairs (`取回/放到/检查` x
   `桌上/抽屉/架上`).
3. True cross-turn memory: at least 2 turns per episode use `"原位"` (home_place) or `"那里"`
   (last_place) with ZERO place tokens in the current turn observation.
4. Held-out compositional split withholds unseen `(colour, object, home_place)` anchor combinations
   while seeing every individual colour, object, verb and place in training.
"""

from __future__ import annotations

from typing import Any

import torch

TASK_VERSION = "taiji-chinese-continuous-v2"
VOCAB: tuple[str, ...] = (
    "<PAD>",
    "请",
    "把",
    "它",
    "从",
    "放",
    "回",
    "到",
    "拿",
    "起",
    "检查",
    "一下",
    "去",
    "再",
    "不要",
    "别",
    "管",
    "，",
    "。",
    "红色",
    "蓝色",
    "绿色",
    "杯子",
    "书",
    "钥匙",
    "桌上",
    "抽屉",
    "架上",
    "原位",
    "那里",
)
VOCAB_TO_ID = {w: i for i, w in enumerate(VOCAB)}
PAD_ID = 0
MAX_TOKENS = 16

VERBS: tuple[str, ...] = ("取回", "放到", "检查")
PLACES: tuple[str, ...] = ("桌上", "抽屉", "架上")
OBJECTS: tuple[str, ...] = ("杯子", "书", "钥匙")
COLOURS: tuple[str, ...] = ("红色", "蓝色", "绿色")
PLACE_TOKEN_IDS = frozenset(VOCAB_TO_ID[p] for p in PLACES)

ACTIONS: tuple[str, ...] = tuple(f"{v}_{p}" for v in VERBS for p in PLACES)
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

HELDOUT_TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("红色", "杯子", "抽屉"),
    ("蓝色", "书", "架上"),
    ("绿色", "钥匙", "桌上"),
    ("绿色", "杯子", "架上"),
    ("红色", "钥匙", "抽屉"),
    ("蓝色", "杯子", "桌上"),
)
HELDOUT_SET = frozenset(HELDOUT_TRIPLES)
TRAIN_TRIPLES: tuple[tuple[str, str, str], ...] = tuple(
    (c, o, p) for c in COLOURS for o in OBJECTS for p in PLACES if (c, o, p) not in HELDOUT_SET
)


class ChineseTaskEnvV2:
    """Deterministic multi-turn Chinese instruction episode with true cross-turn coreference."""

    def __init__(self, seed: int, *, heldout: bool = False) -> None:
        self.gen = torch.Generator().manual_seed(int(seed) % (2**63))
        self.heldout = bool(heldout)
        self.turns_data: list[dict[str, Any]] = []
        self.step_idx = 0
        self.home_place = "桌上"
        self.last_place = "桌上"

    def _rand_idx(self, n: int) -> int:
        return int(torch.randint(0, n, (), generator=self.gen))

    def _build_episode(self) -> list[dict[str, Any]]:
        triples = HELDOUT_TRIPLES if self.heldout else TRAIN_TRIPLES
        colour, obj, p0 = triples[self._rand_idx(len(triples))]
        other_places = [p for p in PLACES if p != p0]
        distractor0 = other_places[self._rand_idx(len(other_places))]
        words0 = ["请", "把", colour, obj, "从", p0, "拿", "起", "。", "别", "管", distractor0]
        home_place = p0
        last_place = p0
        turns: list[dict[str, Any]] = [
            {
                "words": words0,
                "action": f"取回_{p0}",
                "is_coref": False,
                "anchor": (colour, obj, p0),
            }
        ]

        n_turns = 4 + self._rand_idx(2)  # 4 or 5 turns
        # Guarantee at least 2 coreference turns among turns 1..n_turns-1.
        candidate_modes = ["coref_home", "coref_there", "direct_move"]
        follow_modes = ["coref_home", "coref_there"]
        while len(follow_modes) < n_turns - 1:
            follow_modes.append(candidate_modes[self._rand_idx(len(candidate_modes))])
        perm = torch.randperm(len(follow_modes), generator=self.gen).tolist()
        for idx in perm:
            mode = follow_modes[idx]
            if mode == "direct_move":
                p_new = PLACES[self._rand_idx(len(PLACES))]
                d_pool = [p for p in PLACES if p != p_new]
                dist = d_pool[self._rand_idx(len(d_pool))]
                if self._rand_idx(2) == 0:
                    words = ["不要", "去", dist, "，", "请", "把", "它", "放", "到", p_new, "。"]
                else:
                    words = ["请", "把", "它", "放", "到", p_new, "，", "别", "管", dist, "。"]
                last_place = p_new
                turns.append({"words": words, "action": f"放到_{p_new}", "is_coref": False})
            elif mode == "coref_home":
                if self._rand_idx(2) == 0:
                    words = ["请", "再", "把", "它", "放", "回", "原位", "。"]
                    act = f"放到_{home_place}"
                    last_place = home_place
                else:
                    words = ["请", "检查", "一下", "原位", "。"]
                    act = f"检查_{home_place}"
                turns.append({"words": words, "action": act, "is_coref": True})
            else:
                if self._rand_idx(2) == 0:
                    words = ["请", "去", "那里", "检查", "一下", "。"]
                    act = f"检查_{last_place}"
                else:
                    words = ["请", "再", "从", "那里", "拿", "起", "。"]
                    act = f"取回_{last_place}"
                turns.append({"words": words, "action": act, "is_coref": True})
        return turns

    @staticmethod
    def _encode(words: list[str]) -> list[int]:
        if len(words) > MAX_TOKENS:
            raise ValueError(f"utterance exceeds MAX_TOKENS={MAX_TOKENS}: {len(words)}")
        ids = [VOCAB_TO_ID[w] for w in words]  # strict lookup, zero silent OOV
        return ids + [PAD_ID] * (MAX_TOKENS - len(ids))

    def reset(self) -> dict[str, Any]:
        self.turns_data = self._build_episode()
        self.step_idx = 0
        obs_tokens = self._encode(self.turns_data[0]["words"])
        return {
            "tokens": obs_tokens,
            "step": 0,
            "turns": len(self.turns_data),
            "pending": len(self.turns_data) - 1,
        }

    def current_tokens(self) -> list[int]:
        return self._encode(self.turns_data[self.step_idx]["words"])

    @property
    def current_is_coref(self) -> bool:
        return bool(self.turns_data[self.step_idx]["is_coref"])

    @property
    def intended_action(self) -> int:
        return ACTION_TO_ID[self.turns_data[self.step_idx]["action"]]

    def step(self, action_id: int) -> tuple[dict[str, Any], float, bool]:
        reward = 1.0 if action_id == self.intended_action else 0.0
        self.step_idx += 1
        done = self.step_idx >= len(self.turns_data)
        if done:
            return (
                {
                    "tokens": [PAD_ID] * MAX_TOKENS,
                    "step": self.step_idx,
                    "turns": len(self.turns_data),
                    "pending": 0,
                },
                reward,
                True,
            )
        return (
            {
                "tokens": self.current_tokens(),
                "step": self.step_idx,
                "turns": len(self.turns_data),
                "pending": len(self.turns_data) - self.step_idx,
            },
            reward,
            False,
        )


def oracle_policy_v2(env: ChineseTaskEnvV2) -> int:
    """Control only: intended action (never used in learned training)."""
    return env.intended_action


def random_policy_v2(env: ChineseTaskEnvV2, gen: torch.Generator) -> int:
    return int(torch.randint(0, len(ACTIONS), (), generator=gen))


def eval_policy_v2(
    policy,
    seeds: list[int],
    *,
    gen: torch.Generator | None = None,
    heldout: bool = False,
) -> dict[str, Any]:
    correct = total = coref_correct = coref_total = ep_success = 0
    rng = gen or torch.Generator().manual_seed(0)
    for seed in seeds:
        env = ChineseTaskEnvV2(seed, heldout=heldout)
        env.reset()
        all_ok = True
        while True:
            is_coref = env.current_is_coref
            a = policy(env, rng) if policy is random_policy_v2 else policy(env)
            _, reward, done = env.step(a)
            hit = int(reward > 0.0)
            total += 1
            correct += hit
            if is_coref:
                coref_total += 1
                coref_correct += hit
            if not hit:
                all_ok = False
            if done:
                break
        ep_success += int(all_ok)
    return {
        "n_episodes": len(seeds),
        "mean_step_acc": round(correct / max(1, total), 4),
        "coref_acc": round(coref_correct / max(1, coref_total), 4),
        "episode_success": round(ep_success / max(1, len(seeds)), 4),
        "n_steps": total,
        "n_coref_steps": coref_total,
    }
