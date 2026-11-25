"""Closed-vocabulary natural-Chinese continuous-task harness (`taiji-chinese-continuous-v1`).

Zero-training facility for the "自然中文连续任务" axis (coverage matrix gap D). It generates
deterministic multi-turn Chinese instruction episodes over a closed vocabulary:

* each episode has 3-5 turns; every turn is a Chinese instruction made of slots
  (action / object / colour / place) plus one distractor clause;
* the state carries where the object currently is, so later turns can refer to it
  ("放回原位" resolves against the previous place) — i.e. continuous tasks, not independent items;
* held-out slot combinations are reserved for a generalisation split (never seen in train splits);
* observations are fixed-length token ids (closed vocab, no task_id), actions are 8 ids.

A scripted oracle (the intended action) and a uniform random policy are provided for calibration;
no model is trained here.
"""

from __future__ import annotations

import torch

TASK_VERSION = "taiji-chinese-continuous-v1"
VOCAB: tuple[str, ...] = (
    "请",
    "把",
    "拿",
    "放",
    "打开",
    "关闭",
    "检查",
    "记录",
    "等待",
    "求助",
    "红色",
    "蓝色",
    "绿色",
    "杯子",
    "书",
    "钥匙",
    "灯",
    "门",
    "桌上",
    "抽屉",
    "架上",
    "原位",
    "然后",
    "先",
    "再",
    "的",
    "，",
    "。",
    "？",
    "好",
    "不",
    "了",
    "吗",
    "一",
    "二",
    "三",
    "它",
    "那里",
    "旁边",
    "左边",
    "右边",
    "上",
    "下",
    # Action phrases must be single vocab tokens; otherwise render() would silently map them
    # to PAD and the observation would carry no action identity (bug found in NCH.1.05).
    "拿杯子",
    "放杯子",
    "打开灯",
    "关闭灯",
    "检查门",
    "记录清单",
)
VOCAB_TO_ID = {w: i for i, w in enumerate(VOCAB)}
PAD_ID = 0
MAX_TOKENS = 16

ACTIONS: tuple[str, ...] = (
    "拿杯子",
    "放杯子",
    "打开灯",
    "关闭灯",
    "检查门",
    "记录清单",
    "等待",
    "求助",
)
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

OBJECTS = ("杯子", "书", "钥匙")
COLOURS = ("红色", "蓝色", "绿色")
PLACES = ("桌上", "抽屉", "架上")
HELDOUT_COMBOS = frozenset(
    {
        ("红色", "抽屉"),
        ("蓝色", "架上"),
        ("绿色", "抽屉"),
        ("钥匙", "绿色", "桌上"),
        ("书", "红色", "架上"),
    }
)


class ChineseTaskEnv:
    """Deterministic multi-turn Chinese instruction episode."""

    def __init__(self, seed: int, *, heldout: bool = False) -> None:
        self.gen = torch.Generator().manual_seed(int(seed) % (2**63))
        self.heldout = bool(heldout)
        self.plan: list[dict] = []
        self.step_idx = 0
        self.current_place = "桌上"
        self.turn_obs: list[list[int]] = []

    # ------------------------------------------------------------------ episode
    def _sample_turn(self, i: int) -> dict:
        if self.heldout:
            combo = tuple(HELDOUT_COMBOS)[
                int(torch.randint(0, len(HELDOUT_COMBOS), (), generator=self.gen))
            ]
            if len(combo) == 2:
                colour, place = combo
                obj = OBJECTS[int(torch.randint(0, len(OBJECTS), (), generator=self.gen))]
            else:
                obj, colour, place = combo
            action = ACTIONS[int(torch.randint(0, len(ACTIONS), (), generator=self.gen))]
            distractor = "等待" if float(torch.rand((), generator=self.gen)) < 0.5 else "记录"
            return {
                "action": action,
                "object": obj,
                "colour": colour,
                "place": place,
                "distractor": distractor,
            }
        obj = OBJECTS[int(torch.randint(0, len(OBJECTS), (), generator=self.gen))]
        colour = COLOURS[int(torch.randint(0, len(COLOURS), (), generator=self.gen))]
        action = ACTIONS[int(torch.randint(0, len(ACTIONS), (), generator=self.gen))]
        if i == 0:
            place = PLACES[int(torch.randint(0, len(PLACES), (), generator=self.gen))]
        else:
            place = (
                "原位"
                if float(torch.rand((), generator=self.gen)) < 0.5
                else PLACES[int(torch.randint(0, 3, (), generator=self.gen))]
            )
        distractor = "等待" if float(torch.rand((), generator=self.gen)) < 0.5 else "记录"
        return {
            "action": action,
            "object": obj,
            "colour": colour,
            "place": place,
            "distractor": distractor,
        }

    def reset(self) -> dict:
        n_turns = 3 + int(torch.randint(0, 3, (), generator=self.gen))
        self.plan = [self._sample_turn(i) for i in range(n_turns)]
        self.step_idx = 0
        self.current_place = "桌上"
        obs = self._render(0)
        self.turn_obs = [obs]
        return {"tokens": obs, "step": 0, "turns": n_turns, "pending": n_turns - 1}

    def _render(self, i: int) -> list[int]:
        turn = self.plan[i]
        words = ["请", "先", turn["action"]]
        if turn["action"] in ("拿杯子", "放杯子"):
            words += [turn["colour"], turn["object"]]
            if turn["place"] == "原位":
                words += ["放", "回", "原位"]
            else:
                words += ["，", "然后", "放", turn["place"]]
        words += ["。", "不", "要", turn["distractor"]]
        ids = [VOCAB_TO_ID.get(w, PAD_ID) for w in words][:MAX_TOKENS]
        return ids + [PAD_ID] * (MAX_TOKENS - len(ids))

    # ------------------------------------------------------------------ step
    def step(self, action_id: int) -> tuple[dict, float, bool]:
        turn = self.plan[self.step_idx]
        correct = action_id == ACTION_TO_ID[turn["action"]]
        if turn["action"] == "拿杯子":
            self.current_place = "手上"
        elif turn["action"] == "放杯子":
            target = "原位" if turn["place"] == "原位" else turn["place"]
            self.current_place = self.current_place if target == "原位" else target
        reward = 1.0 if correct else 0.0
        self.step_idx += 1
        done = self.step_idx >= len(self.plan)
        if done:
            return (
                {
                    "tokens": [PAD_ID] * MAX_TOKENS,
                    "step": self.step_idx,
                    "turns": len(self.plan),
                    "pending": 0,
                },
                reward,
                True,
            )
        obs = self._render(self.step_idx)
        self.turn_obs.append(obs)
        return (
            {
                "tokens": obs,
                "step": self.step_idx,
                "turns": len(self.plan),
                "pending": len(self.plan) - self.step_idx,
            },
            reward,
            False,
        )

    @property
    def intended_action(self) -> int:
        return ACTION_TO_ID[self.plan[self.step_idx]["action"]]


def scripted_oracle_policy(env: ChineseTaskEnv) -> int:
    """Control arm: the intended action (never given to any learned policy)."""
    return env.intended_action


def random_policy(env: ChineseTaskEnv, gen: torch.Generator) -> int:
    return int(torch.randint(0, len(ACTIONS), (), generator=gen))


def eval_policy(
    policy, seeds: list[int], *, gen: torch.Generator | None = None, heldout: bool = False
) -> dict:
    """Evaluate a step policy: accuracy over turns, episode success, and turn count."""
    correct = total = success = 0
    rng = gen or torch.Generator().manual_seed(0)
    per_seed = []
    for seed in seeds:
        env = ChineseTaskEnv(seed, heldout=heldout)
        env.reset()
        ep_ok = True
        turns = 0
        while True:
            a = policy(env, rng) if policy is random_policy else policy(env)
            obs, reward, done = env.step(a)
            total += 1
            turns += 1
            if reward > 0:
                correct += 1
            else:
                ep_ok = False
            if done:
                break
        success += int(ep_ok)
        per_seed.append({"seed": seed, "turns": turns, "success": ep_ok})
    return {
        "n_episodes": len(seeds),
        "mean_step_acc": round(correct / max(1, total), 4),
        "episode_success": round(success / max(1, len(seeds)), 4),
        "mean_turns": round(sum(p["turns"] for p in per_seed) / max(1, len(per_seed)), 2),
        "per_seed": per_seed[:5],
    }
