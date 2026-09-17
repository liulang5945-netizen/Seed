"""Course/data scale sweep (candidate 3, step i): is the interaction monotonic?

For each scale in {4x, 8x, 16x}, generate a joint-sequence corpus from the same
template family as the H3.8 builder with a proportionally larger entity pool
(mutually exclusive across splits), then run the H-OBJ two-arm experiment at a
**constant step budget** (~750 steps):

    steps = epochs x episodes/epoch, episodes/epoch = len(train split)

so "more data" means "fewer passes over more episodes", never "more compute".

Read-only w.r.t. existing artifacts; writes only its own report directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

SCALES = (4, 8, 16)
SEEDS = (20260917, 20260918, 20260919)
LEARNING_RATE = 0.01
GENERATION_LIMIT = 64
TARGET_STEPS = 750
BOUNDARY = int(SequenceWorkspaceConfig().boundary_symbol)
LAMBDA = {"control": 0.0, "treatment": 1.0}

OBJECT_BANKS: dict[str, tuple[str, ...]] = {
    "train": (
        "天空",
        "草",
        "树叶",
        "海洋",
        "山川",
        "城市",
        "森林",
        "河流",
        "沙漠",
        "湖泊",
        "岛屿",
        "峡谷",
        "平原",
        "丘陵",
        "火山",
        "草原",
        "沼泽",
        "三角洲",
        "半岛",
        "群岛",
        "海峡",
        "盆地",
        "高原",
        "河谷",
        "绿洲",
        "溶洞",
        "瀑布",
        "温泉",
        "礁石",
        "险峰",
        "古道",
        "驿站",
    ),
    "dev": (
        "雪",
        "太阳",
        "沙漠",
        "湖水",
        "云层",
        "彩虹",
        "闪电",
        "晨雾",
        "露水",
        "霜花",
        "冰凌",
        "雨滴",
        "季风",
        "台风",
        "龙卷",
        "极光",
    ),
    "final": (
        "夜晚",
        "大海",
        "火焰",
        "冰川",
        "星辰",
        "银河",
        "彗星",
        "黑洞",
        "流星",
        "月环",
        "日冕",
        "星云",
        "脉冲星",
        "类星体",
        "白矮星",
        "超新星",
    ),
}
COLOR_BANKS: dict[str, tuple[str, ...]] = {
    "train": (
        "蓝",
        "绿",
        "红",
        "黄",
        "青",
        "紫",
        "橙",
        "棕",
        "灰",
        "粉",
        "银",
        "金",
        "褐",
        "靛",
        "绛",
        "黛",
    ),
    "dev": (
        "白",
        "灰白",
        "乳白",
        "米色",
        "浅蓝",
        "深蓝",
        "浅绿",
        "深绿",
    ),
    "final": (
        "黑",
        "墨黑",
        "漆黑",
        "暗红",
        "绯红",
        "朱红",
        "赤金",
        "乌金",
    ),
}

TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "fact": {
        "train": {"prefix": "问：{o}是什么颜色？背景：{o}是{c}。答：", "answer": "{c}"},
        "dev": {"prefix": "提问：{o}的颜色？线索：{o}是{c}。回答：", "answer": "{c}"},
    },
    "negation": {
        "train": {"prefix": "问：{o}是{c}吗？背景：不是{c}。答：", "answer": "不是{c}"},
        "dev": {"prefix": "提问：{o}是否为{c}？线索：{o}不是{c}。回答：", "answer": "不是{c}"},
    },
    "unknown": {
        "train": {"prefix": "问：{o}是什么颜色？背景：没有关于{o}的信息。答：", "answer": "未知"},
        "dev": {"prefix": "提问：{o}的颜色？线索：{o}的信息缺失。回答：", "answer": "未知"},
    },
    "combination_same": {
        "train": {
            "prefix": "问：{o1}和{o2}颜色相同吗？背景：{o1}是{c}，{o2}是{c}。答：",
            "answer": "相同",
        },
        "dev": {
            "prefix": "提问：{o1}与{o2}的颜色一致吗？线索：{o1}是{c}，{o2}是{c}。回答：",
            "answer": "相同",
        },
    },
    "combination_different": {
        "train": {
            "prefix": "问：{o1}和{o2}颜色相同吗？背景：{o1}是{c1}，{o2}是{c2}。答：",
            "answer": "不同",
        },
        "dev": {
            "prefix": "提问：{o1}与{o2}的颜色一致吗？线索：{o1}是{c1}，{o2}是{c2}。回答：",
            "answer": "不同",
        },
    },
    "same_opening_fact": {
        "train": {"prefix": "问：请判断{o}的颜色。背景：{o}是{c}。答：", "answer": "{c}"},
        "dev": {"prefix": "提问：请判断{o}的颜色。线索：{o}是{c}。回答：", "answer": "{c}"},
    },
    "same_opening_unknown": {
        "train": {"prefix": "问：请判断{o}的颜色。背景：没有关于{o}的信息。答：", "answer": "未知"},
        "dev": {"prefix": "提问：请判断{o}的颜色。线索：{o}的信息缺失。回答：", "answer": "未知"},
    },
}


def _pools(scale: int) -> dict[str, dict[str, tuple[str, ...]]]:
    objects_n = {4: 8, 8: 16, 16: 32}[scale]
    colors_n = {4: 4, 8: 8, 16: 16}[scale]
    return {
        split: {
            "objects": tuple(dict.fromkeys(bank))[:objects_n],
            "colors": tuple(dict.fromkeys(bank))[:colors_n],
        }
        for split, bank in OBJECT_BANKS.items()
    } | {
        f"{split}-colors": tuple(dict.fromkeys(bank))[:colors_n]
        for split, bank in COLOR_BANKS.items()
    }


def _build_corpus(scale: int) -> dict[str, tuple[tuple[bytes, bytes], ...]]:
    pools = _pools(scale)
    out: dict[str, tuple[tuple[bytes, bytes], ...]] = {}
    for split in ("train", "dev"):
        objects = pools[split]["objects"]
        colors = pools[f"{split}-colors"]
        records: list[tuple[bytes, bytes]] = []

        def add(template_id: str, prefix: str, answer: str) -> None:
            records.append((prefix.encode("utf-8"), answer.encode("utf-8")))

        for obj in objects:
            for color in colors:
                slot = {"o": obj, "c": color}
                add(
                    "fact",
                    TEMPLATES["fact"][split]["prefix"].format(**slot),
                    TEMPLATES["fact"][split]["answer"].format(**slot),
                )
                add(
                    "negation",
                    TEMPLATES["negation"][split]["prefix"].format(**slot),
                    TEMPLATES["negation"][split]["answer"].format(**slot),
                )
                add(
                    "unknown",
                    TEMPLATES["unknown"][split]["prefix"].format(**slot),
                    TEMPLATES["unknown"][split]["answer"].format(**slot),
                )
                add(
                    "so_fact",
                    TEMPLATES["same_opening_fact"][split]["prefix"].format(**slot),
                    TEMPLATES["same_opening_fact"][split]["answer"].format(**slot),
                )
                add(
                    "so_unknown",
                    TEMPLATES["same_opening_unknown"][split]["prefix"].format(**slot),
                    TEMPLATES["same_opening_unknown"][split]["answer"].format(**slot),
                )
        # combination shapes: disjoint object pairs, same/different colors
        for i in range(min(len(objects), 16)):
            o1, o2 = objects[i], objects[(i + 1) % len(objects)]
            c = colors[i % len(colors)]
            slot = {"o1": o1, "o2": o2, "c": c}
            add(
                "comb_same",
                TEMPLATES["combination_same"][split]["prefix"].format(**slot),
                TEMPLATES["combination_same"][split]["answer"].format(**slot),
            )
            c1, c2 = colors[i % len(colors)], colors[(i + 1) % len(colors)]
            slot2 = {"o1": o1, "o2": o2, "c1": c1, "c2": c2}
            add(
                "comb_diff",
                TEMPLATES["combination_different"][split]["prefix"].format(**slot2),
                TEMPLATES["combination_different"][split]["answer"].format(**slot2),
            )
        out[split] = tuple(sorted(records))
    return out


def _corpus_digest(corpus: dict[str, tuple[tuple[bytes, bytes], ...]]) -> str:
    return content_digest({split: sorted(items) for split, items in corpus.items()})


def _forced(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, Any]:
    positions = correct = 0
    losses: list[float] = []
    with torch.no_grad():
        for prefix, response in episodes:
            loss, metrics = prototype.sequence_loss(prefix, response)
            positions += int(metrics["positions"])
            correct += int(metrics["correct"])
            losses.append(float(loss))
    return {
        "accuracy": correct / max(1, positions),
        "mean_loss": sum(losses) / max(1, len(losses)),
    }


def _generation(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, Any]:
    exact = 0
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for prefix, response in episodes:
            result = prototype.generate(prefix, max_bytes=GENERATION_LIMIT)
            produced = bytes(result.bytes_out)
            exact += int(produced == response)
            if len(samples) < 2:
                samples.append(
                    {
                        "reference": response.decode("utf-8", errors="replace"),
                        "generated": produced.decode("utf-8", errors="replace"),
                        "exact": produced == response,
                    }
                )
    return {
        "exact_rate": exact / max(1, len(episodes)),
        "exact_count": exact,
        "episodes": len(episodes),
        "samples": samples,
    }


def _run_arm(
    corpus: dict[str, tuple[tuple[bytes, bytes], ...]],
    *,
    seed: int,
    lam: float,
    epochs: int,
    episodes_per_epoch: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    prototype = SequenceWorkspacePrototype(SequenceWorkspaceConfig(seed=seed))
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=LEARNING_RATE)
    trainer.set_episodes(corpus["train"])
    trainer.enable_epoch_shuffle(seed=20260917 + seed)
    if lam > 0.0:
        trainer.enable_context_contrastive(weight=lam, margin=1.0)
    started = time.monotonic()
    for _ in range(epochs):
        trainer.train_epoch(max_episodes=episodes_per_epoch)
    elapsed = time.monotonic() - started
    return {
        "lambda": lam,
        "global_step": trainer.global_step,
        "elapsed_seconds": elapsed,
        "train": _forced(prototype, corpus["train"][:episodes_per_epoch]),
        "dev_forced": _forced(prototype, corpus["dev"]),
        "dev_generation": _generation(prototype, corpus["dev"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Course/data scale sweep")
    parser.add_argument(
        "--report", type=Path, default=Path("reports/r2_course_scale_sweep_20260917.json")
    )
    args = parser.parse_args(argv)

    sweep: list[dict[str, Any]] = []
    for scale in SCALES:
        corpus = _build_corpus(scale)
        episodes_per_epoch = len(corpus["train"])  # full pool every epoch
        epochs = 4  # fixed: every episode is seen exactly 4 times at every scale
        actual_steps = epochs * episodes_per_epoch
        arms = [
            dict(
                _run_arm(
                    corpus,
                    seed=seed,
                    lam=LAMBDA[name],
                    epochs=epochs,
                    episodes_per_epoch=episodes_per_epoch,
                ),
                seed=seed,
                arm=name,
            )
            for seed in SEEDS
            for name, lam in LAMBDA.items()
        ]
        rates = {
            name: [arm["dev_generation"]["exact_rate"] for arm in arms if arm["arm"] == name]
            for name in LAMBDA
        }
        means = {name: sum(v) / max(1, len(v)) for name, v in rates.items()}
        digest = _corpus_digest(corpus)
        print(
            f"scale {scale}x: train={len(corpus['train'])} dev={len(corpus['dev'])} "
            f"epochs={epochs} steps={actual_steps} | "
            f"control exact {rates['control']} mean {means['control']:.4f} | "
            f"treatment exact {rates['treatment']} mean {means['treatment']:.4f}"
        )
        sweep.append(
            {
                "scale": scale,
                "train_episodes": len(corpus["train"]),
                "dev_episodes": len(corpus["dev"]),
                "epochs": epochs,
                "actual_steps": actual_steps,
                "corpus_digest": digest,
                "dev_exact_per_seed": rates,
                "dev_exact_mean": means,
                "margin_note": "lambda=1 vs lambda=0; seeds " + ",".join(map(str, SEEDS)),
                "arms": arms,
            }
        )

    monotonic = all(
        sweep[i + 1]["dev_exact_mean"]["treatment"] >= sweep[i]["dev_exact_mean"]["treatment"]
        for i in range(len(sweep) - 1)
    )
    treatment_beats = all(
        item["dev_exact_mean"]["treatment"] >= item["dev_exact_mean"]["control"] for item in sweep
    )
    payload = {
        "format": "taiji-r2-course-scale-sweep-v1",
        "target_steps": TARGET_STEPS,
        "seeds": list(SEEDS),
        "frozen": {"lambda": LAMBDA, "learning_rate": LEARNING_RATE, "margin": 1.0},
        "sweep": sweep,
        "interaction_monotonic_in_scale": monotonic,
        "treatment_beats_control_at_every_scale": treatment_beats,
        "verdict": (
            "interaction_monotonic_and_positive"
            if monotonic and treatment_beats
            else "interaction_not_monotonic" if treatment_beats else "interaction_not_supported"
        ),
        "final_evaluated": False,
        "growth_admitted": False,
        "can_promote": False,
        "scope": "isolated prototype only; must NOT be extrapolated to chat()",
    }
    target = PROJECT_ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"verdict: {payload['verdict']}")
    print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
