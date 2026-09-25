"""生成并**冻结** §4.3a 表层判据的扩展题集（100 题，D 维同构）。

为什么要有这份题集（不要把它当成"新的能力评价集"）：§4.3 的表层子判据要"成句差 ≥3 句
且两次独立取数同向"才允许下结论，而 D+E 原题集只有 36 条文本——±1 条≈2.8pp，
阈值 3 条等于要求 8% 的摆动，**这条判据在那个题集上结构性地不可判**。
本件把表层测量的分母抬到约 208 条文本（104 题 ×〔告知轮的答复＋提问轮的答复〕≈208 条），
只服务一个目的：让"装上复制回路有没有把表层说坏"这件事**可判**。
它**不是**能力判据（严格真命中的主分母仍是 CAP D 维那 16 题，不变）。

冻结纪律：
* 每题的期望答案词必须**逐字出现在该题更早的轮次里**（可复制性由构造保证，测试逐题断言）；
* 三套实体池互不相交：本扩展集 ≠ CAP 评价集（阿岩/杭州/17/Seed…）≠ A2.3 训练表
  （阿蒙/成都/42/星舟…）——防"扩展集把训练过的内容当新题"这种自证；
* 生成完全确定（固定 seed、固定顺序），重跑必须产出同一份字节；
* 2 轮题与 3 轮题各半；每个 (族,实体) 配对**只出现一次**（题数＝配对数）（3 轮题里一半是"另一条告知"＝真正的多事件选择，一半是无关话题问句）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"

#: 8 个告知族；ask 句式与 CAP D 维同构（第一人称告知 → 第一人称提问）。
KINDS: dict[str, tuple[str, str]] = {
    "name": ("我叫{X}。", "我的名字是什么？"),
    "city": ("我住在{X}。", "我住哪？"),
    "number": ("我最喜欢的数字是{X}。", "那个数字是多少？"),
    "project": ("我的项目叫{X}。", "我的项目叫什么？"),
    "drink": ("我平时爱喝{X}。", "我爱喝什么？"),
    "pet": ("我养了一只{X}。", "我养的动物是什么？"),
    "book": ("我最近在读《{X}》。", "我最近在读什么书？"),
    "dish": ("我最爱吃{X}。", "我最爱吃什么？"),
}

#: 每族 13 个实体（8×13=104 个**互不相同**的 (告知,提问) 对——题数与配对数相等，
#: 否则"200 条文本"只是重复题面的假样本量；上一版 100 题里就有 62 题在重复）。
#: 两条硬约束（由合同测试逐条断言，不靠注释）：
#: ① 与 CAP 评价集期望词、A2.3 训练表两套均**不相等**；② 每词长度 ≥2 字
#:    （单字答案词在乱码里靠运气就能命中，上一版有 6 个这样的词——"二""日"那类）。
POOLS: dict[str, tuple[str, ...]] = {
    "name": (
        "明轩",
        "雨桐",
        "承泽",
        "思远",
        "慕安",
        "砚青",
        "洛宁",
        "时予",
        "怀砚",
        "苏晚",
        "沈舟",
        "顾昭",
        "晚吟",
    ),
    "city": (
        "苏州",
        "西安",
        "青岛",
        "长沙",
        "昆明",
        "兰州",
        "宁波",
        "温州",
        "洛阳",
        "扬州",
        "泉州",
        "芜湖",
        "自贡",
    ),
    "number": (
        "36",
        "21",
        "77",
        "1998",
        "1024",
        "47",
        "88",
        "121",
        "256",
        "313",
        "520",
        "618",
        "909",
    ),
    "project": (
        "海图",
        "晨星",
        "石径",
        "归云",
        "听涛",
        "折柳",
        "疏影",
        "临风",
        "宿雨",
        "行舟",
        "望舒",
        "未央",
        "知秋",
    ),
    "drink": (
        "乌龙",
        "苏打",
        "桂花酿",
        "柠檬水",
        "砖茶",
        "普洱",
        "苦荞茶",
        "红枣茶",
        "陈皮水",
        "薄荷饮",
        "山楂水",
        "茉莉茶",
        "荷叶茶",
    ),
    "pet": (
        "雪鸮",
        "浣熊",
        "树袋熊",
        "蓝舌蜥",
        "蜜獾",
        "刺猬",
        "鹦鹉",
        "仓鼠",
        "乌龟",
        "水獭",
        "狐狸",
        "蝙蝠",
        "松鼠",
    ),
    "book": (
        "海客谈瀛",
        "水经注疏",
        "松窗梦语",
        "云笈七签",
        "梦溪笔谈",
        "天工开物",
        "齐民要术",
        "洛阳伽蓝记",
        "东京梦华录",
        "梦粱录",
        "西湖梦寻",
        "陶庵梦忆",
        "徐霞客游记",
    ),
    "dish": (
        "小笼包",
        "辣子鸡",
        "羊肉泡馍",
        "胡辣汤",
        "臭豆腐",
        "糖芋苗",
        "鸭血粉",
        "蟹黄汤包",
        "牛肉面",
        "酸菜鱼",
        "口水鸡",
        "狮子头",
        "夫妻肺片",
    ),
}

#: 3 轮题的两种干扰：告知型（真多事件选择）与话题型（无关问句）。
DISTRACTOR_TELLS: tuple[str, ...] = (
    "冰箱里的牛奶好像过期了。",
    "阳台上的茉莉今年没开花。",
    "地铁上被人踩了一脚。",
    "楼下新开的面馆排了很长队。",
    "旧钥匙串找不到了。",
)
DISTRACTOR_TOPICS: tuple[str, ...] = (
    "请介绍一下青藏高原。",
    "简述一下光合作用的过程。",
    "介绍一下印象派绘画的起源。",
    "说说潮汐是怎么形成的。",
    "请讲讲活字印刷术的来历。",
)


def build(count: int = 104, seed: int = 20260925) -> dict[str, object]:
    rng = random.Random(seed)
    kinds = list(KINDS)
    if count > len(kinds) * min(len(pool) for pool in POOLS.values()):
        raise ValueError(
            f"题数 {count} 超过可用 (族,实体) 配对数 "
            f"{len(kinds) * min(len(pool) for pool in POOLS.values())}——"
            "加题必须先加实体池，否则就是在重复题面上堆假样本量"
        )
    items: list[dict[str, object]] = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        slot = index // len(kinds)
        tell_template, ask = KINDS[kind]
        entity = POOLS[kind][slot]
        tell = tell_template.format(X=entity)
        three_turn = index % 2 == 1
        turns = [tell]
        if three_turn:
            if (index // 2) % 2 == 0:
                turns.append(rng.choice(DISTRACTOR_TELLS))
            else:
                turns.append(rng.choice(DISTRACTOR_TOPICS))
        turns.append(ask)
        items.append(
            {
                "id": f"X{index + 1:03d}",
                "family": "given_then_ask" if not three_turn else "with_distractor",
                "kind": kind,
                "turns": turns,
                "expected_contains": [entity],
            }
        )
    return {
        "format": "r2-copy-surface-extension-v1",
        "frozen_on": "2026-09-25",
        "frozen_before_any_candidate_score": True,
        "purpose": (
            "§4.3a 表层子判据的分母扩展（成句率/可解码率）；**不作能力判据**——"
            "严格真命中的主分母仍是 CAP D 维 16 题（不变）"
        ),
        "prereg": "plans/reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md",
        "generator": "scripts/training/build_r2_copy_surface_extension.py",
        "seed": seed,
        "count": len(items),
        "dimensions": {
            "X": {
                "name": "告知→提问（表层扩展集）",
                "count": len(items),
                "items": items,
            }
        },
        "disjointness_note": (
            "实体池与 CAP 评价集、A2.3 训练表两套均不相交；由 "
            "tests/taiji_native/test_r2_copy_surface_extension_contract.py 逐条断言"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--count", type=int, default=104)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--check", action="store_true", help="只校验现有清单是否与确定性重生成的字节一致"
    )
    args = parser.parse_args()

    payload = build(count=args.count, seed=args.seed)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out = Path(args.out)
    if args.check:
        existing = out.read_text(encoding="utf-8")
        same = existing == text
        print(json.dumps({"check": "ok" if same else "drift", "sha256": digest[:16]}))
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    print(json.dumps({"written": out.name, "count": payload["count"], "sha256": digest[:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
