"""扩展集 **v2**：把干扰换成"同谓语、不同实体"那一族（修订需求登记在 `SPEC-A-21` §6）。

**为什么要有 v2**：v1 的 `with_distractor` 干扰句是我自己写的通用指令句
（"地铁上被人踩了一脚。""简述一下光合作用的过程。"），与提问的**字面交集近乎为空**，
而答案告知必然与提问共享谓语字 ⇒ 免训练的"共享字符"规则键**按构造就会赢**
（实测静态 48/48、表层严格命中 23→39，见 `SPEC-A-17` §14/§15）。
题集偏易会让"学习式选择器有增量"这件事**永远证不出来**——所以先造一把规则骑不动的题。

**每题 3 轮**：`告知(我) → 干扰(他人，同谓语，不同实体) → 提问`。干扰交替取两种主体写法：

| 型 | 干扰写法 | 规则键在该型上的结局 |
|---|---|---|
| (a) 带同一个主语字 | "我表哥住在无锡。" | 与提问的共享字符数**等于**答案告知 ⇒ 平手按 `event_id` 取后入库那条＝**必错** |
| (b) 不带主语字 | "同事住在无锡。" | 少一个共享字 ⇒ 规则**可能对**，但语义上仍要求分清"谁住的" |

混排是故意的：只造"规则必错"的题等于把测量换成另一根定死的棍子，**梯度**才说明得了事。

**冻结前的机检（跑不过就不许落盘）**：
① 答案词逐字出现在**且只**出现在第一条告知里（干扰句、提问句都不得含它）——保证每题只有一个正确答案；
② (告知,干扰,提问) 三元组互不相同；
③ 答案词长度 ≥2 字（单字词在乱码底子里靠运气也能命中）；
④ 实体取自 v1 那套池（与 CAP 评价集、A2.3 训练表两条不相交断言沿用 v1 的合同件）；
⑤ **本件自己在构造层面模拟一遍规则键**（与 `SPEC-A-17` §14 的 `oracle` 档同一排序、同一平手取新规则），
   统计"规则会挑对多少"，**要求 ≤0.60**；超线说明题仍偏易，回头改干扰而不是冻结。

确定性：无随机源，重跑必须产出同一份字节。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT / "scripts" / "training",):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_r2_copy_surface_extension import KINDS, POOLS  # noqa: E402  单一实体来源

OUT_DEFAULT = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v2.json"

#: 每族两种"同谓语不同实体"干扰：(a) 带主语字"我"，(b) 不带。`{Y}` 是同族**另一个**实体。
HARD_DISTRACTORS: dict[str, tuple[str, str]] = {
    "name": ("我表哥叫{Y}。", "表哥叫{Y}。"),
    "city": ("我表哥住在{Y}。", "同事住在{Y}。"),
    "number": ("我妹妹最喜欢的数字是{Y}。", "妹妹最喜欢的数字是{Y}。"),
    "project": ("我同事的项目叫{Y}。", "朋友的项目叫{Y}。"),
    "drink": ("我爱人爱喝{Y}。", "爱人爱喝{Y}。"),
    "pet": ("我家邻居养了一只{Y}。", "邻居养了一只{Y}。"),
    "book": ("我姐姐最近在读《{Y}》。", "姐姐最近在读《{Y}》。"),
    "dish": ("我哥哥最爱吃{Y}。", "哥哥最爱吃{Y}。"),
}

#: 构造层面的规则键上限：超过这条线就不许冻结（题集对免训练规则仍太友好）。
RULE_CEILING = 0.60


def _rule_wins(ask: str, tell: str, distractor: str) -> bool:
    """模拟"共享字符最多者胜、平手取后入库那条"——后入库的是**干扰**，故平手算错。"""
    return len(set(ask) & set(tell)) > len(set(ask) & set(distractor))


def build() -> dict[str, Any]:
    kinds = list(KINDS)
    items: list[dict[str, Any]] = []
    index = 0
    for slot in range(max(len(pool) for pool in POOLS.values())):
        for kind in kinds:
            pool = POOLS[kind]
            if slot >= len(pool):
                continue
            tell_template, ask = KINDS[kind]
            entity = pool[slot]
            distractor_pool = list(pool[1:] + pool[:1])
            other = distractor_pool[slot % len(distractor_pool)]
            hard_a, hard_b = HARD_DISTRACTORS[kind]
            flavour = hard_a if slot % 2 == 0 else hard_b
            distractor = flavour.format(Y=other)
            tell = tell_template.format(X=entity)
            turns = [tell, distractor, ask]
            items.append(
                {
                    "id": f"V{index + 1:03d}",
                    "family": "with_hard_distractor",
                    "flavour": "same_subject_marker" if flavour is hard_a else "no_subject_marker",
                    "kind": kind,
                    "turns": turns,
                    "expected_contains": [entity],
                }
            )
            index += 1
    _validate(items)
    return {
        "format": "r2-copy-surface-extension-v2",
        "frozen_on": "2026-09-26",
        "frozen_before_any_candidate_score": True,
        "purpose": (
            "**能力**题集（不是回归门题集）：把事件选择的干扰换成同谓语不同实体，"
            "使免训练的共享字符规则不再按构造取胜——v1 的表层判决锚点不变，本件不替代它"
        ),
        "prereg": "plans/reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md §6",
        "generator": "scripts/training/build_r2_copy_surface_extension_v2.py",
        "derived_from": "plans/manifests/r2_copy_surface_extension_v1.json",
        "count": len(items),
        "rule_simulated_accuracy": _rule_accuracy(items),
        "dimensions": {
            "X": {
                "name": "告知→同谓语干扰→提问（难干扰扩展集 v2）",
                "count": len(items),
                "items": items,
            }
        },
        "disjointness_note": (
            "实体池沿用 v1（与 CAP 评价集、A2.3 训练表两条不相交断言由 "
            "tests/taiji_native/test_r2_copy_surface_extension_contract.py 逐条断言）"
        ),
    }


def _rule_accuracy(items: list[dict[str, Any]]) -> float:
    return round(sum(1 for item in items if _rule_item(item)) / max(len(items), 1), 4)


def _rule_item(item: dict[str, Any]) -> bool:
    turns = [str(turn) for turn in item["turns"]]
    return _rule_wins(turns[-1], turns[0], turns[1])


def _validate(items: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, ...]] = set()
    for item in items:
        turns = [str(turn) for turn in item["turns"]]
        tokens = [str(token) for token in item["expected_contains"]]
        if len(turns) != 3:
            raise AssertionError(f"{item['id']}: v2 每题必须 3 轮")
        tell, distractor, ask = turns
        for token in tokens:
            if len(token) < 2:
                raise AssertionError(f"{item['id']}: 答案词 '{token}' 少于 2 字")
            if token not in tell:
                raise AssertionError(f"{item['id']}: 答案词没逐字出现在告知里")
            if token in distractor:
                raise AssertionError(f"{item['id']}: 干扰句也含答案词，这题没有唯一正确答案")
            if token in ask:
                raise AssertionError(f"{item['id']}: 提问里就写着答案词（白送）")
        triple = tuple(turns)
        if triple in seen:
            raise AssertionError(f"{item['id']}: (告知,干扰,提问) 三元组重复")
        seen.add(triple)
    accuracy = _rule_accuracy(items)
    if accuracy > RULE_CEILING:
        by_kind: dict[str, list[bool]] = {}
        for item in items:
            by_kind.setdefault(str(item["kind"]), []).append(_rule_item(item))
        detail = {kind: round(sum(v) / len(v), 3) for kind, v in sorted(by_kind.items())}
        raise AssertionError(
            f"免训练规则键在构造层面就能挑对 {accuracy:.0%}（上限 {RULE_CEILING:.0%}）"
            f"——题集仍偏易，别冻结：{detail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument(
        "--check", action="store_true", help="只校验现有清单是否与确定性重生成的字节一致"
    )
    args = parser.parse_args()

    payload = build()
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if args.check:
        same = out.read_text(encoding="utf-8") == text
        print(json.dumps({"check": "same" if same else "drift", "sha256": digest[:16]}))
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "written": out.name,
                "count": payload["count"],
                "rule_simulated_accuracy": payload["rule_simulated_accuracy"],
                "sha256": digest[:16],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
