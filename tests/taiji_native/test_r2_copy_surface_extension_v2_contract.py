"""扩展集 **v2**（难干扰集）的合同。

v2 存在的唯一理由（`SPEC-A-21` §6）：v1 的干扰句与提问字面近乎无交集，
免训练的"共享字符"规则键**按构造就赢**（静态 48/48、表层 23→39）——那样一批题证不出
"学习式选择器有增量"。所以本文件把"规则骑不动"这件事从注释变成断言：
① 构造层面模拟规则键，挑对率必须 ≤ 生成器里的 `RULE_CEILING`；
② 每题**只有一个正确答案**（答案词不得出现在干扰句或提问里）——否则"难"是靠歧义造出来的；
③ 生成确定、与 v1 一样不与两套已用过的表相撞。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v2.json"
BUILD = PROJECT_ROOT / "scripts/training/build_r2_copy_surface_extension_v2.py"
EVAL_SET = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"


def _items() -> list[dict[str, object]]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return list(payload["dimensions"]["X"]["items"])


def _turns(item: dict[str, object]) -> list[str]:
    return [str(turn) for turn in (item.get("turns") or [])]


def _rule_wins(ask: str, tell: str, distractor: str) -> bool:
    from build_r2_copy_surface_extension_v2 import _rule_wins as implementation

    return implementation(ask, tell, distractor)


def test_every_item_is_three_turn_hard_distractor() -> None:
    items = _items()
    assert len(items) == 104, len(items)
    for item in items:
        turns = _turns(item)
        assert len(turns) == 3, item["id"]
        assert item["family"] == "with_hard_distractor", item["id"]


def test_answer_token_is_copyable_and_unique_to_the_telling_turn() -> None:
    """ "难"必须靠同谓语干扰造，不能靠歧义造：答案词只能出现在第一条告知里。"""

    for item in _items():
        tell, distractor, ask = _turns(item)
        for token in item["expected_contains"]:
            assert len(str(token)) >= 2, item["id"]
            assert str(token) in tell, item["id"]
            assert str(token) not in distractor, f"{item['id']}: 干扰句也有正确答案"
            assert str(token) not in ask, f"{item['id']}: 提问里就写着答案词"


def test_distractors_share_the_predicate_with_the_telling_turn() -> None:
    """干扰句必须是**同谓语不同实体**——否则又退回 v1 那种字面无交集的软干扰。"""

    from build_r2_copy_surface_extension_v2 import HARD_DISTRACTORS, POOLS

    #: 每族一个谓语片段，必须**同时**出现在告知与干扰里（难干扰的定义就是这一条）。
    predicates = {
        "name": "叫",
        "city": "住在",
        "number": "最喜欢的数字是",
        "project": "叫",
        "drink": "爱喝",
        "pet": "养了一只",
        "book": "在读",
        "dish": "爱吃",
    }
    items = _items()
    assert {str(item["kind"]) for item in items} == set(predicates)
    for item in items:
        kind = str(item["kind"])
        tell, distractor, _ = _turns(item)
        assert predicates[kind] in tell, f"{item['id']}: 告知里没有谓语片段"
        assert predicates[kind] in distractor, f"{item['id']}: 干扰句没沿用同一谓语"
        assert any(
            distractor == template.format(Y=other)
            for template in HARD_DISTRACTORS[kind]
            for other in POOLS[kind]
        ), f"{item['id']}: 干扰句不是本族模板生成的：{distractor}"


def test_rule_key_cannot_ride_this_set_by_construction() -> None:
    """构造层面模拟规则键（同 §14 的排序与"平手取后入库"）：挑对率必须低于上限。"""

    from build_r2_copy_surface_extension_v2 import RULE_CEILING

    items = _items()
    correct = sum(
        1 for item in items if _rule_wins(_turns(item)[2], _turns(item)[0], _turns(item)[1])
    )
    rate = correct / len(items)
    assert rate <= RULE_CEILING, f"规则键仍挑对 {rate:.0%}（上限 {RULE_CEILING:.0%}）——题偏易"

    same = [item for item in items if item["flavour"] == "same_subject_marker"]
    assert all(
        not _rule_wins(_turns(item)[2], _turns(item)[0], _turns(item)[1]) for item in same
    ), "带同一主语字的干扰应当让规则平手判负（后入库者胜），否则这条设计不成立"


def test_triples_are_unique() -> None:
    items = _items()
    triples = {tuple(_turns(item)) for item in items}
    assert len(triples) == len(items), "三元组重复＝样本量是假的"


def test_entities_stay_disjoint_from_both_used_tables() -> None:
    from train_taiji_r2_copy_circuit import CITIES, NAMES, NUMBERS, PROJECTS

    trained = set(NAMES) | set(CITIES) | set(NUMBERS) | set(PROJECTS)
    evaluation = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    eval_tokens = {
        str(token)
        for dimension in ("C", "D", "E")
        for item in evaluation["dimensions"][dimension]["items"]
        for token in (item.get("expected_contains") or [])
    }
    extension = {str(token) for item in _items() for token in item["expected_contains"]}
    assert not extension & trained, sorted(extension & trained)
    assert not extension & eval_tokens, sorted(extension & eval_tokens)


def test_manifest_is_reproducible_from_the_generator(tmp_path: Path) -> None:
    """冻结件必须能由生成器逐字节重生成——否则后来的读数没人知道题被改过。"""

    rewritten = tmp_path / "v2.json"
    subprocess.run(
        [sys.executable, str(BUILD), "--out", str(rewritten)],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    assert rewritten.read_bytes() == MANIFEST.read_bytes()
    check = subprocess.run(
        [sys.executable, str(BUILD), "--out", str(MANIFEST), "--check"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    assert check.returncode == 0, check.stdout
