"""§4.3a 扩展分母的合同：题集性质、生成确定性、判读函数的正反断言。

题集是给一条**判据**当分母的，所以它自己的性质不能靠生成器注释声明——
"与两套已用过的表不相交""配对不重复""答案词不是单字"这三条必须逐条断言。
第一版就是靠这三条检查才发现 62 题重复、`8`/`日` 与评价集相撞、6 个单字词。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"
EVAL_SET = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"
BUILD = PROJECT_ROOT / "scripts/training/build_r2_copy_surface_extension.py"


def _items() -> list[dict[str, object]]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return list(payload["dimensions"]["X"]["items"])


def _turns(item: dict[str, object]) -> list[str]:
    return [str(turn) for turn in (item.get("turns") or [])]


def test_every_answer_token_is_copyable_from_an_earlier_turn() -> None:
    """表层集要测"告知→提问"，答案词必须逐字出现在更早轮里，否则测的就不是复制。"""

    for item in _items():
        prior = "".join(_turns(item)[:-1])
        assert any(token in prior for token in item["expected_contains"]), item["id"]


def test_pairing_is_unique_and_family_split_is_balanced() -> None:
    """题数必须等于 (告知,提问) 配对数——否则"260 条文本"是重复题面撑出来的假分母。"""

    items = _items()
    pairs = {(_turns(item)[0], _turns(item)[-1]) for item in items}
    assert len(items) == len(pairs) == 104, f"题数 {len(items)}，不同配对 {len(pairs)}"
    families = {str(item["family"]) for item in items}
    assert families == {"given_then_ask", "with_distractor"}
    counts = {family: sum(1 for item in items if item["family"] == family) for family in families}
    assert counts["given_then_ask"] == counts["with_distractor"] == 52, counts


def test_entities_are_disjoint_from_both_already_used_tables() -> None:
    """与 CAP 评价集期望词、A2.3 训练表**都**不相交——否则扩展集会替训练表内容背书。"""

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
    assert not (extension & trained), f"扩展集撞了训练表：{sorted(extension & trained)}"
    assert not (
        extension & eval_tokens
    ), f"扩展集撞了评价集期望词：{sorted(extension & eval_tokens)}"


def test_no_single_character_answer_tokens() -> None:
    """单字答案词在乱码里靠运气就能命中——严格口径不许出现这种弱证据。"""

    short = sorted(
        {
            str(token)
            for item in _items()
            for token in item["expected_contains"]
            if len(str(token)) < 2
        }
    )
    assert short == [], f"出现单字答案词：{short}"


def test_manifest_is_reproducible_from_the_generator(tmp_path: Path) -> None:
    """冻结件必须能由固定 seed 的生成器逐字节重生成（`--check` 语义）。"""

    scratch = tmp_path / "regen.json"
    proc = subprocess.run(
        [sys.executable, str(BUILD), "--out", str(scratch)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert scratch.read_text(encoding="utf-8") == MANIFEST.read_text(encoding="utf-8")
    check = subprocess.run(
        [sys.executable, str(BUILD), "--check"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert '"check": "ok"' in check.stdout, check.stdout


def _arm(texts: int, formed: int, circuit: str | None = None) -> dict[str, object]:
    return {
        "circuit": circuit,
        "texts": texts,
        "well_formed_texts": formed,
        "well_formed_rate": round(formed / max(texts, 1), 4),
        "utf8_decodable_rate": 1.0,
        "strict_hits": 0,
        "rows": [],
    }


def test_rule_refuses_to_judge_on_two_texts_gap_and_on_opposing_circuits() -> None:
    """两个反向用例：差 2 条（阈值 3）与两电路不同向，都**不许**出"劣化/改善"结论。"""

    from score_taiji_r2_copy_surface_extension import rule_verdict

    control = _arm(260, 20)
    small = rule_verdict(control, [_arm(260, 17, "A"), _arm(260, 18, "B")])
    assert small["status"] == "not_resolved"
    opposing = rule_verdict(control, [_arm(260, 14, "A"), _arm(260, 26, "B")])
    assert opposing["status"] == "not_resolved"
    single = rule_verdict(control, [_arm(260, 10, "A")])
    assert single["status"] == "not_resolved"  # 只有一次取数就下判＝违反 §3
    assert "独立取数只有 1 次" in single["verdict"]


def test_rule_judges_only_when_every_circuit_moves_the_same_way_past_the_floor() -> None:
    from score_taiji_r2_copy_surface_extension import rule_verdict

    control = _arm(260, 20)
    worse = rule_verdict(control, [_arm(260, 16, "A"), _arm(260, 15, "B")])
    assert worse["status"] == "degraded"
    assert all(row["meets_min_gap"] for row in worse["per_circuit"])
    better = rule_verdict(control, [_arm(260, 24, "A"), _arm(260, 27, "B")])
    assert better["status"] == "improved"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
