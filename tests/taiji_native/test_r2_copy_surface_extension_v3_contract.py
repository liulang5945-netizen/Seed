"""`v3` 位置随机题集的合同（SPEC-A-22 §18/§19）。

钉的是"位置这条捷径真的被剥掉了"——本案全部反转都出在这一维上：v1/v2 机检规定
"含答案词的告知必为第一轮"，于是"挑最旧"两边满分（实测配对换位后从 36 塌到 2）。
所以这份题集的机检不是走形式：**每一条都对应一种"仍能作弊"的方式**。

另含一条真实回归：生成器第一版把牌堆标签 `"first"/"second"`（字符串）直接当布尔条件用，
"second" 也为真 ⇒ 104 题全落第一轮，是"严格对半"那条断言把它拦下的。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_r2_copy_surface_extension_v3_position_random import (  # noqa: E402
    DEFAULT_OUT,
    SOURCE,
    build,
)

V3 = DEFAULT_OUT


def _items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["dimensions"]["X"]["items"])


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("v3") / "v3.json"
    report = build(SOURCE, out)
    assert report["count"] == 104
    return out


def test_placement_is_an_exact_balanced_deck(built: Path) -> None:
    counter = Counter(row["answer_tell_position"] for row in _items(built))
    assert counter == {0: 52, 1: 52}, counter


def test_no_kind_is_position_determined(built: Path) -> None:
    payload = json.loads(built.read_text(encoding="utf-8"))
    by_kind = payload["placement_by_kind"]
    assert by_kind, "缺 per-kind 分层读数"
    for kind, sides in by_kind.items():
        assert sides.get("first") and sides.get("second"), kind


def test_marker_matches_content_and_is_unique(built: Path) -> None:
    for row in _items(built):
        expected = list(row["expected_contains"])
        index = int(row["answer_tell_position"])
        told = [row["turns"][0], row["turns"][1]]
        assert any(token in told[index] for token in expected), row["id"]
        assert not any(token in told[1 - index] for token in expected), row["id"]


def test_paired_with_parent_and_question_untouched(built: Path) -> None:
    parent = json.loads(SOURCE.read_text(encoding="utf-8"))["dimensions"]["X"]["items"]
    v3_rows = {row["id"]: row for row in _items(built)}
    assert set(v3_rows) == {row["id"] for row in parent}
    for source_row in parent:
        row = v3_rows[source_row["id"]]
        assert row["turns"][2] == source_row["turns"][2], row["id"]
        assert row["expected_contains"] == source_row["expected_contains"], row["id"]
        assert set(row["turns"][:2]) == set(source_row["turns"][:2]), row["id"]


def test_generator_refuses_to_overwrite_a_frozen_manifest(built: Path) -> None:
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build(SOURCE, built)


def test_repo_copy_carries_the_frozen_sha(built: Path) -> None:
    #: 入册的那一份必须与重新生成的一致——种子固定 ⇒ 题集可复现，不许悄悄重洗。
    assert V3.exists()
    assert json.loads(V3.read_text(encoding="utf-8"))["dimensions"]["X"]["items"] == _items(built)
