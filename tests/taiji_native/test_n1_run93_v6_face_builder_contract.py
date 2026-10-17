"""PLAN-N1-93 v6 造面器的合同测：机检必须既能为真也能为假，且未签参数时不落盘。

零模型加载、零查询：全部读数来自内存里造出的面。三条"必须为假"的构造分别打在三条
设计约束上（答案位置轮转、材质不重复、类配平），因为这三条正是"题集的构造约束本身
就是捷径"那条旧教训的可执行形式。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import build_taiji_n1_fresh_confirmation_v6 as v6  # noqa: E402


def draft() -> tuple:
    params = v6.draft_distribution()
    episodes, pair, _sources = v6.build(params)
    return params, episodes, pair


def test_draft_face_passes_every_machine_check() -> None:
    params, episodes, pair = draft()
    result = v6.audit(episodes, params, pair)
    assert result["accepted"] is True, result["checks"]
    assert len(episodes) == 40
    assert all(episode["split"] == "holdout" for episode in episodes)
    assert {len(episode["observations"]) for episode in episodes} == {4}


def test_checks_are_able_to_go_red() -> None:
    params, episodes, pair = draft()

    stacked = copy.deepcopy(episodes)
    for episode in stacked:
        if episode["kind"] in ("lookup", "revision"):
            needle = f" uses {episode['answer']}."
            answer_row = next(row for row in episode["observations"] if needle in row)
            rest = [row for row in episode["observations"] if row != answer_row]
            episode["observations"] = [answer_row] + rest
    assert v6.audit(stacked, params, pair)["accepted"] is False

    doubled = copy.deepcopy(episodes)
    for episode in doubled:
        if episode["kind"] == "lookup":
            needle = f" uses {episode['answer']}."
            slot = next(
                index for index, row in enumerate(episode["observations"]) if needle not in row
            )
            duplicate = f"{episode['entities'][1]} uses {episode['answer']}."
            observations = list(episode["observations"])
            observations[slot] = duplicate
            episode["observations"] = observations
    assert v6.audit(doubled, params, pair)["accepted"] is False

    unbalanced = copy.deepcopy(episodes)
    for episode in unbalanced:
        episode["kind"] = "lookup"
    assert v6.audit(unbalanced, params, pair)["accepted"] is False

    merged = copy.deepcopy(episodes)
    for episode in merged:
        episode["subset"] = "sacrifice"
    assert v6.audit(merged, params, pair)["accepted"] is False

    short_answers = copy.deepcopy(episodes)
    for episode in short_answers:
        if episode["kind"] in ("lookup", "revision") and len(episode["answer"]) == 7:
            episode["answer"] = episode["answer"][:6]
    leak = v6.audit(short_answers, params, pair)["checks"]["J93_6_answer_length_leak"]
    assert leak["ok"] is False, leak


def test_the_abstention_answer_shape_is_not_unique() -> None:
    params, episodes, pair = draft()
    leak = v6.audit(episodes, params, pair)["checks"]["J93_6_answer_length_leak"]
    assert leak["ok"] is True
    assert len(leak["classes_sharing_abstention_length"]) >= 2
    assert v6.ABSTENTION in {episode["answer"] for episode in episodes}


def test_the_two_subsets_are_balanced_and_disjoint() -> None:
    params, episodes, pair = draft()
    reading = v6.audit(episodes, params, pair)
    split = reading["checks"]["J93_5_subset_split"]
    assert split["sizes"] == {"sacrifice": 20, "confirm": 20}
    assert split["overlap"] == []
    counts = split["kinds_per_subset"]
    assert [counts[name][kind] for name in v6.SUBSETS for kind in v6.KINDS] == [5] * 8
    assert {episode["subset"] for episode in episodes} == set(v6.SUBSETS)


def test_unsigned_parameters_refuse_without_writing(tmp_path: Path) -> None:
    face = tmp_path / "v6.json"
    report = tmp_path / "checks.json"
    assert v6.SIGNED_DISTRIBUTION is None, "signature must not be filled by a test"
    monkey_out = tmp_path / "nothing.json"
    assert v6.main(["--check-draft", "--execute-signed"]) == 2
    with pytest.raises(v6.Refuse, match="尚未由 owner 签署"):
        v6.execute_signed(monkey_out)
    assert not face.exists() and not report.exists() and not monkey_out.exists()
    assert v6.main([]) == 2


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def test_the_confusability_knob_lands_at_exactly_the_requested_distance() -> None:
    """同分布的另一条难度维：干扰实体名与主体名只差 d 个字符（句式一维不动）。"""
    params = {**v6.draft_distribution(), "name_neighbor_distance": 2}
    episodes, pair, _sources = v6.build(params)
    reading = v6.audit(episodes, params, pair)
    assert reading["accepted"] is True, reading["checks"]
    block = reading["checks"]["J93_8_name_confusability"]
    assert (block["min_hamming"], block["max_hamming"]) == (2, 2), block
    for episode in episodes:
        subject, *others = episode["entities"]
        assert {hamming(subject, other) for other in others} == {2}, episode["id"]
        assert subject in episode["question"], episode["id"]


def test_a_wrong_distance_makes_the_confusability_check_go_red() -> None:
    params = {**v6.draft_distribution(), "name_neighbor_distance": 2}
    episodes, pair, _sources = v6.build(params)
    episodes[0]["entities"][1] = "z" * len(episodes[0]["entities"][1])
    reading = v6.audit(episodes, params, pair)
    assert reading["checks"]["J93_8_name_confusability"]["ok"] is False, reading["checks"]
    assert reading["accepted"] is False


def test_the_confusability_check_does_not_gate_when_the_knob_is_off() -> None:
    params, episodes, pair = draft()
    block = v6.audit(episodes, params, pair)["checks"]["J93_8_name_confusability"]
    assert block["requested_distance"] is None and block["ok"] is True, block
    with pytest.raises(v6.Refuse, match="name_neighbor_distance"):
        v6.build({**params, "name_neighbor_distance": 0})
