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
