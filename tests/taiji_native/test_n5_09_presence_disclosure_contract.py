"""㊵-621：判读器对过程侧在场计数器的**披露**契约（PLAN-N5-04 G-N5d-4 的读侧对账）。

两支都要走（单向验＝没验）：
① 块里带三枚计数器 ⇒ `presence_counters.status="present"` 且值原样出版；
② 块里不带（㊵-618 之前落盘的所有件，含已入库的 G/H 两臂）⇒ 标 `absent_from_block`，
   **且判据结论与 rc 一律不许变**——本件加的是披露，不是第八枚必需键。

再加一条冻结守卫：`REQUIRED_KEYS` 必须仍是 PLAN-N5-02 那五枚。谁以后把它们扩成八枚，
就会把旧读数追认成 `ran_not_measured`——那是改判据，必须走升版而不是顺手加键。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n5_shadow_gate.py"

FROZEN_REQUIRED = (
    "gate",
    "shadow_gate_requested",
    "candidate_gate",
    "candidate_utility",
    "candidate_counterfactual_utility",
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_gate_judge_621", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _load()


def _face(tmp_path: Path, name: str, block: dict[str, Any]) -> Path:
    path = tmp_path / name
    torch.save({"envelope": {"format": "seed-native-v1", "n5_shadow": block}}, str(path))
    return path


def _block(**over: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "candidate_id": "r4-candidate:abc",
        "bridge_id": "predictive_residual.bridge",
        "gate": 1.0,
        "unit_count": 97,
        "shadow_gate_requested": 1.0,
        "candidate_gate": 0.5,
        "candidate_utility": 0.31,
        "candidate_counterfactual_utility": 0.11,
    }
    block.update(over)
    return block


def _judge(tmp_path: Path, block: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "verdict.json"
    face = _face(tmp_path, "t.pt", block)
    rc = JUDGE.main(["--arm-treated", str(face), "--out", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_required_keys_stay_the_five_frozen_ones() -> None:
    assert JUDGE.REQUIRED_KEYS == FROZEN_REQUIRED


def test_presence_counters_are_published_when_the_block_carries_them(tmp_path: Path) -> None:
    rc, payload = _judge(
        tmp_path,
        _block(shadow_forward_hits=11735, shadow_learn_hits=11735, shadow_branch_hits=23470),
    )
    arm = payload["arms"]["treated"]
    face = arm["presence_counters"]
    assert face["status"] == "present", face
    assert face["missing"] == []
    assert face["values"] == {
        "shadow_forward_hits": 11735,
        "shadow_learn_hits": 11735,
        "shadow_branch_hits": 23470,
    }
    assert arm["j_n5b_1"] == "present"
    assert arm["j_n5b_2"] == "shadow_learned"
    assert rc == 0


def test_absence_is_labelled_and_does_not_move_the_verdict(tmp_path: Path) -> None:
    """不带计数器的旧件：判据结论与 rc 必须与今天一致（披露不是新必需键）。"""
    rc, payload = _judge(tmp_path, _block())
    arm = payload["arms"]["treated"]
    face = arm["presence_counters"]
    assert face["status"] == "absent_from_block", face
    assert set(face["missing"]) == {
        "shadow_forward_hits",
        "shadow_learn_hits",
        "shadow_branch_hits",
    }
    assert face["values"] == {}
    assert arm["j_n5b_1"] == "present"
    assert arm["j_n5b_2"] == "shadow_learned"
    assert rc == 0


def test_partial_presence_is_absent_not_zero(tmp_path: Path) -> None:
    """只带两枚也不能当成"第三枚＝0"——整组降级为 absent 并点名缺哪一枚。"""
    _, payload = _judge(tmp_path, _block(shadow_forward_hits=3, shadow_learn_hits=3))
    face = payload["arms"]["treated"]["presence_counters"]
    assert face["status"] == "absent_from_block"
    assert face["missing"] == ["shadow_branch_hits"]
