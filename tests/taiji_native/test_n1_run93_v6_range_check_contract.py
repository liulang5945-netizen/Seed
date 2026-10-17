"""PLAN-N1-93 范围检验仪的合同测：子集把关与两条线都必须双向可判。

零模型加载、零查询——全部断言跑在合成数据与临时文件上。这个仪器的价值全在"确认子集不
被偷看"和"尺没有动态范围时它必须拒绝说通过"，所以两支都为假的情形都要在测里走一遍。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import verify_taiji_n1_v6_range_check as gate  # noqa: E402

KINDS = ("lookup", "revision", "transition", "unknown")


def endpoint_rows(seed: int, arm: str, error_share: float) -> dict:
    rows = []
    for index in range(20):
        kind = KINDS[index % 4]
        wrong = error_share > 0 and (index % 10) / 10 <= error_share
        rows.append({"id": f"v6-{index:03d}", "kind": kind, "free_exact": not wrong})
    return {"seed": seed, "arm": arm, "rows": rows}


def nine(error_share: float) -> list:
    return [endpoint_rows(seed, arm, error_share) for seed in gate.SEEDS for arm in gate.ARMS]


def test_the_range_lines_go_both_ways() -> None:
    rich = gate.range_lines(nine(0.6))
    assert rich["range_check_passed"] is True, rich
    assert rich["endpoints_total"] == 9
    assert rich["endpoints_with_triple_errors"] == 9

    saturated = gate.range_lines(nine(0.0))
    assert saturated["range_check_passed"] is False, saturated
    assert saturated["endpoints_with_triple_errors"] == 0
    assert all(block["mean_errors"] == 0.0 for block in saturated["per_class"].values())

    partial = gate.range_lines(nine(0.6)[:5])
    assert partial["range_check_passed"] is False, "a short run must never read as passed"
    assert partial["endpoints_total"] == 5


def test_a_single_endpoint_short_of_the_floor_is_counted_not_rounded() -> None:
    results = nine(0.6)
    for row in results[0]["rows"]:
        row["free_exact"] = True
    reading = gate.range_lines(results)
    assert reading["endpoints_with_triple_errors"] == 8, reading


def test_unsigned_or_broken_faces_are_refused(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "v6.json"
    monkeypatch.setattr(gate, "FACE", missing)
    with pytest.raises(gate.Refuse, match="signed v6 face is missing"):
        gate.load_face()

    face = tmp_path / "face.json"
    face.write_text(json.dumps({"episodes": [], "subsets": {}}), encoding="utf-8")
    checks = tmp_path / "checks.json"
    checks.write_text(
        json.dumps({"face_sha256": "0" * 64, "accepted": True, "queries": 0}), encoding="utf-8"
    )
    monkeypatch.setattr(gate, "FACE", face)
    monkeypatch.setattr(gate, "CHECKS", checks)
    with pytest.raises(gate.Refuse, match="differs from its check report"):
        gate.load_face()


def test_a_leaked_confirmation_item_stops_the_check() -> None:
    face = {
        "episodes": [
            {"id": "a", "kind": "lookup"},
            {"id": "b", "kind": "revision"},
        ],
        "subsets": {"sacrifice": ["a", "b"], "confirm": ["b"]},
    }
    with pytest.raises(gate.Refuse, match="leaked into the range check"):
        gate.queried_ids(face)
    honest = {
        "episodes": face["episodes"],
        "subsets": {"sacrifice": ["a"], "confirm": ["b"]},
    }
    assert gate.queried_ids(honest) == ["a"]


def test_the_deadline_guard_has_both_branches() -> None:
    now = time.process_time()
    assert gate.Deadline(now, 60).guard(5) is None
    with pytest.raises(gate.Refuse, match="CPU reserve"):
        gate.Deadline(now, -1).guard(0)


def test_cli_modes_are_exclusive(tmp_path: Path, monkeypatch) -> None:
    assert gate.main([]) == 2
    assert gate.main(["--readiness", "--execute-approved"]) == 2
    sealed = tmp_path / "range.json"
    sealed.write_text("{}", encoding="utf-8")
    before = sealed.read_bytes()
    monkeypatch.setattr(gate, "OUT", sealed)
    assert gate.main(["--execute-approved"]) == 2
    assert sealed.read_bytes() == before, "refusal must not rewrite the sealed output"
