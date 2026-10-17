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


def signed_face(tmp_path: Path, monkeypatch) -> None:
    """一份最小自洽的已签题面：40 题、两半不重叠、检查件与题面指纹同源。"""
    episodes = [
        {"id": f"v6-{index:03d}", "kind": KINDS[index % 4], "split": "holdout"}
        for index in range(40)
    ]
    ids = [episode["id"] for episode in episodes]
    face = tmp_path / "face.json"
    face.write_text(
        json.dumps(
            {"episodes": episodes, "subsets": {"sacrifice": ids[:20], "confirm": ids[20:]}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    checks = tmp_path / "checks.json"
    checks.write_text(
        json.dumps({"face_sha256": gate.sha(face), "accepted": True, "queries": 0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "FACE", face)
    monkeypatch.setattr(gate, "CHECKS", checks)
    monkeypatch.setattr(gate, "OUT", tmp_path / "range.json")


def stub_measure(monkeypatch, rows_for) -> None:
    endpoints = {(seed, arm): Path("endpoint.pt") for seed in gate.SEEDS for arm in gate.ARMS}
    monkeypatch.setattr(gate.bind, "family_receipts", lambda: {})
    monkeypatch.setattr(gate.bind, "endpoints", lambda receipts: (endpoints, {}))

    def fake(seed, arm, path, items, budget):
        return {"seed": seed, "arm": arm, "rows": rows_for(seed, arm, len(items))}

    monkeypatch.setattr(gate, "measure_endpoint", fake)


def synthetic_rows(seed: int, arm: str, count: int) -> list:
    wrong_by_kind = {"lookup": 3, "revision": 2, "transition": 2, "unknown": 1}
    return [
        {
            "id": f"v6-{index:03d}",
            "kind": KINDS[index % 4],
            "free_exact": (index // 4) >= wrong_by_kind[KINDS[index % 4]],
        }
        for index in range(count)
    ]


def test_the_published_columns_reproduce_the_adjudication(tmp_path: Path, monkeypatch) -> None:
    """判读用的逐类错误必须进件：否则 `usable` 只能由件外的内存态算出，复算仪无从独立复算。"""
    signed_face(tmp_path, monkeypatch)
    stub_measure(monkeypatch, synthetic_rows)
    assert gate.run_approved() == 0
    reading = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))
    assert reading["format"] == "taiji-n1-93-v6-range-check-v2"
    assert reading["generation_cap_site"].endswith(".py:114")
    for entry in reading["endpoints"]:
        split = entry["errors_by_kind"]
        assert sum(split.values()) == entry["errors"], entry
        assert entry["addressable_errors"] == entry["errors"] - split["unknown"], entry
        assert entry["free_exact"] + entry["errors"] == 20, entry
    usable = sum(
        1
        for entry in reading["endpoints"]
        if entry["addressable_errors"] >= gate.TRIPLE_ERROR_FLOOR
    )
    assert usable == reading["adjudication"]["endpoints_with_triple_errors"], reading


def test_an_unlisted_kind_breaks_the_split_guard(tmp_path: Path, monkeypatch) -> None:
    """守卫要能为假：一行落在 KINDS 之外时，逐类拆分漏计错误，本件必须拒绝出版。"""

    def rows_with_stray_kind(seed, arm, count):
        rows = synthetic_rows(seed, arm, count)
        if (seed, arm) == (gate.SEEDS[0], gate.ARMS[0]):
            rows[0]["kind"] = "reasoning"
            rows[0]["free_exact"] = False
        return rows

    signed_face(tmp_path, monkeypatch)
    stub_measure(monkeypatch, rows_with_stray_kind)
    with pytest.raises(gate.Refuse, match="class split misses an error"):
        gate.run_approved()
    assert not (tmp_path / "range.json").exists(), "a refused run must not leave a report behind"
