"""PLAN-N1-94 执行器的合同测：四门逐门双向、双口径摘要分得清"口径"与"真改"、前置缺一即拒。

零模型装载、零查询——全部断言跑在合成行、临时文件与临时题面上。这个执行器一旦发射就永久消费
那 20 题，所以"什么情况下它必须拒绝发射"要在这里全部走一遍，而不是等到有题面的那天。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.training import eval_taiji_n1_transition_factorial_confirmation_v6 as v6  # noqa: E402
from scripts.training import verify_taiji_n1_v6_range_check as range_gate  # noqa: E402


def rows_for(cell_counts: dict) -> list:
    rows = []
    for (seed, arm), correct in cell_counts.items():
        tally = 0
        for index in range(v6.ITEMS_PER_MODEL):
            kind = v6.KINDS[index % 4]
            right = tally < correct
            if right:
                tally += 1
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "id": f"v6-confirm-{index:03d}",
                    "kind": kind,
                    "free_exact": right,
                }
            )
    return rows


def passing_cells() -> dict:
    return {(seed, arm): (14 if arm == "position" else 4) for seed in v6.SEEDS for arm in v6.ARMS}


def test_all_four_gates_pass_on_a_clear_lead() -> None:
    reading = v6.evaluate_gates(rows_for(passing_cells()))
    gates = reading["gates"]
    assert gates["each_position_seed"]["passed"] is True
    assert gates["each_position_kind"]["passed"] is True
    assert gates["advantage_terminal"]["value"] == 30
    assert gates["advantage_cut"]["value"] == 30
    assert reading["all_four_gates"] is True


def test_each_gate_alone_can_break_the_conjunction() -> None:
    base = passing_cells()
    weak = dict(base)
    weak[(13, "position")] = v6.GATES["each_position_seed"] - 1
    reading = v6.evaluate_gates(rows_for(weak))
    assert reading["gates"]["each_position_seed"]["passed"] is False
    assert reading["all_four_gates"] is False

    base = passing_cells()
    kindless = dict(base)
    kindless[(11, "position")] = 17
    kindless_rows = rows_for(kindless)
    for row in kindless_rows:
        if row["seed"] == 11 and row["arm"] == "position" and row["kind"] == "transition":
            row["free_exact"] = False
    reading = v6.evaluate_gates(kindless_rows)
    assert reading["gates"]["each_position_kind"]["passed"] is False
    assert reading["gates"]["each_position_seed"]["passed"] is True

    cut_lead = dict(passing_cells())
    cut_lead[(11, "cut")] = 14
    cut_lead[(12, "cut")] = 14
    reading = v6.evaluate_gates(rows_for(cut_lead))
    assert reading["gates"]["advantage_cut"]["passed"] is False
    assert reading["gates"]["advantage_terminal"]["passed"] is True
    assert reading["all_four_gates"] is False


def test_a_lead_exactly_on_the_line_counts_and_one_less_does_not() -> None:
    on_line = dict(passing_cells())
    for seed in v6.SEEDS:
        on_line[(seed, "cut")] = 10
    reading = v6.evaluate_gates(rows_for(on_line))
    assert reading["gates"]["advantage_cut"]["value"] == v6.GATES["advantage_cut"]
    assert reading["gates"]["advantage_cut"]["passed"] is True

    one_less = dict(passing_cells())
    for seed in v6.SEEDS:
        one_less[(seed, "cut")] = 11
    reading = v6.evaluate_gates(rows_for(one_less))
    assert reading["gates"]["advantage_cut"]["value"] < v6.GATES["advantage_cut"]
    assert reading["gates"]["advantage_cut"]["passed"] is False


def test_a_short_cell_is_refused_rather_than_rescaled() -> None:
    rows = rows_for(passing_cells())[:-1]
    with pytest.raises(v6.Refuse, match="queried 19 items|expected nine cells"):
        v6.evaluate_gates(rows)


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_face(tmp_path: Path) -> dict:
    episodes = []
    for index in range(40):
        kind = v6.KINDS[index % 4]
        episodes.append(
            {
                "id": f"v6-confirm-{index:03d}",
                "split": "holdout",
                "kind": kind,
                "observations": [f"e{index} uses abcdef."],
                "answer": "abcdef",
            }
        )
    face = {
        "episodes": episodes,
        "subsets": {
            "sacrifice": [e["id"] for e in episodes[:20]],
            "confirm": [e["id"] for e in episodes[20:]],
        },
    }
    write(tmp_path / "face.json", face)
    checks = {"face_sha256": range_gate.sha(tmp_path / "face.json"), "accepted": True, "queries": 0}
    write(tmp_path / "checks.json", checks)
    return {
        "FACE": tmp_path / "face.json",
        "CHECKS": tmp_path / "checks.json",
        "face": face,
    }


def test_preconditions_refuse_loudly(tmp_path: Path, monkeypatch) -> None:
    fixture = make_face(tmp_path)
    monkeypatch.setattr(v6, "FACE", fixture["FACE"])
    monkeypatch.setattr(v6, "CHECKS", fixture["CHECKS"])
    monkeypatch.setattr(range_gate, "FACE", fixture["FACE"])
    monkeypatch.setattr(range_gate, "CHECKS", fixture["CHECKS"])

    good = {
        "adjudication": {"range_check_passed": True},
        "confirm_subset_queried": False,
        "subsets_queried": ["sacrifice"],
        "face_sha256": range_gate.sha(fixture["FACE"]),
    }
    report = write(tmp_path / "range.json", good)
    monkeypatch.setattr(v6, "RANGE", report)
    monkeypatch.setattr(v6, "REPORT", tmp_path / "nope.json")
    monkeypatch.setattr(v6.bind, "family_receipts", lambda: {})
    nine = {(seed, arm): tmp_path / f"{seed}-{arm}.pt" for seed in v6.SEEDS for arm in v6.ARMS}
    monkeypatch.setattr(v6.bind, "endpoints", lambda _receipts: (nine, {}))
    monkeypatch.setattr(v6.bind, "cross_family_guard_conflicts", lambda _receipts: {})
    context = v6.preconditions()
    assert len(context["items"]) == v6.ITEMS_PER_MODEL

    for field, value, message in (
        ("confirm_subset_queried", True, "already queried"),
        ("subsets_queried", ["sacrifice", "confirm"], "sacrifice subset"),
        ("face_sha256", "0" * 64, "different face"),
    ):
        broken = copy.deepcopy(good)
        broken[field] = value
        write(tmp_path / "range.json", broken)
        with pytest.raises(v6.Refuse, match=message):
            v6.preconditions()

    failed = copy.deepcopy(good)
    failed["adjudication"]["range_check_passed"] = False
    write(tmp_path / "range.json", failed)
    with pytest.raises(v6.Refuse, match="never shown to have range"):
        v6.preconditions()


def test_digests_separate_a_convention_gap_from_a_real_change(tmp_path: Path) -> None:
    json_path = tmp_path / "guard.json"
    json_path.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    binary = tmp_path / "guard.pt"
    binary.write_bytes(b"\x00\x01\x02\r\n")
    recorded = {
        str(json_path): hashlib.sha256(json_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        str(binary): hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    reading = v6.source_guard_digests(recorded)
    assert reading["sources_unchanged"] is True
    assert reading["convention_only_count"] == 1
    assert reading["real_change_count"] == 0

    changed = dict(recorded)
    changed[str(binary)] = "f" * 64
    reading = v6.source_guard_digests(changed)
    assert reading["sources_unchanged"] is False
    assert reading["judged_mismatch_count"] == 1
    assert reading["real_change_count"] == 1


def test_cli_refusals(tmp_path: Path, monkeypatch) -> None:
    assert v6.main([]) == 2
    assert v6.main(["--readiness", "--execute-approved"]) == 2
    fixture = make_face(tmp_path)
    monkeypatch.setattr(v6, "FACE", fixture["FACE"])
    monkeypatch.setattr(v6, "CHECKS", fixture["CHECKS"])
    monkeypatch.setattr(range_gate, "FACE", fixture["FACE"])
    monkeypatch.setattr(range_gate, "CHECKS", fixture["CHECKS"])
    good = write(
        tmp_path / "range.json",
        {
            "adjudication": {"range_check_passed": True},
            "confirm_subset_queried": False,
            "subsets_queried": ["sacrifice"],
            "face_sha256": range_gate.sha(fixture["FACE"]),
        },
    )
    monkeypatch.setattr(v6, "RANGE", good)
    nine = {(seed, arm): tmp_path / f"{seed}-{arm}.pt" for seed in v6.SEEDS for arm in v6.ARMS}
    monkeypatch.setattr(v6.bind, "family_receipts", lambda: {})
    monkeypatch.setattr(v6.bind, "endpoints", lambda _receipts: (nine, {}))
    monkeypatch.setattr(v6.bind, "cross_family_guard_conflicts", lambda _receipts: {})

    sealed = tmp_path / "report.json"
    sealed.write_text("{}", encoding="utf-8")
    before = sealed.read_bytes()
    monkeypatch.setattr(v6, "REPORT", sealed)
    assert v6.main(["--execute-approved"]) == 2
    assert sealed.read_bytes() == before, "refusal must not rewrite the sealed report"


def test_the_unseen_claim_column_is_derived_and_can_go_false() -> None:
    """DEBT-G155 修法：这一列原先是字面量 `True`，`status=interrupted` 那支也照出版它。"""
    full = rows_for(passing_cells())
    neutral = [{"unchanged": True} for _ in v6.SEEDS]
    assert v6.unseen_claim("published", full, len(full), neutral) is True
    assert v6.unseen_claim("interrupted", full, len(full), neutral) is False
    assert v6.unseen_claim("published", full[:-1], len(full) - 1, neutral) is False
    assert v6.unseen_claim("published", full, len(full) - 5, neutral) is False
    assert v6.unseen_claim("published", full, len(full), []) is False
    broken = [{"unchanged": True}, {"unchanged": False}]
    assert v6.unseen_claim("published", full, len(full), broken) is False


def test_today_s_published_face_still_reads_true_under_the_derivation() -> None:
    """换算法之后先打在真件上：已出版的那一枚读数不因修法而翻。"""
    entry = json.loads(v6.REPORT.read_text(encoding="utf-8"))
    assert (
        v6.unseen_claim(
            entry["status"],
            entry["rows"],
            entry["confirmation_queries"],
            entry["endpoint_neutrality"],
        )
        is True
    )
    assert entry["retry_allowed"] is False and entry["parameter_updates"] == 0
