"""CAP F 维复判的合同测试：证明它读的是**门是否真的过**，不是证据文件在不在盘上。

F 维旧实现只记 `report_present`（文件存在性布尔）。这类"只读封存件"的检查永远不会红，
本仓已在 DEBT-I4/普查 §1e 反复登记过同一失效形态，所以这里的断言全部朝"能否把它跑红"来写。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.training.eval_taiji_cap0_baseline import (
    _dimension_items,
    _eval_set,
    _f_dimension_block,
)
from scripts.training.eval_taiji_cap0_f_dimension import (
    F_ADJUDICATORS,
    adjudicate_f_items,
    adjudicate_unified_entry,
    f_dimension_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
F01_REPORT = PROJECT_ROOT / "reports/taiji_b0_b1_representation_20260915.json"
UNIFIED_ENTRY_SEALED = PROJECT_ROOT / "reports/taiji_unified_entry_evidence_20260919.json"


def _reference_of(item_id: str) -> str:
    """被引报告路径取自现行评价集，测试里不再抄第三份字面串。"""

    return next(
        str(item["reference"])
        for item in _dimension_items(_eval_set(), "F")
        if str(item["id"]) == item_id
    )


F04_REF = _reference_of("F04")
F04_REPORT = PROJECT_ROOT / F04_REF


def _f_items() -> list[dict[str, object]]:
    return _dimension_items(_eval_set(), "F")


def _rows() -> dict[str, dict[str, object]]:
    return {str(row["id"]): row for row in adjudicate_f_items(_f_items())}


# --- 复判读内容，不读在场 --------------------------------------------------


def test_every_f_item_has_a_recomputable_gate_and_the_dispatch_exists() -> None:
    items = _f_items()
    assert [str(i["id"]) for i in items] == ["F01", "F02", "F03", "F04"]
    assert set(F_ADJUDICATORS) == {str(i["id"]) for i in items}


def test_f01_passes_on_recomputed_numbers_not_on_file_presence() -> None:
    row = _rows()["F01"]
    assert row["report_present"] is True
    assert row["gate_verdict"] == "pass"
    held = {str(c["clause"]): c["held"] for c in row["clauses"]}
    assert all(held.values()), held


def test_f02_negative_result_stays_red_and_is_not_rescued() -> None:
    row = _rows()["F02"]
    assert row["verdict"] == "fail"
    failed = [c for c in row["clauses"] if c["held"] is False]
    assert failed, "G4/G5 未过必须被复算出来"
    assert "G4_task_gate_H1" in failed[0]["detail"]


def test_f02_consistency_clause_actually_compares_instead_of_being_always_true() -> None:
    """钉住那条"报告自带结论与门一致"的子句：它必须能红，而不是恒真。

    曾经写成 `bool(...) is not None` —— bool() 永远不是 None，于是这条子句永久为真，
    一个"门全过但自带结论说没过"的自相矛盾载荷也照样放行。
    """

    base = {
        "gates": {f"G{i}_x": True for i in range(1, 7)},
        "experiment_passed": True,
    }
    consistent = _verdict_of_rows(F_ADJUDICATORS["F02"](base))
    assert consistent == "pass"

    contradictory = {
        "gates": {f"G{i}_x": True for i in range(1, 7)},
        "experiment_passed": False,
    }
    clauses = F_ADJUDICATORS["F02"](contradictory)
    agreement = [c for c in clauses if "experiment_passed" in c["clause"]][0]
    assert agreement["held"] is False, "自带结论与门不一致时必须红"
    assert _verdict_of_rows(clauses) == "fail"


def test_a_report_that_is_present_but_failed_its_gate_never_reads_as_pass() -> None:
    """同一份文件仍在场，只把 F01 的一道门改成不过 ⇒ 复算必须翻成 fail。"""

    payload = json.loads(F01_REPORT.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(payload)
    tampered["gates"]["no_leakage"] = False
    assert _verdict_of_rows(F_ADJUDICATORS["F01"](tampered)) == "fail"
    assert _verdict_of_rows(F_ADJUDICATORS["F01"](payload)) == "pass"


def _verdict_of_rows(clauses: list[dict[str, object]]) -> str:
    from scripts.training.eval_taiji_cap0_f_dimension import _verdict_of

    return _verdict_of(clauses)


def test_missing_reference_report_is_unavailable_not_zero_and_not_pass(
    tmp_path: Path,
) -> None:
    item = {"id": "F01", "capability": "x", "gate": "g", "reference": "nope/absent.json"}
    row = adjudicate_f_items([item], root=tmp_path)[0]
    assert row["report_present"] is False
    assert row["verdict"] == "unavailable"


def test_unreadable_reference_report_fails_closed(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    item = {"id": "F01", "capability": "x", "gate": "g", "reference": "broken.json"}
    row = adjudicate_f_items([item], root=tmp_path)[0]
    assert row["report_present"] is True
    assert row["verdict"] == "unavailable"


def test_unmachine_checkable_gate_clause_caps_the_item_at_partial() -> None:
    row = _rows()["F03"]
    assert row["gate_verdict"] == "partial"
    unverified = [c for c in row["clauses"] if c["held"] is None]
    labels = " ".join(str(c["clause"]) for c in unverified)
    assert "+2.000" in labels and "interleaved" in labels


def test_must_show_without_a_machine_check_caps_an_item_at_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """过门 ≠ 通过：抽掉某项 must_show 的机检实现，它就只剩 partial（不得被当成已验收）。"""

    from scripts.training import eval_taiji_cap0_f_dimension as fmod

    assert _rows()["F01"]["verdict"] == "pass"
    monkeypatch.delitem(fmod._MUST_SHOW_ADJUDICATORS, "F01")
    row = adjudicate_f_items(_f_items())[0]
    assert row["id"] == "F01"
    assert row["gate_verdict"] == "pass"
    assert row["verdict"] == "partial"
    assert row["must_show_clauses"][0]["held"] is None


def test_f01_must_show_demands_actual_execution_not_only_a_fit_probe() -> None:
    assert _rows()["F01"]["verdict"] == "pass"

    payload = json.loads(F01_REPORT.read_text(encoding="utf-8"))
    stripped = copy.deepcopy(payload)
    stripped["discrimination"]["candidate_surface"]["actual_gain_vs_all_singleton_oracle"] = {}
    from scripts.training.eval_taiji_cap0_f_dimension import _must_show_f01

    assert any(c["held"] is False for c in _must_show_f01(stripped))


# --- 统一入口的 L1–L4 复算 --------------------------------------------------


def test_unified_entry_recomputation_agrees_with_the_sealed_conclusion() -> None:
    report = json.loads(UNIFIED_ENTRY_SEALED.read_text(encoding="utf-8"))
    adjudged = adjudicate_unified_entry(report)
    assert adjudged["lines_match_report"] is True
    assert adjudged["recomputed_all_pass"] is True
    assert adjudged["unverifiable_clauses"], "L3 代理与 L4 digest 缺口必须留在账上"


def test_unified_entry_recomputation_goes_red_when_a_recorded_line_is_inflated() -> None:
    """把 full 臂的成功率下调（消融臂仍有 1.0）⇒ L1 必须翻假，且与报告里记的 true 不一致。"""

    report = json.loads(UNIFIED_ENTRY_SEALED.read_text(encoding="utf-8"))
    report["per_arm"] = [
        {**row, "main_success_rate": 0.0} if row["arm"] == "full" else row
        for row in report["per_arm"]
    ]
    adjudged = adjudicate_unified_entry(report)
    assert adjudged["recomputed_lines"]["L1_full_beats_all_ablations"] is False
    assert adjudged["recomputed_all_pass"] is False
    assert adjudged["lines_match_report"] is False


def test_unified_entry_recomputation_flags_report_and_numbers_disagreeing() -> None:
    report = json.loads(UNIFIED_ENTRY_SEALED.read_text(encoding="utf-8"))
    report["lines"]["L2_handoff_resource_line"] = False
    adjudged = adjudicate_unified_entry(report)
    assert adjudged["lines_match_report"] is False


def test_a_missing_arm_is_not_silently_a_pass() -> None:
    report = json.loads(UNIFIED_ENTRY_SEALED.read_text(encoding="utf-8"))
    report["per_arm"] = [r for r in report["per_arm"] if r["arm"] != "disable_selection"]
    adjudged = adjudicate_unified_entry(report)
    assert adjudged["recomputed_lines"]["L2_handoff_resource_line"] is False


# --- 维度门与健康报告接线 ----------------------------------------------------


def _live_executed() -> dict[str, object]:
    report = json.loads(UNIFIED_ENTRY_SEALED.read_text(encoding="utf-8"))
    return {
        "status": "executed",
        **adjudicate_unified_entry(report),
        "matches_sealed_report": True,
    }


def test_dimension_gate_requires_the_passing_item_to_show_its_own_chain() -> None:
    rows = adjudicate_f_items(_f_items())
    gate = f_dimension_gate(rows, _live_executed())
    assert gate["items_passing_frozen_gate"] == ["F01"]
    assert gate["items_passing_gate_and_must_show"] == ["F01"]
    assert gate["verdict"] == "pass"
    assert gate["end_to_end_demonstration"]["counts_as_item_chain"] is False


def test_bundle_level_demo_alone_never_lights_the_dimension_up() -> None:
    rows = adjudicate_f_items(_f_items())
    for row in rows:
        row["verdict"] = "partial"
    gate = f_dimension_gate(rows, _live_executed())
    assert gate["verdict"] == "partial"
    assert gate["reason"]


def test_not_executed_live_run_is_recorded_as_such() -> None:
    gate = f_dimension_gate(adjudicate_f_items(_f_items()), {"status": "not_executed"})
    assert gate["end_to_end_demonstration"]["unified_entry_reproduced"] is False


def test_health_f_block_carries_recomputed_rows_and_no_longer_a_bare_list() -> None:
    block = _f_dimension_block(f_live_evidence=False)
    assert "contracts" not in block
    assert [row["id"] for row in block["items"]] == ["F01", "F02", "F03", "F04"]
    assert block["live_entry_evidence"]["status"] == "not_executed"
    assert block["dimension_gate"]["verdict"] in {"pass", "partial"}
    for row in block["items"]:
        assert "clauses" in row and "verdict" in row
        assert "must_show_clauses" in row


def test_f04_on_the_current_eval_set_is_not_flagged_stale() -> None:
    """v2 把 F04 的引用挪到新底 inventory ⇒ 时效守卫必须放行（否则换底没做成）。"""

    from scripts.training.eval_taiji_cap0_baseline import EVAL_SET_PATH

    assert EVAL_SET_PATH.name == "cap0_eval_set_v2.json"
    row = _rows()["F04"]
    staleness = [c for c in row["clauses"] if "当前产品默认基座" in c["clause"]]
    assert staleness and staleness[0]["held"] is True
    assert row["verdict"] != "stale_reference"
    #: 换底没有把模板回显改掉，所以 F04 仍不是 pass —— 引用换了不等于门过了。
    assert row["verdict"] == "fail"


def test_a_reference_describing_another_substrate_reads_stale_not_failed(tmp_path) -> None:
    """被引报告若描述别的基座，判过/判不过都算冒称 ⇒ 必须是第三种状态 stale_reference。"""

    payload = json.loads(F04_REPORT.read_text(encoding="utf-8"))
    shifted = copy.deepcopy(payload)
    shifted["model_reality"]["default_checkpoint"] = "seed_corpus.pt"
    shifted["model_reality"]["default_tick"] = 2
    shifted["model_reality"]["wiring_defect"] = True
    (tmp_path / "inv.json").write_text(json.dumps(shifted), encoding="utf-8")
    item = {
        "id": "F04",
        "capability": "x",
        "gate": "g",
        "reference": "inv.json",
        "must_show": "m",
    }
    row = adjudicate_f_items([item], root=tmp_path)[0]
    assert row["verdict"] == "stale_reference"
    assert row["gate_verdict"] == "stale_reference"


def test_the_stale_check_is_inert_when_the_provenance_record_is_unreadable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读不到现行登记时不许"默认未过期"——守卫本身失效也得留在读数里。"""

    from scripts.training import eval_taiji_cap0_f_dimension as fmod

    monkeypatch.setattr(fmod, "_current_product_default_name", lambda: None)
    row = adjudicate_f_items(
        [{"id": "F04", "capability": "x", "gate": "g", "reference": F04_REF, "must_show": "m"}]
    )[0]
    staleness = [c for c in row["clauses"] if "当前产品默认基座" in c["clause"]][0]
    assert staleness["held"] is None
    assert row["verdict"] != "pass"


def test_the_frozen_gate_text_used_for_recomputation_is_the_manifests_own(tmp_path: Path) -> None:
    """复判读的是冻结评价集里的门文本与被引报告，runner 不再自己抄一份（旧 F_CONTRACTS 已漂移）。"""

    manifest = {str(i["id"]): str(i["gate"]) for i in _f_items()}
    rows = {str(r["id"]): str(r["gate"]) for r in adjudicate_f_items(_f_items())}
    assert rows == manifest
