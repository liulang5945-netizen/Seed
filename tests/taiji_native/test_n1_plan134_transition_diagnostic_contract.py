from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scripts.training import build_taiji_n1_v16_transition_direction_position_diagnostic as v16
from scripts.training import eval_taiji_n1_v16_transition_direction_position_diagnostic as evaluator


def _sealed_pins() -> list[str]:
    """封存 readiness 要复算在位字节的全部对象：来源哈希表的键＋endpoint checkpoint 路径。"""
    record = v16.read_json(v16.N133_READINESS)
    return list(record.get("source_hashes", {})) + [
        row["path"] for row in record.get("endpoint_restores", [])
    ]


def _machine_bound(raw: str) -> bool:
    #: 有仓库身份（在 HEAD 索引里，或仓内当前在位）的钉点换机也能取证——缺件＝工作树损坏，
    #: 必须响亮红，不许豁免；只有仓外一次性预检文件和未入库 checkpoint 这种既无仓库身份
    #: 又不在盘上的，才是"离了建账机器就复算不了"。
    if v16.seal_ast.repo_relative(raw) is not None:
        return False
    return not Path(raw).is_file()


def test_v16_face_crosses_all_directions_and_target_positions() -> None:
    excluded_names: set[str] = set()
    excluded_words = set(v16.DIRECTIONS)
    face = v16.build_v16_face(excluded_names, excluded_words)
    checks = v16.audit_v16_face(face, excluded_names=excluded_names, excluded_words=excluded_words)

    assert checks["accepted"] is True
    assert face["schema"] == "n1-transition-direction-position-diagnostic-v1"
    assert face["seed"] == 116
    assert len(face["episodes"]) == 40
    assert Counter(
        (item["answer_direction"], item["target_start_index"]) for item in face["episodes"]
    ) == Counter(
        {(direction, start): 2 for direction in v16.DIRECTIONS for start in v16.TARGET_STARTS}
    )
    assert checks["checks"]["left_right_balanced_in_each_cell"] is True
    assert all(len(item["observations"]) == 16 for item in face["episodes"])


def test_v16_face_audit_rejects_wrong_answer_direction() -> None:
    excluded_names: set[str] = set()
    excluded_words = set(v16.DIRECTIONS)
    face = v16.build_v16_face(excluded_names, excluded_words)
    face["episodes"][0]["answer"] = "south"

    checks = v16.audit_v16_face(face, excluded_names=excluded_names, excluded_words=excluded_words)

    assert checks["accepted"] is False
    assert checks["checks"]["two_target_rows_and_direction_consistency"] is False


def test_v16_source_union_is_bound_to_consumed_v15_and_frozen_endpoints() -> None:
    bound = [raw for raw in _sealed_pins() if _machine_bound(raw)]
    if bound:
        pytest.skip(
            f"N1-134 这条契约复算封存记录钉住的在位字节，本机有 {len(bound)} 枚钉点既无仓库身份也不在盘上"
            f"（建账机器的一次性预检文件／未入库 checkpoint）；按批 C 裁定 skipif 化，"
            f"产物入库即自动恢复全量守护。示例：{bound[:2]}"
        )
    names, words, source_hashes, counts, endpoints = v16._prior_sources(require_unused_seed=False)

    assert len(names) == 7938
    assert len(words) == 6288
    assert counts["source_hash_count"] == len(source_hashes)
    assert len(endpoints) == 12
    assert sum(item["source"] == "N1-132" for item in endpoints) == 3
    assert sum(item["source"] == "N1-120" for item in endpoints) == 9
    assert v16.FACE.is_relative_to(v16.ROOT)
    assert v16.RUN_REPORT.is_relative_to(v16.ROOT)
    if v16.RUN_REPORT.exists():
        run_report = v16.read_json(v16.RUN_REPORT)
        assert run_report.get("complete") is True
        assert v16.seal_ast.receipt_binding_matches(run_report.get("face_sha256"), v16.FACE)
    else:
        assert not v16.FACE.exists()


def test_machine_bound_predicate_only_excuses_unrecoverable_pins(tmp_path: Path) -> None:
    """豁免判据的四个角都要能指出来：在册缺件绝不豁免，仓外预检件在本机在位时也不豁免。"""

    tracked = str(v16.N133_READINESS)
    relative = v16.seal_ast.repo_relative(tracked)
    assert relative is not None and _machine_bound(tracked) is False

    ghost = f"/home/runner/work/Seed/Seed/{relative}"
    assert not Path(ghost).is_file()
    assert _machine_bound(ghost) is False

    scratch = tmp_path / "face_conf-s4-d1-draw91.json"
    assert v16.seal_ast.repo_relative(str(scratch)) is None
    assert _machine_bound(str(scratch)) is True
    scratch.write_text("{}", encoding="utf-8")
    assert _machine_bound(str(scratch)) is False


def test_v16_direction_position_summary_keeps_full_cell_denominator() -> None:
    face = v16.build_v16_face(set(), set(v16.DIRECTIONS))
    endpoint_rows = []
    for source, arms in (("N1-132", ("position",)), ("N1-120", ("position", "terminal", "cut"))):
        for seed in (11, 12, 13):
            for arm in arms:
                rows = [
                    {
                        "id": item["id"],
                        "free_exact": (index % 2 == 0),
                    }
                    for index, item in enumerate(face["episodes"])
                ]
                endpoint_rows.append({"source": source, "seed": seed, "arm": arm, "rows": rows})

    cell_summary, paired = evaluator._summarize_direction_position(endpoint_rows, face)

    assert len(cell_summary) == 12
    assert all(
        cell["items"] == 2
        for endpoint in cell_summary.values()
        for cell in endpoint["by_direction_and_target_start"].values()
    )
    assert len(paired) == 9
    assert all(result["overall"]["candidate_minus_control"] == 0 for result in paired.values())
