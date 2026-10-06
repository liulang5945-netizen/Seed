"""A30 逐字段比较器的守卫：验收式「默认关闭 ⇒ 逐位不变」必须**既判得相同也判得不同**。

来历：DECISION-A30 §3 甲。写这条守卫的理由是今天踩过的两件事——
① 只验 fail-closed 分支的判读器等于没验（成功路径第一次走就炸在解包上）；
② 跨仪器版本整块比 JSON 会把**后加的披露字段**当成行为差异（实测第一处差异是
   `treated_with_circuit.answer_bytes_quantiles` 一边 `None` 一边 `dict`）。
⇒ 本文件同时钉住"能为真""能为假""schema 与 behavior 分得开"三件事。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from compare_taiji_a30_report_identity import compare_files, diff_reports, main  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
SEEDB_ANCHOR = REPORTS / "taiji_a30_surface_tradeoff_seedB_anchor_20261002.json"
SEEDB_WINDOW = REPORTS / "taiji_a30_surface_tradeoff_seedB_window64_20261002.json"
CAP_FULL = REPORTS / "taiji_a30_cap_dual_arm_chunked_short_budget256_20261002.json"
CAP_WINDOW = REPORTS / "taiji_a30_cap_dual_arm_chunked_short_window64_20261002.json"


def test_same_config_control_arms_read_as_identical() -> None:
    """两次独立运行的**未挂回路对照**必须逐字段相同——这是"默认关闭 ⇒ 逐位不变"的可行性证据。"""
    result = compare_files(SEEDB_ANCHOR, SEEDB_WINDOW, subtree="control_no_circuit")
    assert result["identical"] is True, result
    assert result["behavior_diff_count"] == 0, result


def test_different_config_arms_are_detected_as_different() -> None:
    """守卫必须能为假：全剂量臂与窗口 K=64 臂在同一面上行为不同。"""
    result = compare_files(CAP_FULL, CAP_WINDOW, subtree="treated_with_circuit")
    assert result["identical"] is False, result
    assert result["behavior_diff_count"] >= 1, result
    paths = [str(result["first_behavior_diff"])[0:60]]
    assert paths, result


def test_later_added_disclosure_field_is_schema_not_behavior(tmp_path: Path) -> None:
    """后加的披露字段（旧件为 `None`／新件有值，或一侧缺键）必须归到 schema 那一堆。"""
    old = {"correct": 6, "answer_bytes_quantiles": None, "format": "v1"}
    new = {"correct": 6, "answer_bytes_quantiles": {"median": 12}, "format": "v1", "prereg": "x"}
    behavior, schema = diff_reports(old, new)
    assert behavior == [], behavior
    kinds = {path for path, _ in schema}
    assert "answer_bytes_quantiles" in kinds, schema
    assert "prereg" in kinds, schema
    # 同样的两份东西，strict 一开必须翻成"不相同"——否则 --strict 是个装饰性开关。
    left = tmp_path / "old.json"
    right = tmp_path / "new.json"
    left.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    right.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    assert compare_files(left, right)["identical"] is True
    assert compare_files(left, right, strict=True)["identical"] is False


def test_cli_exit_code_is_the_verdict(tmp_path: Path) -> None:
    """rc 就是结论：相同 ⇒ 0，不同 ⇒ 1（验收要能直接挂门，不能靠人读 JSON）。"""
    same_a = tmp_path / "a.json"
    same_b = tmp_path / "b.json"
    other = tmp_path / "c.json"
    same_a.write_text(json.dumps({"control_no_circuit": {"correct": 0}}), encoding="utf-8")
    same_b.write_text(json.dumps({"control_no_circuit": {"correct": 0}}), encoding="utf-8")
    other.write_text(json.dumps({"control_no_circuit": {"correct": 7}}), encoding="utf-8")
    assert (
        main(["--left", str(same_a), "--right", str(same_b), "--subtree", "control_no_circuit"])
        == 0
    )
    assert (
        main(["--left", str(same_a), "--right", str(other), "--subtree", "control_no_circuit"]) == 1
    )


def test_two_none_values_are_equality_not_a_schema_difference(tmp_path: Path) -> None:
    #: 实测踩到的 bug：旧写法 `left is None or right is None` 把两边同为 `None` 的字段
    #: （如 `control_no_circuit.circuit`）报成 schema 差异 ⇒ `--strict` 会把两份**完全相同**的件判成不相同。
    both_none = {"circuit": None, "picked_cosine": None, "correct": 0}
    left = tmp_path / "l.json"
    right = tmp_path / "r.json"
    left.write_text(json.dumps(both_none), encoding="utf-8")
    right.write_text(json.dumps(both_none), encoding="utf-8")
    behavior, schema = diff_reports(both_none, dict(both_none))
    assert behavior == [] and schema == [], (behavior, schema)
    assert compare_files(left, right, strict=True)["identical"] is True
    #: 而"一边 None 一边有值"仍必须归进 schema（这条式子两方向都要成立）。
    assert diff_reports({"x": None}, {"x": 1})[1] == [("x", "null_filled_other_side_has_value")]


#: 2026-10-03 实测到的第二件事：同一配置、只差一个仪器版本的两个档，`behavior_diff_count` 是 **2**，
#: 而第一处就是 `format` 本身（v31 对 v32）⇒ "只加字段不改行为"这句话在跨版本比对里**永远证不出来**。
#: 下面三条把这个洞堵上：豁免要有效、要可见、而且**不能吞掉真的行为差**。
def test_disclosure_only_differences_are_ignored_and_still_counted(tmp_path: Path) -> None:
    old = {
        "format": "taiji-a30-stop-failure-v31",
        "started_utc": "2026-10-03T02:51:00",
        "per_item": [{"steps": 756}],
        "control_no_circuit": {"correct": 0},
    }
    new = {
        "format": "taiji-a30-stop-failure-v32",
        "started_utc": "2026-10-03T03:27:00",
        "format_note_v32": "加字段说明",
        "per_item": [{"steps": 756}],
        "control_no_circuit": {"correct": 0},
    }
    left = tmp_path / "old.json"
    right = tmp_path / "new.json"
    left.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    right.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    waived = compare_files(left, right)
    assert waived["identical"] is True, waived
    assert waived["behavior_diff_count"] == 0, waived
    #: 豁免必须可见——否则就是无声放宽门柱
    assert waived["ignored_disclosure_count"] == 3, waived
    assert waived["first_ignored_disclosure"] in (
        "format",
        "format_note_v32",
        "started_utc",
    ), waived
    #: 关掉豁免必须翻回"不相同"，证明这条豁免确实在起作用（不是恒真式）
    strict_old = compare_files(left, right, ignore_disclosure=False)
    assert strict_old["identical"] is False, strict_old
    assert strict_old["behavior_diff_count"] == 2, strict_old


def test_a_real_behavior_change_survives_the_disclosure_exemption(tmp_path: Path) -> None:
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text(json.dumps({"format": "v31", "per_item": [{"steps": 756}]}), encoding="utf-8")
    right.write_text(json.dumps({"format": "v32", "per_item": [{"steps": 757}]}), encoding="utf-8")
    result = compare_files(left, right)
    assert result["identical"] is False, result
    assert result["behavior_diff_count"] == 1, result
    assert "per_item[0].steps" in str(result["first_behavior_diff"]), result
    assert main(["--left", str(left), "--right", str(right)]) == 1, "rc 必须是结论"


def test_cross_version_same_config_pair_reads_as_behaviourally_identical() -> None:
    """真件证据：装机底"挂 seed-A、门关闭"这一配置在 v31 与 v32 两版仪器上跑出的两份件，
    剥掉披露字段后**行为零差异**——这才是 `DECISION-A30` §3 甲要的那种可机检陈述。"""
    v31 = REPORTS / "taiji_a30_stop_failure_self_v31_winner_circuitseedA_20261003.json"
    v32 = REPORTS / "taiji_a30_stop_failure_self_v32_lfcontext_off_20261003.json"
    if not (v31.exists() and v32.exists()):  #: 缺件不静默通过：让它红，我才去查
        raise AssertionError("两份跨版本对照件必须在仓内")
    result = compare_files(v31, v32)
    assert result["identical"] is True, result
    assert result["behavior_diff_count"] == 0, result
    assert result["ignored_disclosure_count"] >= 1, result
