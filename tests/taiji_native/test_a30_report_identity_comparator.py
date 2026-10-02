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
    result = compare_files(SEEDB_ANCHOR, SEEDB_WINDOW, subtree='control_no_circuit')
    assert result['identical'] is True, result
    assert result['behavior_diff_count'] == 0, result


def test_different_config_arms_are_detected_as_different() -> None:
    """守卫必须能为假：全剂量臂与窗口 K=64 臂在同一面上行为不同。"""
    result = compare_files(CAP_FULL, CAP_WINDOW, subtree='treated_with_circuit')
    assert result['identical'] is False, result
    assert result['behavior_diff_count'] >= 1, result
    paths = [str(result['first_behavior_diff'])[0:60]]
    assert paths, result


def test_later_added_disclosure_field_is_schema_not_behavior(tmp_path: Path) -> None:
    """后加的披露字段（旧件为 `None`／新件有值，或一侧缺键）必须归到 schema 那一堆。"""
    old = {'correct': 6, 'answer_bytes_quantiles': None, 'format': 'v1'}
    new = {'correct': 6, 'answer_bytes_quantiles': {'median': 12}, 'format': 'v1', 'prereg': 'x'}
    behavior, schema = diff_reports(old, new)
    assert behavior == [], behavior
    kinds = {path for path, _ in schema}
    assert 'answer_bytes_quantiles' in kinds, schema
    assert 'prereg' in kinds, schema
    # 同样的两份东西，strict 一开必须翻成"不相同"——否则 --strict 是个装饰性开关。
    left = tmp_path / 'old.json'
    right = tmp_path / 'new.json'
    left.write_text(json.dumps(old, ensure_ascii=False), encoding='utf-8')
    right.write_text(json.dumps(new, ensure_ascii=False), encoding='utf-8')
    assert compare_files(left, right)['identical'] is True
    assert compare_files(left, right, strict=True)['identical'] is False


def test_cli_exit_code_is_the_verdict(tmp_path: Path) -> None:
    """rc 就是结论：相同 ⇒ 0，不同 ⇒ 1（验收要能直接挂门，不能靠人读 JSON）。"""
    same_a = tmp_path / 'a.json'
    same_b = tmp_path / 'b.json'
    other = tmp_path / 'c.json'
    same_a.write_text(json.dumps({'control_no_circuit': {'correct': 0}}), encoding='utf-8')
    same_b.write_text(json.dumps({'control_no_circuit': {'correct': 0}}), encoding='utf-8')
    other.write_text(json.dumps({'control_no_circuit': {'correct': 7}}), encoding='utf-8')
    assert main(['--left', str(same_a), '--right', str(same_b), '--subtree', 'control_no_circuit']) == 0
    assert main(['--left', str(same_a), '--right', str(other), '--subtree', 'control_no_circuit']) == 1
