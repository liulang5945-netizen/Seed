"""A30 stop_failure v6 守卫：面比较放到**同一表示层**（深帧复现纠正追加二的误诊）。

诊断更正（§2bb-追加三）：`V019` 首轮的 53 次 observe＝1 告知＋1 边界＋50 prompt＋**1 生成**——
prompt 全部在案，1 字节是生成环的真实产量（边界符胜出后 break，正是 L2 要测的早停成功）；
追加二读到的那张"完整成句"答复是**器官占位句模板**（`_fallback_text`），不是模型输出。
v6 的重放走完整产品面链（decode → marker 切割 → 同一个器官 emit），本文件钉它的三面行为：
极短/不可读 raw ⇒ 占位句（早停不再被记成面违规）、可读 raw ⇒ 原样通过、raw 内含 marker ⇒ 切割。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from probe_taiji_a30_stop_failure import replay_surface_from_fed  # noqa: E402

from taiji.language_organ import NativeReadableTextLanguageOrgan  # noqa: E402

TURN_MARKERS = ("\n问：", "问：")
TURN = "我妹妹最喜欢的数字是1997"


def _replay(fed: list[int]) -> dict:
    return replay_surface_from_fed(
        fed,
        turn=TURN,
        history=[],
        organ=NativeReadableTextLanguageOrgan(),
        turn_markers=TURN_MARKERS,
    )


def test_early_stop_lone_lead_byte_takes_the_organ_fallback() -> None:
    """V019 的形状：模型只吐 1 个 CJK 首字节就撞边界 ⇒ decode 出 U+FFFD ⇒ 器官占位句。

    占位句模板里带 prompt 原文——这正是 v5 把早停记成面违规的原因：拿 1 字节 raw
    去比 44 字占位句，必然不等。v6 重放走同一器官 ⇒ 与产品答复同表示。
    """

    result = _replay([0xE4])
    #: 孤立 CJK 首字节是**不完整序列** ⇒ 产品口径的 `trim_partial_tail` 先裁掉 ⇒ native 为空。
    assert result["native_replay"] == ""
    assert result["replay_surface"] == (
        f"我已收到你的问题：“{TURN}”。当前原生语言表层正在形成稳定表达。"
    )


def test_readable_raw_passes_through_unchanged() -> None:
    result = _replay(list("你好".encode()))
    assert result["replay_surface"] == "你好"


def test_turn_marker_is_cut_before_the_organ() -> None:
    result = _replay(list("你好\n问：下一题".encode()))
    assert result["marker_at"] == 2
    assert result["replay_surface"] == "你好"


def test_empty_generation_takes_the_fallback_too() -> None:
    """fed 为空（第一字节就是边界符、break 在 observe 之前）——同样是早停成功样本。"""

    result = _replay([])
    assert result["native_replay"] == ""
    assert "当前原生语言表层正在形成稳定表达" in result["replay_surface"]


def test_instrument_carries_v6_and_the_correction_note() -> None:
    """格式面：v12 在案（v6–v11 的语义逐条仍在）、更正说明点名追加二的误诊与深帧证据，严格守卫公式未动。

    2026-10-02 该仪器先升 **v7**（加性自述 `surface_gate_state`／`write_back_gate_last_reason`），
    同日再升 **v8**（加性旗标 `--no-copy-evidence-gate` ＋守卫 `evidence_gate_flag_honored`），
    v6 的每一条判据**原样保留在下面**——升版改的是这一行的版本号，不改任何比较公式与计数列。
    """

    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(
        encoding="utf-8"
    )
    assert '"format": "taiji-a30-stop-failure-v12"' in source
    assert all(f"format_note_v{v}" in source for v in (6, 7, 8, 9, 10, 11)), (
        "升版只许加列，历史说明必须逐版留在件里"
    )
    assert "format_note_v6" in source
    assert "追加二" in source and "深帧复现" in source
    assert "all_surfaces_are_replayed_raw" in source
    #: 早停计数更正：eating 按 fed≥预算判、新增边界自停列（追加四）。
    assert '"generations_boundary_self_stop"' in source
    assert 'if check["fed_bytes"] >= args.max_length' in source
    #: v7 的两位自述必须真的在报（回写门槛状态与最近一次回写判决）。
    assert '"surface_gate_state": runtime.surface_gate_state' in source
    #: v8 的旗标与"被走到"守卫（拆"挂载回路"里捆绑的两样东西；行为面另有真实测试）。
    assert '"--no-copy-evidence-gate"' in source
    assert "evidence_gate_flag_honored" in source
    #: v9 的剂量乘数与"被消费"守卫。
    assert '"--copy-evidence-alpha"' in source
    assert "evidence_alpha_consumed" in source
    #: v10 的 sha 级配对：字段与"必须报出来"的守卫都在。
    assert '"checkpoint_sha256": sha_before[:16]' in source
    assert "checkpoint_sha_recorded" in source
    #: v11 的自信度衰减旗标与"被消费"守卫。
    assert '"--relevance-ceiling-c"' in source
    assert "relevance_ceiling_consumed" in source
    assert '"write_back_gate_last_reason"' in source
    #: 旧分类退役：新比较只产生 None／surface_differs_from_replay 两种取值
    #: （历史件里的 replay_tiny_feed 字段仍在，读旧件不受影响）。
    assert 'else "surface_differs_from_replay"' in source

    #: v12（2026-10-02）：**"包装器被走到"与"过滤器开过枪"是两件事**——前者只数调用次数，
    #: 一个从未命中的 c 会给出与全剂量同值的读数却看不见自己是空的。钉三样：更严的守卫、
    #: 每档的静音计数、以及静音计数必须**穿过真正的生成循环**（挂在 priming 上不算）。
    assert (
        '"relevance_ceiling_fired": (args.relevance_ceiling_c is None or loop_silenced[1] > 0)'
        in source
    )
    assert '"relevance_ceiling_silenced_calls": loop_silenced[1]' in source
    assert "scaled, loop_silenced = _observe_silencing(" in source
    #: 环内归因的判据：取调用**之前**最后一条在案帧的 `in_generation_loop`（`records` 每轮清空）。
    assert 'if prev is not None and prev["in_generation_loop"]:' in source
    #: 退役：只证明"被消费"的那一把不得留下——它是 v12 要替换掉的那把松尺子。
    assert '"relevance_ceiling_consumed"' not in source
