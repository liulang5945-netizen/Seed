"""A30 stop_failure v6 守卫：面比较放到**同一表示层**（深帧复现纠正追加二的误诊）。

诊断更正（§2bb-追加三）：`V019` 首轮的 53 次 observe＝1 告知＋1 边界＋50 prompt＋**1 生成**——
prompt 全部在案，1 字节是生成环的真实产量（边界符胜出后 break，正是 L2 要测的早停成功）；
追加二读到的那张"完整成句"答复是**器官占位句模板**（`_fallback_text`），不是模型输出。
v6 的重放走完整产品面链（decode → marker 切割 → 同一个器官 emit），本文件钉它的三面行为：
极短/不可读 raw ⇒ 占位句（早停不再被记成面违规）、可读 raw ⇒ 原样通过、raw 内含 marker ⇒ 切割。
"""

from __future__ import annotations

import re
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
    """格式面：v19 在案（v6–v18 的语义逐条仍在）、更正说明点名追加二的误诊与深帧证据，严格守卫公式未动。

    2026-10-02 该仪器先升 **v7**（加性自述 `surface_gate_state`／`write_back_gate_last_reason`），
    同日再升 **v8**（加性旗标 `--no-copy-evidence-gate` ＋守卫 `evidence_gate_flag_honored`），
    v6 的每一条判据**原样保留在下面**——升版改的是这一行的版本号，不改任何比较公式与计数列。
    """

    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(
        encoding="utf-8"
    )
    assert '"format": "taiji-a30-stop-failure-v21"' in source
    assert all(
        f"format_note_v{v}" in source for v in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    ), "升版只许加列，历史说明必须逐版留在件里"
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

    #: v13（2026-10-02，owner 裁"不立项、先追轨迹面由什么在管"）：内容／硬度分离档必须自带**硬度守恒**这道
    #: 前提守卫——守恒不成立时两个臂差的不止内容，整档作废。钉的是"逐次相对差最大值"这一式，不是注释。
    assert '"content_arm_magnitude_preserved"' in source

    #: v16/v17 的式子级断言（先前一次提交误记为"已加"——那次脚本里写了 `if False`，实为 no-op；
    #: 这次补上并在台账里更正）。钉三件事：
    #: ① 停止位置的**正确刻度**：自停生成的 `fed_bytes`，不得再按"在案行的 boundary_is_argmax"数
    #:    （边界符胜出那一步 `break` 在 `observe` 之前、根本不入案，那个取法结构上恒为 0）。
    #: ② 资格档必须**两侧都出现过**（只发不静音＝没这档；只静音不发＝恒零档）。
    #: ③ 分堆函数与件级聚合必须在场（K 只能落在 2 的幂刻度上，杜绝事后挑刚好过线的 K）。
    assert 'if check["fed_bytes"] < args.max_length' in source
    #: （反面只钉"按 enumerate 数位置"那个取法本身；`boundary_is_argmax` 这个键在
    #: `steps_boundary_is_argmax` 那一列还在用，那是合法的另一种统计，不该被这条断言误伤。）
    assert "for position, row in enumerate(" not in source
    assert '"window_both_sides_seen"' in source
    assert "def _position_histogram(positions: list[int]) -> dict[str, int]:" in source
    assert '"boundary_win_position_hist": _position_histogram(' in source

    #: v18：检索侧资格档必须**被走到**且自述候选集大小；实现只许住在共用 helper 里（又一处"别写第二把尺子"）。
    assert '"store_scope_consumed"' in source
    assert "= _make_store_scoped_arm(" in source
    #: 反面：仪器侧不得自带一份 store 操作（clear/record 只许出现在共用实现里）。
    assert "store.clear()" not in source and "store.record(" not in source

    #: v19：oracle 选择档必须被走到并披露"选对率"（found／fell_through）——
    #: 只有 calls 而没有 found，就分不清"选对了"与"库里根本没有正确事件"。
    assert '"oracle_found": oracle_state["found"]' in source
    assert '"oracle_fell_through": oracle_state["fell_through"]' in source
    #: 反面：oracle 的实现只许住在共用 helper 里，仪器侧不得自带一份 `best_match` 逻辑。
    assert "def oracle_best_match" not in source
    #: v15：内容档**搬到剂量探针里与复述面共用** ⇒ 那两条"式子级"断言跟着搬走（不是删掉）：
    #: 守恒用的是逐次相对差最大值，冻结源必须是"通道真的在发"的那一次。
    dose = (
        PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_copy_evidence_dose.py"
    ).read_text(encoding="utf-8")
    assert "counters[3] = max(counters[3], abs(new - single) / single)" in dose
    assert "if not frozen and single > 0.0:" in dose
    #: 反面：轨迹面仪器里不得再留一份自己的内容档实现（两把尺子的老坑）。
    assert "perm_cache" not in source
    assert 'choices=("permutation", "frozen")' in source
    #: 反面：不得用"两趟累加之差"当守恒判据（n=1 冒烟证明那量的是求和顺序的表示层噪声，1e-4 级）。
    assert "abs(content_guard[1] - content_guard[2])" not in source

    #: v14：**冻结源必须是"通道真的在发"的那一次**——v13 的 `frozen` 冻结到第一条调用，而那时 store 还空，
    #: 于是整档测的是"永久关掉通道"（66/72 对不挂回路的 66/72，`max_rel_l1_diff=1.0` 是指纹）。
    #: 钉的是"非零才算冻结源"这一式，以及冻结点必须被披露。
    assert '"content_arm_frozen_at_call"' in source and '"content_arm_frozen_l1"' in source
    assert '"relevance_ceiling_consumed"' not in source


def test_the_content_arm_has_exactly_one_implementation_across_instruments() -> None:
    """两半必须做**同一个**置换／冻结操作：内容档只许住在剂量探针里，别处只许 import。

    复述面与轨迹面各写一份，就是两把尺子量同一件事——本项目已经为此废过读数。
    """

    shared = (
        PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_copy_evidence_dose.py"
    ).read_text(encoding="utf-8")
    assert shared.count("def _make_content_armed_evidence") == 1

    for name in ("probe_taiji_a30_stop_failure.py", "score_taiji_r2_copy_circuit_chat_cap.py"):
        src = (PROJECT_ROOT / "scripts" / "training" / name).read_text(encoding="utf-8")
        assert "import _make_content_armed_evidence" in src, name
        #: 反面：这两台仪器里不得再出现自己的置换／冻结机械（`index_select` 只在共用实现里有）。
        assert "index_select" not in src, name
        #: 第四十三次停靠的 decoy 档同理：换内容的动作只许住在共用 helper 里。
        assert "replace(base, content=" not in src, name
        assert "rng.shuffle" not in src, name
        #: v19 的 oracle 档同理：`best_match` 的替身只许住在共用 helper 里。
        assert "def oracle_best_match" not in src, name


def test_the_three_a30_instruments_self_describe_checkpoint_sha() -> None:
    #: DEBT-G21（已修，提交 1049b25b）：跨工件配对必须按**内容**钉底座——只记 `checkpoint` 路径会被
    #: 后来的长跑覆盖（cap 仪器自己注释里就写着这条，理由同样适用于回路）。字段退回去 ⇒ 红。
    for name in (
        "probe_taiji_a30_stop_failure.py",
        "score_taiji_r2_copy_circuit_chat_cap.py",
        "score_taiji_r2_copy_surface_extension.py",
    ):
        source = (PROJECT_ROOT / "scripts" / "training" / name).read_text(encoding="utf-8")
        assert '"checkpoint_sha256":' in source, name


def test_recall_numbers_after_docking50_name_the_circuit() -> None:
    #: §第五十/五十一 那两格立的规矩：**「两全」必须带枚数说**——同一个 D 读数在 seed-A 上是 6→7、
    #: 在 seed-B 上是 3→3，所以不点名回路的 `N/16` 句子会造出假事实。本文件从"第五十次停靠"起逐行扫，
    #: 任何含 `N/16` 的行必须同时出现 seed-A／seed-B／circuit／枚 之一。写这条时它先抓到了我自己两行。
    plan = (
        PROJECT_ROOT
        / "plans"
        / "reference"
        / "PLAN-A-30_surface_repetition_localization_20260928.md"
    ).read_text(encoding="utf-8")
    lines = plan.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("### 第五十")]
    assert starts, "docking 50 heading missing"
    section = lines[min(starts):]
    offenders = [
        line
        for line in section
        if re.search(r"\d+/16", line)
        and not any(token in line for token in ("seed-A", "seed-B", "circuit", "枚"))
    ]
    assert not offenders, offenders[:3]


def test_the_product_gate_arm_is_distinct_from_the_instrument_stand_in() -> None:
    #: v20（owner 2026-10-02 裁「立项进产品」）：验收面必须走**产品原生门**，
    #: 不能拿本仪器 v17 那副 monkeypatch 替身当代答——替身证的是"这样修有用"，不是"产品里就是这么修的"。
    #: 三样钉住：旗标存在且调产品 setter、两档互斥、门开过枪要自证（`steps_seen=0` ⇒ 假档）。
    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(encoding="utf-8")
    assert '"--product-window-steps"' in source
    assert "substrate.set_copy_evidence_window_steps(args.product_window_steps)" in source
    assert "产品门与仪器替身档不能同开" in source
    assert '"product_window_fired"' in source
    #: 产品侧的门：默认必须关闭（None），否则就不是"逐位不变"的立项形状。
    from taiji.config import TaijiConfig

    assert TaijiConfig().copy_evidence_window_steps is None
    #: 门的"前 K 步发、之后静音"必须真的分两堆计数——只报 emitted 会看不见它从未静音过。
    model_source = (PROJECT_ROOT / "taiji" / "model.py").read_text(encoding="utf-8")
    assert "within_window = self._copy_evidence_step < window" in model_source
    assert '"silenced_steps": self._copy_evidence_window_silenced' in model_source
    assert "reset_copy_evidence_window()" in model_source


def test_the_stand_alone_loop_caller_must_reset_the_product_window() -> None:
    #: 2026-10-02 实测踩到：L2 仪器自己驱动生成环，产品门在 `generate()` 里的复位对它不生效，
    #: 于是 K=64 变成"整批只发前 64 步"，读数与不挂回路同值却被当成增益。
    #: 钉住调用方必须显式复位（且只在产品档下）。
    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(encoding="utf-8")
    assert "substrate.reset_copy_evidence_window()" in source
    assert source.count("if args.product_window_steps is not None:") >= 2


def test_the_product_window_resets_after_the_prompt_feed_not_before_it() -> None:
    #: DEBT-G22 的静态那半：答复相计步的前提是"prompt 喂完就复位"。
    #: 复位早于 prompt 环 ⇒ K 被 prompt 吃掉（实测过），所以这里钉**次序**而不只是钉存在。
    model_source = (PROJECT_ROOT / "taiji" / "model.py").read_text(encoding="utf-8")
    feed = model_source.index("        for symbol in prompt:")
    first_reset = model_source.index("        self.reset_copy_evidence_window()", feed)
    answer_loop = model_source.index("            for _ in range(length):", first_reset)
    assert feed < first_reset < answer_loop


def test_every_a30_face_discloses_which_window_path_produced_it() -> None:
    #: DEBT-G23：表层那两件产品档读数当时**件里说不出自己来自哪条路径**（只有 cap 自述了旗标），
    #: 而同一个信封里 `window_arm` 在替身档未开时恒为 0 ⇒ 光看件分不开"产品门"与"替身档"。
    #: 钉三台仪器都必须自述旗标与被走到计数。
    holders = {
        "probe_taiji_a30_stop_failure.py": ('"product_window_steps": args.product_window_steps',
                                            '"product_window_stats"'),
        "score_taiji_r2_copy_circuit_chat_cap.py": ('"product_window_steps": args.product_window_steps',
                                                    '"product_window_stats": product_window_stats'),
        "score_taiji_r2_copy_surface_extension.py": ('"product_window_steps": product_window_steps',
                                                     '"product_window_stats": product_window_stats'),
    }
    for name, needles in holders.items():
        source = (PROJECT_ROOT / "scripts" / "training" / name).read_text(encoding="utf-8")
        for needle in needles:
            assert needle and needle in source, (name, needle)
