"""A30 stop_failure v6 守卫：面比较放到**同一表示层**（深帧复现纠正追加二的误诊）。

诊断更正（§2bb-追加三）：`V019` 首轮的 53 次 observe＝1 告知＋1 边界＋50 prompt＋**1 生成**——
prompt 全部在案，1 字节是生成环的真实产量（边界符胜出后 break，正是 L2 要测的早停成功）；
追加二读到的那张"完整成句"答复是**器官占位句模板**（`_fallback_text`），不是模型输出。
v6 的重放走完整产品面链（decode → marker 切割 → 同一个器官 emit），本文件钉它的三面行为：
极短/不可读 raw ⇒ 占位句（早停不再被记成面违规）、可读 raw ⇒ 原样通过、raw 内含 marker ⇒ 切割。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

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
    assert '"format": "taiji-a30-stop-failure-v40"' in source
    assert all(
        f"format_note_v{v}" in source for v in range(6, 41)
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
    #: v38（DEBT-G33）更正：这一列在**没开上限**的档里必须报 `null`——v37 及以前报 0，
    #: 与"开了上限但一次都没静音"在件里同形，而这两种读数的下一步动作完全不同。
    assert '"relevance_ceiling_silenced_calls": (' in source
    assert "loop_silenced[1] if args.relevance_ceiling_c is not None else None" in source
    assert "scaled, loop_silenced = _observe_silencing(" in source
    #: 环内归因的判据：取调用**之前**最后一条在案帧的 `in_generation_loop`（`records` 每轮清空）。
    assert 'if prev is not None and prev["in_generation_loop"]:' in source
    #: v38（DEBT-G33）：**环内调用计数不许再依赖 α／上限旗标**，且"没在测"必须写成 `null` 而不是 0。
    #: 钉的是"观察者安装条件＝回路在场"＋旗标列＋缺观察者时 `null`，并反向钉住那个会自我欺骗的旧写法。
    assert "evidence_observer_installed = substrate.copy_circuit is not None" in source
    assert 'if evidence_observer_installed:\n        scaled, loop_silenced = _observe_silencing(' in source
    assert '"evidence_observer_installed": evidence_observer_installed,' in source
    assert "loop_silenced[0] if evidence_observer_installed else None" in source
    assert '"evidence_calls_in_generation_loop": loop_silenced[0],' not in source
    assert '"relevance_ceiling_silenced_calls_all_chains": (' in source
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
        if re.search(r"(?<!\d)\d+/16(?!\d)(?!/)", line)
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
                                                     '"product_window_stats": substrate.copy_evidence_window_stats()'),
    }
    for name, needles in holders.items():
        source = (PROJECT_ROOT / "scripts" / "training" / name).read_text(encoding="utf-8")
        for needle in needles:
            assert needle and needle in source, (name, needle)


def test_the_window_counters_are_read_after_the_run_not_before_it() -> None:
    #: 实测踩过的披露陷阱：把 `copy_evidence_window_stats()` 取在生成之前 ⇒ 件里永远是全零快照，
    #: 而全零恰好会"证明门没开过枪"——一个会把成功读数说成空档的自证。两台仪器都必须在返回时才取。
    import ast as _ast
    for name, needle in (
        ("score_taiji_r2_copy_surface_extension.py", "copy_evidence_window_stats"),
        ("score_taiji_r2_copy_circuit_chat_cap.py", "copy_evidence_window_stats"),
    ):
        source = (PROJECT_ROOT / "scripts" / "training" / name).read_text(encoding="utf-8")
        tree = _ast.parse(source)
        fn = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef) and n.name == "run_arm")
        lines = [n.lineno for n in _ast.walk(fn) if isinstance(n, _ast.Call)
                 and isinstance(n.func, _ast.Attribute) and n.func.attr == needle]
        assert lines, name
        starts = [x.lineno for x in _ast.walk(fn) if isinstance(x, _ast.Return)]
        assert starts, name
        # 计数必须在**返回那一刻或更晚**取（离 return 起始行不超过 3 行），否则就是跑前快照。
        assert max(lines) >= min(starts) - 3, (name, lines, min(starts))


def test_the_endstep_probe_groups_by_generation_not_by_item() -> None:
    """§59 的四标量必须**按生成**算——按 item 串接算就退回"跨代混算的中位数"那把没判别力的尺子。

    用合成行钉两件事：① `step` 归零即换代；② `in_run` 为假的行不参与。
    """
    from probe_taiji_a30_stop_failure import (
        _endstep_probe_per_generation,
        _group_rows_by_generation,
    )

    def row(step, p, rank, in_run=True, byte=65):
        return {"step": step, "p_boundary": p, "boundary_rank_in_legal": rank,
                "legal_candidates": 40, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 3.0, "in_run": in_run,
                "emitted_byte": byte, "utf8_byte_class": "ascii", "utf8_state_before": [0, 0]}

    rows = [row(0, 0.1, 9), row(1, 0.4, 3), row(2, 0.2, 7), row(0, 0.9, 1), row(1, 0.5, 2), row(3, 0.7, 5, in_run=False)]
    groups = _group_rows_by_generation(rows)
    #: v23：全部行都参与换代分组（v22 误把 `in_run` 当"在生成内"用 ⇒ 72 代只剩 12／0）。
    assert [len(g) for g in groups] == [3, 3], groups
    probe = _endstep_probe_per_generation(rows, max_length=256, terminals=[None, None])
    assert len(probe) == 2, probe
    assert probe[0]["p_boundary_max"] == 0.4 and probe[0]["p_boundary_argmax_step"] == 1, probe[0]
    assert probe[0]["steps_in_repeat_run"] == 3 and probe[1]["steps_in_repeat_run"] == 2, probe
    assert probe[0]["boundary_rank_at_peak_step"] == 3, probe[0]
    assert probe[0]["peak_is_last_step"] is False, probe[0]      # 峰值在 step 1，该代最后一步是 step 2 ⇒ 看到了还在走
    assert probe[1]["p_boundary_max"] == 0.9 and probe[1]["boundary_rank_at_peak_step"] == 1, probe[1]
    assert probe[1]["peak_is_last_step"] is False, probe[1]
    assert all(entry["ate_full_budget"] is False for entry in probe), probe
    #: 第三代的真值样例：峰值落在**最后一个在案步**。⚠ §六十九收回的那句话正是把这件事读成
    #: "停在峰值上"——产品环体 `break` 在 `observe` 之前，最后一个在案步是**停下前一步**，
    #: 真正的终止决策只在 v27 的 `terminal_decision` 里（下面那条守卫钉住）。
    last_step_peak = _endstep_probe_per_generation(
        [row(0, .1, 9), row(1, .3, 4), row(2, .8, 1)], max_length=256, terminals=[None]
    )
    assert last_step_peak[0]["peak_is_last_step"] is True, last_step_peak
    assert last_step_peak[0]["boundary_rank_at_peak_step"] == 1, last_step_peak
    assert last_step_peak[0]["terminal_decision"] is None, last_step_peak   # 没补终止行 ⇒ 这一代"停在哪儿"不可答
    # 吃满预算的判据：末步 +1 >= max_length（不依赖外部真值）
    long_run = _endstep_probe_per_generation(
        [row(0, .1, 9), row(1, .2, 8)], max_length=2, terminals=[None]
    )
    assert long_run[0]["ate_full_budget"] is True, long_run


def test_the_fixed_step_probe_records_only_reached_steps() -> None:
    """§63 的固定步位：只记"走得到的"步，且必须带上分母可核的三个量。"""
    from probe_taiji_a30_stop_failure import _FIXED_STEPS, _endstep_probe_per_generation

    def row(step, p, rank, legal=40):
        return {"step": step, "p_boundary": p, "boundary_rank_in_legal": rank,
                "legal_candidates": legal, "legal_candidates_including_boundary": legal + 1,
                "ratio_best_over_boundary": 4.0, "in_run": True,
                "emitted_byte": 97, "utf8_byte_class": "ascii", "utf8_state_before": [0, 0]}

    short = _endstep_probe_per_generation(
        [row(s, 0.01, 9) for s in range(10)], max_length=256, terminals=[None]
    )
    assert [entry["step"] for entry in short[0]["at_steps"]] == [8], short[0]["at_steps"]
    long = _endstep_probe_per_generation(
        [row(s, 0.02, 7) for s in range(200)], max_length=256, terminals=[None]
    )
    assert [entry["step"] for entry in long[0]["at_steps"]] == list(_FIXED_STEPS), long[0]["at_steps"]
    assert all({"p_boundary", "boundary_rank_in_legal", "legal_candidates",
                "legal_candidates_including_boundary"} <= set(entry) for entry in long[0]["at_steps"])
    #: DEBT-G24：名次必须被**含边界符**那一列界定
    assert all(entry["boundary_rank_in_legal"] <= entry["legal_candidates_including_boundary"]
               for entry in long[0]["at_steps"]), long[0]["at_steps"]


def test_boundary_rank_is_bounded_by_the_denominator_it_belongs_to() -> None:
    """DEBT-G24：`legal_candidates` 少算边界符，名次可比它大 1 ⇒ 钉住两条口径不等式。

    旧列不动（历史件同格可比），但从此任何"名次／候选数"的比值都必须用**含边界符**那一列。
    """
    import torch
    from probe_taiji_a30_stop_failure import replay_step

    for legal_size in (2, 5, 40):
        probs = torch.zeros(257)
        boundary, emitted = 256, 65
        probs[boundary] = 0.01
        for index in range(legal_size):
            probs[100 + index] = 0.5 + 0.001 * index
        probs[emitted] = 0.5005
        row = replay_step(probs, emitted, boundary, (0, 0), b"", 0.0, 8)
        assert row["boundary_rank_in_legal"] <= row["legal_candidates_including_boundary"], row
        assert row["boundary_rank_in_legal"] <= row["legal_candidates"] + 1, row
        assert row["legal_candidates_including_boundary"] == row["legal_candidates"] + 1, row


def test_window_stats_report_cumulative_and_last_turn_separately() -> None:
    """DEBT-G25：末趟计数会把成功的档自我否证（一趟只有 43 步、K=128 时不可能静音过）。

    所以件内必须**同时**有全程累计与末趟两组，且累计只增不清。这里在进程内测两件事：
    门控在"跨两趟答复"后累计 silenced>0，而 `reset_copy_evidence_window()` 只清末趟。
    """
    model_src = (PROJECT_ROOT / "taiji" / "model.py").read_text(encoding="utf-8")
    assert "_copy_evidence_window_emitted_total" in model_src
    assert '"emitted_steps_total"' in model_src and '"silenced_steps_total"' in model_src
    reset_body = model_src.split("def reset_copy_evidence_window(self) -> None:", 1)[1].split("    def ", 1)[0]
    assert "_copy_evidence_step = 0" in reset_body
    assert "_total = 0" not in reset_body          # 累计量绝不在复位里被清零


def _synthetic_row(step: int, p: float, rank: int, legal: int = 40, byte: int = 0x80) -> dict:
    return {
        "step": step,
        "p_boundary": p,
        "boundary_rank_in_legal": rank,
        "legal_candidates": legal,
        "legal_candidates_including_boundary": legal + 1,
        "ratio_best_over_boundary": 5.0,
        "in_run": False,
        "boundary_is_argmax": rank == 1,
        "emitted_byte": byte,
        "utf8_byte_class": _class_of(byte),
        "utf8_state_before": [0, 0],
    }


def _class_of(byte: int) -> str:
    if byte < 0x80:
        return "ascii"
    if byte < 0xC0:
        return "continuation"
    if byte < 0xE0:
        return "lead2"
    if byte < 0xF0:
        return "lead3"
    return "lead4"


def test_the_terminal_row_is_the_one_step_the_generation_loop_hides() -> None:
    """DEBT-G26：`break` 在 `observe` 之前 ⇒ 停下那一步过去**没有行**，v27 必须补出来。

    进程内钉四件事（全部两向：能在成功样本为真，也能为假）：
    ① 只有边界自停的代有 `terminal_decision`，吃满预算的代给出 `absent_reason`；
    ② 终止步＝最后一个在案步的**下一步**（这正是过去不可见的那一格）；
    ③ `terminal_over_recorded_peak` 按在案峰值归一；
    ④ `terminals` 与分组数量不一致必须抛错（静默错配比抛错更难查）。
    """
    from probe_taiji_a30_stop_failure import _endstep_probe_per_generation

    stop_rows = [_synthetic_row(0, 0.001, 20), _synthetic_row(1, 0.004, 12)]
    eat_rows = [_synthetic_row(0, 0.0005, 60), _synthetic_row(1, 0.0006, 61)]
    terminal = {"step": 2, "p_boundary": 0.02, "p_boundary_before_penalty": 0.02,
                "boundary_rank_in_legal": 1, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 1.0, "boundary_is_argmax": True,
                "utf8_state_before": [0, 0]}
    rows = stop_rows + eat_rows
    probe = _endstep_probe_per_generation(rows, max_length=2, terminals=[terminal, None])
    assert probe[0]["terminal_decision"] is not None, probe[0]
    assert probe[0]["terminal_decision"]["step"] == probe[0]["last_recorded_row"]["step"] + 1, probe
    assert probe[0]["terminal_decision"]["terminal_over_recorded_peak"] == 5.0, probe[0]
    assert probe[0]["terminal_decision_absent_reason"] is None, probe[0]
    assert probe[1]["terminal_decision"] is None, probe[1]
    assert probe[1]["terminal_decision_absent_reason"] == "ate_full_budget", probe[1]
    #: `last_recorded_row` 就是"最后一个在案步"——它**不是**停下那一步（§六十九收回的那条误读）。
    assert probe[0]["last_recorded_row"]["boundary_rank_in_legal"] == 12, probe[0]
    with pytest.raises(RuntimeError, match="终止决策行与生成分组数量不一致"):
        _endstep_probe_per_generation(rows, max_length=2, terminals=[terminal])


def test_the_terminal_summary_can_report_a_face_violation_and_a_bad_pairing() -> None:
    """件级汇总的两个自检都必须**能为假**：名次非 1＝重放与产品环分岔；配不上数＝分组错。"""
    from probe_taiji_a30_stop_failure import _terminal_summary_v27

    def generation(terminal_p, rank, recorded_peak, last_rank, ate):
        return {
            "generation_steps": 2,
            "last_step": 1,
            "p_boundary_max": recorded_peak,
            "ate_full_budget": ate,
            "last_recorded_row": {"step": 1, "boundary_rank_in_legal": last_rank,
                                  "p_boundary": 0.001, "legal_candidates_including_boundary": 41,
                                  "ratio_best_over_boundary": 5.0},
            "terminal_decision": (
                None
                if terminal_p is None
                else {"step": 2, "p_boundary": terminal_p, "p_boundary_before_penalty": terminal_p,
                      "boundary_rank_in_legal": rank, "legal_candidates_including_boundary": 41,
                      "ratio_best_over_boundary": 1.0, "boundary_is_argmax": rank == 1,
                      "terminal_over_recorded_peak": round(terminal_p / recorded_peak, 4)}
            ),
            "terminal_decision_absent_reason": None if terminal_p is not None else "ate_full_budget",
        }

    good = [generation(0.02, 1, 0.004, 12, False), generation(None, 1, 0.001, 60, True)]
    per_item = [{"endstep_probe_v22": good, "generations_boundary_self_stop": 1,
                 "generations_eating_full_budget": 1}]
    summary = _terminal_summary_v27(per_item)
    assert summary["pairing_ok"] is True, summary
    assert summary["terminal_rank_not_one_count"] == 0, summary
    assert summary["terminal_above_recorded_peak_count"] == 1, summary
    assert summary["terminal_above_recorded_peak_share"] == 1.0, summary
    assert summary["last_recorded_median_rank_for_eaters"] == 60, summary

    #: 能为假①：终止名次不是 1 ⇒ 必须数出来（那说明重放口径与产品环不是一条）
    bad_rank = [generation(0.02, 4, 0.004, 12, False)]
    flipped = _terminal_summary_v27([{"endstep_probe_v22": bad_rank,
                                      "generations_boundary_self_stop": 1,
                                      "generations_eating_full_budget": 0}])
    assert flipped["terminal_rank_not_one_count"] == 1, flipped
    assert flipped["terminal_rank_values_seen"] == [4], flipped
    #: 能为假②：件内点数与分组对不上 ⇒ pairing_ok 必须为假，不能默认成立
    misaligned = _terminal_summary_v27([{"endstep_probe_v22": good,
                                         "generations_boundary_self_stop": 3,
                                         "generations_eating_full_budget": 0}])
    assert misaligned["pairing_ok"] is False, misaligned


def test_the_terminal_row_stays_out_of_the_per_step_aggregates() -> None:
    """补行**只加字段**：终止行不许混进 `item_rows`，否则既有列（名次中位、argmax 步数）语义会漂。"""
    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(
        encoding="utf-8"
    )
    assert source.count("item_rows.append(") == 1, "逐步聚合列只许有一个写入点"
    assert "terminal_rows.append(" in source
    assert 'item_rows.append(terminal' not in source
    assert '"steps_boundary_is_argmax": sum(1 for row in item_rows if row["boundary_is_argmax"])' in source
    assert '"terminal_decision_summary_v27": _terminal_summary_v27(per_item)' in source


def test_the_stop_step_is_argmax_by_construction_so_rank_one_is_a_check_not_an_assumption() -> None:
    """§六十九能成立的那半：终止那一步**按构造**是掩码后合法集的第一名 ⇒ `rank != 1` 只能是仪器分岔。

    次序照 `Taiji.generate` 环体：`argmax()` → 边界即 `break` → 才 `observe`；
    产品链两个旗标都在（`stop_at_boundary=True`／`utf8_strict=True`），否则这句话没有依据。
    """
    model_src = (PROJECT_ROOT / "taiji" / "model.py").read_text(encoding="utf-8")
    body = model_src.split("def generate(", 1)[1].split("\n    @staticmethod", 1)[0]
    argmax_at = body.index("next_symbol = int(probabilities.argmax().item())")
    break_at = body.index("if next_symbol == self.config.boundary_symbol and stop_at_boundary:")
    observe_at = body.index("step = self.observe(", break_at)
    assert argmax_at < break_at < observe_at, "环体次序变了，§六十九的推理要重读"
    runtime_src = (PROJECT_ROOT / "api" / "seed_runtime.py").read_text(encoding="utf-8")
    assert "stop_at_boundary=True" in runtime_src
    assert "utf8_strict=True" in runtime_src


def test_the_tail_identity_records_what_was_emitted_right_before_the_stop() -> None:
    """§72 的 v28：每代必须带"停之前那一步发了哪个字节、当时在字的哪一段"，且分布按组各算分母。

    含两向：① 缺 `emitted_byte` 的旧形状行必须**抛错**（不静默交空 ⇒ §72 资格前置②的来路）；
    ② 位置类占比与"自停减吃满"的差都能为负（这里就故意排成负数，防止有人把方向写反）。
    """
    from probe_taiji_a30_stop_failure import (
        _endstep_probe_per_generation,
        _tail_identity_summary_v28,
    )

    def terminal(step_value: int):
        return {"step": step_value, "p_boundary": 0.03, "p_boundary_before_penalty": 0.03,
                "boundary_rank_in_legal": 1, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 1.0, "boundary_is_argmax": True,
                "utf8_state_before": [0, 0]}

    rows, terminals = [], []
    for byte in (0xE4, 0xE4, 0x41):                            # 自停三代：两个 lead3、一个 ascii
        rows.extend([_synthetic_row(0, 0.001, 9, byte=byte), _synthetic_row(1, 0.002, 7, byte=byte)])
        terminals.append(terminal(2))
    for byte in (0x80, 0x80, 0x80):                            # 吃满三代：全是续字节
        rows.extend(
            [_synthetic_row(0, 0.001, 30, byte=byte), _synthetic_row(1, 0.002, 31, byte=byte),
             _synthetic_row(2, 0.003, 32, byte=byte)]
        )
        terminals.append(None)
    #: 预算取 3：自停的代只有 2 行（没吃满），吃满的代有 3 行——**上一版我在这里写错**
    #: （预算 2 让"自停"代同时满足 `ate_full_budget` ⇒ 两组重叠、`top1_share` 被算成 0.5），
    #: 所以那两条健康度计数就是用来当场抓住这种重叠的。
    probes = _endstep_probe_per_generation(rows, max_length=3, terminals=terminals)
    assert [p["tail_identity"]["prev_step_utf8_byte_class"] for p in probes] == (
        ["lead3", "lead3", "ascii"] + ["continuation"] * 3
    ), [p["tail_identity"] for p in probes]
    assert probes[0]["tail_identity"]["prev_step_emitted_byte"] == 0xE4, probes[0]
    assert [p["ate_full_budget"] for p in probes] == [False] * 3 + [True] * 3, probes

    per_item = [{"endstep_probe_v22": probes, "generations_boundary_self_stop": 3,
                 "generations_eating_full_budget": 3}]
    summary = _tail_identity_summary_v28(per_item)
    assert summary["tail_identity_present_for_all_generations"] is True, summary
    assert summary["stoppers_that_also_ate_full_budget"] == 0, summary
    assert summary["generations_neither_stop_nor_eater"] == 0, summary
    assert summary["stoppers_prev_step_byte_classes"]["top1_share"] == 0.6667, summary
    assert summary["eaters_prev_step_byte_classes"]["top1_share"] == 1.0, summary
    #: 差为负 ⇒ 方向没写反（自停组更分散时它就该是负的）
    assert summary["class_share_stoppers_minus_eaters"] == -0.3333, summary
    assert summary["stoppers_prev_step_bytes"]["distinct"] == 2, summary
    assert summary["eaters_prev_step_bytes"]["n"] == 3, summary

    stale = dict(_synthetic_row(0, 0.001, 9))
    stale.pop("emitted_byte")
    with pytest.raises(RuntimeError, match="emitted_byte"):
        _endstep_probe_per_generation([stale], max_length=256, terminals=[None])


def test_the_v28_tail_identity_is_pinned_at_the_row_and_in_the_envelope() -> None:
    """字面身份要在**写行的那一处**落地，件里必须有对应的汇总键（否则只是我又写了个故事）。"""
    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(
        encoding="utf-8"
    )
    assert 'row["emitted_byte"] = byte' in source
    assert 'row["utf8_byte_class"] = _utf8_byte_class(byte)' in source
    assert '"tail_identity_summary_v28": _tail_identity_summary_v28(per_item)' in source
    assert source.count("def _utf8_byte_class(") == 1


def test_the_lf_followthrough_summary_separates_no_lf_from_lf_not_captured() -> None:
    """§73 末段冻线要的那把尺：**发了 LF 却没停** 与 **整条没发 LF** 是两种修法方向，必须分得开。

    两向都测：① "停紧跟在 LF 后"的占比能小于 1（不是把 118/118 那种巧合写成恒真）；
    ② 拖写组的 `with_lf_share` 由分母现算，且末 LF 的位置堆按"离该代末尾多远"归堆。
    """
    from probe_taiji_a30_stop_failure import (
        _endstep_probe_per_generation,
        _lf_followthrough_summary_v29,
    )

    def gen(bytes_of_generation):
        return [
            _synthetic_row(step, 0.001 + 0.0001 * step, 20 + step, byte=byte)
            for step, byte in enumerate(bytes_of_generation)
        ]

    def terminal_at(step_value: int):
        return {"step": step_value, "p_boundary": 0.03, "p_boundary_before_penalty": 0.03,
                "boundary_rank_in_legal": 1, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 1.0, "boundary_is_argmax": True,
                "utf8_state_before": [0, 0]}

    rows = gen((0x41, 0x42, 10)) + gen((10, 0x42, 0x43)) + gen((10, 0x42, 0x43, 0x44)) + gen(
        (0x41, 0x42, 0x43, 0x44)
    )
    probes = _endstep_probe_per_generation(
        rows, max_length=4, terminals=[terminal_at(3), terminal_at(3), None, None]
    )
    assert [p["lf_trace_v29"]["lf_step_count"] for p in probes] == [1, 1, 1, 0], probes
    per_item = [{"endstep_probe_v22": probes, "generations_boundary_self_stop": 2,
                 "generations_eating_full_budget": 2}]
    summary = _lf_followthrough_summary_v29(per_item)
    assert summary["stoppers"]["n"] == 2 and summary["stoppers"]["with_lf_share"] == 1.0, summary
    #: 只有一代是"停紧跟 LF"（另一代的 LF 在第 0 步）⇒ 这一支能为假，不是恒真
    assert summary["stoppers_last_lf_is_last_step"] == 1, summary
    assert summary["stoppers_last_lf_is_last_step_share"] == 0.5, summary
    assert summary["eaters"]["n"] == 2 and summary["eaters"]["with_lf"] == 1, summary
    assert summary["eaters"]["with_lf_share"] == 0.5, summary
    assert summary["eaters_last_lf_position_buckets"] == {"far-from-end(>25%)": 1}, summary


def test_the_lf_next_probe_measures_leverage_and_refuses_small_denominators() -> None:
    """§75 的 v30：LF 之后那一步的边界名次——自停侧必须恒 1，拖写侧才算杠杆；分母 <5 不许判。

    两向：① 自停的代里『LF 的下一步』是**终止行**（不是下一个在案步），取错就恒假不了；
    ② `denominator_at_least_5` 必须随分母翻转（2 枚 ⇒ False、5 枚 ⇒ True），
       这条开关是 §74 那个丙的成因（分母太小）被写进仪器里的形状。
    """
    from probe_taiji_a30_stop_failure import (
        _endstep_probe_per_generation,
        _lf_next_summary_v30,
    )

    def terminal_at(step_value: int, rank: int = 1):
        return {"step": step_value, "p_boundary": 0.3, "p_boundary_before_penalty": 0.3,
                "boundary_rank_in_legal": rank, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 1.0, "boundary_is_argmax": rank == 1,
                "utf8_state_before": [0, 0]}

    stopper = [_synthetic_row(0, 0.001, 40, byte=0x41), _synthetic_row(1, 0.002, 39, byte=0x42),
               _synthetic_row(2, 0.003, 38, byte=10)]
    deep_eater = [_synthetic_row(s, 0.001, 20 if s == 1 else 44, byte=(10 if s == 0 else 0x42))
                  for s in range(4)]
    near_eater = [_synthetic_row(s, 0.001, 2 if s == 1 else 44, byte=(10 if s == 0 else 0x42))
                  for s in range(4)]
    probes = _endstep_probe_per_generation(
        stopper + deep_eater + near_eater,
        max_length=4,
        terminals=[terminal_at(3), None, None],
    )
    assert probes[0]["lf_next_probe_v30"]["best_boundary_rank_after_lf"] == 1, probes[0]
    assert probes[1]["lf_next_probe_v30"]["best_boundary_rank_after_lf"] == 20, probes[1]
    assert probes[2]["lf_next_probe_v30"]["best_boundary_rank_after_lf"] == 2, probes[2]
    summary = _lf_next_summary_v30(
        [{"endstep_probe_v22": probes, "generations_boundary_self_stop": 1,
          "generations_eating_full_budget": 2}]
    )
    assert summary["stoppers_all_rank_one"] is True, summary
    assert summary["stoppers_rank_values_seen"] == [1], summary
    assert summary["eaters_with_lf_and_next_step"] == 2, summary
    assert summary["denominator_at_least_5"] is False, summary
    assert summary["eaters_share_rank_le_3"] == 0.5, summary
    assert summary["eaters_share_rank_ge_10"] == 0.5, summary

    five = probes[1:] + probes[1:2] * 3
    flipped = _lf_next_summary_v30(
        [{"endstep_probe_v22": five, "generations_boundary_self_stop": 0,
          "generations_eating_full_budget": 5}]
    )
    assert flipped["eaters_with_lf_and_next_step"] == 5, flipped
    assert flipped["denominator_at_least_5"] is True, flipped
    assert flipped["stoppers_all_rank_one"] is False, flipped


def test_the_lf_competitor_summary_ignores_the_terminal_row_and_counts_the_winner() -> None:
    """§76 末段的 v31：**LF 之后那一步是谁压住边界**——胜出字节分布只数拖写侧，终止行记 boundary 不掺进去。

    两向：① 自停的代其"LF 下一步"是终止行 ⇒ 它**不进**拖写分布（否则 top1_share 会被 118 个 boundary 顶成 1.0，
    那就是我差点写出来的那种假结论）；② 拖写侧同字节时 top1_share 必须为 1.0、distinct 为 1。
    """
    from probe_taiji_a30_stop_failure import (
        _endstep_probe_per_generation,
        _lf_competitor_summary_v31,
        _winner_label,
    )

    assert _winner_label({"emitted_byte": 65}) == 65
    assert _winner_label({"step": 3}) == "boundary"
    assert _winner_label(None) is None

    def terminal_at(step_value: int):
        return {"step": step_value, "p_boundary": 0.3, "p_boundary_before_penalty": 0.3,
                "boundary_rank_in_legal": 1, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 1.0, "boundary_is_argmax": True,
                "utf8_state_before": [0, 0]}

    stopper = [_synthetic_row(0, 0.001, 40, byte=0x41), _synthetic_row(1, 0.002, 39, byte=0x42),
               _synthetic_row(2, 0.003, 38, byte=10)]
    eater_a = [_synthetic_row(0, 0.001, 44, byte=10), _synthetic_row(1, 0.002, 2, byte=0x41),
               _synthetic_row(2, 0.003, 40, byte=0x43), _synthetic_row(3, 0.004, 41, byte=0x44)]
    eater_b = [_synthetic_row(0, 0.001, 44, byte=10), _synthetic_row(1, 0.002, 3, byte=0x41),
               _synthetic_row(2, 0.003, 40, byte=0x43), _synthetic_row(3, 0.004, 41, byte=0x44)]
    probes = _endstep_probe_per_generation(
        stopper + eater_a + eater_b, max_length=4, terminals=[terminal_at(3), None, None]
    )
    assert probes[0]["lf_next_probe_v30"]["winner_byte_at_best_rank_step"] == "boundary", probes[0]
    summary = _lf_competitor_summary_v31(
        [{"endstep_probe_v22": probes, "generations_boundary_self_stop": 1,
          "generations_eating_full_budget": 2}]
    )
    assert summary["n_eaters_with_post_lf_step"] == 2, summary
    assert summary["denominator_at_least_5"] is False, summary
    assert summary["winner_byte_distinct"] == 1, summary
    assert summary["winner_byte_at_best_rank_top"] == [{"key": 65, "count": 2, "share": 1.0}], summary
    assert summary["winner_utf8_class_distinct"] == 1, summary
    #: 所有 LF 之后步的胜出字节：拖写两代各 1 个 LF ⇒ 2 个，仍全是 65；自停代的 boundary 不在内
    assert summary["every_post_lf_winner_count"] == 2, summary
    assert summary["every_post_lf_winner_top"] == [{"key": 65, "count": 2, "share": 1.0}], summary


def test_the_lf_repeat_context_counts_newlines_inside_the_repeat_run_only() -> None:
    """§85 的 v32：`in_run` 的本义是"落在同字重复段内"，这里只许按本义用（v22 误用过的地方）。

    两向：① 占比能取 0.0／0.5／1.0（不是恒零列也不是恒真式）；
    ② `discriminator_denominator_at_least_10` 必须随拖写组里"发过 LF 的代数"翻转——
       §85 的分母门槛（≥10 才允许判）写进仪器，不靠我记得。
    """
    from probe_taiji_a30_stop_failure import (
        _endstep_probe_per_generation,
        _lf_repeat_context_summary_v32,
    )

    def terminal_at(step_value: int):
        return {"step": step_value, "p_boundary": 0.3, "p_boundary_before_penalty": 0.3,
                "boundary_rank_in_legal": 1, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 1.0, "boundary_is_argmax": True,
                "utf8_state_before": [0, 0]}

    def group(lf_inside: tuple, *, last_byte: int = 0x41):
        #: lf_inside 里每一项是 (该 LF 是否在同字重复段内)；末尾再补一个非 LF 步
        rows = []
        for index, inside in enumerate(lf_inside):
            row = _synthetic_row(index, 0.001, 20 + index, byte=10)
            row["in_run"] = bool(inside)
            rows.append(row)
        tail = _synthetic_row(len(rows), 0.002, 30, byte=last_byte)
        rows.append(tail)
        return rows

    stopper = group((True, False))                      #: 2 个 LF，一半在重复段内（共 3 步＜预算 ⇒ 自停）
    eater = group((False, False, False))               #: 3 个 LF，全在段外（共 4 步＝吃满预算）
    probes = _endstep_probe_per_generation(
        stopper + eater, max_length=4, terminals=[terminal_at(2), None]
    )
    assert [p["ate_full_budget"] for p in probes] == [False, True], probes
    assert probes[0]["lf_repeat_context_v32"] == {
        "lf_steps_total": 2, "lf_steps_inside_repeat_run": 1, "lf_share_inside_repeat_run": 0.5,
    }, probes[0]
    assert probes[1]["lf_repeat_context_v32"]["lf_share_inside_repeat_run"] == 0.0, probes[1]

    summary = _lf_repeat_context_summary_v32(
        [{"endstep_probe_v22": probes, "generations_boundary_self_stop": 1,
          "generations_eating_full_budget": 1}]
    )
    assert summary["stoppers_median_share_inside_repeat_run"] == 0.5, summary
    assert summary["eaters_median_share_inside_repeat_run"] == 0.0, summary
    assert summary["discriminator_denominator_at_least_10"] is False, summary

    #: 把拖写组里"发过 LF 的代"推到 10 代 ⇒ 开关必须翻成 True（§85 的门槛不是口头承诺）
    many = probes[:1] + [probes[1] for _ in range(10)]
    flipped = _lf_repeat_context_summary_v32(
        [{"endstep_probe_v22": many, "generations_boundary_self_stop": 1,
          "generations_eating_full_budget": 10}]
    )
    assert flipped["eaters_with_lf"] == 10, flipped
    assert flipped["discriminator_denominator_at_least_10"] is True, flipped


def test_the_dynamic_range_flag_refuses_a_ruler_that_cannot_discriminate() -> None:
    """§86 的教训落成拦得住的东西：一把只能取端点的尺必须被 `ruler_usable=false` 挡下。

    两向都要有：① §85 那种退化形状（每组各自只有一个端点值）⇒ `usable` False、
    `endpoint_only` True ⇒ 判据不成立；② 有真实散布时 ⇒ True。
    没有②的话这条守卫等于"永远拒绝"，那是另一种坏。
    """
    from probe_taiji_a30_stop_failure import _dynamic_range, _lf_repeat_context_summary_v32

    degenerate = _dynamic_range([0.0, 0.0, 1.0])
    assert degenerate["usable"] is False and degenerate["endpoint_only"] is True, degenerate
    assert degenerate["distinct"] == 2 and degenerate["n_values"] == 3, degenerate
    spread = _dynamic_range([0.0, 0.5, 1.0])
    assert spread["usable"] is True and spread["endpoint_only"] is False, spread
    empty = _dynamic_range([None, None])
    assert empty["usable"] is False and empty["n_values"] == 0, empty

    def gen(share: float, *, ate: bool):
        return {
            "generation_steps": 4,
            "last_step": 3,
            "p_boundary_max": 0.001,
            "ate_full_budget": ate,
            "last_recorded_row": {"step": 3, "boundary_rank_in_legal": 30, "p_boundary": 0.001,
                                  "legal_candidates_including_boundary": 41,
                                  "ratio_best_over_boundary": 9.0},
            "terminal_decision": None if ate else {
                "step": 4, "p_boundary": 0.3, "p_boundary_before_penalty": 0.3,
                "boundary_rank_in_legal": 1, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": 1.0, "boundary_is_argmax": True,
                "terminal_utf8_state_before": [0, 0],
                "terminal_over_recorded_peak": 300.0},
            "terminal_decision_absent_reason": "ate_full_budget" if ate else None,
            "tail_identity": {"prev_step_emitted_byte": 10, "prev_step_utf8_byte_class": "ascii",
                              "prev_step_utf8_state_before": [0, 0], "prev_step_boundary_rank": 30,
                              "prev_step_p_boundary": 0.001},
            "lf_trace_v29": {"lf_step_count": 2, "last_lf_step": 2, "generation_last_step": 3},
            "lf_next_probe_v30": {"lf_count": 2, "lf_next_steps": 2,
                                  "best_boundary_rank_after_lf": 2,
                                  "p_boundary_at_best_rank_step": 0.02,
                                  "min_ratio_best_over_boundary_after_lf": 1.1,
                                  "winner_byte_at_best_rank_step": 231,
                                  "utf8_class_at_best_rank_step": "lead3",
                                  "winner_bytes_after_lf": [231, 231]},
            "lf_repeat_context_v32": {"lf_steps_total": 2, "lf_steps_inside_repeat_run": int(share),
                                      "lf_share_inside_repeat_run": share},
        }

    #: §85 真实形状：自停组全 0.0、拖写组全 0.0 ⇒ 尺子没有动态范围
    stuck = _lf_repeat_context_summary_v32(
        [{"endstep_probe_v22": [gen(0.0, ate=False), gen(0.0, ate=False)] + [gen(0.0, ate=True) for _ in range(11)],
          "generations_boundary_self_stop": 2, "generations_eating_full_budget": 11}]
    )
    assert stuck["ruler_usable"] is False, stuck
    assert stuck["column_dynamic_range_eaters"]["endpoint_only"] is True, stuck
    assert stuck["discriminator_denominator_at_least_10"] is True, stuck      #: 分母够但尺子不够

    #: 有散布时开关必须放行（否则这条守卫会永远拒绝，那是另一种设计错）
    varied = _lf_repeat_context_summary_v32(
        [{"endstep_probe_v22": [gen(0.0, ate=False), gen(0.5, ate=False)]
          + [gen(0.5, ate=True) for _ in range(10)],
          "generations_boundary_self_stop": 2, "generations_eating_full_budget": 10}]
    )
    assert varied["ruler_usable"] is True, varied


def test_the_lf_margin_ruler_has_dynamic_range_and_shares_one_implementation() -> None:
    """§89 的 v34：换的那把尺必须**自己报告有没有动态范围**，且 LF+1 的取法只许住一处。

    三件事一起钉：① 每次 LF+1 的比值要能散布（不是 §86 那种端点尺）——
       合成档里 1.2 与 9.0 并存时 `usable` 必须 True；全为空时 `usable` False；
    ② `_rows_after_lf()` 计数＝1（v30 与 v34 共用，两处各写一份就是两把尺子）；
    ③ 每次 LF 的明细最多留 6 条（件不许被逐字节大数组撑爆，§63 那条"只存点不存序列"的规矩）。
    """
    from probe_taiji_a30_stop_failure import (
        _endstep_probe_per_generation,
        _lf_margin_summary_v34,
    )

    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(
        encoding="utf-8"
    )
    assert source.count("def _rows_after_lf(") == 1, "LF 之后那一步的取法只许住一处"

    def terminal_at(step_value: int, ratio: float = 1.0):
        return {"step": step_value, "p_boundary": 0.3, "p_boundary_before_penalty": 0.3,
                "boundary_rank_in_legal": 1, "legal_candidates_including_boundary": 41,
                "ratio_best_over_boundary": ratio, "boundary_is_argmax": True,
                "utf8_state_before": [0, 0]}

    def gen(spec):
        #: spec 是 (步序, 名次, 该步胜出者比边界强多少, 发出的字节) 的列表；
        #: LF 的"下一步"才是要看的那一步，所以比值挂在**跟随步**上。
        rows = []
        for step, rank, ratio, byte in spec:
            row = _synthetic_row(step, 0.001 + 0.0001 * step, rank, byte=byte)
            row["ratio_best_over_boundary"] = ratio
            rows.append(row)
        return rows

    a_rows = gen([(0, 44, 20.0, 10), (1, 5, 1.2, 0x42), (2, 44, 20.0, 10), (3, 3, 9.0, 0x43)])
    b_rows = gen([(0, 30, 7.0, 0x41), (1, 44, 20.0, 10), (2, 2, 1.3, 0x42), (3, 40, 12.0, 0x43)])
    probes = _endstep_probe_per_generation(a_rows + b_rows, max_length=4, terminals=[None, None])
    margins = [p["lf_margins_v34"] for p in probes]
    assert [m["lf_steps"] for m in margins] == [2, 1], margins
    assert [m["lf_plus_one_observed"] for m in margins] == [2, 1], margins
    assert margins[0]["min_ratio"] == 1.2 and margins[0]["worst_ratio"] == 9.0, margins[0]
    assert margins[0]["best_rank"] == 3, margins[0]
    assert len(margins[0]["per_step"]) <= 6, margins[0]
    assert all(p["ate_full_budget"] for p in probes), probes
    summary = _lf_margin_summary_v34(
        [{"endstep_probe_v22": probes, "generations_boundary_self_stop": 0,
          "generations_eating_full_budget": 2}]
    )
    assert summary["eaters_with_lf_next"] == 2, summary
    assert summary["eaters_dynamic_range"]["usable"] is True, summary
    assert summary["eaters_median_min_ratio"] in (1.2, 1.3), summary
    assert summary["eaters_denominator_at_least_20"] is False, summary
    #: 尺子退化的形状（全 None／全同值）必须被报成 usable=False
    empty = _lf_margin_summary_v34(
        [{"endstep_probe_v22": [dict(p, lf_margins_v34={**p["lf_margins_v34"], "min_ratio": 2.0})
                                for p in probes],
          "generations_boundary_self_stop": 0, "generations_eating_full_budget": 2}]
    )
    assert empty["eaters_dynamic_range"]["usable"] is False, empty


def test_the_report_carries_the_prompt_set_fingerprint_and_no_duplicate_keys() -> None:
    """v35（DEBT-G31）：配对档要能**从件内**自证同题面；顺手钉一条"字典字面量不许有重复键"。

    两条都是我自己写坏过的形状：①我曾以为"`items: 96` ＋ manifest 路径相同"就算同题面——不是，
    那只证明读过同一个文件，题面内容有没有变过要另算；②升 v35 时我在同一个 report 字典里写下了
    两次 `"format_note_v34"`，Python **静默取最后一份**、前一份成了死文本，`py_compile` 与 ruff 都不报
    ——是 AST 扫描抓住的，于是把那个扫描本身写成守卫。
    """

    source = (PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py").read_text(
        encoding="utf-8"
    )
    assert '"items_sha256": _items_fingerprint(items)' in source
    assert '"format_note_v35"' in source and source.count('"format_note_v34"') == 1

    duplicates: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        seen: set[str] = set()
        for key in node.keys:
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if key.value in seen:
                duplicates.append(key.value)
            seen.add(key.value)
    assert duplicates == [], duplicates

    #: 指纹只吃题面内容（id／turns／expected_contains），不吃生成结果 ⇒ 同题面必同值、改一题必变值。
    from probe_taiji_a30_stop_failure import _items_fingerprint

    sample = [{"id": "V001", "turns": ["问：甲。", "答：乙"], "expected_contains": ["乙"], "family": "x"}]
    changed = [{"id": "V001", "turns": ["问：甲。", "答：丙"], "expected_contains": ["乙"], "family": "x"}]
    assert _items_fingerprint(sample) == _items_fingerprint([dict(sample[0])])
    assert _items_fingerprint(sample) != _items_fingerprint(changed)
    #: `family` 这类**不参与题面**的列改动不该改指纹（否则同题面会因为元数据而被判成两批）。
    meta = [{"id": "V001", "turns": ["问：甲。", "答：乙"], "expected_contains": ["乙"], "family": "y"}]
    assert _items_fingerprint(sample) == _items_fingerprint(meta)


def test_the_peak_step_row_names_what_was_emitted_where_stopping_came_closest(tmp_path: Path) -> None:
    """§106 的守卫：峰值那一步要带**实发字节**与"是否等于前一步的字"，且**分母要一起报**。

    能为假的三处：①首步没有前一步可比 ⇒ 该格必须是 `None`（不许当成"不同字"，那会把占比虚高地抬起来）；
    ②占比只按可比代算，且 `denominator_with_previous_step` 与 `n` 同时出现在件里；
    ③分群边界：`ate_full_budget ∧ lf_step_count==0` 才进 `eaters_never_lf`（§101 的定义），
      发过 LF 的一律不许混进来——混进来会让 §106 的甲/乙分支读的是另一个总体。
    """

    from probe_taiji_a30_stop_failure import _peak_step_summary_v36

    def gen(*, stopped, lf, peak_byte, prev_byte, step=5):
        return {
            "terminal_decision": {"step": step} if stopped else None,
            "ate_full_budget": not stopped,
            "lf_trace_v29": {"lf_step_count": lf},
            "peak_step_row_v36": {
                "step": step,
                "emitted_byte": peak_byte,
                "emitted_byte_class": "ascii",
                "previous_step_emitted_byte": prev_byte,
                "same_byte_as_previous_step": (None if prev_byte is None else peak_byte == prev_byte),
                "ratio_best_over_boundary": 2.0,
                "p_boundary": 0.004,
                "boundary_rank_in_legal": 14,
            },
        }

    per_item = [
        {
            "endstep_probe_v22": [
                gen(stopped=True, lf=1, peak_byte=10, prev_byte=10),
                gen(stopped=False, lf=2, peak_byte=7, prev_byte=7),
                gen(stopped=False, lf=0, peak_byte=228, prev_byte=228),
                gen(stopped=False, lf=0, peak_byte=229, prev_byte=None),
                gen(stopped=False, lf=0, peak_byte=65, prev_byte=66),
            ]
        }
    ]
    out = _peak_step_summary_v36(per_item)
    assert out["stoppers_n"] == 1 and out["eaters_with_lf_n"] == 1 and out["eaters_never_lf_n"] == 3, out
    #: 群三：3 代里 2 代可比（一代无 prev），可比里 1 代同字 ⇒ 1/2＝0.5，**不是** 1/3。
    assert out["eaters_never_lf_denominator_with_previous_step"] == 2, out
    assert out["eaters_never_lf_same_byte_at_peak_share"] == 0.5, out
    assert out["eaters_with_lf_same_byte_at_peak_share"] == 1.0, out
    assert out["eaters_never_lf_median_boundary_rank_at_peak"] == 14, out
    top = out["eaters_never_lf_winner_bytes_top5"]
    assert {item["byte"] for item in top} == {228, 229, 65}, top
    #: 全代都不可比 ⇒ 占比必须 `None`（不许写 0.0 让人读成"没有一代同字"）。
    none_cmp = _peak_step_summary_v36([{"endstep_probe_v22": [gen(stopped=False, lf=0, peak_byte=1, prev_byte=None)]}])
    assert none_cmp["eaters_never_lf_same_byte_at_peak_share"] is None, none_cmp
    assert none_cmp["eaters_never_lf_denominator_with_previous_step"] == 0, none_cmp


def test_the_character_level_selector_is_not_the_byte_adjacency_one(tmp_path: Path) -> None:
    """§110 的守卫：把 §106 那个错**编码成测试**——三字节重复词的相邻字节永远不等。

    所以"字级在段内"与"字节相邻相等"必须能取到相反的真值；一台只会看字节的仪器**没资格**回答
    "崩塌环是否吃掉了近停态"。这条测试即使 v37 被改坏也只红在这一格，不影响其他读数。
    """

    from probe_taiji_a30_stop_failure import _peak_run_summary_v37

    def gen(*, peak_in_run, steps_in_run, byte_eq_prev, stopped=False, lf=0):
        return {
            "terminal_decision": {"step": 3} if stopped else None,
            "ate_full_budget": not stopped,
            "lf_trace_v29": {"lf_step_count": lf},
            "steps_in_repeat_run": steps_in_run,
            "repeat_run_at_peak_step_v37": peak_in_run,
            "peak_step_row_v36": {
                "step": 3,
                "emitted_byte": 0xA5,
                "previous_step_emitted_byte": 0x93,
                "same_byte_as_previous_step": byte_eq_prev,
            },
        }

    # 哥哥哥：峰值步发 A5、上一步发 93 ⇒ 字节相邻 False，但字级在同字段内 True
    repeated_cjk = gen(peak_in_run=True, steps_in_run=120, byte_eq_prev=False)
    out = _peak_run_summary_v37([{"endstep_probe_v22": [repeated_cjk]}])
    assert out["eaters_never_lf_peak_in_run_share"] == 1.0, out
    assert out["eaters_never_lf_n"] == 1
    assert repeated_cjk["peak_step_row_v36"]["same_byte_as_previous_step"] is False
    assert out["incoherent_zero_run_but_peak_in_run"] == 0, out

    #: 自检真能为假：整代同字段步数为 0 却报"峰值在段内" ⇒ 违例数必须记下来（列接错就红在这里）。
    wrong = gen(peak_in_run=True, steps_in_run=0, byte_eq_prev=False)
    out2 = _peak_run_summary_v37([{"endstep_probe_v22": [wrong]}])
    assert out2["incoherent_zero_run_but_peak_in_run"] == 1, out2

    #: 群分母用全部代数（不是"有可比前步"那种残缺分母）；空群报 None 而不是 0.0。
    mixed = _peak_run_summary_v37(
        [{"endstep_probe_v22": [gen(peak_in_run=False, steps_in_run=40, byte_eq_prev=False),
                                 gen(peak_in_run=True, steps_in_run=90, byte_eq_prev=False)]}]
    )
    assert mixed["eaters_never_lf_n"] == 2 and mixed["eaters_never_lf_peak_in_run_count"] == 1, mixed
    assert mixed["eaters_never_lf_peak_in_run_share"] == 0.5, mixed
    assert mixed["stoppers_peak_in_run_share"] is None, mixed

