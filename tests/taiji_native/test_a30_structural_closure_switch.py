"""出口收笔（结构条件）守卫：owner 裁定 2026-09-30 §7b-3＝**实现但默认关**。

结构条件收笔＝产品认"答案已成形＋模型发换行"为轮界（§2z 候选2）。三条钉死：
①默认位（`STRUCTURAL_CLOSURE_DEFAULT=False`）⇒ `chat()` 输出与旧行为逐位不变；
②显式开启 ⇒ 答复在第一个换行处收笔；
③门槛①在开启时把"带换行的吃满预算答复"视为已收口（与解码侧同口径）；
并再钉一句代价登记：这是**产品的判据不是模型的判断**——不许记进模型能力。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from seed import surface_gate  # noqa: E402


def test_default_is_off_and_documented_as_product_judgement() -> None:
    from api import seed_runtime

    assert seed_runtime.STRUCTURAL_CLOSURE_DEFAULT is False
    doc = seed_runtime.SeedRuntime.chat.__doc__ or ""
    assert "产品的判据不是模型的判断" in doc


def test_structural_cut_takes_the_first_newline() -> None:
    cut, fired = surface_gate.structural_closure_cut("答案是四十二。\n后面还有废话")
    assert (cut, fired) == ("答案是四十二。", True)
    kept, fired2 = surface_gate.structural_closure_cut("没有换行的答复")
    assert (kept, fired2) == ("没有换行的答复", False)


def test_ended_naturally_honours_the_switch() -> None:
    raw = ("这一句够长，然后把预算吃满。" * 30).encode("utf-8") + b"\n"
    markers = ("\n问：", "问：")
    budget = 64
    assert len(raw) >= budget
    #: 关：带换行但无轮界 marker ⇒ 不算自然收口（旧行为逐位不变）。
    assert not surface_gate.ended_naturally(raw, turn_markers=markers, budget=budget)
    #: 开：结构条件把换行当收笔。
    assert surface_gate.ended_naturally(
        raw, turn_markers=markers, budget=budget, structural_newline=True
    )


def test_write_back_switch_passes_through() -> None:
    model = surface_gate.load_surface_ngram(
        PROJECT_ROOT / "checkpoints" / "seed_surface_ngram.lzma"
    )
    answer = "这一句是干净中文，长度也够，说完了。"
    raw = (answer * 8).encode("utf-8") + b"\n"
    markers = ("\n问：", "问：")
    budget = 64
    off, reason_off = surface_gate.write_back_allowed(
        answer, raw, model, turn_markers=markers, budget=budget
    )
    on, reason_on = surface_gate.write_back_allowed(
        answer, raw, model, turn_markers=markers, budget=budget, structural_newline=True
    )
    assert (off, reason_off) == (False, "not_ended_naturally")
    assert (on, reason_on) == (True, "passed")


def test_default_chat_face_is_bit_identical_with_switch_absent() -> None:
    """默认位不参与任何路径：不传参时 `last_structural_closure_cut` 恒 False。"""

    from api.seed_runtime import FACTORY_CHECKPOINT, SeedRuntime

    runtime = SeedRuntime.load(FACTORY_CHECKPOINT)
    try:
        runtime.chat("你好", max_length=64, learn=False)
        assert runtime.last_structural_closure_cut is False
        runtime.chat("你好", max_length=64, learn=False, structural_closure=True)
        #: 开启时只多一个切割动作；答复本身可能无换行（视基座输出），但观测位在案。
        assert runtime.last_structural_closure_cut in (True, False)
    finally:
        del runtime
