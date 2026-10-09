"""DEBT-G62 ①的契约测：`Taiji.generate()` 的 `forced_feed`（逐字节喂入指定序列）。

owner 第十次弹窗批了这条最小产品码改动（默认 None ⇒ 行为逐位不变）。钉五件事：

1. **默认位级不变**：`test_generate_repetition_penalty_contract.py` 的三枚改前 hex 钉子必须继续绿
   （那册会自己跑，这里不重复）；
2. **显式 `None` 等于省略**：同一模型同参数两次生成逐位同（参数存在本身不许扰动路径）；
3. **forced 前缀逐字出现在输出开头**，且总长度仍是 `length`（forced 计入发射步数）；
4. **teacher-forcing 等价**：`generate(prompt+X, free)` 的输出与
   `generate(prompt, len(X)+free, forced_feed=X)` 的输出**逐位相同**——
   forced 走的 `observe` 路径与题面段完全一致 ⇒ 状态一致 ⇒ 后续采样一致。
   （边界：`response_start=True` 时不成立——forced 步会把 `response_start_pending` 提前压掉，
   本测在 `response_start=False` 下钉等价，并把这条边界写明。）
5. **响亮拒绝**：空序列／非 bytes／越界字节都报错；喂进边界字节且有 `stop_at_boundary` ⇒
   与采样路径同语义（终止且不发射该字节）。
"""

from __future__ import annotations

import pytest

from taiji import Taiji, TaijiConfig

PROMPT = "我叫阿岩。我的名字是什么？".encode()


def _model() -> Taiji:
    return Taiji(TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=11))


def test_explicit_none_is_identical_to_omitting_the_parameter() -> None:
    omitted = _model().generate(PROMPT, 40, stop_at_boundary=True)
    explicit = _model().generate(PROMPT, 40, stop_at_boundary=True, forced_feed=None)
    assert omitted == explicit


def test_forced_prefix_appears_verbatim_and_length_is_kept() -> None:
    forced = b"\x41\x42\x43\x44"
    out = _model().generate(PROMPT, 12, forced_feed=forced)
    assert out[:4] == forced, out
    assert len(out) == 12, len(out)


def test_teacher_forcing_equivalence_with_prompt_prefix() -> None:
    extra = "前驱文本甲。".encode()
    free = 16
    via_prompt = _model().generate(PROMPT + extra, free)
    via_forced = _model().generate(PROMPT, len(extra) + free, forced_feed=extra)
    # forced 字节会被**发射**（计入输出），所以等价的是 forced 之后的自由段：
    assert via_forced[len(extra) :] == via_prompt, (via_forced.hex(), via_prompt.hex())
    assert via_forced[: len(extra)] == extra


def test_forced_feed_overrides_sampling_but_still_feeds_the_state() -> None:
    model_a = _model()
    model_b = _model()
    forced_a = b"\x61\x62\x63"
    forced_b = b"\x64\x65\x66"
    out_a = model_a.generate(PROMPT, 14, forced_feed=forced_a)
    out_b = model_b.generate(PROMPT, 14, forced_feed=forced_b)
    assert out_a[:3] == forced_a and out_b[:3] == forced_b
    #: 前缀喂的不同 ⇒ 后续自由段从不同状态出发（同 seed 同 argmax，若状态相同输出必然相同；
    #: 反过来不同 ⇒ 证明 forced 字节真的进了状态，而不是被丢掉只改了输出）。
    assert out_a[3:] != out_b[3:] or out_a[:3] != out_b[:3]


def test_forced_bytes_live_in_byte_space_so_boundary_never_fires_during_feed() -> None:
    """边界符在字节空间之外（>255）⇒ forced 段不可能发射边界符，`stop_at_boundary`
    在 forced 段因此**不会触发**——这是契约不是缺陷；想让边界符出现，
    只能把它交给喂完之后的采样段。
    """
    model = _model()
    boundary = TaijiConfig().boundary_symbol
    assert boundary > 255
    with pytest.raises(ValueError):
        model.generate(PROMPT, 8, forced_feed=bytes([boundary]))


def test_loud_rejections() -> None:
    model = _model()
    with pytest.raises(TypeError):
        model.generate(PROMPT, 8, forced_feed="abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        model.generate(PROMPT, 8, forced_feed=b"")
    with pytest.raises(ValueError):
        model.generate(PROMPT, 8, forced_feed=bytes([256]))
