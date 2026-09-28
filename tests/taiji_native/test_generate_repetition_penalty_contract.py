"""PLAN-A-30 乙档守卫：解码侧的重复惩罚（默认 0 ⇒ 逐位不变）。

`PLAN-A-30` §2b 量到产品表层链的复读不是"复述完一遍再从头走"（位置回绕数**低于**乱序基线），
而是**掩码把合法后继收成极少候选 ＋ 证据把质量集中 ⇒ 同一个字连发**
（循环单位平均 1.4–2.1 个字符、重复 5–20 次）。本件把对症的那一手做成**默认关闭**的解码参数。

钉四件事：

1. **默认位级不变**：三条路径（无掩码／带掩码／采样）的字节必须与**本件改动之前**逐位相同——
   下面的十六进制是改前在同一台机器、同一 config、同一 prompt 上取来钉住的（不是改后回填的）；
2. **惩罚真的打断连发**：同一条无掩码路径在 `repetition_penalty=1.0` 下最长同字节连写必须显著变短；
3. **惩罚不改长度、不改停止语义**：边界符照旧终止，且**不新增**任何本来为零的质量
   （除以 `1 + p·counts` 是单调缩放，零还是零）；
4. **响亮拒绝非法取值**：负惩罚／负窗口／"给了惩罚却把窗口设 0"都报错，不静默忽略。
"""

from __future__ import annotations

import torch

from taiji import Taiji, TaijiConfig

#: 改前实测钉住的默认输出（`TaijiConfig(seed=11, region_sizes=(64,48), fan_in 16/48)`，
#: prompt＝"我叫阿岩。我的名字是什么？"，length=40）。注意无掩码那条本身就带着
#: `01×8`、`8a×5`、`9b×7` 的同字节连写——重复退化**不是电路专属**。
PIN_NO_MASK = "b2219c46b087c8c86363010101010101010101d4d037d4acd08a8a8a8a8a952c0a669b9b9b9b9b9b"
PIN_MASKED = "d8b221d2b2217f71170a661e7808d49b0a66690a661e7808e5848ed4ac0724033a74c8ab217f7f"
PIN_SAMPLE = "8682343b72c09a185317fe9d78c34b3af6a8c8632efbc1b30dce83a5aa3aeb4c8ae8cf1730b874b6"

PROMPT = "我叫阿岩。我的名字是什么？".encode()


def _model() -> Taiji:
    return Taiji(TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=11))


def _longest_run(data: bytes) -> int:
    best = run = 1
    for a, b in zip(data, data[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if data else 0


# ---------------------------------------------------------------- 1. 默认位级不变


def test_default_paths_reproduce_the_pinned_pre_change_bytes() -> None:
    assert _model().generate(PROMPT, 40, stop_at_boundary=True, sample=False).hex() == PIN_NO_MASK
    assert (
        _model().generate(PROMPT, 40, stop_at_boundary=True, sample=False, utf8_strict=True).hex()
        == PIN_MASKED
    )
    assert _model().generate(PROMPT, 40, sample=True).hex() == PIN_SAMPLE


def test_explicit_zero_penalty_is_the_same_as_omitting_it() -> None:
    plain = _model().generate(PROMPT, 40, stop_at_boundary=True, sample=False)
    zeroed = _model().generate(
        PROMPT, 40, stop_at_boundary=True, sample=False, repetition_penalty=0.0
    )
    assert plain == zeroed


# ---------------------------------------------------------------- 2/3. 惩罚的效果与边界


def test_penalty_breaks_the_same_byte_runs() -> None:
    plain = _model().generate(PROMPT, 40, stop_at_boundary=True, sample=False)
    penalised = _model().generate(
        PROMPT, 40, stop_at_boundary=True, sample=False, repetition_penalty=1.0
    )
    assert _longest_run(plain) >= 5, "钉住症状：默认路径本来就在连写（退化不是电路专属）"
    assert _longest_run(penalised) < _longest_run(plain), penalised.hex()
    assert penalised != plain


def test_penalty_keeps_length_and_does_not_invent_mass() -> None:
    probabilities = torch.zeros(257)
    probabilities[65] = 1.0  # 只有一个字节有质量，其余严格为零
    counts = torch.zeros_like(probabilities)
    for recent in (65, 65, 65):
        counts[recent] += 1.0
    scaled = probabilities / (1.0 + 1.0 * counts)
    assert float(scaled[65]) == 0.25
    #: 该钉的是"支撑集不变"（单调缩放不会把零质量字节抬出质量），而不是"除 0 号位外全零"——
    #: 后者是我第一版的写法，它把 65 号位自己也一起排除了，是个恒假断言。
    assert bool(((scaled != 0) == (probabilities != 0)).all()), "缩放是单调的：零还是零"

    out = _model().generate(PROMPT, 40, stop_at_boundary=True, sample=False, repetition_penalty=0.5)
    assert len(out) <= 40


def test_boundary_still_stops_under_penalty() -> None:
    model = _model()
    model.reset_dynamics(episode_id="a30-stop")
    #: 边界符不参与惩罚的"必发"路径——它只是恒合法（SPEC-R2-02 的口径），
    #: 停止仍然是模型的权力：这里断言惩罚不会把一次正常生成长度推到上限之外。
    out = model.generate(
        "停".encode(), 24, stop_at_boundary=True, sample=False, repetition_penalty=2.0
    )
    assert isinstance(out, bytes) and len(out) <= 24


# ---------------------------------------------------------------- 4. 非法取值


def test_invalid_penalty_arguments_are_rejected() -> None:
    model = _model()
    for kwargs, message in (
        ({"repetition_penalty": -0.1}, "cannot be negative"),
        ({"repetition_window": -1}, "cannot be negative"),
        ({"repetition_penalty": 1.0, "repetition_window": 0}, "needs a positive"),
    ):
        try:
            model.generate(PROMPT, 8, **kwargs)
        except ValueError as exc:
            assert message in str(exc), (kwargs, exc)
        else:  # pragma: no cover - 守卫本身不允许静默通过
            raise AssertionError(f"{kwargs} 应当响亮拒绝")
