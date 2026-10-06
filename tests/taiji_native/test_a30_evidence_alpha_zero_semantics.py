"""α=0 那一档的全部依据：证据通道**仍被调用、但贡献为零**——这条语义此前没有任何测试覆盖过。

来历（2026-10-03，PLAN-A-30 §第一百一十四次停靠的先置守卫）：下一格要跑"回路在载、把加性证据整体乘零"
的对照档（#53），用它把"通道在被问"与"通道在抬分"拆开。那一档的**前提**就是
`_make_scaled_evidence(..., alpha=0.0)` 满足三件事：①原函数照旧被调用（计数器仍涨），
②返回值逐元素为零，③ α=1.0 时是**恒等**（不是近似）。这三件此前只有源码形状可查，没有读数。
守卫里第 2 条（零）与第 3 条（恒等）都是**能反向失败**的：把 `* alpha` 改成 `+ alpha` 或把计数器挪到
提前返回之后，立刻会红。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from probe_taiji_a30_copy_evidence_dose import _make_scaled_evidence  # noqa: E402


def _fixture(value: float = 3.0, size: int = 5) -> tuple[list[int], object]:
    """返回（被调用次数计数器, 包好的 evidence）。原函数固定给一个非零向量。"""

    calls = [0]

    def original(**kwargs: object) -> torch.Tensor:
        calls[0] += 1
        return torch.full((size,), value)

    return calls, original


def test_alpha_one_is_the_identity_not_an_approximation() -> None:
    _calls, original = _fixture()
    scaled, counters = _make_scaled_evidence(original, 1.0)
    out = scaled()
    torch.testing.assert_close(out, torch.full((5,), 3.0), rtol=0.0, atol=0.0)
    assert counters[0] == 1, "计数器没涨 ⇒ 包装器根本没被走到，后面所有 α 档都是假档"


def test_alpha_zero_silences_the_contribution_but_still_consults_the_channel() -> None:
    _calls, original = _fixture()
    scaled, counters = _make_scaled_evidence(original, 0.0)
    out = scaled()
    assert torch.count_nonzero(out) == 0, f"α=0 却留下非零贡献：{out}"
    assert counters[0] == 1, "α=0 时若连调用都省掉，那这一档测的是'不发通道'而不是'发而不计分'"
    # 没开上下限时，两把过滤器计数必须如实为零（不是 null——这一列在这档里本就是"没开"）
    assert counters[1] == 0 and counters[2] == 0


def test_alpha_zero_and_alpha_one_differ_on_every_element() -> None:
    """单变量对照成立的前提：两档之间差的**只有**那一个乘数，且差是可见的。"""

    _c1, o1 = _fixture()
    _c2, o2 = _fixture()
    zero, _ = _make_scaled_evidence(o1, 0.0)
    full, _ = _make_scaled_evidence(o2, 1.0)
    diff = (zero() - full()).abs()
    assert int((diff > 0).sum()) == 5, "两档输出没有逐元素差异 ⇒ α 没起作用，这一档不成立"


def test_partial_dose_scales_exactly_and_accumulates_calls() -> None:
    _calls, original = _fixture(value=2.0)
    scaled, counters = _make_scaled_evidence(original, 0.25)
    for _ in range(3):
        out = scaled()
    torch.testing.assert_close(out, torch.full((5,), 0.5), rtol=0.0, atol=0.0)
    assert counters[0] == 3 == _calls[0]


def test_the_identity_assertion_actually_discriminates_a_nonzero_dose() -> None:
    """负对照：上面那条"α=1.0 逐元素相等"断言**必须**能为假——否则它是装饰不是检验。

    做法＝把 α 挪开一格（0.99）再用同一条断言去比原值；它应当抛错。
    （本仓的硬规矩：新守卫要有一次"实测能为假"的证据，否则它就是装饰。）
    """

    _calls, original = _fixture()
    scaled, _ = _make_scaled_evidence(original, 0.99)
    raised = False
    try:
        torch.testing.assert_close(scaled(), torch.full((5,), 3.0), rtol=0.0, atol=0.0)
    except AssertionError:
        raised = True
    assert raised, "α=0.99 也能通过恒等断言 ⇒ 那条断言量不到乘数，是恒真式"
