"""守卫：`install_constrained_decode` 的包装签名必须**覆盖产品链现在会传的全部 kwarg**。

来由（2026-10-01，第一次全量 H 标定撞出来的真缺陷）：该 monkeypatch 的签名停在 SPEC-R2-02 之前，
而产品 `chat()` 自 09-27 起总带 `utf8_strict=True` ⇒ 任何 constrained_decode 链路一律
`TypeError: generate() got an unexpected keyword argument 'utf8_strict'`；H 校准/健康支
（required 链含 constrained_decode）因此**静默断档**（仪器自身把读数记成 n=0，没人报错到面上）。
本守卫把"签名面"钉成机检：产品 `Taiji.generate` 的 keyword 参数将来再加，这里先红，
而不是等到某次标定又给出一张空表。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def test_patched_generate_accepts_every_keyword_of_the_product_generate() -> None:
    import functools

    from scripts.training.probe_taiji_cap0_byte_output import install_constrained_decode
    from taiji.adapter import Taiji

    saved = Taiji.generate
    try:
        info = install_constrained_decode()
        assert info["patched"] is True
        product_params = {
            name
            for name, param in inspect.signature(Taiji.generate).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        patched_params = {
            name
            for name, param in inspect.signature(Taiji.generate).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        missing = product_params - patched_params
        assert missing == set(), f"包装签名缺产品链的 kwarg：{sorted(missing)}"
        assert "utf8_strict" in info["ignored_kwargs"]
        assert isinstance(Taiji.generate, functools.partial) or callable(Taiji.generate)
    finally:
        Taiji.generate = saved


def test_patched_generate_refuses_non_default_repetition_penalty_loudly() -> None:
    """对齐不了的一律响亮拒绝（与 boundary/authorization 同一处置）——绝不静默丢语义。"""

    import pytest

    from scripts.training.probe_taiji_cap0_byte_output import install_constrained_decode
    from taiji.adapter import Taiji

    saved = Taiji.generate
    try:
        install_constrained_decode()

        class _Stub:
            pass

        stub = _Stub()
        with pytest.raises(RuntimeError, match="repetition_penalty"):
            Taiji.generate(stub, b"hi", 4, repetition_penalty=2.0)
        with pytest.raises(RuntimeError, match="response_start"):
            Taiji.generate(stub, b"hi", 4, response_phase=True)
    finally:
        Taiji.generate = saved
