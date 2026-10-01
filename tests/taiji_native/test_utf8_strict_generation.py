"""SPEC-R2-02 守卫：解码掩码产品化（UTF-8 硬约束进产品生成链）。

四组断言，对应预注册 §4 的守卫列：
1. **委托等价**：`taiji.utf8_state` 与 `LanguageAlignmentTrainer._utf8_allowed/​_advance_utf8`
   对全状态空间（remaining 0..3 × lead 全 256）逐字节相同——一份实现不许漂移。
2. **off 位逐位不变**：`generate(utf8_strict=False)` 与改前实现逐位相同（golden 摘要在
   合入**当前**代码后、由小模型实测钉下——口径同 `MOUNT_BASELINE_DIGEST_BY_CONFIG_SEED`）。
3. **on 构造性合法**：贪心与采样两路、含对抗提示（随机字节/引导字节），输出 100% 合法
   UTF-8 且无悬空尾缀。
4. **trim_partial_tail**：半字符截断、完整序列不动、纯 ASCII 不动。
"""

from __future__ import annotations

import hashlib
import random

from taiji import Taiji, TaijiConfig
from taiji.language_alignment import LanguageAlignmentTrainer
from taiji.utf8_state import advance_utf8, trim_partial_tail, utf8_allowed

S = 7
CONFIG = TaijiConfig(
    region_sizes=(48, 32),
    synapse_fan_in=12,
    motor_fan_in=32,
    seed=S,
)

#: 在 `utf8_strict` 参数加入后、**默认位行为逐位未变**的前提下实测钉下的输出摘要
#: （口径同 `MOUNT_BASELINE_DIGEST_BY_CONFIG_SEED`：off 位任何改动都会在这里红）。
OFF_GOLDEN = {
    "你好": "b3be1844aae4085eecc033f691313c3a580bd530b9b03db6c80e39c1d8ff422f",
    "问：你叫什么名字？\n答：": "ec217c7a05866d2b7ffe8b1a036fb4ca71c9d6842659edbf891966f16e2fd365",
    "": "2f3657e392fa121cbe718527af7fda2396ab511f0518b528d2a3349b41ec6c86",
    "Ã(": "e88b91139856119a3045201a84fc96436986f06d16cd2ab133765bdaa1f77d51",
}


def _model() -> Taiji:
    return Taiji(CONFIG, episode_id="utf8-strict-guard")


def test_shared_state_machine_matches_delegate_exhaustively() -> None:
    """委托前后逐字节相同：全 remaining × 全 lead 状态空间。"""

    for remaining in range(4):
        for lead in range(256):
            assert LanguageAlignmentTrainer._utf8_allowed(remaining, lead) == utf8_allowed(
                remaining, lead
            ), (remaining, lead)
            for symbol in range(256):
                assert LanguageAlignmentTrainer._advance_utf8(
                    remaining, lead, symbol
                ) == advance_utf8(remaining, lead, symbol)


def test_utf8_state_matches_instrument_probe_copy() -> None:
    """判读仪器（FROZEN，不许改）那份实现与本件逐字节相同——同源确认。"""

    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "training"
        / "probe_taiji_cap0_byte_output.py"
    )
    spec = importlib.util.spec_from_file_location("probe_cap0", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    for remaining in range(4):
        for lead in range(256):
            assert probe._utf8_allowed(remaining, lead) == utf8_allowed(remaining, lead)


def test_off_default_is_bitwise_unchanged() -> None:
    """默认位（utf8_strict 不传）与显式 False 逐位相同，且与入册 golden 摘要一致。"""

    prompts = ["你好", "问：你叫什么名字？\n答：", "", b"\xc3\x28".decode("latin-1")]
    for prompt in prompts:
        model_a = _model()
        model_b = _model()
        out_default = model_a.generate(
            prompt.encode("utf-8"), 32, stop_at_boundary=False, sample=False, use_memory=False
        )
        out_off = model_b.generate(
            prompt.encode("utf-8"),
            32,
            stop_at_boundary=False,
            sample=False,
            use_memory=False,
            utf8_strict=False,
        )
        digest = hashlib.sha256(out_default).hexdigest()
        assert out_default == out_off, prompt
        assert len(digest) == 64, digest
        OFF_GOLDEN.setdefault(prompt, digest)
        assert digest == OFF_GOLDEN[prompt], (prompt, digest)


def test_on_path_outputs_are_valid_utf8_greedy() -> None:
    """贪心＋对抗提示：输出必须整串可解码（构造保证，不许有 \ufffd 或悬空前缀）。"""

    rng = random.Random(20260927)
    adversarial = [
        b"",
        b"\xc3",  # 2 字节引导后立刻停
        b"\xe6\x88",  # 3 字节引导＋一个续字节
        b"\xf0\x9f",  # 4 字节引导半途
        b"\xed\xa0",  # 代理区被掩码禁止
        b"\xf4\x90",  # 超范围被掩码禁止
    ]
    for _ in range(40):
        adversarial.append(bytes(rng.randrange(256) for _ in range(rng.randrange(1, 24))))
    model = _model()
    for prompt in adversarial:
        raw = model.generate(
            prompt, 48, stop_at_boundary=False, sample=False, use_memory=False, utf8_strict=True
        )
        raw.decode("utf-8")  # raises on any invalid byte or dangling prefix


def test_on_path_outputs_are_valid_utf8_sampling() -> None:
    """采样路同样构造合法（且必须消耗同一 RNG——默认位不受影响由上一测钉）。"""

    model = _model()
    rng = random.Random(11)
    prompts = [bytes(rng.randrange(256) for _ in range(8)) for _ in range(20)]
    for prompt in prompts:
        raw = model.generate(
            prompt, 40, stop_at_boundary=False, sample=True, use_memory=False, utf8_strict=True
        )
        raw.decode("utf-8")


def test_trim_partial_tail_semantics() -> None:
    assert trim_partial_tail("你好".encode()) == "你好".encode()
    assert trim_partial_tail(b"a\xc3") == b"a"
    assert trim_partial_tail(b"a\xe6\x88") == b"a"
    assert trim_partial_tail(b"\xc3\x28") == b""  # 非法且截无可截⇒交空串，不交非法串
    assert trim_partial_tail(b"") == b""
