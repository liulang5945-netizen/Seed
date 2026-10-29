"""F1 生成口径：`_generate_from_joint_cache` 的 `logit_bias` 必须**默认不改变**旧生成。

两支对照：
①不给 bias ⇒ 与原贪心完全一致；
②给一个把另一字节抬高的 bias ⇒ 选中的字节当场改掉（否则这个入口是死参数）。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import taiji.conditional_language_v4 as v4_module


class _Joint:
    def __init__(self) -> None:
        self.language_state = SimpleNamespace(generated=b"", utf8_remaining=0, utf8_lead=b"")

    def with_language_state(self, state):
        return SimpleNamespace(language_state=state)

    def complete_response(self):
        raise AssertionError("a two-byte probe must not hit the end marker")


def _model(monkeypatch, logits: torch.Tensor) -> v4_module.ConditionalLanguageCandidateV4:
    monkeypatch.setattr(v4_module, "utf8_allowed", lambda remaining, lead: [0, 1])
    model = v4_module.ConditionalLanguageCandidateV4.__new__(
        v4_module.ConditionalLanguageCandidateV4
    )
    model.config = SimpleNamespace(max_generation_bytes=8)
    model.next_logits = lambda state, cache=None: (logits, None)

    def observe(state, symbol, hidden):
        state.generated = state.generated + bytes([int(symbol)])
        return state

    model.observe_response = observe
    return model


def test_generation_without_bias_keeps_the_greedy_choice(monkeypatch) -> None:
    model = _model(monkeypatch, torch.tensor([0.0, 1.0]))
    produced, _joint = model._generate_from_joint_cache(_Joint(), None, max_new_bytes=2)
    assert produced == b"\x01\x01"


def test_a_bias_changes_the_next_byte(monkeypatch) -> None:
    model = _model(monkeypatch, torch.tensor([0.0, 1.0]))
    bias = torch.tensor([5.0, 0.0])
    produced, _joint = model._generate_from_joint_cache(
        _Joint(), None, max_new_bytes=2, logit_bias=bias
    )
    assert produced == b"\x00\x00"
