"""F1 生成口径：`logit_bias` 必须**默认不改变**旧生成，给了就要当场改选中字节。

两支入口都要测：joint-cache 那条（`_generate_from_joint_cache`，F1 评分器用）与
普通 `generate`（旧保持探针用）。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import taiji.conditional_language_v2 as v2_module
import taiji.conditional_language_v4 as v4_module


class _Joint:
    def __init__(self) -> None:
        self.language_state = SimpleNamespace(generated=b"", utf8_remaining=0, utf8_lead=b"")

    def with_language_state(self, state):
        return SimpleNamespace(language_state=state)

    def complete_response(self):
        raise AssertionError("a two-byte probe must not hit the end marker")


def _observe(state, symbol, hidden):
    state.generated = state.generated + bytes([int(symbol)])
    return state


def _joint_model(monkeypatch, logits: torch.Tensor):
    monkeypatch.setattr(v4_module, "utf8_allowed", lambda remaining, lead: [0, 1])
    model = v4_module.ConditionalLanguageCandidateV4.__new__(
        v4_module.ConditionalLanguageCandidateV4
    )
    model.config = SimpleNamespace(max_generation_bytes=8)
    model.next_logits = lambda state, cache=None: (logits, None)
    model.observe_response = _observe
    return model


def _plain_model(monkeypatch, logits: torch.Tensor):
    monkeypatch.setattr(v2_module, "utf8_allowed", lambda remaining, lead: [0, 1])
    model = v2_module.ConditionalLanguageCandidateV2.__new__(
        v2_module.ConditionalLanguageCandidateV2
    )
    model.config = SimpleNamespace(max_generation_bytes=8)
    model.next_logits = lambda state, cache=None: (logits, None)
    model.evidence_cache = lambda state: None
    model.observe_response = _observe
    return model


def _fresh_state():
    return SimpleNamespace(generated=b"", utf8_remaining=0, utf8_lead=b"")


def test_joint_generation_without_bias_keeps_the_greedy_choice(monkeypatch) -> None:
    model = _joint_model(monkeypatch, torch.tensor([0.0, 1.0]))
    produced, _joint = model._generate_from_joint_cache(_Joint(), None, max_new_bytes=2)
    assert produced == bytes([1, 1])


def test_joint_generation_honours_the_bias(monkeypatch) -> None:
    model = _joint_model(monkeypatch, torch.tensor([0.0, 1.0]))
    produced, _joint = model._generate_from_joint_cache(
        _Joint(), None, max_new_bytes=2, logit_bias=torch.tensor([5.0, 0.0])
    )
    assert produced == bytes([0, 0])


def test_plain_generation_is_unchanged_without_bias(monkeypatch) -> None:
    model = _plain_model(monkeypatch, torch.tensor([0.0, 1.0]))
    assert model.generate(_fresh_state(), max_new_bytes=2).generated == bytes([1, 1])


def test_plain_generation_honours_the_bias(monkeypatch) -> None:
    model = _plain_model(monkeypatch, torch.tensor([0.0, 1.0]))
    generated = model.generate(
        _fresh_state(), max_new_bytes=2, logit_bias=torch.tensor([5.0, 0.0])
    ).generated
    assert generated == bytes([0, 0])
