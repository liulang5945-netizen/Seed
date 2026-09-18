"""R2-D5 implementation gates: graph v7 copy persistence + fixture v2.

Contract:
plans/reference/M5_R2_D5_MULTIBYTE_FACTORIAL_CONTRACT_FROZEN_20260918.md

Zero-capability gates: the flag-off and bias-zero paths are bit-identical to
the frozen v6 graph; the pointer rule (start/end/zero-read), the injected
bonus behavior, gradient reach of the single new parameter, checkpoint
round-trips across versions, and the v2 fixture alignment (train-only value
substitution with byte-identical dev/final).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from taiji.sequence_workspace import (
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_V1 = PROJECT_ROOT / "tests/fixtures/r2_d1_measurement_v1.jsonl"
FIXTURE_V2 = PROJECT_ROOT / "tests/fixtures/r2_d1_measurement_v2.jsonl"

NEW_COLORS = ("琥珀", "珊瑚", "翡翠")


def _v6(seed: int = 20260917) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source="per_position",
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
        )
    )


def _v7(seed: int = 20260917, *, flag: bool = True) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source="per_position",
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
            copy_persistence=flag,
        )
    )


def _prefix() -> bytes:
    return "提问：雪的颜色？线索：雪是琥珀。回答：".encode("utf-8")


def _response() -> bytes:
    return "琥珀".encode("utf-8")


# --------------------------------------------------------------------------- #
# Gate 1: inventories and bit-identity against the frozen v6 graph
# --------------------------------------------------------------------------- #


def test_gate1_v7_inventory() -> None:
    assert SEQUENCE_WORKSPACE_VERSION == 7
    persisted = _v7()
    assert persisted.parameter_count() == 96_035
    names = tuple(name for name, _ in persisted.named_parameters())
    assert "copy_persist_bias" in names
    assert float(persisted._parameters["copy_persist_bias"].detach().sum()) == 0.0
    flag_off = _v7(flag=False)
    assert flag_off.parameter_count() == 96_034
    assert "copy_persist_bias" not in tuple(name for name, _ in flag_off.named_parameters())


def test_gate1_flag_off_and_zero_bias_bitwise_equal_v6() -> None:
    v6 = _v6()
    off = _v7(flag=False)
    on = _v7(flag=True)  # bias initializes at exactly zero
    prefix, response = _prefix(), _response()
    with torch.no_grad():
        for name, prototype in (("flag_off", off), ("zero_bias", on)):
            assert torch.equal(
                prototype.teacher_forced_logits(prefix, response),
                v6.teacher_forced_logits(prefix, response),
            ), name
            mixed, copied = prototype.teacher_forced_mixture_and_copy(prefix, response)
            v6_mixed, v6_copied = v6.teacher_forced_mixture_and_copy(prefix, response)
            assert torch.equal(mixed, v6_mixed) and torch.equal(copied, v6_copied), name
            assert prototype.sequence_loss(prefix, response)[0].tolist() == v6.sequence_loss(
                prefix, response
            )[0].tolist()
            assert (
                prototype.generate(prefix, max_bytes=8).bytes_out
                == v6.generate(prefix, max_bytes=8).bytes_out
            )


# --------------------------------------------------------------------------- #
# Gate 2: pointer rule structure
# --------------------------------------------------------------------------- #


def test_gate2_first_step_has_no_pointer() -> None:
    prototype = _v7()
    state = prototype.begin_episode(_prefix())
    assert state.last_copy_position is None
    assert prototype._persist_offset(state) is None


def _hit_row(prototype: SequenceWorkspacePrototype, state) -> int:
    """Entry index the cross-head addressing weights argue for (pointer rule)."""

    weights = prototype.addressing_weights(state, detach=True)
    stacked = weights if weights.ndim == 2 else weights.unsqueeze(0)
    return int(stacked.mean(dim=0).argmax())


def test_gate2_large_bonus_forces_next_entry() -> None:
    from dataclasses import replace as dc_replace

    prototype = _v7()
    with torch.no_grad():
        prototype._parameters["copy_persist_bias"].fill_(20.0)
    state = prototype.begin_episode(_prefix())
    hit = _hit_row(prototype, state)
    entries = state.entry_bytes
    assert entries is not None
    with torch.no_grad():
        biased = dc_replace(state, last_copy_position=hit)
        weights = prototype.addressing_weights(biased, detach=False)
        per_head_argmax = weights.argmax(dim=-1)
        target = hit + 1
        if target < len(entries):
            assert torch.all(per_head_argmax == target)
        else:
            # last entry: no wrap-around, so the bonus must not appear at 0.
            assert prototype._persist_offset(biased) is None


def test_gate2_pointer_follows_emission() -> None:
    prototype = _v7()
    with torch.no_grad():
        prototype._parameters["copy_persist_bias"].fill_(20.0)
    state = prototype.begin_episode(_prefix())
    boundary = int(prototype.config.boundary_symbol)
    # Replay the function's internal addressing to know the contracted hit row.
    new_state, _ = prototype.step(state, boundary)
    weights = prototype.addressing_weights(new_state, detach=True)
    stacked = weights if weights.ndim == 2 else weights.unsqueeze(0)
    hit = int(stacked.mean(dim=0).argmax())
    _, mixture, _ = prototype._step_mixture_and_copy(state, boundary)
    emitted = int(mixture.argmax())
    follow, _, _ = prototype._step_mixture_and_copy(state, boundary, emitted_symbol=emitted)
    expected = hit if int(state.entry_bytes[hit]) == emitted else None  # type: ignore[index]
    assert follow.last_copy_position == expected
    mismatched, _, _ = prototype._step_mixture_and_copy(state, boundary, emitted_symbol=boundary)
    assert mismatched.last_copy_position is None  # 256 never occurs inside UTF-8 prefixes


def test_gate2_zero_read_has_no_pointer() -> None:
    prototype = _v7()
    state = prototype.begin_episode(_prefix())
    new_state, probability, p_copy = prototype._step_mixture_and_copy(
        state, int(prototype.config.boundary_symbol), zero_read=True
    )
    assert p_copy is None
    assert new_state.last_copy_position is None


def test_config_rejects_persistence_without_copy() -> None:
    import pytest

    with pytest.raises(ValueError):
        SequenceWorkspaceConfig(copy_persistence=True)


# --------------------------------------------------------------------------- #
# Gate 3: gradient reach of the single new parameter
# --------------------------------------------------------------------------- #


def test_gate3_bias_receives_finite_gradient() -> None:
    from dataclasses import replace as dc_replace

    prototype = _v7()
    state = prototype.begin_episode(_prefix())
    biased = dc_replace(state, last_copy_position=1)
    weights = prototype.addressing_weights(biased, detach=False)
    (grad,) = torch.autograd.grad(
        weights.sum(), [prototype._parameters["copy_persist_bias"]], retain_graph=True
    )
    assert torch.isfinite(grad).all()
    # The bonus enters every head's target logit before the softmax, so the
    # sum-of-weights derivative w.r.t. the shared scalar is exactly zero by
    # normalization; a non-constant readout objective must be finite.
    read = (weights @ state.workspace_value).mean()  # type: ignore[operator]
    (grad_read,) = torch.autograd.grad(read, [prototype._parameters["copy_persist_bias"]])
    assert torch.isfinite(grad_read).all()
    response = "琥珀".encode("utf-8")
    total, metrics = prototype.sequence_loss(
        _prefix(), response, copy_value_weight=1.0, value_mask=[True] * len(response)
    )
    (grad_loss,) = torch.autograd.grad(
        total, [prototype._parameters["copy_persist_bias"]], allow_unused=True
    )
    # The teacher-forced path may only touch the bias from step two onward; when
    # the first step's copy argmax misses the true byte the pointer stays None,
    # which is the contracted behaviour, so ``None`` is accepted here.
    assert grad_loss is None or torch.isfinite(grad_loss).all()
    assert metrics["positions"] > 0


# --------------------------------------------------------------------------- #
# Gate 4: checkpoint round-trips
# --------------------------------------------------------------------------- #


def test_gate4_v7_roundtrip_and_v6_readability(tmp_path: Path) -> None:
    trainer = SequenceWorkspaceTrainer(_v7(), learning_rate=0.01, code_revision="gate")
    trainer.set_episodes([(_prefix(), _response())])
    payload = trainer.checkpoint()
    assert payload["version"] == SEQUENCE_WORKSPACE_VERSION
    restored = SequenceWorkspaceTrainer.from_checkpoint(payload)
    assert restored.prototype.config.copy_persistence is True
    before = trainer.prototype.generate(_prefix(), max_bytes=6).bytes_out
    after = restored.prototype.generate(_prefix(), max_bytes=6).bytes_out
    assert before == after
    legacy = SequenceWorkspaceTrainer(_v6(), learning_rate=0.01, code_revision="legacy")
    legacy_path = tmp_path / "v6.pt"
    legacy.save(legacy_path)
    revived = SequenceWorkspaceTrainer.from_checkpoint(
        torch.load(legacy_path, map_location="cpu", weights_only=False)
    )
    assert revived.prototype.config.copy_persistence is False
    assert "copy_persist_bias" not in dict(
        (name, _) for name, _ in revived.prototype.named_parameters()
    )


# --------------------------------------------------------------------------- #
# Gate 5: fixture v2 alignment (contract §2.1)
# --------------------------------------------------------------------------- #


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_gate5_v2_dev_final_identical_and_train_row_aligned() -> None:
    v1 = _rows(FIXTURE_V1)
    v2 = _rows(FIXTURE_V2)
    assert [r for r in v1 if r["split"] != "train"] == [r for r in v2 if r["split"] != "train"]
    t1 = [r for r in v1 if r["split"] == "train"]
    t2 = [r for r in v2 if r["split"] == "train"]
    assert len(t1) == len(t2) == 174
    changed = 0
    for a, b in zip(t1, t2, strict=True):
        assert a["id"] == b["id"] and a["shape"] == b["shape"]
        assert a["pair_id"] == b["pair_id"] and a["pair_type"] == b["pair_type"]
        if a["prefix"] != b["prefix"]:
            changed += 1
    # 144 value rows + 14 combo materials changed; the 16 unknown rows are intact.
    assert changed == 158
    # Every v2 train response is either a content-independent constant or one
    # of the pool colors (optionally negation-prefixed).
    pool = {"黄", "青", "紫", *NEW_COLORS}
    for record in t2:
        response = record["response"]
        if record["shape"] in ("unknown", "same_opening_unknown"):
            assert response == "未知"
        elif record["shape"] == "combination_same":
            assert response == "相同"
        elif record["shape"] == "combination_different":
            assert response == "不同"
        elif record["shape"] == "negation":
            assert response[2:] in pool and response[:2] == "不是"
        else:
            assert response in pool
    assert sum(1 for r in t2 if any(color in r["response"] for color in NEW_COLORS)) == 72
    # New characters never leak outside value positions.
    all_text = "".join(r["prefix"] + r["response"] for r in v1)
    for color in NEW_COLORS:
        for char in color:
            assert char not in all_text
