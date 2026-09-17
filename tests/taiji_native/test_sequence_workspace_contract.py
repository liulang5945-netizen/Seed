"""R2-H3.8 implementation gates for the isolated joint-sequence-credit prototype.

Contract: plans/reference/M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md
section 2.  These tests are the entry requirement: they run no capability
training and prove only structure, gradients, masking, restoration, and wiring.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from taiji.internalization import content_digest
from taiji.sequence_workspace import (
    SEQUENCE_WORKSPACE_BASELINE_PARAMETERS,
    SEQUENCE_WORKSPACE_PARAMETERS,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prototype(seed: int = 7) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(SequenceWorkspaceConfig(seed=seed))


def _batch() -> tuple[tuple[bytes, bytes], ...]:
    return (
        ("问：天空的颜色？".encode(), "蓝色<|end|>".encode()),
        ("问：今天天气？".encode(), "未知<|end|>".encode()),
    )


# --------------------------------------------------------------------------- #
# Gate 1: isolation and inventory
# --------------------------------------------------------------------------- #


def test_gate1_parameter_inventory_matches_the_declared_list() -> None:
    prototype = _prototype()
    live = tuple(name for name, _ in prototype.named_parameters())
    assert live == SEQUENCE_WORKSPACE_PARAMETERS
    assert prototype.declared_parameter_names() == SEQUENCE_WORKSPACE_PARAMETERS
    assert len(set(live)) == len(live)
    total = prototype.parameter_count()
    assert total == sum(item.numel() for item in prototype.parameters())
    assert total > 0


def test_gate1_prototype_is_isolated_from_the_default_entry() -> None:
    native = importlib.import_module("taiji")
    init_source = Path(native.__file__).read_text(encoding="utf-8")
    assert "sequence_workspace" not in init_source
    for module_name in ("taiji.model", "taiji.organs"):
        source = Path(importlib.import_module(module_name).__file__).read_text(encoding="utf-8")
        assert "sequence_workspace" not in source, module_name


def test_gate1_baseline_arm_inventory_and_shared_initialization() -> None:
    baseline = SequenceWorkspacePrototype(SequenceWorkspaceConfig(seed=7, workspace_enabled=False))
    assert baseline.declared_parameter_names() == SEQUENCE_WORKSPACE_BASELINE_PARAMETERS
    assert tuple(name for name, _ in baseline.named_parameters()) == tuple(
        SEQUENCE_WORKSPACE_BASELINE_PARAMETERS
    )
    workspace = _prototype()
    assert baseline.parameter_count() < workspace.parameter_count()
    # shared tensors carry identical initial values so the two arms differ only
    # by the workspace path (contract section 3's same-seed requirement)
    for name, parameter in baseline.named_parameters():
        if name == "decoder":
            continue
        torch.testing.assert_close(parameter, workspace.named_parameter(name), rtol=0, atol=0)
    loss, metrics = baseline.sequence_loss(*_batch()[0])
    assert bool(torch.isfinite(loss))
    assert metrics["positions"] > 0


# --------------------------------------------------------------------------- #
# Gate 2: synthetic gradient checks
# --------------------------------------------------------------------------- #


def test_gate2_every_declared_parameter_receives_finite_nonzero_gradient() -> None:
    prototype = _prototype()
    prototype.zero_grads_for_test()
    loss, _ = prototype.sequence_loss(*_batch()[0])
    loss.backward()
    for name, parameter in prototype.named_parameters():
        gradient = parameter.grad
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all()), name
        assert float(gradient.abs().sum()) > 0.0, name


def test_gate2_numerical_finite_difference_matches_autograd() -> None:
    prototype = _prototype()
    prefix, response = _batch()[0]
    prototype.zero_grads_for_test()
    loss, _ = prototype.sequence_loss(prefix, response)
    loss.backward()

    epsilon = 2e-3
    checked = 0
    for name, parameter in prototype.named_parameters():
        flat = parameter.detach().reshape(-1)
        step = max(1, flat.numel() // 3)
        for index in range(0, flat.numel(), step):
            original = float(flat[index])
            with torch.no_grad():
                flat[index] = original + epsilon
            plus, _ = prototype.sequence_loss(prefix, response)
            with torch.no_grad():
                flat[index] = original - epsilon
            minus, _ = prototype.sequence_loss(prefix, response)
            with torch.no_grad():
                flat[index] = original
            numerical = (float(plus.detach()) - float(minus.detach())) / (2.0 * epsilon)
            analytical = float(parameter.grad.reshape(-1)[index])
            assert abs(analytical - numerical) <= 2e-3 * max(1.0, abs(analytical)), (
                name,
                index,
                analytical,
                numerical,
            )
            checked += 1
    assert checked >= len(SEQUENCE_WORKSPACE_PARAMETERS)


def test_gate2_disabling_the_workspace_connection_removes_its_gradient() -> None:
    prototype = _prototype()
    prototype.zero_grads_for_test()
    state = prototype.begin_episode("问：".encode())
    loss_terms = []
    previous = prototype.config.boundary_symbol
    for symbol in "蓝".encode():
        state, logits = prototype.step(state, previous, detach_workspace_read=True)
        loss_terms.append(
            torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([int(symbol)]))
        )
        previous = int(symbol)
    torch.stack(loss_terms).mean().backward()
    for name in ("workspace_key", "workspace_value"):
        gradient = prototype.named_parameter(name).grad
        assert gradient is None or float(gradient.abs().sum()) == 0.0, name
    # the renderer path itself still receives gradient when W is detached
    assert float(prototype.named_parameter("renderer_input").grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# Gate 3: causal mask and teacher-forced equivalence
# --------------------------------------------------------------------------- #


def test_gate3_future_labels_do_not_change_current_logits() -> None:
    prototype = _prototype()
    prefix = "问：天空的颜色？".encode()
    response = "蓝色<|end|>".encode()
    full = prototype.teacher_forced_logits(prefix, response)
    cut = prototype.teacher_forced_logits(prefix, response[:3])
    torch.testing.assert_close(full[:3], cut[:3], rtol=0, atol=0)


def test_gate3_step_by_step_matches_teacher_forced() -> None:
    prototype = _prototype()
    prefix = "问：空😀".encode()
    response = "答：有".encode()
    forced = prototype.teacher_forced_logits(prefix, response)
    state = prototype.begin_episode(prefix)
    previous = prototype.config.boundary_symbol
    for index, symbol in enumerate(response):
        state, logits = prototype.step(state, previous)
        torch.testing.assert_close(logits, forced[index], rtol=0, atol=0)
        previous = int(symbol)
    _state, final_logits = prototype.step(state, previous)
    torch.testing.assert_close(final_logits, forced[len(response)], rtol=0, atol=0)


def test_gate3_empty_context_end_marker_and_max_length() -> None:
    prototype = _prototype()
    empty_logits = prototype.teacher_forced_logits(b"", b"")
    assert empty_logits.shape == (1, prototype.config.alphabet_size)
    # the end marker is a real target after every response
    _loss, metrics = prototype.sequence_loss(b"", b"a")
    assert metrics["positions"] == 2
    generated = prototype.generate(b"", max_bytes=4)
    assert generated.steps >= 1
    with_length = SequenceWorkspaceConfig(seed=7, max_sequence_bytes=8)
    long_prototype = SequenceWorkspacePrototype(with_length)
    try:
        long_prototype.teacher_forced_logits(b"x" * 9, b"")
    except ValueError as error:
        assert "truncation needs its own contract" in str(error)
    else:  # pragma: no cover - the contract demands an explicit refusal
        raise AssertionError("over-length sequence was not rejected")


# --------------------------------------------------------------------------- #
# Gate 4: zero-step save, fresh-process restore, one-step continuation
# --------------------------------------------------------------------------- #


def _logits_digest(prototype: SequenceWorkspacePrototype) -> str:
    prefix, response = _batch()[0]
    logits = prototype.teacher_forced_logits(prefix, response)
    return content_digest(logits.detach())


def test_gate4_zero_step_save_fresh_process_and_continuation(tmp_path: Path) -> None:
    prototype = _prototype()
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=0.05, code_revision="gate")
    trainer.set_episodes(_batch())
    expected_logits = _logits_digest(prototype)
    parent_path = trainer.save(tmp_path / "zero.pt")

    # restored parameters reproduce the parent logits exactly (zero-step restore)
    restored = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    assert _logits_digest(restored.prototype) == expected_logits

    # continuation after restore equals uninterrupted continuation
    uninterrupted = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(_batch())
    expected_continued = str(uninterrupted.checkpoint()["checkpoint_digest"])
    restored.train_step(_batch())
    assert str(restored.checkpoint()["checkpoint_digest"]) == expected_continued

    # a fresh process loads the same file: logits digest identical, one more
    # optimizer step lands on the same digest
    prefix, response = _batch()[0]
    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.internalization import content_digest;"
        "from taiji.sequence_workspace import SequenceWorkspaceTrainer;"
        f"payload = torch.load(r'{str(parent_path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceWorkspaceTrainer.from_checkpoint(payload);"
        f"logits = trainer.prototype.teacher_forced_logits({prefix!r}, {response!r});"
        f"trainer.train_step({list(_batch())!r});"
        "print(json.dumps({'logits': content_digest(logits.detach()),"
        " 'continued': str(trainer.checkpoint()['checkpoint_digest'])}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["logits"] == expected_logits
    assert payload["continued"] == expected_continued


# --------------------------------------------------------------------------- #
# Gate 5: wiring checks (prove connections only, never capability)
# --------------------------------------------------------------------------- #


def test_gate5_missing_parameters_and_digest_tampering_are_rejected() -> None:
    prototype = _prototype()
    trainer = SequenceWorkspaceTrainer(prototype)
    trainer.set_episodes(_batch())
    payload = trainer.checkpoint()
    broken = dict(payload)
    broken["parameters"] = dict(payload["parameters"])
    del broken["parameters"]["decoder_bias"]
    try:
        SequenceWorkspaceTrainer.from_checkpoint(broken)
    except ValueError as error:
        assert "missing" in str(error) or "digest" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing parameter was not rejected")
    tampered = dict(payload)
    tampered["cursor"] = int(payload["cursor"]) + 1
    try:
        SequenceWorkspaceTrainer.from_checkpoint(tampered)
    except ValueError as error:
        assert "digest" in str(error)
    else:  # pragma: no cover
        raise AssertionError("tampered checkpoint was not rejected")


def test_gate5_prefix_and_workspace_both_affect_the_output() -> None:
    prototype = _prototype()
    response = "蓝".encode()
    first = prototype.teacher_forced_logits("问：天空的颜色？".encode(), response)
    shuffled = prototype.teacher_forced_logits("？色颜的空天：问".encode(), response)
    assert not torch.allclose(first, shuffled), "shuffling the prefix must change the output"

    mutated = _prototype()
    with torch.no_grad():
        mutated.named_parameter("workspace_key").zero_()
        mutated.named_parameter("workspace_value").zero_()
    masked = mutated.teacher_forced_logits("问：天空的颜色？".encode(), response)
    assert not torch.allclose(first, masked), "masking the workspace must remove its contribution"
