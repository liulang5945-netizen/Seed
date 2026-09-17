"""R2-D2 implementation gates for graph v3 (per-position evidence memory).

Contract:
plans/reference/M5_R2_D2_PER_POSITION_EVIDENCE_PREREGISTRATION_FROZEN_20260918.md
section 5.1.  No capability training happens here: these gates prove the
inventory, the single-channel structure, per-position provenance, gradients,
causality, lesion wiring and checkpoint restoration of the v3 graph.  The
context-removal construction gate (5.1-7) is exercised by the implementation
gate validator script, since it operates on D1 fixture text rather than on the
prototype.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from taiji.internalization import content_digest
from taiji.sequence_workspace import (
    EVIDENCE_BROADCAST_FINAL,
    EVIDENCE_FINAL_STATE_SLOTS,
    EVIDENCE_PER_POSITION,
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _arm(source: str, seed: int = 20260917) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(SequenceWorkspaceConfig(seed=seed, evidence_source=source))


def _episode() -> tuple[bytes, bytes]:
    return ("提问：雪的颜色？线索：雪是白。回答：".encode(), "白".encode())


# --------------------------------------------------------------------------- #
# Gate 1: inventory, counts, shared initialization, default identity
# --------------------------------------------------------------------------- #


def test_gate1_v3_inventories_and_exact_parameter_counts() -> None:
    arm_a = _arm(EVIDENCE_PER_POSITION)
    arm_c1 = _arm(EVIDENCE_BROADCAST_FINAL)
    arm_c0 = _arm(EVIDENCE_FINAL_STATE_SLOTS)
    assert arm_a.parameter_count() == 82_593
    assert arm_c1.parameter_count() == 82_593
    assert arm_c0.parameter_count() == 101_025
    names_a = tuple(name for name, _ in arm_a.named_parameters())
    assert names_a == tuple(name for name, _ in arm_c1.named_parameters())
    assert "evidence_key" in names_a and "evidence_value" in names_a
    assert "workspace_key" not in names_a and "workspace_value" not in names_a
    c0_names = tuple(name for name, _ in arm_c0.named_parameters())
    assert "workspace_key" in c0_names and "workspace_value" in c0_names
    # default config is bit-for-bit the v2 graph
    defaulted = SequenceWorkspacePrototype(SequenceWorkspaceConfig())
    assert defaulted.config.evidence_source == EVIDENCE_FINAL_STATE_SLOTS
    assert defaulted.parameter_count() == 101_025
    # invalid provenance and baseline+variant combinations are rejected
    try:
        SequenceWorkspaceConfig(evidence_source="nonsense")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid evidence_source was not rejected")
    try:
        SequenceWorkspaceConfig(workspace_enabled=False, evidence_source=EVIDENCE_PER_POSITION)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("variant on the baseline arm was not rejected")


def test_gate1_shared_names_keep_identical_initial_values() -> None:
    arm_a = _arm(EVIDENCE_PER_POSITION, seed=42)
    arm_c1 = _arm(EVIDENCE_BROADCAST_FINAL, seed=42)
    arm_c0 = _arm(EVIDENCE_FINAL_STATE_SLOTS, seed=42)
    for name in arm_a.declared_parameter_names():
        torch.testing.assert_close(
            arm_a.named_parameter(name), arm_c1.named_parameter(name), rtol=0, atol=0
        )
    # tensors shared with the v2 lineage anchor keep identical values too
    for name in (
        "prefix_embedding",
        "prefix_input",
        "start_vector",
        "address_query",
        "decoder",
        "decoder_bias",
    ):
        torch.testing.assert_close(
            arm_a.named_parameter(name), arm_c0.named_parameter(name), rtol=0, atol=0
        )


# --------------------------------------------------------------------------- #
# Gate 2: single-channel structure preserved
# --------------------------------------------------------------------------- #


def test_gate2_zeroed_evidence_makes_v3_logits_prefix_invariant() -> None:
    zeroed = _arm(EVIDENCE_PER_POSITION)
    with torch.no_grad():
        zeroed.named_parameter("evidence_key").zero_()
        zeroed.named_parameter("evidence_value").zero_()
    response = "白".encode()
    left = zeroed.teacher_forced_logits("提问：雪的颜色？线索：雪是白。回答：".encode(), response)
    right = zeroed.teacher_forced_logits("完全不相干的另一段前文，长度不同。".encode(), response)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    live = _arm(EVIDENCE_PER_POSITION)
    assert not torch.allclose(
        live.teacher_forced_logits("提问：雪的颜色？线索：雪是白。回答：".encode(), response),
        live.teacher_forced_logits("完全不相干的另一段前文，长度不同。".encode(), response),
    )


# --------------------------------------------------------------------------- #
# Gate 3: per-position identity and broadcast control
# --------------------------------------------------------------------------- #


def test_gate3_per_position_entries_are_position_specific_and_causally_ordered() -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    prefix = "提问：雪的颜色？线索：雪是白。回答：".encode()
    state = prototype.begin_episode(prefix)
    key, value = state.workspace_key, state.workspace_value
    assert key.shape[0] == len(prefix) and value.shape[0] == len(prefix)
    # entries are pairwise distinguishable (no collapsed rows)
    for i in range(key.shape[0]):
        for j in range(i + 1, min(i + 4, key.shape[0])):
            assert not torch.allclose(key[i], key[j]) or not torch.allclose(value[i], value[j])
    # causal ordering: mutating byte i leaves every entry j < i bit-identical
    # and changes entry i
    mutated = bytearray(prefix)
    mutated[10] = mutated[10] ^ 0x01
    state_b = prototype.begin_episode(bytes(mutated))
    for j in range(10):
        torch.testing.assert_close(key[j], state_b.workspace_key[j], rtol=0, atol=0)
    assert not torch.allclose(key[10], state_b.workspace_key[10])
    # a query aligned with entry i's key direction addresses that entry when
    # norms are held equal (cosine selectability; production softmax uses raw
    # keys, so this probes distinctness of direction rather than length)
    i = 12
    unit = key / key.norm(dim=1, keepdim=True)
    scores = unit @ unit[i]
    assert int(scores.argmax()) == i


def test_gate3_broadcast_control_rows_are_bit_identical() -> None:
    control = _arm(EVIDENCE_BROADCAST_FINAL)
    state = control.begin_episode(_episode()[0])
    n = state.workspace_key.shape[0]
    assert n == len(_episode()[0])
    torch.testing.assert_close(
        state.workspace_key, state.workspace_key[:1].expand(n, -1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        state.workspace_value, state.workspace_value[:1].expand(n, -1), rtol=0, atol=0
    )
    # uniform addressing collapses the read to the single broadcast value
    # (softmax weights over identical scores are all 1/L; the weighted sum
    # matches the row up to floating-point accumulation order)
    episode_state = control.begin_episode(_episode()[0])
    read = control._content_read(episode_state, detach=False)
    expected = episode_state.workspace_value[0]
    torch.testing.assert_close(read, expected, rtol=1e-6, atol=1e-6)
    # and this is far from what per-position entries would read
    treatment = _arm(EVIDENCE_PER_POSITION)
    treatment_read = treatment._content_read(treatment.begin_episode(_episode()[0]), detach=False)
    assert float((treatment_read - expected).abs().max().detach()) > 1e-3


# --------------------------------------------------------------------------- #
# Gate 4: gradients, zero-read ablation, causality, teacher-forced equivalence
# --------------------------------------------------------------------------- #


def test_gate4_v3_evidence_parameters_receive_finite_nonzero_gradients() -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    prototype.zero_grads_for_test()
    loss, _ = prototype.sequence_loss(*_episode())
    loss.backward()
    for name in (
        "evidence_key",
        "evidence_value",
        "prefix_embedding",
        "prefix_recur",
        "address_query",
        "renderer_input",
    ):
        gradient = prototype.named_parameter(name).grad
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all()), name
        assert float(gradient.abs().sum()) > 0.0, name


def test_gate4_numerical_finite_difference_on_evidence_projections() -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    prefix, response = _episode()
    epsilon = 2e-3
    for name in ("evidence_key", "evidence_value"):
        parameter = prototype.named_parameter(name)
        prototype.zero_grads_for_test()
        loss, _ = prototype.sequence_loss(prefix, response)
        loss.backward()
        index = parameter.numel() // 2
        flat = parameter.detach().reshape(-1)
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
        assert abs(analytical - numerical) <= 2e-3 * max(1.0, abs(analytical)), name


def test_gate4_zero_read_removes_evidence_gradients_but_keeps_renderer_gradients() -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    prototype.zero_grads_for_test()
    state = prototype.begin_episode(_episode()[0])
    losses = []
    previous = prototype.config.boundary_symbol
    for symbol in _episode()[1]:
        state, logits = prototype.step(state, previous, zero_read=True)
        losses.append(
            torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([int(symbol)]))
        )
        previous = int(symbol)
    torch.stack(losses).mean().backward()
    for name in ("evidence_key", "evidence_value", "address_query"):
        gradient = prototype.named_parameter(name).grad
        assert gradient is None or float(gradient.abs().sum()) == 0.0, name
    assert float(prototype.named_parameter("renderer_input").grad.abs().sum()) > 0.0
    assert float(prototype.named_parameter("renderer_embedding").grad.abs().sum()) > 0.0


def test_gate4_v3_causal_mask_and_step_by_step_equivalence() -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    prefix, response = _episode()
    full = prototype.teacher_forced_logits(prefix, response + "尾".encode())
    cut = prototype.teacher_forced_logits(prefix, response)
    torch.testing.assert_close(full[: len(response) + 1], cut, rtol=0, atol=0)
    forced = prototype.teacher_forced_logits(prefix, response)
    state = prototype.begin_episode(prefix)
    previous = prototype.config.boundary_symbol
    for index, symbol in enumerate(response):
        state, logits = prototype.step(state, previous)
        torch.testing.assert_close(logits, forced[index], rtol=0, atol=0)
        previous = int(symbol)
    _state, final_logits = prototype.step(state, previous)
    torch.testing.assert_close(final_logits, forced[len(response)], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Gate 5: lesion wiring
# --------------------------------------------------------------------------- #


def test_gate5_misbind_rotation_is_deterministic_and_preserves_keys_and_content() -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    prefix = _episode()[0]
    intact = prototype.begin_episode(prefix)
    rotation = len(prefix) // 2
    first = prototype.begin_episode(prefix, value_rotation=rotation)
    second = prototype.begin_episode(prefix, value_rotation=rotation)
    # deterministic across calls
    torch.testing.assert_close(first.workspace_value, second.workspace_value, rtol=0, atol=0)
    torch.testing.assert_close(first.workspace_key, intact.workspace_key, rtol=0, atol=0)
    assert not torch.allclose(first.workspace_value, intact.workspace_value)
    # content is preserved: rotated rows are the same multiset, reordered
    intact_rows = [tuple(row.tolist()) for row in intact.workspace_value]
    rotated_rows = [tuple(row.tolist()) for row in first.workspace_value]
    assert sorted(intact_rows) == sorted(rotated_rows)
    # zero rotation is the intact graph
    none = prototype.begin_episode(prefix, value_rotation=0)
    torch.testing.assert_close(none.workspace_value, intact.workspace_value, rtol=0, atol=0)


def test_gate5_lesions_rewire_the_read_at_logit_level() -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    boundary = int(prototype.config.boundary_symbol)
    prefix = _episode()[0]

    intact_state = prototype.begin_episode(prefix)
    _state, intact_logits = prototype.step(intact_state, boundary)
    misbind_state = prototype.begin_episode(prefix, value_rotation=len(prefix) // 2)
    _state, misbind_logits = prototype.step(misbind_state, boundary)
    _state, zero_logits = prototype.step(intact_state, boundary, zero_read=True)

    assert not torch.allclose(misbind_logits, intact_logits)
    assert not torch.allclose(zero_logits, intact_logits)
    # zero-read is prefix-invariant: two prefixes produce the same first logits
    other_state = prototype.begin_episode("另一段完全不同的问题与线索。".encode())
    _state, other_zero_logits = prototype.step(other_state, boundary, zero_read=True)
    torch.testing.assert_close(zero_logits, other_zero_logits, rtol=0, atol=0)
    # and the public generation path accepts both lesions
    generated = prototype.generate(
        prefix, max_bytes=6, value_rotation=len(prefix) // 2, zero_read=False
    )
    assert generated.steps >= 1


# --------------------------------------------------------------------------- #
# Gate 6: checkpoint dual-version restoration
# --------------------------------------------------------------------------- #


def _logits_digest(prototype: SequenceWorkspacePrototype) -> str:
    prefix, response = _episode()
    return content_digest(prototype.teacher_forced_logits(prefix, response).detach())


def test_gate6_v3_roundtrip_restore_and_continuation(tmp_path: Path) -> None:
    prototype = _arm(EVIDENCE_PER_POSITION)
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=0.05, code_revision="d2gate")
    batch = (_episode(), ("提问：太阳的颜色？线索：太阳是红。回答：".encode(), "红".encode()))
    trainer.set_episodes(batch)
    assert int(trainer.checkpoint()["version"]) == SEQUENCE_WORKSPACE_VERSION
    expected = _logits_digest(prototype)
    path = trainer.save(tmp_path / "v3.pt")
    restored = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    assert restored.prototype.config.evidence_source == EVIDENCE_PER_POSITION
    assert _logits_digest(restored.prototype) == expected
    uninterrupted = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(batch)
    continued = str(uninterrupted.checkpoint()["checkpoint_digest"])
    restored.train_step(batch)
    assert str(restored.checkpoint()["checkpoint_digest"]) == continued

    # fresh-process load of the saved file
    prefix, response = _episode()
    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.internalization import content_digest;"
        "from taiji.sequence_workspace import SequenceWorkspaceTrainer;"
        f"payload = torch.load(r'{str(path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceWorkspaceTrainer.from_checkpoint(payload);"
        f"logits = trainer.prototype.teacher_forced_logits({prefix!r}, {response!r});"
        f"trainer.train_step({list(batch)!r});"
        "print(json.dumps({'logits': content_digest(logits.detach()),"
        " 'continued': str(trainer.checkpoint()['checkpoint_digest']),"
        " 'source': trainer.prototype.config.evidence_source}))"
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
    assert payload["logits"] == expected
    assert payload["continued"] == continued
    assert payload["source"] == EVIDENCE_PER_POSITION


def test_gate6_version_two_payloads_still_load_and_version_one_refused(tmp_path: Path) -> None:
    # a v2-arm writer emits current-version payloads carrying the v2 graph;
    # simulate the historical v2 envelope and confirm the dual read path
    v2 = SequenceWorkspaceTrainer(_arm(EVIDENCE_FINAL_STATE_SLOTS), code_revision="v2graph")
    v2.set_episodes((_episode(),))
    payload = v2.checkpoint()
    assert payload["config"]["evidence_source"] == EVIDENCE_FINAL_STATE_SLOTS
    historical = dict(payload)
    historical["version"] = 2
    historical["checkpoint_digest"] = content_digest(
        {key: value for key, value in historical.items() if key != "checkpoint_digest"}
    )
    restored = SequenceWorkspaceTrainer.from_checkpoint(historical)
    assert restored.prototype.config.evidence_source == EVIDENCE_FINAL_STATE_SLOTS
    assert _logits_digest(restored.prototype) == _logits_digest(v2.prototype)
    ancient = dict(historical)
    ancient["version"] = 1
    ancient["checkpoint_digest"] = content_digest(
        {key: value for key, value in ancient.items() if key != "checkpoint_digest"}
    )
    try:
        SequenceWorkspaceTrainer.from_checkpoint(ancient)
    except ValueError as error:
        assert "version" in str(error)
    else:  # pragma: no cover
        raise AssertionError("version-1 checkpoint was not refused")
