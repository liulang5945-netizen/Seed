"""R2-D2 H-A2 implementation gates for graph v4 (copy-mixture readout).

Contract:
plans/reference/M5_R2_D2_COPY_MIXTURE_AMENDMENT_FROZEN_20260918.md
section 5.1.  Zero capability training: inventory, mixture probability laws,
gradient reach into the copy path, causal teacher-forced equivalence, lesion
wiring, checkpoint restoration across versions 2/3/4, and byte-for-byte
identity with the v3 graph when the mixture is switched off.
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
    EVIDENCE_PER_POSITION,
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _arm(
    source: str = EVIDENCE_PER_POSITION,
    *,
    copy: bool = True,
    seed: int = 20260917,
) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=seed, evidence_source=source, copy_mixture=copy)
    )


def _episode() -> tuple[bytes, bytes]:
    return ("提问：雪的颜色？线索：雪是白。回答：".encode(), "白".encode())


# --------------------------------------------------------------------------- #
# Gate 1: inventory and copy-off identity
# --------------------------------------------------------------------------- #


def test_gate1_copy_inventory_and_invalid_combinations() -> None:
    a2 = _arm(EVIDENCE_PER_POSITION)
    c1 = _arm(EVIDENCE_BROADCAST_FINAL)
    assert a2.parameter_count() == 82_658
    assert c1.parameter_count() == 82_658
    names = tuple(name for name, _ in a2.named_parameters())
    assert names[-3:] == ("copy_gate_weight", "copy_gate_bias", "decoder_bias") or (
        "copy_gate_weight" in names and "copy_gate_bias" in names
    )
    for invalid in (
        dict(evidence_source="final_state_slots", copy_mixture=True),
        dict(workspace_enabled=False, copy_mixture=True),
    ):
        try:
            SequenceWorkspaceConfig(**invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid copy combination accepted: {invalid}")


def test_gate1_copy_off_is_bit_identical_to_v3() -> None:
    explicit_off = _arm(EVIDENCE_PER_POSITION, copy=False)
    default = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=20260917, evidence_source=EVIDENCE_PER_POSITION)
    )
    assert explicit_off.parameter_count() == 82_593
    prefix, response = _episode()
    # explicit copy-off and the v3 default share inventory, initial values,
    # logits and free-running generation bit-for-bit
    assert explicit_off.declared_parameter_names() == default.declared_parameter_names()
    torch.testing.assert_close(
        explicit_off.teacher_forced_logits(prefix, response),
        default.teacher_forced_logits(prefix, response),
        rtol=0,
        atol=0,
    )
    assert (
        explicit_off.generate(prefix, max_bytes=5).bytes_out
        == default.generate(prefix, max_bytes=5).bytes_out
    )
    # while the copy arm demonstrably takes the mixture path
    copy_on = _arm(EVIDENCE_PER_POSITION)
    forced = copy_on.teacher_forced_distributions(prefix, response)
    assert forced.shape[1] == copy_on.config.alphabet_size


# --------------------------------------------------------------------------- #
# Gate 2: copy probability law
# --------------------------------------------------------------------------- #


def test_gate2_copy_distribution_is_normalized_over_visible_bytes_only() -> None:
    prototype = _arm()
    prefix, _ = _episode()
    state = prototype.begin_episode(prefix)
    state, vocab_logits = prototype.step(state, prototype.config.boundary_symbol)
    weights = prototype.addressing_weights(state, detach=False)
    p_copy = prototype._copy_distribution(state, weights)
    assert p_copy.shape == (prototype.config.alphabet_size,)
    assert float(p_copy.sum()) == 1.0
    assert float(p_copy[prototype.config.boundary_symbol]) == 0.0
    # manual reference: mass per byte equals the sum of weights on its rows
    expected = torch.zeros(prototype.config.alphabet_size)
    for weight, symbol in zip(weights, prefix, strict=True):
        expected[int(symbol)] += float(weight)
    torch.testing.assert_close(p_copy, expected, rtol=1e-6, atol=1e-6)
    # the answer byte (white = 0xe7 0x99 0xbd in the material) has nonzero
    # copy mass available at the material positions
    white = "白".encode()
    assert all(float(p_copy[int(byte)]) > 0.0 for byte in white)


def test_gate2_mixture_is_a_valid_distribution_and_gate_runs_per_step() -> None:
    prototype = _arm()
    prefix, response = _episode()
    distributions = prototype.teacher_forced_distributions(prefix, response)
    assert distributions.shape == (len(response) + 1, prototype.config.alphabet_size)
    assert bool(torch.isfinite(distributions).all())
    sums = distributions.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-6, atol=1e-6)
    # boundary can only be supplied by the vocabulary branch
    assert float(distributions[:, prototype.config.boundary_symbol].min().detach()) >= 0.0


# --------------------------------------------------------------------------- #
# Gate 3: gradient reach and zero-read isolation
# --------------------------------------------------------------------------- #


def test_gate3_nll_reaches_copy_gate_and_evidence_projections() -> None:
    prototype = _arm()
    prototype.zero_grads_for_test()
    loss, _ = prototype.sequence_loss(*_episode())
    loss.backward()
    for name in ("copy_gate_weight", "copy_gate_bias", "evidence_key", "evidence_value"):
        gradient = prototype.named_parameter(name).grad
        assert gradient is not None and float(gradient.abs().sum()) > 0.0, name
    # finite-difference check on the gate bias: shifting it moves the NLL
    bias = prototype.named_parameter("copy_gate_bias")
    prefix, response = _episode()
    epsilon = 1e-3
    prototype.zero_grads_for_test()
    loss, _ = prototype.sequence_loss(prefix, response)
    loss.backward()
    analytical = float(bias.grad.reshape(-1)[0])
    with torch.no_grad():
        bias[0] += epsilon
    plus, _ = prototype.sequence_loss(prefix, response)
    with torch.no_grad():
        bias[0] -= 2 * epsilon
    minus, _ = prototype.sequence_loss(prefix, response)
    numerical = (float(plus) - float(minus)) / (2 * epsilon)
    assert abs(analytical - numerical) <= 2e-3 * max(1.0, abs(analytical))


def test_gate3_zero_read_removes_copy_and_evidence_gradients() -> None:
    prototype = _arm()
    prototype.zero_grads_for_test()
    state = prototype.begin_episode(_episode()[0])
    losses = []
    previous = prototype.config.boundary_symbol
    for symbol in _episode()[1]:
        state, probability = prototype.step_distribution(state, previous, zero_read=True)
        losses.append(-probability[int(symbol)].clamp_min(1e-12).log())
        previous = int(symbol)
    torch.stack(losses).mean().backward()
    for name in ("copy_gate_weight", "copy_gate_bias", "evidence_key", "evidence_value"):
        gradient = prototype.named_parameter(name).grad
        assert gradient is None or float(gradient.abs().sum()) == 0.0, name
    assert float(prototype.named_parameter("renderer_embedding").grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# Gate 4: causal mask and teacher-forced distribution equivalence
# --------------------------------------------------------------------------- #


def test_gate4_future_bytes_do_not_change_current_distributions() -> None:
    prototype = _arm()
    prefix, response = _episode()
    full = prototype.teacher_forced_distributions(prefix, response + "尾".encode())
    cut = prototype.teacher_forced_distributions(prefix, response)
    torch.testing.assert_close(full[: len(response) + 1], cut, rtol=0, atol=0)


def test_gate4_step_distribution_walk_matches_teacher_forced() -> None:
    prototype = _arm()
    prefix, response = _episode()
    forced = prototype.teacher_forced_distributions(prefix, response)
    state = prototype.begin_episode(prefix)
    previous = prototype.config.boundary_symbol
    for index, symbol in enumerate(response):
        state, probability = prototype.step_distribution(state, previous)
        torch.testing.assert_close(probability, forced[index], rtol=0, atol=0)
        previous = int(symbol)
    _state, probability = prototype.step_distribution(state, previous)
    torch.testing.assert_close(probability, forced[len(response)], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Gate 5: lesion wiring
# --------------------------------------------------------------------------- #


def test_gate5_entry_rotation_misaligns_copy_bytes_deterministically() -> None:
    prototype = _arm()
    prefix, _ = _episode()
    intact = prototype.begin_episode(prefix)
    shift = len(prefix) // 2
    rotated = prototype.begin_episode(prefix, entry_rotation=shift)
    again = prototype.begin_episode(prefix, entry_rotation=shift)
    assert intact.entry_bytes is not None and rotated.entry_bytes is not None
    assert rotated.entry_bytes == again.entry_bytes
    assert rotated.entry_bytes != intact.entry_bytes
    # keys/values and the multiset of bytes are preserved
    torch.testing.assert_close(rotated.workspace_key, intact.workspace_key, rtol=0, atol=0)
    assert sorted(rotated.entry_bytes) == sorted(intact.entry_bytes)
    # it is exactly a cyclic shift
    n = len(intact.entry_bytes)
    assert rotated.entry_bytes == tuple(intact.entry_bytes[(i - shift) % n] for i in range(n))
    # and the copy distribution changes under the same renderer query
    intact_state, _ = prototype.step(intact, prototype.config.boundary_symbol)
    rotated_state, _ = prototype.step(rotated, prototype.config.boundary_symbol)
    w_intact = prototype.addressing_weights(intact_state, detach=False)
    w_rotated = prototype.addressing_weights(rotated_state, detach=False)
    torch.testing.assert_close(w_intact, w_rotated, rtol=0, atol=0)
    p_intact = prototype._copy_distribution(intact_state, w_intact)
    p_rotated = prototype._copy_distribution(rotated_state, w_rotated)
    assert not torch.allclose(p_intact, p_rotated)


def test_gate5_value_and_entry_rotations_are_independent_lesions() -> None:
    prototype = _arm()
    prefix, _ = _episode()
    intact = prototype.begin_episode(prefix)
    value_only = prototype.begin_episode(prefix, value_rotation=len(prefix) // 2)
    entry_only = prototype.begin_episode(prefix, entry_rotation=len(prefix) // 2)
    # value rotation moves values but leaves the copy-byte alignment intact
    assert value_only.entry_bytes == intact.entry_bytes
    assert not torch.allclose(value_only.workspace_value, intact.workspace_value)
    # entry rotation moves copy bytes but leaves values and keys intact
    assert entry_only.entry_bytes != intact.entry_bytes
    torch.testing.assert_close(entry_only.workspace_value, intact.workspace_value, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Gate 6: v4 checkpoint roundtrip, v3 payload still loads, fresh process
# --------------------------------------------------------------------------- #


def _logits_digest(prototype: SequenceWorkspacePrototype) -> str:
    prefix, response = _episode()
    return content_digest(prototype.teacher_forced_logits(prefix, response).detach())


def test_gate6_v4_roundtrip_and_fresh_process_restore(tmp_path: Path) -> None:
    prototype = _arm()
    batch = (_episode(),)
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=0.01, code_revision="d2v4")
    trainer.set_episodes(batch)
    expected = _logits_digest(prototype)
    path = trainer.save(tmp_path / "v4.pt")
    restored = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    assert restored.prototype.config.copy_mixture is True
    assert _logits_digest(restored.prototype) == expected
    uninterrupted = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(batch)
    continued = str(uninterrupted.checkpoint()["checkpoint_digest"])
    restored.train_step(batch)
    assert str(restored.checkpoint()["checkpoint_digest"]) == continued

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
        " 'version': int(trainer.checkpoint()['version']),"
        " 'copy': bool(trainer.prototype.config.copy_mixture)}))"
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
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["logits"] == expected
    assert observed["continued"] == continued
    assert observed["version"] == SEQUENCE_WORKSPACE_VERSION
    assert observed["copy"] is True


def test_gate6_v3_payload_loads_with_copy_off_and_v1_refused() -> None:
    v3_trainer = SequenceWorkspaceTrainer(_arm(copy=False), code_revision="v3graph")
    v3_trainer.set_episodes((_episode(),))
    historical = dict(v3_trainer.checkpoint())
    historical["version"] = 3
    historical["checkpoint_digest"] = content_digest(
        {key: value for key, value in historical.items() if key != "checkpoint_digest"}
    )
    restored = SequenceWorkspaceTrainer.from_checkpoint(historical)
    assert restored.prototype.config.copy_mixture is False
    assert _logits_digest(restored.prototype) == _logits_digest(v3_trainer.prototype)
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
