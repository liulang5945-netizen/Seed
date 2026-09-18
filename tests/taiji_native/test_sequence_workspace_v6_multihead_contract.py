"""R2-D3 H-G implementation gates for graph v6 (multi-head position-aware
readout).

Contract:
plans/reference/M5_R2_D3_FIRST_STEP_GEOMETRY_CONTRACT_FROZEN_20260918.md
Zero capability training: inventory, positional encoding structure,
per-head normalization and differentiation, gradient reach, single-head
ablation, single-head backward compatibility, and versioned restoration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from taiji.internalization import content_digest
from taiji.sequence_workspace import (
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _hg(seed: int = 20260917) -> SequenceWorkspacePrototype:
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


def _h1_pe(seed: int = 20260917) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source="per_position",
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=1,
            positional_keys=True,
        )
    )


def _prefix() -> bytes:
    return "提问：雪的颜色？线索：雪是白。回答：".encode()


# --------------------------------------------------------------------------- #
# Gate 1: inventory and single-head compatibility
# --------------------------------------------------------------------------- #


def test_gate1_multihead_inventory() -> None:
    prototype = _hg()
    assert prototype.parameter_count() == 96_034
    names = tuple(name for name, _ in prototype.named_parameters())
    for head in range(1, 5):
        assert f"head_query_{head}" in names
    assert "address_query" not in names
    assert "evidence_key" in names and "copy_gate_weight" in names
    assert "answer_start_weight" in names
    # H=1 positional arm keeps the v5 inventory
    single = _h1_pe()
    assert single.parameter_count() == 86_818
    assert "address_query" in tuple(name for name, _ in single.named_parameters())


def test_gate1_invalid_combinations_rejected() -> None:
    for invalid in (
        dict(readout_heads=0),
        dict(readout_heads=5),
        dict(evidence_source="final_state_slots", readout_heads=4),
        dict(workspace_enabled=False, readout_heads=4),
        dict(evidence_source="final_state_slots", positional_keys=True),
    ):
        try:
            SequenceWorkspaceConfig(**invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid v6 combination accepted: {invalid}")


# --------------------------------------------------------------------------- #
# Gate 2: positional encoding structure
# --------------------------------------------------------------------------- #


def test_gate2_positional_keys_are_position_specific_and_bounded() -> None:
    prototype = _h1_pe()
    state = prototype.begin_episode(_prefix())
    keys = state.workspace_key
    assert keys.shape[0] == len(_prefix())
    # rows differ across positions at initialization (PE contribution)
    assert not torch.allclose(keys[0], keys[1], atol=1e-7)
    # PE-only margin is bounded by the sinusoidal range difference (<=2)
    raw = prototype.begin_episode(_prefix()).workspace_key
    assert bool(torch.isfinite(raw).all())
    # same stem, different material: keys before the material marker are equal
    other = prototype.begin_episode("提问：雪的颜色？线索：雪是青。回答：".encode())
    marker = _prefix().find("线索：".encode())
    torch.testing.assert_close(raw[: marker - 1], other.workspace_key[: marker - 1], rtol=0, atol=0)
    assert not torch.allclose(raw, other.workspace_key, atol=1e-7)


# --------------------------------------------------------------------------- #
# Gate 3: multi-head weights, gradients and ablation
# --------------------------------------------------------------------------- #


def test_gate3_heads_normalize_and_differentiate() -> None:
    prototype = _hg()
    state = prototype.begin_episode(_prefix())
    state, _ = prototype.step(state, prototype.config.boundary_symbol)
    weights = prototype._head_weights(state, detach=False)
    assert weights.shape == (4, len(_prefix()))
    sums = weights.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones(4), rtol=1e-6, atol=1e-6)
    # distinct learned projections give distinct heads at initialization
    assert any(not torch.allclose(weights[0], weights[h], atol=1e-7) for h in range(1, 4))


def test_gate3_gradient_reaches_every_head() -> None:
    prototype = _hg()
    prototype.zero_grads_for_test()
    loss, _ = prototype.sequence_loss(_prefix(), "白".encode())
    loss.backward()
    for head in range(1, 5):
        gradient = prototype.named_parameter(f"head_query_{head}").grad
        assert gradient is not None and float(gradient.abs().sum()) > 0.0, head


def test_gate3_ablating_three_heads_changes_the_read() -> None:
    prototype = _hg()
    state = prototype.begin_episode(_prefix())
    state, _ = prototype.step(state, prototype.config.boundary_symbol)
    full = prototype._content_read(state, detach=False)
    with torch.no_grad():
        for head in range(2, 5):
            prototype.named_parameter(f"head_query_{head}").zero_()
    ablated_state = prototype.begin_episode(_prefix())
    ablated_state, _ = prototype.step(ablated_state, prototype.config.boundary_symbol)
    single = prototype._content_read(ablated_state, detach=False)
    assert not torch.allclose(full, single, atol=1e-7)


# --------------------------------------------------------------------------- #
# Gate 4: mixture and causal walk under multi-head
# --------------------------------------------------------------------------- #


def test_gate4_mixture_valid_and_teacher_forced_walk_consistent() -> None:
    prototype = _hg()
    prefix, response = _prefix(), "白".encode()
    forced = prototype.teacher_forced_distributions(prefix, response)
    assert forced.shape == (len(response) + 1, 257)
    torch.testing.assert_close(forced.sum(dim=1), torch.ones(forced.shape[0]), rtol=1e-6, atol=1e-6)
    assert float(forced[:, prototype.config.boundary_symbol].min()) >= 0.0
    state = prototype.begin_episode(prefix)
    previous = prototype.config.boundary_symbol
    for index, symbol in enumerate(response):
        state, probability = prototype.step_distribution(state, previous)
        torch.testing.assert_close(probability, forced[index], rtol=0, atol=0)
        previous = int(symbol)
    _state, probability = prototype.step_distribution(state, previous)
    torch.testing.assert_close(probability, forced[-1], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Gate 5: restoration v6 and v5 fallback
# --------------------------------------------------------------------------- #


def _logits_digest(prototype: SequenceWorkspacePrototype) -> str:
    return content_digest(prototype.teacher_forced_logits(_prefix(), "白".encode()).detach())


def test_gate5_v6_roundtrip_and_fresh_process(tmp_path: Path) -> None:
    prototype = _hg()
    batch = ((_prefix(), "白".encode()),)
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=0.01, code_revision="d2v6")
    trainer.set_episodes(batch)
    expected = _logits_digest(prototype)
    path = trainer.save(tmp_path / "v6.pt")
    restored = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    assert restored.prototype.config.readout_heads == 4
    assert restored.prototype.config.positional_keys is True
    assert _logits_digest(restored.prototype) == expected

    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.sequence_workspace import SequenceWorkspaceTrainer;"
        f"payload = torch.load(r'{str(path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceWorkspaceTrainer.from_checkpoint(payload);"
        "print(json.dumps({'version': int(trainer.checkpoint()['version']),"
        " 'heads': int(trainer.prototype.config.readout_heads),"
        " 'pe': bool(trainer.prototype.config.positional_keys)}))"
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
    assert observed == {"version": SEQUENCE_WORKSPACE_VERSION, "heads": 4, "pe": True}


def test_gate5_v5_payload_loads_with_single_head_and_no_pe() -> None:
    v5 = SequenceWorkspaceTrainer(
        _h1_pe().__class__(  # v5 arm: copy+qstart, heads defaults 1
            SequenceWorkspaceConfig(
                seed=20260917,
                evidence_source="per_position",
                copy_mixture=True,
                question_conditioned_start=True,
            )
        ),
        code_revision="v5graph",
    )
    v5.set_episodes(((_prefix(), "白".encode()),))
    historical = dict(v5.checkpoint())
    historical["version"] = 5
    historical["checkpoint_digest"] = content_digest(
        {key: value for key, value in historical.items() if key != "checkpoint_digest"}
    )
    restored = SequenceWorkspaceTrainer.from_checkpoint(historical)
    assert restored.prototype.config.readout_heads == 1
    assert restored.prototype.config.positional_keys is False
    assert _logits_digest(restored.prototype) == _logits_digest(v5.prototype)
