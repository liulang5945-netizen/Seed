"""R2 content-binding v2 implementation gates: pair-contrastive objective.

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_V2_FROZEN_20260919.md §D1/§D3

Gates on the single v2 design variable (group-counterfactual contrastive
auxiliary): margin mathematics (equal-answer groups degenerate to a constant
with zero gradient; flipped members produce the correct sign), gradient reach
into both arms' content modules, group-structure enforcement, checkpoint
identity (v2 revision recorded; v1 payloads stay loadable), and a behavioural
sanity check that the pair term decreases under training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from taiji.sequence_char_workspace import CharVocab
from taiji.sequence_content_workspace import (
    SequenceContentConfig,
    SequenceContentTrainer,
    SequenceContentWorkspace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_content_binding_v1_train.jsonl"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in TRAIN_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def vocab() -> CharVocab:
    rows = _rows()
    return CharVocab("".join(r["question"] + r["material"] + r["response"] for r in rows))


def _group_batch(rows: list[dict], cls: str) -> list[dict]:
    """Two whole members of one group, carrying group_id (v2 batch shape)."""

    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["group_class"] == cls:
            groups.setdefault(row["group_id"], []).append(row)
    members = next(iter(groups.values()))
    return [
        {
            "group_id": members[0]["group_id"],
            "question": members[0]["question"],
            "material": members[0]["material"],
            "response": members[0]["response"],
            "copy_mask": members[0]["copy_mask"],
        },
        {
            "group_id": members[1]["group_id"],
            "question": members[1]["question"],
            "material": members[1]["material"],
            "response": members[1]["response"],
            "copy_mask": members[1]["copy_mask"],
        },
    ]


def _trainer(vocab: CharVocab, arm: str = "B", seed: int = 17) -> SequenceContentTrainer:
    workspace = SequenceContentWorkspace(vocab, SequenceContentConfig(arm=arm, seed=seed))
    trainer = SequenceContentTrainer(workspace, learning_rate=0.001, total_updates=10)
    trainer.enable_pair_contrastive(weight=1.0, margin=1.0)
    return trainer


# --------------------------------------------------------------------------- #
# Gate 1: margin mathematics (contract section D1)
# --------------------------------------------------------------------------- #


def test_gate1_margin_math_on_floats() -> None:
    """Hinge: softplus(s_crossed - s_own + gamma); satisfied margin leaves the
    softplus(gamma) residual, violated margins grow linearly."""

    def hinge(s_own: float, s_crossed: float, gamma: float = 1.0) -> float:
        return float(
            torch.nn.functional.softplus(torch.tensor(s_crossed - s_own + gamma))
        )

    assert hinge(-0.5, -2.5) == pytest.approx(0.313262)  # softplus(-1): margin partly kept
    assert hinge(0.0, -5.0) == pytest.approx(0.018149927)  # softplus(-4): own wins big
    assert hinge(-5.0, 0.0) == pytest.approx(6.002476)  # softplus(6): violated, ~linear
    assert hinge(-5.0, 0.0) > hinge(0.0, -5.0)  # monotone in the violation


def test_gate1_equal_answer_group_degenerates_to_constant(vocab) -> None:
    """distractor_invariant members share the same answer string: s_yx == s_xx
    so both hinges are constants (softplus(gamma)) with zero gradient."""

    rows = _rows()
    batch = _group_batch(rows, "distractor_invariant")
    assert batch[0]["response"] == batch[1]["response"]
    for arm in ("A", "B"):
        trainer = _trainer(vocab, arm=arm)
        s_xx, _ = trainer.workspace.sequence_loglik(
            batch[0]["question"], batch[0]["material"], batch[0]["response"]
        )
        s_yx, _ = trainer.workspace.sequence_loglik(
            batch[0]["question"], batch[0]["material"], batch[1]["response"]
        )
        assert torch.allclose(s_yx, s_xx, atol=1e-6)
        pair_term, groups = trainer._pair_contrastive_term(batch)
        assert groups == 1
        grads = torch.autograd.grad(
            pair_term, trainer.workspace.parameters(), allow_unused=True
        )
        for grad in grads:
            if grad is not None:
                # two identical graphs cancel; float noise stays below 1e-4
                assert float(grad.abs().sum()) < 1e-4
        # structural: identical answer strings -> the term equals 2*softplus(gamma)
        assert float(pair_term.detach()) == pytest.approx(
            2 * float(torch.nn.functional.softplus(torch.tensor(1.0))), abs=1e-4
        )


# --------------------------------------------------------------------------- #
# Gate 2: gradient reach into the content modules (contract section D3)
# --------------------------------------------------------------------------- #


def test_gate2_pair_gradients_reach_content_modules(vocab) -> None:
    rows = _rows()
    batch = _group_batch(rows, "object_swap")
    for arm, names in (
        ("B", ["slot_init", "relation_mlp_input", "copy_query_content"]),
        ("A", ["head_query_1", "a_content_mlp_input", "copy_query_content"]),
    ):
        trainer = _trainer(vocab, arm=arm)
        pair_term, _ = trainer._pair_contrastive_term(batch)
        grads = torch.autograd.grad(
            pair_term,
            [trainer.workspace._parameters[name] for name in names],
            allow_unused=True,
        )
        for name, grad in zip(names, grads):
            assert grad is not None, (arm, name)
            assert torch.isfinite(grad).all(), (arm, name)


def test_gate2_train_step_includes_pair_term(vocab) -> None:
    rows = _rows()
    batch = _group_batch(rows, "fact_flip") + _group_batch(rows, "relation_flip")
    trainer = _trainer(vocab)
    stats = trainer.train_step(batch)
    assert stats["pair_contrastive"] >= 0.0
    assert stats["loss"] > 0.0
    assert trainer.trainer_revision == "v2-pair-contrastive"


# --------------------------------------------------------------------------- #
# Gate 3: group-structure enforcement (contract section D1)
# --------------------------------------------------------------------------- #


def test_gate3_batch_structure_enforced(vocab) -> None:
    rows = _rows()
    batch = _group_batch(rows, "fact_flip")
    trainer = _trainer(vocab)
    broken = [dict(batch[0]), dict(batch[1])]
    broken[1]["group_id"] = "other-group"
    with pytest.raises(ValueError):
        trainer.train_step(broken)
    single = [dict(batch[0])]
    with pytest.raises(ValueError):
        trainer.train_step(single)
    no_id = [{k: v for k, v in batch[0].items() if k != "group_id"}]
    with pytest.raises(ValueError):
        trainer.train_step(no_id)


# --------------------------------------------------------------------------- #
# Gate 4: checkpoint identity and backward compatibility (contract section D3)
# --------------------------------------------------------------------------- #


def test_gate4_v2_checkpoint_roundtrip(vocab, tmp_path) -> None:
    rows = _rows()
    batch = _group_batch(rows, "fact_flip")
    trainer = _trainer(vocab)
    trainer.train_step(batch)
    path = trainer.save(tmp_path / "v2.pt")
    restored = SequenceContentTrainer.from_checkpoint(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    assert restored.trainer_revision == "v2-pair-contrastive"
    assert restored.pair_contrastive_weight == pytest.approx(1.0)
    assert restored.pair_contrastive_margin == pytest.approx(1.0)
    stats = restored.train_step(batch)
    assert stats["pair_contrastive"] >= 0.0


def test_gate4_v1_payload_still_loads(vocab, tmp_path) -> None:
    """A v1-style payload without the pair fields loads with the objective off."""

    rows = _rows()
    item = next(r for r in rows if r["group_class"] == "fact_flip")
    workspace = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=5))
    trainer = SequenceContentTrainer(workspace, learning_rate=0.001, total_updates=5)
    trainer.train_step(
        [
            {
                "question": item["question"],
                "material": item["material"],
                "response": item["response"],
                "copy_mask": item["copy_mask"],
            }
        ]
    )
    payload = trainer.checkpoint()
    # simulate a v1-era payload: strip the v2 fields and re-sign the body
    from taiji.internalization import content_digest

    legacy = {
        k: v
        for k, v in payload.items()
        if k not in (
            "trainer_revision",
            "pair_contrastive_weight",
            "pair_contrastive_margin",
            "checkpoint_digest",
        )
    }
    legacy["checkpoint_digest"] = content_digest(legacy)
    restored = SequenceContentTrainer.from_checkpoint(legacy)
    assert restored.trainer_revision == "v1"
    assert restored.pair_contrastive_weight == 0.0


# --------------------------------------------------------------------------- #
# Gate 5: behavioural sanity -- the pair term decreases under training
# --------------------------------------------------------------------------- #


def test_gate5_gradient_step_reduces_pair_term(vocab) -> None:
    """Isolate the auxiliary objective: one first-order step on the pair term
    alone must reduce it (the objective is locally tractable)."""

    rows = _rows()
    batch = _group_batch(rows, "object_swap")
    workspace = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=17))
    trainer = SequenceContentTrainer(workspace, learning_rate=0.003, total_updates=50)
    trainer.enable_pair_contrastive(weight=1.0, margin=1.0)
    before, _ = trainer._pair_contrastive_term(batch)
    optimizer = torch.optim.SGD(trainer.workspace.parameters(), lr=0.003)
    optimizer.zero_grad(set_to_none=True)
    before.backward()
    optimizer.step()
    after, _ = trainer._pair_contrastive_term(batch)
    assert float(after) < float(before), (float(before), float(after))
