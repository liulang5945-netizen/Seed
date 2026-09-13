from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts.training.verify_taiji_e4_artifact_internalization import build_fixture, run_gate
from taiji import (
    ArtifactInternalizationTrainer,
    ArtifactKnowledgeEncoder,
    SemanticArtifactKnowledgeEncoder,
)
from taiji.internalization import content_digest


def test_e4_artifact_internalization_gate_passes() -> None:
    result = run_gate()

    assert result["gate"] == "taiji-e4-artifact-internalization"
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    report = result["report"]
    assert report["admitted"] is True
    assert report["procedural_holdout_accuracy"] > report["procedural_lesion_holdout_accuracy"]
    assert report["affordance_native_holdout_mse"] < report["affordance_frozen_holdout_mse"]


def test_artifact_encoder_excludes_source_identity_across_scope() -> None:
    train_artifacts, holdout_artifacts, *_ = build_fixture()
    train = next(item for item in train_artifacts if item.unit_kind == "procedure")
    holdout = next(item for item in holdout_artifacts if item.unit_kind == "procedure")
    encoder = ArtifactKnowledgeEncoder(feature_dim=64)

    assert encoder.encode(train).equal(encoder.encode(holdout))


def test_artifact_internalization_checkpoint_is_tamper_evident_and_boundary_is_strict() -> None:
    train_artifacts, *_ = build_fixture()
    trainer = ArtifactInternalizationTrainer()
    checkpoint = trainer.checkpoint()
    restored = ArtifactInternalizationTrainer.from_checkpoint(checkpoint)
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)

    tampered = deepcopy(checkpoint)
    tampered["revision"] = 7
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        ArtifactInternalizationTrainer.from_checkpoint(tampered)

    with pytest.raises(ValueError, match="admitted"):
        trainer.consolidate(
            (train_artifacts[0].with_status("candidate"),),
            holdout_artifacts=train_artifacts[1:2],
            retention_artifacts=train_artifacts[2:3],
            train_experiences=(),
            holdout_experiences=(),
            retention_experiences=(),
        )


class _FakeEmbedder:
    """Duck-typed embedder instrument (keeps this test free of transformers)."""

    model_id = "fake/anchor-model"
    revision = "fake-rev-1"
    config_digest = "fake-config-digest"
    dimension = 8

    def embed(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros((len(texts), self.dimension))


def test_semantic_encoder_requires_injected_embedder() -> None:
    """DEBT-A1/A2: taiji must not construct an HF-backed embedder implicitly."""

    with pytest.raises(TypeError):
        SemanticArtifactKnowledgeEncoder()
    with pytest.raises(ValueError, match="injected embedder"):
        SemanticArtifactKnowledgeEncoder(embedder=None)


def test_semantic_encoder_checkpoint_roundtrip_is_anchor_verified() -> None:
    embedder = _FakeEmbedder()
    encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)
    payload = encoder.checkpoint()

    restored = SemanticArtifactKnowledgeEncoder.from_checkpoint(payload, embedder=embedder)
    assert restored.embedder is embedder
    assert restored.feature_dim == embedder.dimension

    tampered = deepcopy(payload)
    tampered["embedder_revision"] = f"{tampered['embedder_revision']}-tampered"
    with pytest.raises(ValueError, match="embedder_revision drift"):
        SemanticArtifactKnowledgeEncoder.from_checkpoint(tampered, embedder=embedder)


def test_trainer_from_checkpoint_semantic_payload_requires_embedder() -> None:
    trainer = ArtifactInternalizationTrainer(
        feature_dim=_FakeEmbedder.dimension,
        encoder=SemanticArtifactKnowledgeEncoder(embedder=_FakeEmbedder()),
    )
    payload = trainer.checkpoint()
    assert payload["encoder"]["format"] == SemanticArtifactKnowledgeEncoder.FORMAT

    with pytest.raises(ValueError, match="requires an injected embedder"):
        ArtifactInternalizationTrainer.from_checkpoint(payload)
    restored = ArtifactInternalizationTrainer.from_checkpoint(payload, embedder=_FakeEmbedder())
    assert isinstance(restored.encoder, SemanticArtifactKnowledgeEncoder)

    # Native-encoder checkpoints keep restoring with no embedder argument.
    native_payload = ArtifactInternalizationTrainer().checkpoint()
    native_restored = ArtifactInternalizationTrainer.from_checkpoint(native_payload)
    assert isinstance(native_restored.encoder, ArtifactKnowledgeEncoder)
