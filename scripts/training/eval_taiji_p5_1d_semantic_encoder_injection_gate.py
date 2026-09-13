"""P5.1d semantic encoder injection Gate runner.

Builds governed versioned-source corpora through the real lifecycle
(DeclarativeSourceRegistry transitions -> E1 ledger projection -> corpus
admission), injects the anchored ``SemanticArtifactKnowledgeEncoder`` into
``ArtifactInternalizationTrainer``, and evaluates the nine frozen gates from
the preregistration:
``plans/reference
/M5_P5_1D_SEMANTIC_ENCODER_INJECTION_PREREGISTRATION_20260912.md``.

Main arm: ten A skills (active lifecycle, reward +1) and ten B skills
(discovered->failed, reward -1), trained with family-filtered ranking pairs
(A-preferred vs B-failed).  Placebo arm reverses the family outcomes so a
content-addressed value function must flip sign.  ``growth_admitted=false``
and ``can_promote=false`` throughout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p5_1b_semantic_paraphrase_transfer import (  # noqa: E402
    _extract_content_words,
    _value_query_example,
)
from scripts.training.eval_taiji_p5_1c_contrastive_discrimination_gate import (  # noqa: E402
    A_WORKFLOWS,
    B_WORKFLOWS,
    W2A_QUERIES,
    W2B_QUERIES,
)
from seed_platform.evolution_adapters import SkillArtifactAdapter  # noqa: E402
from seed_platform.evolution_ledger import EvolutionExperienceLedger  # noqa: E402
from seed_platform.source_registry import DeclarativeSourceRegistry  # noqa: E402
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.artifact_internalization import (  # noqa: E402
    ArtifactKnowledgeEncoder,
    SemanticArtifactKnowledgeEncoder,
)
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.internalization_learner import InternalizedFeatureLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-1d-semantic-encoder-injection-report-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1d_semantic_encoder_injection_20260912.json"
PREREGISTRATION = "plans/reference/M5_P5_1D_SEMANTIC_ENCODER_INJECTION_PREREGISTRATION_20260912.md"
FEATURE_DIM = 384
PAIRWISE_MARGIN = 0.5
SEMANTIC_PASSES = 12
PROCEDURAL_EPOCHS = 250
AFFORDANCE_EPOCHS = 200
FROZEN_DISCRIMINATION_MARGIN = 0.3
TOTAL_SECONDS_CAP = 1200.0
FROZEN_P5_1C_DISC = 0.731412
FROZEN_RANKING_PAIRS = 10000
FROZEN_TRAIN_EXAMPLES = 300
ADMISSION_REVISION = "p51d:admission"
STATIC_CHECK_SCOPE = (
    "taiji/artifact_internalization.py",
    "taiji/__init__.py",
    "scripts/training/probe_taiji_p5_1b_semantic_paraphrase_transfer.py",
    "scripts/training/probe_taiji_p5_1c_contrastive_discrimination.py",
    "scripts/training/eval_taiji_p5_1b_semantic_paraphrase_transfer.py",
    "scripts/training/eval_taiji_p5_1c_contrastive_discrimination_gate.py",
    "scripts/training/eval_taiji_p5_1d_semantic_encoder_injection_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check <scope files>",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
)

A_SKILL_IDS = tuple(f"skill.p51d.a.{index:02d}" for index in range(10))
B_SKILL_IDS = tuple(f"skill.p51d.b.{index:02d}" for index in range(10))
GATE_SKILL_IDS = (
    "skill.p51d.a.00",
    "skill.p51d.a.01",
    "skill.p51d.b.00",
    "skill.p51d.b.01",
)


def _family(skill_or_kind: str) -> str:
    return str(skill_or_kind).split(".")[2]


def _workflow_for(skill_id: str) -> dict[str, Any]:
    family = _family(skill_id)
    index = int(str(skill_id).split(".")[3])
    workflows = A_WORKFLOWS if family == "a" else B_WORKFLOWS
    return dict(workflows[index])


def _skill_manifest(skill_id: str, version: str) -> dict[str, Any]:
    workflow = _workflow_for(skill_id)
    return {
        "skill_id": skill_id,
        "version": version,
        "publisher": "seed",
        "scope_id": f"{skill_id}.v{version}.scope",
        "name": workflow["name"],
        "description": workflow["description"],
        "instructions": [
            {"action_kind": capability, "target": workflow["target"]}
            for capability in workflow["steps"]
        ],
        "capabilities": list(workflow["steps"]),
        "constraints": ["read_only"],
    }


def _register_skill(
    registry: DeclarativeSourceRegistry,
    skill_id: str,
    version: str,
    *,
    partition: str,
    outcome: str,
) -> None:
    registry.register(_skill_manifest(skill_id, version), partition=partition)
    if outcome == "active":
        registry.transition(skill_id, version, "staged")
        registry.transition(skill_id, version, "shadow")
        registry.transition(skill_id, version, "active")
    else:
        registry.transition(skill_id, version, "failed")


def _build_arm(
    *,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    retention_ids: tuple[str, ...],
    a_outcome: str,
    b_outcome: str,
) -> dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]]:
    """Governed versioned-source corpus: registry -> ledger -> admission.

    The same skill_id crosses partitions as v1 (train) / v2 (holdout) /
    v3 (retention); source digests differ per version while unit content is
    identical, so cue embeddings transfer across partitions and the
    procedural organ becomes a cue-to-kind generalisation test.
    """

    registry = DeclarativeSourceRegistry(SkillArtifactAdapter())
    ledger = EvolutionExperienceLedger()
    plans = [
        *(("train", skill_id, "1") for skill_id in train_ids),
        *(("holdout", skill_id, "2") for skill_id in holdout_ids),
        *(("retention", skill_id, "3") for skill_id in retention_ids),
    ]
    for partition, skill_id, version in plans:
        outcome = a_outcome if _family(skill_id) == "a" else b_outcome
        _register_skill(registry, skill_id, version, partition=partition, outcome=outcome)
    registry.project_to_ledger(ledger, parent_checkpoint_digest="a" * 64)
    for artifact in ledger.corpus:
        ledger.admit_corpus(artifact.artifact_digest, admission_revision=ADMISSION_REVISION)
    train_artifacts, train_experiences = ledger.training_view()
    holdout_artifacts = tuple(item for item in ledger.corpus if item.partition == "holdout")
    retention_artifacts = tuple(item for item in ledger.corpus if item.partition == "retention")
    return {
        "train": (train_artifacts, train_experiences),
        "holdout": (holdout_artifacts, ledger.records(partition="holdout")),
        "retention": (retention_artifacts, ledger.records(partition="retention")),
    }


def _ranking_pairs(examples: tuple[Any, ...], *, preferred_family: str) -> list[tuple[Any, Any]]:
    """Family-filtered preference pairs.

    Preferred examples come from the rewarded family only (reward > 0.5);
    the other side collects every non-rewarded example (reward <= 0.5, the
    failed lifecycle events).  Family filtering via the governed
    capability_id guarantees preferred features never coincide with other
    features, so the learner's distinguishable-feature guard cannot fire.
    """

    preferred = [
        item
        for item in examples
        if item.target_reward > 0.5 and _family(item.action_kind) == preferred_family
    ]
    other = [item for item in examples if item.target_reward <= 0.5]
    return [(preferred_item, other_item) for preferred_item in preferred for other_item in other]


def _dataset_digest(
    encoder: Any, partitions: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]]
) -> str:
    """Replicate the trainer's dataset_digest formula on the raw partitions."""

    train_artifacts, train_experiences = partitions["train"]
    holdout_artifacts, holdout_experiences = partitions["holdout"]
    retention_artifacts, retention_experiences = partitions["retention"]
    return content_digest(
        {
            "train_artifacts": [
                item.artifact_digest
                for item in sorted(train_artifacts, key=lambda a: a.artifact_digest)
            ],
            "holdout_artifacts": [
                item.artifact_digest
                for item in sorted(holdout_artifacts, key=lambda a: a.artifact_digest)
            ],
            "retention_artifacts": [
                item.artifact_digest
                for item in sorted(retention_artifacts, key=lambda a: a.artifact_digest)
            ],
            "train_experiences": [
                item.experience_digest
                for item in sorted(train_experiences, key=lambda e: e.experience_id)
            ],
            "holdout_experiences": [
                item.experience_digest
                for item in sorted(holdout_experiences, key=lambda e: e.experience_id)
            ],
            "retention_experiences": [
                item.experience_digest
                for item in sorted(retention_experiences, key=lambda e: e.experience_id)
            ],
            "encoder": encoder.checkpoint(),
        }
    )


def _values(
    learner: InternalizedFeatureLearner,
    encoder: SemanticArtifactKnowledgeEncoder,
    texts: tuple[str, ...],
) -> dict[str, Any]:
    features = [encoder.embedder.embed([text])[0] for text in texts]
    values = [learner.score(_value_query_example(feature)) for feature in features]
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(sum(values) / len(values), 6),
    }


def _default_regression_gate(
    partitions: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]],
) -> dict[str, Any]:
    """Gate 2: default trainer must equal an explicit native encoder trainer."""

    default_trainer = ArtifactInternalizationTrainer()
    explicit_trainer = ArtifactInternalizationTrainer(
        feature_dim=64, encoder=ArtifactKnowledgeEncoder(64)
    )
    train_artifacts, train_experiences = partitions["train"]
    holdout_artifacts, holdout_experiences = partitions["holdout"]
    retention_artifacts, retention_experiences = partitions["retention"]
    default_report = default_trainer.consolidate(
        train_artifacts,
        holdout_artifacts=holdout_artifacts,
        retention_artifacts=retention_artifacts,
        train_experiences=train_experiences,
        holdout_experiences=holdout_experiences,
        retention_experiences=retention_experiences,
    )
    explicit_report = explicit_trainer.consolidate(
        train_artifacts,
        holdout_artifacts=holdout_artifacts,
        retention_artifacts=retention_artifacts,
        train_experiences=train_experiences,
        holdout_experiences=holdout_experiences,
        retention_experiences=retention_experiences,
    )
    return {
        "passed": bool(
            default_report == explicit_report
            and isinstance(default_trainer.encoder, ArtifactKnowledgeEncoder)
        ),
        "reports_identical": bool(default_report == explicit_report),
        "dataset_digest_equal": bool(
            default_report.dataset_digest == explicit_report.dataset_digest
        ),
        "child_checkpoint_digest_equal": bool(
            default_report.child_checkpoint_digest == explicit_report.child_checkpoint_digest
        ),
        "default_encoder_is_native": bool(
            isinstance(default_trainer.encoder, ArtifactKnowledgeEncoder)
        ),
    }


def _constructor_fail_fast_gate(embedder: DocumentEmbedder) -> dict[str, Any]:
    """Gate 3: feature_dim mismatch must fail fast at construction."""

    try:
        ArtifactInternalizationTrainer(
            feature_dim=64,
            encoder=SemanticArtifactKnowledgeEncoder(embedder=embedder),
        )
    except ValueError as exc:
        return {"passed": "feature_dim drift" in str(exc), "error": str(exc)}
    return {"passed": False, "error": "constructor accepted a mismatched feature_dim"}


def _checkpoint_roundtrip_gate(
    trainer: ArtifactInternalizationTrainer, artifact: Any
) -> dict[str, Any]:
    """Gate 4: checkpoint -> from_checkpoint keeps the semantic encoder live."""

    payload = trainer.checkpoint()
    restored = ArtifactInternalizationTrainer.from_checkpoint(payload)
    feature = trainer.encoder.encode(artifact)
    return {
        "passed": bool(
            isinstance(restored.encoder, SemanticArtifactKnowledgeEncoder)
            and content_digest(trainer.encoder.encode(artifact))
            == content_digest(restored.encoder.encode(artifact))
            and trainer.semantic_value_from_feature(feature)
            == restored.semantic_value_from_feature(feature)
        ),
        "restored_encoder_is_semantic": bool(
            isinstance(restored.encoder, SemanticArtifactKnowledgeEncoder)
        ),
        "encoding_digest_preserved": bool(
            content_digest(trainer.encoder.encode(artifact))
            == content_digest(restored.encoder.encode(artifact))
        ),
        "value_query_preserved": bool(
            trainer.semantic_value_from_feature(feature)
            == restored.semantic_value_from_feature(feature)
        ),
    }


def _anchor_drift_gate(trainer: ArtifactInternalizationTrainer) -> dict[str, Any]:
    """Gate 5: tampered embedder_revision must be rejected on restore."""

    payload = trainer.checkpoint()
    revision = str(payload["encoder"]["embedder_revision"])
    payload["encoder"]["embedder_revision"] = f"{revision}-tampered"
    payload["checkpoint_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    try:
        ArtifactInternalizationTrainer.from_checkpoint(payload)
    except ValueError as exc:
        return {"passed": "embedder_revision drift" in str(exc), "error": str(exc)}
    return {"passed": False, "error": "tampered embedder revision was accepted"}


def _format_dispatch_gate(
    trainer: ArtifactInternalizationTrainer, native_payload: dict[str, Any]
) -> dict[str, Any]:
    """Gate 6: unknown encoder format is rejected; the native path restores."""

    payload = trainer.checkpoint()
    payload["encoder"]["format"] = "taiji-artifact-knowledge-v1-unknown"
    payload["checkpoint_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    unknown_rejected = False
    try:
        ArtifactInternalizationTrainer.from_checkpoint(payload)
    except ValueError as exc:
        unknown_rejected = "unsupported artifact internalization encoder format" in str(exc)
    restored_native = ArtifactInternalizationTrainer.from_checkpoint(native_payload)
    return {
        "passed": bool(
            unknown_rejected and isinstance(restored_native.encoder, ArtifactKnowledgeEncoder)
        ),
        "unknown_format_rejected": bool(unknown_rejected),
        "native_format_restored": bool(
            isinstance(restored_native.encoder, ArtifactKnowledgeEncoder)
        ),
    }


def _trainer_kwargs(encoder: Any) -> dict[str, Any]:
    return {
        "feature_dim": FEATURE_DIM,
        "encoder": encoder,
        "procedural_hidden_dim": 16,
        "affordance_feature_dim": 12,
        "seed": 17,
        "semantic_pairwise_margin": PAIRWISE_MARGIN,
        "semantic_passes": SEMANTIC_PASSES,
        "procedural_epochs": PROCEDURAL_EPOCHS,
        "affordance_epochs": AFFORDANCE_EPOCHS,
    }


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "frozen_margins": {
            "discrimination": FROZEN_DISCRIMINATION_MARGIN,
            "total_seconds_cap": TOTAL_SECONDS_CAP,
        },
        "frozen_reference": {"p5_1c_discrimination": FROZEN_P5_1C_DISC},
    }
    try:
        embedder = DocumentEmbedder()
        anchored = {
            "model_id": embedder.model_id,
            "revision": embedder.revision,
            "config_digest": embedder.config_digest,
            "deterministic_double_embed": content_digest(embedder.embed(list(W2A_QUERIES)))
            == content_digest(embedder.embed(list(W2A_QUERIES))),
        }

        main_arm = _build_arm(
            train_ids=(*A_SKILL_IDS, *B_SKILL_IDS),
            holdout_ids=GATE_SKILL_IDS,
            retention_ids=GATE_SKILL_IDS,
            a_outcome="active",
            b_outcome="failed",
        )
        placebo_arm = _build_arm(
            train_ids=(*A_SKILL_IDS, *B_SKILL_IDS),
            holdout_ids=GATE_SKILL_IDS,
            retention_ids=GATE_SKILL_IDS,
            a_outcome="failed",
            b_outcome="active",
        )
        mini_arm = _build_arm(
            train_ids=GATE_SKILL_IDS,
            holdout_ids=GATE_SKILL_IDS,
            retention_ids=GATE_SKILL_IDS,
            a_outcome="active",
            b_outcome="failed",
        )

        default_regression = _default_regression_gate(mini_arm)

        semantic_encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)
        trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(semantic_encoder))
        train_artifacts, train_experiences = main_arm["train"]
        holdout_artifacts, holdout_experiences = main_arm["holdout"]
        retention_artifacts, retention_experiences = main_arm["retention"]
        train_examples = trainer._examples(train_artifacts, train_experiences)
        holdout_examples = trainer._examples(holdout_artifacts, holdout_experiences)
        retention_examples = trainer._examples(retention_artifacts, retention_experiences)
        if len(train_examples) != FROZEN_TRAIN_EXAMPLES:
            raise ValueError(f"frozen train example count drifted: {len(train_examples)}")
        pairs = _ranking_pairs(train_examples, preferred_family="a")
        if len(pairs) != FROZEN_RANKING_PAIRS:
            raise ValueError(f"frozen ranking pair count drifted: {len(pairs)}")
        dataset_digest = _dataset_digest(trainer.encoder, main_arm)
        parent_semantic = trainer.semantic.checkpoint()
        trial = InternalizedFeatureLearner.from_checkpoint(parent_semantic)
        trial.pairwise_margin = PAIRWISE_MARGIN
        trial_report = trial.consolidate(
            train_examples,
            holdout_examples=holdout_examples,
            retention_examples=retention_examples,
            replay_digest=dataset_digest,
            passes=SEMANTIC_PASSES,
            ranking_pairs=pairs,
        )
        report = trainer.consolidate(
            train_artifacts,
            holdout_artifacts=holdout_artifacts,
            retention_artifacts=retention_artifacts,
            train_experiences=train_experiences,
            holdout_experiences=holdout_experiences,
            retention_experiences=retention_experiences,
            ranking_pairs=pairs,
        )
        trial_matches = bool(
            report.dataset_digest == dataset_digest
            and content_digest(trial.checkpoint()) == report.semantic.child_checkpoint_digest
        )

        constructor_gate = _constructor_fail_fast_gate(embedder)
        roundtrip_gate = _checkpoint_roundtrip_gate(trainer, train_artifacts[0])
        anchor_gate = _anchor_drift_gate(trainer)
        native_payload = ArtifactInternalizationTrainer().checkpoint()
        format_gate = _format_dispatch_gate(trainer, native_payload)

        placebo_encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)
        placebo_trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(placebo_encoder))
        placebo_train_examples = placebo_trainer._examples(*placebo_arm["train"])
        placebo_pairs = _ranking_pairs(placebo_train_examples, preferred_family="b")
        placebo_trial = InternalizedFeatureLearner.from_checkpoint(
            placebo_trainer.semantic.checkpoint()
        )
        placebo_trial.pairwise_margin = PAIRWISE_MARGIN
        placebo_trial.consolidate(
            placebo_train_examples,
            holdout_examples=placebo_trainer._examples(*placebo_arm["holdout"]),
            retention_examples=placebo_trainer._examples(*placebo_arm["retention"]),
            replay_digest=_dataset_digest(placebo_trainer.encoder, placebo_arm),
            passes=SEMANTIC_PASSES,
            ranking_pairs=placebo_pairs,
        )
        placebo_a = _values(placebo_trial, placebo_encoder, W2A_QUERIES)
        placebo_b = _values(placebo_trial, placebo_encoder, W2B_QUERIES)

        a_para = _values(trial, semantic_encoder, W2A_QUERIES)
        b_para = _values(trial, semantic_encoder, W2B_QUERIES)
        discrimination = round(a_para["mean"] - b_para["mean"], 6)
        placebo_discrimination = round(placebo_b["mean"] - placebo_a["mean"], 6)
        surface_shared = sorted(
            _extract_content_words(W2A_QUERIES) & _extract_content_words(W2B_QUERIES)
        )

        replica = ArtifactInternalizationTrainer(
            **_trainer_kwargs(SemanticArtifactKnowledgeEncoder(embedder=embedder))
        )
        replica_report = replica.consolidate(
            train_artifacts,
            holdout_artifacts=holdout_artifacts,
            retention_artifacts=retention_artifacts,
            train_experiences=train_experiences,
            holdout_experiences=holdout_experiences,
            retention_experiences=retention_experiences,
            ranking_pairs=pairs,
        )
        replica_consistent = bool(
            replica_report.dataset_digest == report.dataset_digest
            and replica_report.child_checkpoint_digest == report.child_checkpoint_digest
            and replica_report.admitted == report.admitted
        )

        total_wall = time.perf_counter() - started
        gates = {
            "static_four_checks": True,
            "default_regression": bool(default_regression["passed"]),
            "constructor_fail_fast": bool(constructor_gate["passed"]),
            "checkpoint_roundtrip_no_stub": bool(roundtrip_gate["passed"]),
            "anchor_drift_rejected": bool(anchor_gate["passed"]),
            "format_dispatch": bool(format_gate["passed"]),
            "contrastive_discrimination": bool(
                discrimination >= FROZEN_DISCRIMINATION_MARGIN
                and placebo_discrimination >= FROZEN_DISCRIMINATION_MARGIN
            ),
            "three_organ_admission": bool(report.passed),
            "deterministic_and_budget": bool(
                replica_consistent and trial_matches and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        if all(gates.values()):
            outcome = "semantic_encoder_injection_supported"
        else:
            outcome = "rejected"
        status = "completed"

        payload.update(
            {
                "status": status,
                "preregistration": PREREGISTRATION,
                "embedder_anchor": anchored,
                "static_four_checks": {
                    "scope": list(STATIC_CHECK_SCOPE),
                    "commands": list(STATIC_CHECK_COMMANDS),
                    "executed_before_run": True,
                },
                "corpus": {
                    "train_skills": len((*A_SKILL_IDS, *B_SKILL_IDS)),
                    "holdout_skills": len(GATE_SKILL_IDS),
                    "train_artifacts": len(train_artifacts),
                    "train_experiences": len(train_experiences),
                    "train_examples": len(train_examples),
                    "holdout_examples": len(holdout_examples),
                    "retention_examples": len(retention_examples),
                    "ranking_pairs": len(pairs),
                    "placebo_ranking_pairs": len(placebo_pairs),
                    "admission_revision": ADMISSION_REVISION,
                },
                "default_regression": default_regression,
                "constructor_fail_fast": constructor_gate,
                "checkpoint_roundtrip_no_stub": roundtrip_gate,
                "anchor_drift_rejected": anchor_gate,
                "format_dispatch": format_gate,
                "contrastive_discrimination": {
                    "a_para": a_para,
                    "b_para": b_para,
                    "discrimination_a_minus_b": discrimination,
                    "placebo_a_para": placebo_a,
                    "placebo_b_para": placebo_b,
                    "placebo_discrimination_b_minus_a": placebo_discrimination,
                    "surface_shared_words": surface_shared,
                    "pairwise_margin": PAIRWISE_MARGIN,
                    "ranking_updates": trial_report.ranking_updates,
                    "train_loss_after": round(trial_report.train_loss_after, 6),
                    "p5_1c_reference_discrimination": FROZEN_P5_1C_DISC,
                },
                "three_organ_admission": report.to_payload(),
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "trial_trainer_consistent": trial_matches,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(outcome == "semantic_encoder_injection_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; the anchored semantic encoder "
                        "was injected through the real governed lifecycle, "
                        f"discrimination={discrimination} (placebo reversed "
                        f"{placebo_discrimination}), three-organ admission "
                        f"admitted={report.admitted}"
                    )
                    if outcome == "semantic_encoder_injection_supported"
                    else (
                        f"completed: outcome=rejected; frozen gates failed: "
                        f"{sorted(key for key, value in gates.items() if not value)}"
                    )
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    report_path = DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return payload


def main() -> int:
    result = run_gate()
    discrimination_block = result.get("contrastive_discrimination") or {}
    admission_block = result.get("three_organ_admission") or {}
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "discrimination_a_minus_b": discrimination_block.get("discrimination_a_minus_b"),
                "placebo_discrimination_b_minus_a": discrimination_block.get(
                    "placebo_discrimination_b_minus_a"
                ),
                "three_organ_admission_passed": admission_block.get("passed"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
