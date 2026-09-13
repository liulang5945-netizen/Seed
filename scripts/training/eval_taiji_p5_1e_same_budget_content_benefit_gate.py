"""P5.1e same-budget content benefit Gate runner.

Two family-exclusive arms under an identical trainer budget
(``ArtifactInternalizationTrainer`` + ``SemanticArtifactKnowledgeEncoder``,
384-dim MiniLM anchor, all trainer hyper-parameters bit-identical):

- sourced arm: 12 A-family skills (capability vocabulary
  ``editor.open/read/inspect``), 8 active + 4 failed lifecycle outcomes;
- placebo arm: 12 B-family skills (``network.search/index.scan/cache.fetch``),
  8 active + 4 failed (isomorphic outcome structure).

The single variable is the trained content family.  Benefit is measured as
the procedural accuracy delta on 4 fresh A-family gate skills (new
subject/verb text, same capability steps) evaluated evaluation-only on each
arm's committed trainer (appendix 2 designated-label records; appendix 3
cross-partition ids and the ``holdout`` gate projection label).  Nine frozen
gates live in ``plans/reference
/M5_P5_1E_SAME_BUDGET_CONTENT_BENEFIT_PREREGISTRATION_20260912.md``.
``growth_admitted=false`` and ``can_promote=false`` throughout.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.evolution_adapters import (  # noqa: E402
    ArtifactCorpusProjection,
    SkillArtifactAdapter,
)
from seed_platform.evolution_ledger import EvolutionExperienceLedger  # noqa: E402
from seed_platform.source_registry import DeclarativeSourceRegistry  # noqa: E402
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.artifact_internalization import (  # noqa: E402
    SemanticArtifactKnowledgeEncoder,
)
from taiji.contracts import ActionIntent, EpisodicMemoryRecord  # noqa: E402
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-p5-1e-same-budget-content-benefit-report-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1e_same_budget_content_benefit_20260912.json"
PREREGISTRATION = "plans/reference/M5_P5_1E_SAME_BUDGET_CONTENT_BENEFIT_PREREGISTRATION_20260912.md"
FEATURE_DIM = 384
PAIRWISE_MARGIN = 0.5
SEMANTIC_PASSES = 12
PROCEDURAL_EPOCHS = 250
AFFORDANCE_EPOCHS = 200
FROZEN_CONTENT_TRANSFER_MARGIN = 0.15
TOTAL_SECONDS_CAP = 1200.0
FROZEN_P5_1D_DISC = 0.870033
ADMISSION_REVISION = "p51e:admission"
ACTIVE_SKILL_COUNT = 8
# Appendix 3: EVOLUTION_PARTITIONS has no "gate"; the gate projection is an
# evaluation-only construct outside registry/ledger/admission flows, so the
# inert "holdout" label only lets the corpus artifact validate.
GATE_PARTITION = "holdout"
GATE_RECORDS_PER_SKILL = 4
STATIC_CHECK_SCOPE = (
    "taiji/artifact_internalization.py",
    "taiji/procedural_memory.py",
    "taiji/evolution_experience.py",
    "seed_platform/evolution_adapters.py",
    "seed_platform/source_registry.py",
    "seed_platform/evolution_ledger.py",
    "scripts/training/eval_taiji_p5_1e_same_budget_content_benefit_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check <scope files>",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
)

A_SKILL_IDS = tuple(f"skill.p51e.a.{index:02d}" for index in range(1, 13))
B_SKILL_IDS = tuple(f"skill.p51e.b.{index:02d}" for index in range(1, 13))
A_GATE_SKILL_IDS = tuple(f"skill.p51e.a.{index:02d}" for index in range(13, 17))
B_GATE_SKILL_IDS = tuple(f"skill.p51e.b.{index:02d}" for index in range(13, 17))
A_FAMILY_STEPS = ("editor.open", "editor.read", "editor.inspect")
B_FAMILY_STEPS = ("network.search", "index.scan", "cache.fetch")

A_WORKFLOW_TEXTS = (
    ("draft memo", "Draft"),
    ("review sheet", "Review"),
    ("staged note", "Stage"),
    ("pinned record", "Pin"),
    ("queued entry", "Queue"),
    ("flagged item", "Flag"),
    ("tracked change", "Track"),
    ("versioned draft", "Version"),
    ("signed note", "Sign"),
    ("sealed memo", "Seal"),
    ("tagged entry", "Tag"),
    ("indexed page", "Index"),
    ("cached sheet", "Cache"),
    ("masked note", "Mask"),
    ("scoped entry", "Scope"),
    ("bundled record", "Bundle"),
)
B_WORKFLOW_TEXTS = (
    ("remote index", "Search"),
    ("mirror shard", "Sweep"),
    ("relay feed", "Probe"),
    ("edge cache", "Scan"),
    ("peer catalogue", "Comb"),
    ("proxy table", "Fetch"),
    ("gateway log", "Pull"),
    ("upstream delta", "Collect"),
    ("offset slice", "Query"),
    ("warm pool", "Harvest"),
    ("cold archive", "Mirror"),
    ("sync stream", "Sync"),
    ("quota ledger", "Relay"),
    ("route map", "Route"),
    ("ticket queue", "Batch"),
    ("batch header", "Stream"),
)


def _family(skill_or_kind: str) -> str:
    return str(skill_or_kind).split(".")[2]


def _skill_index(skill_id: str) -> int:
    return int(str(skill_id).split(".")[3]) - 1


def _workflows_for(skill_id: str) -> dict[str, Any]:
    family = _family(skill_id)
    index = _skill_index(skill_id)
    texts = A_WORKFLOW_TEXTS if family == "a" else B_WORKFLOW_TEXTS
    steps = A_FAMILY_STEPS if family == "a" else B_FAMILY_STEPS
    subject, verb = texts[index]
    if family == "a":
        description = (
            f"{verb} the {subject} in bounded steps: bring the {subject} up, "
            "view its contents, and check the outcome."
        )
    else:
        description = (
            f"{verb} the {subject} for matches, inspect the matched slices, "
            "then pull the stored copies."
        )
    return {
        "name": f"{verb} {subject}",
        "description": description,
        "steps": steps,
        "target": subject,
    }


def _skill_manifest(skill_id: str, version: str) -> dict[str, Any]:
    workflow = _workflows_for(skill_id)
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


def _build_arm(*, family: str) -> dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]]:
    """Governed versioned-source corpus for one family-exclusive arm.

    All 12 skill_ids of the family cross the three partitions as v1 (train) /
    v2 (holdout) / v3 (retention) with identical unit content (appendix 3
    reading of the frozen section 2 table); source digests differ per version,
    so the procedural organ is a cue-to-kind generalisation test.  Outcome
    structure is 8 active + 4 failed per arm in both arms (isomorphic).
    """

    registry = DeclarativeSourceRegistry(SkillArtifactAdapter())
    ledger = EvolutionExperienceLedger()
    skill_ids = A_SKILL_IDS if family == "a" else B_SKILL_IDS
    active_ids = set(skill_ids[:ACTIVE_SKILL_COUNT])
    plans = [
        *(("train", skill_id, "1") for skill_id in skill_ids),
        *(("holdout", skill_id, "2") for skill_id in skill_ids),
        *(("retention", skill_id, "3") for skill_id in skill_ids),
    ]
    for partition, skill_id, version in plans:
        outcome = "active" if skill_id in active_ids else "failed"
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
    """Family-filtered preference pairs with the appendix-1 pruning.

    Preferred examples carry reward > 0.5 and the preferred family; the other
    side collects every non-rewarded example (reward <= 0.5).  On top of the
    full cartesian product the pruning skips pairs whose grounded features are
    bit-identical (``grounding.equal``): same-skill discovered/failed events
    share one source digest, and same-family affordance units share the
    ``{"value": <capability>}`` content, so un-pruned zero-norm pairs would
    hard-fail the learner's distinguishable-feature guard.
    """

    preferred = [
        item
        for item in examples
        if item.target_reward > 0.5 and _family(item.action_kind) == preferred_family
    ]
    other = [item for item in examples if item.target_reward <= 0.5]
    return [
        (preferred_item, other_item)
        for preferred_item in preferred
        for other_item in other
        if not preferred_item.grounding.equal(other_item.grounding)
    ]


def _consolidate_arm(
    trainer: ArtifactInternalizationTrainer,
    arm: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]],
    pairs: list[tuple[Any, Any]],
) -> Any:
    train_artifacts, train_experiences = arm["train"]
    holdout_artifacts, holdout_experiences = arm["holdout"]
    retention_artifacts, retention_experiences = arm["retention"]
    return trainer.consolidate(
        train_artifacts,
        holdout_artifacts=holdout_artifacts,
        retention_artifacts=retention_artifacts,
        train_experiences=train_experiences,
        holdout_experiences=holdout_experiences,
        retention_experiences=retention_experiences,
        ranking_pairs=pairs,
    )


def _gate_projection(gate_id: str) -> ArtifactCorpusProjection:
    return SkillArtifactAdapter().project(_skill_manifest(gate_id, "1"), partition=GATE_PARTITION)


def _gate_units(gate_id: str) -> tuple[Any, ...]:
    return tuple(_gate_projection(gate_id).corpus)


def _gate_records(
    gate_ids: tuple[str, ...],
    *,
    family: str,
    encoder: Any,
) -> tuple[EpisodicMemoryRecord, ...]:
    """Designated-label evaluation records (appendix 2).

    Each gate skill yields 4 records sharing one procedure-unit cue; the
    actual kind is the index-aligned in-vocabulary training id
    (``skill.p51e.<family>.01..04``), so ``_sequence_accuracy`` compares
    against the arm's committed readout vocabulary.  ``outcome=None``: no
    fabricated rewards.
    """

    training_ids = A_SKILL_IDS if family == "a" else B_SKILL_IDS
    records: list[EpisodicMemoryRecord] = []
    for offset, gate_id in enumerate(gate_ids):
        procedure = next(unit for unit in _gate_units(gate_id) if unit.unit_kind == "procedure")
        cue = encoder.encode(procedure)
        episode_id = f"p51e-gate:{gate_id}"
        for tick in range(1, GATE_RECORDS_PER_SKILL + 1):
            records.append(
                EpisodicMemoryRecord(
                    memory_id=f"p51e-gate-memory:{gate_id}:{tick:02d}",
                    episode_id=episode_id,
                    tick=tick,
                    cue=cue,
                    action_intent=ActionIntent(
                        f"p51e-gate-intent:{gate_id}:{tick:02d}",
                        training_ids[offset],
                        tick=tick - 1,
                    ),
                    outcome=None,
                    provenance="p51e-gate-evaluation",
                )
            )
    return tuple(records)


def _gate_accuracy(
    trainer: ArtifactInternalizationTrainer, records: tuple[EpisodicMemoryRecord, ...]
) -> dict[str, Any]:
    if not trainer.procedural.action_kinds:
        return {
            "measurable": False,
            "reason": "procedural readout has no committed vocabulary (admission not reached)",
        }
    return {
        "measurable": True,
        "accuracy": round(
            float(ArtifactInternalizationTrainer._sequence_accuracy(trainer.procedural, records)),
            6,
        ),
    }


def _capability_vocabulary(artifacts: tuple[Any, ...]) -> tuple[str, ...]:
    values = {
        str(unit.content["value"])
        for unit in artifacts
        if unit.unit_kind == "affordance" and "value" in unit.content
    }
    return tuple(sorted(values))


def _arm_structure(
    arm: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]],
    *,
    pairs: list[tuple[Any, Any]],
    a_gate_records: int,
    b_gate_records: int,
) -> dict[str, Any]:
    structure: dict[str, Any] = {}
    for partition in ("train", "holdout", "retention"):
        artifacts, experiences = arm[partition]
        structure[f"{partition}_artifacts"] = len(artifacts)
        structure[f"{partition}_experiences"] = len(experiences)
        structure[f"{partition}_unit_kinds"] = dict(
            sorted(Counter(unit.unit_kind for unit in artifacts).items())
        )
    structure["ranking_pairs"] = len(pairs)
    structure["a_gate_records"] = a_gate_records
    structure["b_gate_records"] = b_gate_records
    return structure


def _checkpoint_roundtrip(trainer: ArtifactInternalizationTrainer, artifact: Any) -> dict[str, Any]:
    payload = trainer.checkpoint()
    restored = ArtifactInternalizationTrainer.from_checkpoint(payload)
    feature = trainer.encoder.encode(artifact)
    return {
        "checkpoint_digest_preserved": bool(
            restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"]
        ),
        "encoding_digest_preserved": bool(
            content_digest(feature) == content_digest(restored.encoder.encode(artifact))
        ),
        "semantic_value_preserved": bool(
            trainer.semantic_value_from_feature(feature)
            == restored.semantic_value_from_feature(feature)
        ),
    }


def _semantic_surface(
    trainer: ArtifactInternalizationTrainer, units: tuple[Any, ...]
) -> dict[str, Any]:
    """Arm-native semantic face over the gate units (appendix 2 item 3).

    No ``_examples`` pass (gate skills have no experiences and no fabricated
    outcomes); values are disclosed, not gated.
    """

    values = [
        round(trainer.semantic_value_from_feature(trainer.encoder.encode(unit)), 6)
        for unit in sorted(units, key=lambda item: item.corpus_id)
    ]
    mean = sum(values) / len(values) if values else 0.0
    return {"gate_unit_values": values, "mean": round(mean, 6)}


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
            "content_transfer": FROZEN_CONTENT_TRANSFER_MARGIN,
            "total_seconds_cap": TOTAL_SECONDS_CAP,
        },
        "frozen_reference": {"p5_1d_discrimination": FROZEN_P5_1D_DISC},
    }
    try:
        embedder = DocumentEmbedder()
        anchor_texts = [
            _workflows_for(gate_id)["name"] for gate_id in (*A_GATE_SKILL_IDS, *B_GATE_SKILL_IDS)
        ]
        anchored = {
            "model_id": embedder.model_id,
            "revision": embedder.revision,
            "config_digest": embedder.config_digest,
            "deterministic_double_embed": content_digest(embedder.embed(anchor_texts))
            == content_digest(embedder.embed(anchor_texts)),
        }

        sourced_arm = _build_arm(family="a")
        placebo_arm = _build_arm(family="b")
        sourced_encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)
        sourced_trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(sourced_encoder))
        placebo_encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)
        placebo_trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(placebo_encoder))

        sourced_train_examples = sourced_trainer._examples(*sourced_arm["train"])
        placebo_train_examples = placebo_trainer._examples(*placebo_arm["train"])
        sourced_pairs = _ranking_pairs(sourced_train_examples, preferred_family="a")
        placebo_pairs = _ranking_pairs(placebo_train_examples, preferred_family="b")
        sourced_report = _consolidate_arm(sourced_trainer, sourced_arm, sourced_pairs)
        placebo_report = _consolidate_arm(placebo_trainer, placebo_arm, placebo_pairs)

        a_gate_units = tuple(unit for gate_id in A_GATE_SKILL_IDS for unit in _gate_units(gate_id))
        b_gate_units = tuple(unit for gate_id in B_GATE_SKILL_IDS for unit in _gate_units(gate_id))
        a_gate_records_sourced = _gate_records(
            A_GATE_SKILL_IDS, family="a", encoder=sourced_encoder
        )
        a_gate_records_placebo = _gate_records(
            A_GATE_SKILL_IDS, family="a", encoder=placebo_encoder
        )
        b_gate_records_sourced = _gate_records(
            B_GATE_SKILL_IDS, family="b", encoder=sourced_encoder
        )
        b_gate_records_placebo = _gate_records(
            B_GATE_SKILL_IDS, family="b", encoder=placebo_encoder
        )
        a_gate_sourced = _gate_accuracy(sourced_trainer, a_gate_records_sourced)
        a_gate_placebo = _gate_accuracy(placebo_trainer, a_gate_records_placebo)
        b_gate_sourced = _gate_accuracy(sourced_trainer, b_gate_records_sourced)
        b_gate_placebo = _gate_accuracy(placebo_trainer, b_gate_records_placebo)

        if a_gate_sourced["measurable"] and a_gate_placebo["measurable"]:
            content_transfer_delta: float | None = round(
                float(a_gate_sourced["accuracy"]) - float(a_gate_placebo["accuracy"]), 6
            )
        else:
            content_transfer_delta = None

        sourced_structure = _arm_structure(
            sourced_arm,
            pairs=sourced_pairs,
            a_gate_records=len(a_gate_records_sourced),
            b_gate_records=len(b_gate_records_sourced),
        )
        placebo_structure = _arm_structure(
            placebo_arm,
            pairs=placebo_pairs,
            a_gate_records=len(a_gate_records_placebo),
            b_gate_records=len(b_gate_records_placebo),
        )
        sourced_vocabulary = _capability_vocabulary(sourced_arm["train"][0])
        placebo_vocabulary = _capability_vocabulary(placebo_arm["train"][0])
        a_gate_vocabulary = _capability_vocabulary(a_gate_units)
        b_gate_vocabulary = _capability_vocabulary(b_gate_units)

        sourced_replica = ArtifactInternalizationTrainer(
            **_trainer_kwargs(SemanticArtifactKnowledgeEncoder(embedder=embedder))
        )
        sourced_replica_report = _consolidate_arm(sourced_replica, sourced_arm, sourced_pairs)
        placebo_replica = ArtifactInternalizationTrainer(
            **_trainer_kwargs(SemanticArtifactKnowledgeEncoder(embedder=embedder))
        )
        placebo_replica_report = _consolidate_arm(placebo_replica, placebo_arm, placebo_pairs)
        replica_consistent = bool(
            sourced_replica_report.dataset_digest == sourced_report.dataset_digest
            and sourced_replica_report.child_checkpoint_digest
            == sourced_report.child_checkpoint_digest
            and sourced_replica_report.admitted == sourced_report.admitted
            and placebo_replica_report.dataset_digest == placebo_report.dataset_digest
            and placebo_replica_report.child_checkpoint_digest
            == placebo_report.child_checkpoint_digest
            and placebo_replica_report.admitted == placebo_report.admitted
        )

        sourced_roundtrip = _checkpoint_roundtrip(sourced_trainer, sourced_arm["train"][0][0])
        placebo_roundtrip = _checkpoint_roundtrip(placebo_trainer, placebo_arm["train"][0][0])
        sourced_surface = _semantic_surface(sourced_trainer, (*a_gate_units, *b_gate_units))
        placebo_surface = _semantic_surface(placebo_trainer, (*a_gate_units, *b_gate_units))

        total_wall = time.perf_counter() - started
        gates = {
            "static_four_checks": True,
            "same_budget_enforced": bool(sourced_structure == placebo_structure),
            "capability_vocabulary_disjoint": bool(
                set(sourced_vocabulary) >= set(a_gate_vocabulary)
                and not set(placebo_vocabulary) & set(a_gate_vocabulary)
            ),
            "arm_sanity_both_arms": bool(sourced_report.passed and placebo_report.passed),
            "content_transfer_margin": bool(
                content_transfer_delta is not None
                and content_transfer_delta >= FROZEN_CONTENT_TRANSFER_MARGIN
            ),
            "sourced_beats_lesion": bool(
                sourced_report.procedural_holdout_accuracy
                > sourced_report.procedural_lesion_holdout_accuracy
            ),
            "affordance_content_specificity": bool(
                sourced_report.affordance_native_holdout_mse
                < sourced_report.affordance_frozen_holdout_mse
                and sourced_report.affordance_native_holdout_mse
                < placebo_report.affordance_native_holdout_mse
            ),
            "checkpoint_roundtrip_both_arms": bool(
                all(sourced_roundtrip.values()) and all(placebo_roundtrip.values())
            ),
            "deterministic_and_budget": bool(
                replica_consistent and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        structural_keys = tuple(key for key in gates if key != "content_transfer_margin")
        if all(gates.values()):
            outcome = "same_budget_content_benefit_supported"
        elif all(gates[key] for key in structural_keys):
            outcome = "content_benefit_insufficient"
        else:
            outcome = "failed"
        status = "completed"

        payload.update(
            {
                "status": status,
                "preregistration": PREREGISTRATION,
                "preregistration_revisions": [
                    "appendix-1-pair-pruning",
                    "appendix-2-gate-records",
                    "appendix-3-partitions-and-gate-partition-label",
                ],
                "embedder_anchor": anchored,
                "static_four_checks": {
                    "scope": list(STATIC_CHECK_SCOPE),
                    "commands": list(STATIC_CHECK_COMMANDS),
                    "executed_before_run": True,
                },
                "corpus": {
                    "skills_per_family": len(A_SKILL_IDS),
                    "active_skills_per_family": ACTIVE_SKILL_COUNT,
                    "failed_skills_per_family": len(A_SKILL_IDS) - ACTIVE_SKILL_COUNT,
                    "partitions": ["train", "holdout", "retention"],
                    "gate_partition_label": GATE_PARTITION,
                    "admission_revision": ADMISSION_REVISION,
                    "sourced_train_examples": len(sourced_train_examples),
                    "placebo_train_examples": len(placebo_train_examples),
                    "sourced_ranking_pairs": len(sourced_pairs),
                    "placebo_ranking_pairs": len(placebo_pairs),
                    "gate_records_per_arm": {
                        "a_gate": len(a_gate_records_sourced),
                        "b_gate": len(b_gate_records_sourced),
                    },
                },
                "arms": {
                    "sourced": {
                        "preferred_family": "a",
                        "structure": sourced_structure,
                        "capability_vocabulary": list(sourced_vocabulary),
                        "gate_vocabulary": {
                            "a_gate": list(a_gate_vocabulary),
                            "b_gate": list(b_gate_vocabulary),
                        },
                        "consolidation": sourced_report.to_payload(),
                        "gate_accuracies": {
                            "a_gate": a_gate_sourced,
                            "b_gate": b_gate_sourced,
                        },
                        "checkpoint_roundtrip": sourced_roundtrip,
                        "semantic_surface": sourced_surface,
                    },
                    "placebo": {
                        "preferred_family": "b",
                        "structure": placebo_structure,
                        "capability_vocabulary": list(placebo_vocabulary),
                        "gate_vocabulary": {
                            "a_gate": list(a_gate_vocabulary),
                            "b_gate": list(b_gate_vocabulary),
                        },
                        "consolidation": placebo_report.to_payload(),
                        "gate_accuracies": {
                            "a_gate": a_gate_placebo,
                            "b_gate": b_gate_placebo,
                        },
                        "checkpoint_roundtrip": placebo_roundtrip,
                        "semantic_surface": placebo_surface,
                    },
                },
                "content_transfer": {
                    "a_gate_sourced_accuracy": a_gate_sourced.get("accuracy"),
                    "a_gate_placebo_accuracy": a_gate_placebo.get("accuracy"),
                    "delta_sourced_minus_placebo": content_transfer_delta,
                    "frozen_margin": FROZEN_CONTENT_TRANSFER_MARGIN,
                    "disclosure": (
                        "placebo A-gate accuracy is structurally 0 when admitted: the "
                        "designated labels are sourced-family training ids outside the "
                        "placebo readout vocabulary; b-gate is the symmetric disclosure "
                        "on the sourced arm."
                    ),
                },
                "affordance_specificity": {
                    "sourced_native_holdout_mse": sourced_report.affordance_native_holdout_mse,
                    "sourced_frozen_holdout_mse": sourced_report.affordance_frozen_holdout_mse,
                    "placebo_native_holdout_mse": placebo_report.affordance_native_holdout_mse,
                },
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(outcome == "same_budget_content_benefit_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        "completed: outcome=same_budget_content_benefit_supported; "
                        f"a-gate accuracy sourced={a_gate_sourced.get('accuracy')} "
                        f"placebo={a_gate_placebo.get('accuracy')} "
                        f"delta={content_transfer_delta} >= "
                        f"{FROZEN_CONTENT_TRANSFER_MARGIN} under identical budgets"
                    )
                    if outcome == "same_budget_content_benefit_supported"
                    else (
                        f"completed: outcome={outcome}; frozen gates failed: "
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
    transfer_block = result.get("content_transfer") or {}
    arms = result.get("arms") or {}
    sourced_gate = (arms.get("sourced") or {}).get("gate_accuracies", {}).get("a_gate") or {}
    placebo_gate = (arms.get("placebo") or {}).get("gate_accuracies", {}).get("a_gate") or {}
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "a_gate_accuracy_sourced": sourced_gate.get("accuracy"),
                "a_gate_accuracy_placebo": placebo_gate.get("accuracy"),
                "content_transfer_delta": transfer_block.get("delta_sourced_minus_placebo"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
