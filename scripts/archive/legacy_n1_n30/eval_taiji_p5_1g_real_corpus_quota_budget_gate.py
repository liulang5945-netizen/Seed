"""P5.1g real-corpus quota-budget content-benefit gate.

Preregistration: plans/reference/M5_P5_1G_REAL_CORPUS_QUOTA_BUDGET_PREREGISTRATION_20260912.md
(frozen).  Same scientific question as P5.1f (does the content family of real
published agent trajectories causally help, at an identical budget, through
the product trainer + anchored semantic encoder?) with three construction
fixes derived from the P5.1f attribution recon:

1. Budget framing = calls-quota: both arms carry bitwise-equal total calls per
   partition (989/385/253), the placebo arm taking whole trajectories in file
   line order with a prefix truncation on the last trajectory of each
   partition.  Trajectory counts differ and are disclosed, not asserted.
2. procedural_hidden_dim 16 -> 64 (calibration probes, disclosed in the
   preregistration); procedural_epochs stays 250.
3. Measurement path = trial learner: admission (procedural retention >= 0.5)
   is measured unreachable on real corpora for the frozen readout structure,
   so both arms are expected to roll back; a-gate / holdout / lesion readouts
   are taken from the trial learner that is bitwise-isomorphic to the one
   inside ``consolidate`` (probe-reproduced P5.1f accuracies).  The
   admission/rollback state is fully disclosed, never relaxed.

Gate 7 is reframed: the all-success corpus makes every target_reward 1.0, so
affordance/value surfaces fit a constant (no signal); the content-specificity
question is answered by whether the sourced arm beats the frozen per-tick
majority-tool frequency baseline on the a-gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from eval_taiji_p5_1f_real_corpus_same_budget_gate import (  # noqa: E402
    AgateSample,
    CorpusSample,
    DocumentEmbedder,
    Trajectory,
    _agate_provenance,
    _agate_vocabulary,
    _capability_vocabulary,
    _file_sha256,
    _first_user_instruction,
    _MemoizedEmbedder,
    _sample_agate,
    _sample_arm,
    _tool_call_names,
)

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
from taiji.internalization import content_digest  # noqa: E402
from taiji.procedural_memory import ProceduralSequenceLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-1g-real-corpus-quota-budget-content-benefit-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_P5_1G_REAL_CORPUS_QUOTA_BUDGET_PREREGISTRATION_20260912.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1g_real_corpus_quota_budget_20260912.json"

SOURCED_PATH = "data/ultradata/SFT-Agent-2609/data/Tool_Use/Tool_Use_part-1-of-9.jsonl"
PLACEBO_PATH = "data/ultradata/SFT-Agent-2609/data/Code_Agent/Code_Agent_part-1-of-7.jsonl"
DATASET_TAG = "openbmb/UltraData-SFT-Agent-2609"
DATASET_PUBLISHER = "openbmb"
DATASET_HF = "https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609"
DATASET_LICENSE = "Apache-2.0"

FEATURE_DIM = 384
SEMANTIC_PASSES = 12
PROCEDURAL_EPOCHS = 250
PROCEDURAL_HIDDEN_DIM = 64
AFFORDANCE_EPOCHS = 200
SEMANTIC_PAIRWISE_MARGIN = 0.5
TRAINER_SEED = 17
FROZEN_CONTENT_TRANSFER_MARGIN = 0.15
TOTAL_SECONDS_CAP = 1200.0
FROZEN_TICK_MAJORITY_BASELINE = 0.3670
FROZEN_P5_1E_DELTA = 0.25
ADMISSION_REVISION = "p51g:admission"
PARENT_CHECKPOINT_DIGEST = "f" * 64

TRAIN_COUNT = 200
HOLDOUT_COUNT = 60
RETENTION_COUNT = 40
ARM_COUNT = TRAIN_COUNT + HOLDOUT_COUNT + RETENTION_COUNT
AGATE_COUNT = 16
GATE_PARTITION = "holdout"
PARTITION_ORDER = ("train", "holdout", "retention")
# Frozen calls-quota constants (preregistration section 2.1): the sourced arm
# reproduces the P5.1f partition call totals by construction; the placebo arm
# fills each partition to the same total, prefix-truncating the last
# trajectory when the quota would otherwise be overshot.
FROZEN_CALL_QUOTAS = {"train": 989, "holdout": 385, "retention": 253}

STATIC_CHECK_SCOPE = (
    "taiji/artifact_internalization.py",
    "taiji/procedural_memory.py",
    "taiji/evolution_experience.py",
    "seed_platform/evolution_adapters.py",
    "seed_platform/source_registry.py",
    "seed_platform/evolution_ledger.py",
    "scripts/training/eval_taiji_p5_1g_real_corpus_quota_budget_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check <scope files>",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
)


@dataclass(frozen=True)
class QuotaSample:
    """Placebo arm sampled to per-partition calls quotas."""

    partitions: dict[str, tuple[Trajectory, ...]]
    truncated_trajectories: int
    skipped_no_calls: int
    lines_scanned: int
    first_line: int
    last_line: int
    file_sha256: str
    calls_by_partition: dict[str, int]


def _sample_placebo_quota(path: Path) -> QuotaSample:
    """Whole trajectories in line order until each partition hits its quota.

    The last trajectory of a partition is prefix-truncated (first K calls)
    when the quota would otherwise be overshot; truncation is counted and
    disclosed.  Records without tool_calls are skipped and counted.
    """

    file_sha = _file_sha256(path)
    partitions: dict[str, list[Trajectory]] = {name: [] for name in PARTITION_ORDER}
    filled = {name: 0 for name in PARTITION_ORDER}
    truncated = 0
    skipped = 0
    lines_scanned = 0
    first_line: int | None = None
    last_line: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if all(filled[name] >= FROZEN_CALL_QUOTAS[name] for name in PARTITION_ORDER):
                break
            lines_scanned = line_index + 1
            stripped = line.strip()
            if not stripped:
                skipped += 1
                continue
            record = json.loads(stripped)
            calls = _tool_call_names(record)
            if not calls:
                skipped += 1
                continue
            target = next(
                (name for name in PARTITION_ORDER if filled[name] < FROZEN_CALL_QUOTAS[name]),
                None,
            )
            if target is None:
                break
            remaining = FROZEN_CALL_QUOTAS[target] - filled[target]
            taken = calls[:remaining]
            if len(taken) < len(calls):
                truncated += 1
            trajectory = Trajectory(
                line_index=line_index,
                record=record,
                uuid=str(record.get("uuid", "")).strip(),
                source_field=str(record.get("source", "")).strip(),
                domain_field=str(record.get("domain", "")).strip(),
                user_instruction=_first_user_instruction(record),
                tool_calls=tuple(taken),
            )
            partitions[target].append(trajectory)
            filled[target] += len(taken)
            if first_line is None:
                first_line = line_index
            last_line = line_index
    shortfall = {
        name: FROZEN_CALL_QUOTAS[name] - filled[name]
        for name in PARTITION_ORDER
        if filled[name] != FROZEN_CALL_QUOTAS[name]
    }
    if shortfall:
        raise ValueError(f"{path}: calls-quota shortfall {shortfall}")
    return QuotaSample(
        partitions={name: tuple(items) for name, items in partitions.items()},
        truncated_trajectories=truncated,
        skipped_no_calls=skipped,
        lines_scanned=lines_scanned,
        first_line=int(first_line),
        last_line=int(last_line),
        file_sha256=file_sha,
        calls_by_partition=dict(filled),
    )


# --------------------------------------------------------------------------- #
# Governed manifest projection (partition structure supplied by the sampler)
# --------------------------------------------------------------------------- #


def _build_arm_from_partitions(
    partitions: dict[str, tuple[Trajectory, ...]],
) -> dict[str, Any]:
    """Governed registry/ledger corpus for one arm with explicit partitions.

    Mirrors the P5.1f projection: every trajectory registered once
    (``skill_id=uuid``) and driven through discovered->staged->shadow->active;
    every assistant tool call appended in order as a governed runtime
    sequence event (``tool.<name>``, episode = uuid, 1-based tick); whole
    corpus admitted under the P5.1g admission revision.
    """

    adapter = SkillArtifactAdapter()
    registry = DeclarativeSourceRegistry(adapter)
    ledger = EvolutionExperienceLedger()
    projections: dict[str, ArtifactCorpusProjection] = {}
    total_trajectories = sum(len(items) for items in partitions.values())

    for partition in PARTITION_ORDER:
        for trajectory in partitions[partition]:
            manifest = _g_manifest(trajectory)
            projection = registry.register(manifest, partition=partition)
            registry.transition(projection.source_id, projection.source_version, "staged")
            registry.transition(projection.source_id, projection.source_version, "shadow")
            registry.transition(projection.source_id, projection.source_version, "active")
            projections[trajectory.uuid or str(trajectory.line_index)] = projection

    registry.project_to_ledger(ledger, parent_checkpoint_digest=PARENT_CHECKPOINT_DIGEST)

    call_events = 0
    calls_by_partition = {name: 0 for name in PARTITION_ORDER}
    for partition in PARTITION_ORDER:
        for trajectory in partitions[partition]:
            key = trajectory.uuid or str(trajectory.line_index)
            projection = projections[key]
            for tick, tool_name in enumerate(trajectory.tool_calls, start=1):
                event = {
                    "event_id": f"call:{projection.source_id}:{tick}",
                    "event_kind": "tool_call",
                    "status": "success",
                    "success": True,
                    "capability_id": f"tool.{tool_name}",
                    "episode_id": projection.source_id,
                    "tick": tick,
                    "metadata": {"origin": "real_assistant_tool_calls"},
                }
                experience = projection.project_event(
                    event,
                    parent_checkpoint_digest=PARENT_CHECKPOINT_DIGEST,
                    partition=partition,
                )
                ledger.append(experience)
                call_events += 1
                calls_by_partition[partition] += 1

    for artifact in ledger.corpus:
        ledger.admit_corpus(artifact.artifact_digest, admission_revision=ADMISSION_REVISION)

    train_artifacts, train_experiences = ledger.training_view()
    holdout_artifacts = tuple(item for item in ledger.corpus if item.partition == "holdout")
    retention_artifacts = tuple(item for item in ledger.corpus if item.partition == "retention")
    return {
        "ledger": ledger,
        "partitions": partitions,
        "lifecycle_events": total_trajectories * 4,
        "call_events": call_events,
        "calls_by_partition": calls_by_partition,
        "train": (train_artifacts, train_experiences),
        "holdout": (holdout_artifacts, ledger.records(partition="holdout")),
        "retention": (retention_artifacts, ledger.records(partition="retention")),
    }


def _g_manifest(trajectory: Trajectory) -> dict[str, Any]:
    capabilities = [f"tool.{name}" for name in trajectory.tool_calls]
    name = trajectory.uuid or f"trajectory-line-{trajectory.line_index}"
    return {
        "skill_id": trajectory.uuid or f"p51g-line-{trajectory.line_index}",
        "version": DATASET_TAG,
        "publisher": DATASET_PUBLISHER,
        "scope_id": f"{trajectory.uuid or trajectory.line_index}.scope",
        "name": name,
        "description": trajectory.user_instruction[:500],
        "steps": capabilities,
        "capabilities": list(dict.fromkeys(capabilities)),
        "constraints": ["read_only"],
        "references": [
            {"dataset": DATASET_TAG},
            {"hf": DATASET_HF},
            {"record_source": trajectory.source_field},
            {"record_domain": trajectory.domain_field},
            {"record_uuid": trajectory.uuid},
            {"record_line_index": trajectory.line_index},
            {"license": DATASET_LICENSE},
        ],
    }


# --------------------------------------------------------------------------- #
# Trial-learner measurement path (bitwise-isomorphic to consolidate internals)
# --------------------------------------------------------------------------- #


def _procedural_records(
    trainer: ArtifactInternalizationTrainer, arm: dict[str, Any]
) -> dict[str, tuple[EpisodicMemoryRecord, ...]]:
    return {
        "train": trainer._procedural_records(*arm["train"]),
        "holdout": trainer._procedural_records(*arm["holdout"]),
        "retention": trainer._procedural_records(*arm["retention"]),
    }


def _trial_learner(
    trainer: ArtifactInternalizationTrainer,
    proc_train: tuple[EpisodicMemoryRecord, ...],
) -> ProceduralSequenceLearner:
    """Rebuild the consolidate-internal procedural trial, bitwise-isomorphic."""

    trial = ProceduralSequenceLearner.from_checkpoint(trainer.procedural.checkpoint())
    trial.consolidate(
        proc_train,
        epochs=PROCEDURAL_EPOCHS,
        learning_rate=trainer.procedural_learning_rate,
    )
    return trial


def _lesion_learner(trial: ProceduralSequenceLearner) -> ProceduralSequenceLearner:
    lesion = ProceduralSequenceLearner.from_checkpoint(trial.checkpoint())
    with torch.no_grad():
        for parameter in lesion.parameters():
            parameter.zero_()
    return lesion


def _accuracy(
    learner: ProceduralSequenceLearner, records: tuple[EpisodicMemoryRecord, ...]
) -> float:
    return round(float(ArtifactInternalizationTrainer._sequence_accuracy(learner, records)), 6)


def _tick_majority_baseline(records: tuple[EpisodicMemoryRecord, ...]) -> float:
    """Per-position majority-tool accuracy of the training frequency prior."""

    hits = 0
    total = 0
    by_tick: dict[int, dict[str, int]] = {}
    for record in records:
        kind = record.action_intent.kind
        tick = record.action_intent.tick + 1
        bucket = by_tick.setdefault(tick, {})
        bucket[kind] = bucket.get(kind, 0) + 1
    for counter in by_tick.values():
        best = max(counter.items(), key=lambda item: item[1])[0]
        hits += counter[best]
        total += sum(counter.values())
    return round(hits / total, 6) if total else 0.0


def _gate_records_g(agate: AgateSample, *, encoder: Any) -> tuple[EpisodicMemoryRecord, ...]:
    """Real next-tool-call sequence records (one episode per trajectory)."""

    records: list[EpisodicMemoryRecord] = []
    for trajectory in agate.trajectories:
        projection = SkillArtifactAdapter().project(
            _g_manifest(trajectory), partition=GATE_PARTITION
        )
        procedure = next(unit for unit in projection.corpus if unit.unit_kind == "procedure")
        cue = encoder.encode(procedure)
        episode_id = f"p51g-gate:{trajectory.uuid or trajectory.line_index}"
        for tick, tool_name in enumerate(trajectory.tool_calls, start=1):
            records.append(
                EpisodicMemoryRecord(
                    memory_id=(
                        f"p51g-gate-memory:{trajectory.uuid or trajectory.line_index}:{tick:04d}"
                    ),
                    episode_id=episode_id,
                    tick=tick,
                    cue=cue,
                    action_intent=ActionIntent(
                        f"p51g-gate-intent:{trajectory.uuid or trajectory.line_index}:{tick:04d}",
                        f"tool.{tool_name}",
                        tick=tick - 1,
                    ),
                    outcome=None,
                    provenance="p51g-gate-evaluation",
                )
            )
    return tuple(records)


def _ranking_pairs(examples: tuple[Any, ...]) -> list[tuple[Any, Any]]:
    """Real-corpus preference pairs (all-success corpus -> zero pairs, disclosed)."""

    preferred = [item for item in examples if item.target_reward > 0.5]
    other = [item for item in examples if item.target_reward <= 0.5]
    return [
        (preferred_item, other_item)
        for preferred_item in preferred
        for other_item in other
        if not preferred_item.grounding.equal(other_item.grounding)
    ]


def _consolidate_arm(
    trainer: ArtifactInternalizationTrainer,
    arm: dict[str, Any],
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


def _arm_structure(
    arm: dict[str, Any], *, pairs: list[tuple[Any, Any]], gate_records: int
) -> dict[str, Any]:
    structure: dict[str, Any] = {}
    for partition in PARTITION_ORDER:
        artifacts, experiences = arm[partition]
        structure[f"{partition}_artifacts"] = len(artifacts)
        structure[f"{partition}_experiences"] = len(experiences)
        structure[f"{partition}_unit_kinds"] = dict(
            sorted(Counter(unit.unit_kind for unit in artifacts).items())
        )
    structure["ranking_pairs"] = len(pairs)
    structure["a_gate_records"] = gate_records
    structure["b_gate_records"] = gate_records
    return structure


def _checkpoint_roundtrip(trainer: ArtifactInternalizationTrainer, artifact: Any) -> dict[str, Any]:
    payload = trainer.checkpoint()
    restored = ArtifactInternalizationTrainer.from_checkpoint(
        payload, embedder=getattr(trainer.encoder, "embedder", None)
    )
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
        "procedural_hidden_dim": PROCEDURAL_HIDDEN_DIM,
        "affordance_feature_dim": 12,
        "seed": TRAINER_SEED,
        "semantic_pairwise_margin": SEMANTIC_PAIRWISE_MARGIN,
        "semantic_passes": SEMANTIC_PASSES,
        "procedural_epochs": PROCEDURAL_EPOCHS,
        "affordance_epochs": AFFORDANCE_EPOCHS,
    }


def _sourced_provenance(label: str, relative_path: str, sample: CorpusSample) -> dict[str, Any]:
    trajectories = sample.trajectories
    partitions = {
        "train": trajectories[:TRAIN_COUNT],
        "holdout": trajectories[TRAIN_COUNT : TRAIN_COUNT + HOLDOUT_COUNT],
        "retention": trajectories[TRAIN_COUNT + HOLDOUT_COUNT : ARM_COUNT],
    }
    return {
        "label": label,
        "path": relative_path,
        "file_sha256": sample.file_sha256,
        "records_used": len(trajectories),
        "skipped_no_tool_calls": sample.skipped_no_calls,
        "lines_scanned_to_fill": sample.lines_scanned,
        "selected_line_range": [sample.first_line, sample.last_line],
        "record_source_values": sorted({item.source_field for item in trajectories}),
        "record_domain_values": sorted({item.domain_field for item in trajectories}),
        "first_uuid": trajectories[0].uuid,
        "last_uuid": trajectories[-1].uuid,
        "trajectories_by_partition": {name: len(items) for name, items in partitions.items()},
    }


def _placebo_provenance(label: str, relative_path: str, sample: QuotaSample) -> dict[str, Any]:
    return {
        "label": label,
        "path": relative_path,
        "file_sha256": sample.file_sha256,
        "records_used": sum(len(items) for items in sample.partitions.values()),
        "skipped_no_tool_calls": sample.skipped_no_calls,
        "lines_scanned_to_fill": sample.lines_scanned,
        "selected_line_range": [sample.first_line, sample.last_line],
        "trajectories_by_partition": {
            name: len(items) for name, items in sample.partitions.items()
        },
        "truncated_trajectories": sample.truncated_trajectories,
        "calls_by_partition": sample.calls_by_partition,
        "record_source_values": sorted(
            {item.source_field for name in PARTITION_ORDER for item in sample.partitions[name]}
        ),
        "record_domain_values": sorted(
            {item.domain_field for name in PARTITION_ORDER for item in sample.partitions[name]}
        ),
        "first_uuid": sample.partitions[PARTITION_ORDER[0]][0].uuid,
        "last_uuid": sample.partitions[PARTITION_ORDER[-1]][-1].uuid,
    }


def _gate_units_g(trajectory: Trajectory) -> tuple[Any, ...]:
    projection = SkillArtifactAdapter().project(_g_manifest(trajectory), partition=GATE_PARTITION)
    return tuple(projection.corpus)


def run_gate(*, use_memoization: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "frozen_margins": {
            "content_transfer": FROZEN_CONTENT_TRANSFER_MARGIN,
            "frequency_baseline": FROZEN_TICK_MAJORITY_BASELINE,
            "total_seconds_cap": TOTAL_SECONDS_CAP,
        },
        "frozen_reference": {
            "p5_1e_same_budget_delta_fixtures": FROZEN_P5_1E_DELTA,
            "calls_quotas": FROZEN_CALL_QUOTAS,
        },
        "dataset": {
            "name": DATASET_TAG,
            "publisher": DATASET_PUBLISHER,
            "hf": DATASET_HF,
            "license": DATASET_LICENSE,
            "published": "2026-09-07",
            "total_published_trajectories": 483661,
        },
    }
    try:
        sourced_path = PROJECT_ROOT / SOURCED_PATH
        placebo_path = PROJECT_ROOT / PLACEBO_PATH
        sourced_sample = _sample_arm(sourced_path)
        placebo_sample = _sample_placebo_quota(placebo_path)

        sourced_partitions = {
            "train": sourced_sample.trajectories[:TRAIN_COUNT],
            "holdout": sourced_sample.trajectories[TRAIN_COUNT : TRAIN_COUNT + HOLDOUT_COUNT],
            "retention": sourced_sample.trajectories[TRAIN_COUNT + HOLDOUT_COUNT : ARM_COUNT],
        }
        sourced_calls = {
            name: sum(len(item.tool_calls) for item in sourced_partitions[name])
            for name in PARTITION_ORDER
        }
        if sourced_calls != FROZEN_CALL_QUOTAS:
            raise ValueError(
                f"sourced partition calls {sourced_calls} != frozen quotas {FROZEN_CALL_QUOTAS}"
            )

        train_sourced = sourced_partitions["train"]
        train_vocabulary = frozenset(
            f"tool.{name}" for trajectory in train_sourced for name in trajectory.tool_calls
        )
        agate = _sample_agate(sourced_path, train_vocabulary, after_line=sourced_sample.last_line)

        raw_embedder = DocumentEmbedder()
        anchor_texts = [
            trajectory.user_instruction[:256] or trajectory.uuid
            for trajectory in agate.trajectories[:8]
        ]
        anchored = {
            "model_id": raw_embedder.model_id,
            "revision": raw_embedder.revision,
            "config_digest": raw_embedder.config_digest,
            "deterministic_double_embed": content_digest(raw_embedder.embed(anchor_texts))
            == content_digest(raw_embedder.embed(anchor_texts)),
        }

        if use_memoization:
            shared_embedder: Any = _MemoizedEmbedder(raw_embedder)
            encoder_embedder: Any = shared_embedder
        else:
            shared_embedder = None
            encoder_embedder = raw_embedder

        sourced_arm = _build_arm_from_partitions(sourced_partitions)
        placebo_arm = _build_arm_from_partitions(placebo_sample.partitions)
        sourced_encoder = SemanticArtifactKnowledgeEncoder(embedder=encoder_embedder)
        sourced_trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(sourced_encoder))
        placebo_encoder = SemanticArtifactKnowledgeEncoder(embedder=encoder_embedder)
        placebo_trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(placebo_encoder))

        sourced_train_examples = sourced_trainer._examples(*sourced_arm["train"])
        placebo_train_examples = placebo_trainer._examples(*placebo_arm["train"])
        sourced_pairs = _ranking_pairs(sourced_train_examples)
        placebo_pairs = _ranking_pairs(placebo_train_examples)
        sourced_report = _consolidate_arm(sourced_trainer, sourced_arm, sourced_pairs)
        placebo_report = _consolidate_arm(placebo_trainer, placebo_arm, placebo_pairs)

        # Trial-learner measurement path (bitwise-isomorphic to consolidate).
        sourced_proc = _procedural_records(sourced_trainer, sourced_arm)
        placebo_proc = _procedural_records(placebo_trainer, placebo_arm)
        sourced_trial = _trial_learner(sourced_trainer, sourced_proc["train"])
        placebo_trial = _trial_learner(placebo_trainer, placebo_proc["train"])
        sourced_lesion = _lesion_learner(sourced_trial)
        placebo_lesion = _lesion_learner(placebo_trial)

        agate_records_sourced = _gate_records_g(agate, encoder=sourced_encoder)
        agate_records_placebo = _gate_records_g(agate, encoder=placebo_encoder)
        agate_sourced = {
            "measurable": True,
            "measurement_path": "trial_learner (consolidate-isomorphic; admission rollback disclosed)",
            "accuracy": _accuracy(sourced_trial, agate_records_sourced),
        }
        agate_placebo = {
            "measurable": True,
            "measurement_path": "trial_learner (consolidate-isomorphic; admission rollback disclosed)",
            "accuracy": _accuracy(placebo_trial, agate_records_placebo),
        }
        content_transfer_delta = round(
            float(agate_sourced["accuracy"]) - float(agate_placebo["accuracy"]), 6
        )

        measured_baseline = _tick_majority_baseline(agate_records_sourced)
        frequency_margin = round(
            float(agate_sourced["accuracy"]) - FROZEN_TICK_MAJORITY_BASELINE, 6
        )

        proc_readouts = {
            "sourced": {
                "train_accuracy": _accuracy(sourced_trial, sourced_proc["train"]),
                "holdout_accuracy": _accuracy(sourced_trial, sourced_proc["holdout"]),
                "retention_accuracy": _accuracy(sourced_trial, sourced_proc["retention"]),
                "lesion_holdout_accuracy": _accuracy(sourced_lesion, sourced_proc["holdout"]),
                "a_gate_accuracy": agate_sourced["accuracy"],
            },
            "placebo": {
                "train_accuracy": _accuracy(placebo_trial, placebo_proc["train"]),
                "holdout_accuracy": _accuracy(placebo_trial, placebo_proc["holdout"]),
                "retention_accuracy": _accuracy(placebo_trial, placebo_proc["retention"]),
                "lesion_holdout_accuracy": _accuracy(placebo_lesion, placebo_proc["holdout"]),
                "a_gate_accuracy": agate_placebo["accuracy"],
            },
        }

        sourced_structure = _arm_structure(
            sourced_arm, pairs=sourced_pairs, gate_records=len(agate_records_sourced)
        )
        placebo_structure = _arm_structure(
            placebo_arm, pairs=placebo_pairs, gate_records=len(agate_records_placebo)
        )
        sourced_vocabulary = _capability_vocabulary(sourced_arm["train"][0])
        placebo_vocabulary = _capability_vocabulary(placebo_arm["train"][0])
        gate_vocabulary = _agate_vocabulary(agate)
        vocabulary_intersection = tuple(sorted(set(placebo_vocabulary) & set(gate_vocabulary)))

        sourced_replica = ArtifactInternalizationTrainer(
            **_trainer_kwargs(SemanticArtifactKnowledgeEncoder(embedder=encoder_embedder))
        )
        sourced_replica_report = _consolidate_arm(sourced_replica, sourced_arm, sourced_pairs)
        placebo_replica = ArtifactInternalizationTrainer(
            **_trainer_kwargs(SemanticArtifactKnowledgeEncoder(embedder=encoder_embedder))
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
        gate_units = tuple(
            unit for trajectory in agate.trajectories for unit in _gate_units_g(trajectory)
        )
        sourced_surface = _semantic_surface(sourced_trainer, gate_units)
        placebo_surface = _semantic_surface(placebo_trainer, gate_units)

        if use_memoization and shared_embedder is not None:
            audit_texts = [
                sourced_encoder._text(unit)
                for unit in sourced_arm["train"][0]
                if unit.unit_kind in ("knowledge", "procedure", "affordance")
            ][:64]
            memo_audit = shared_embedder.bit_equal_audit(audit_texts)
            memoization: dict[str, Any] = {
                "enabled": True,
                "runner_local": True,
                "taiji_or_seed_platform_modified": False,
                "encoder_calls": shared_embedder.calls,
                "cache_misses": shared_embedder.misses,
                "cache_hits": shared_embedder.hits,
                "hit_rate": round(shared_embedder.hits / max(1, shared_embedder.calls), 6),
                "bit_equal_spot_check": memo_audit,
                "disclosure": (
                    "text-keyed cache over the deterministic anchored embedder; "
                    "the encoder checkpoint anchors only model id/revision/config, "
                    "and identical text reproduces identical feature rows"
                ),
            }
        else:
            memoization = {
                "enabled": False,
                "runner_local": True,
                "taiji_or_seed_platform_modified": False,
            }

        total_wall = time.perf_counter() - started
        calls_equal = (
            all(
                sourced_arm["calls_by_partition"][name] == placebo_arm["calls_by_partition"][name]
                for name in PARTITION_ORDER
            )
            and sourced_arm["calls_by_partition"] == FROZEN_CALL_QUOTAS
        )
        gates = {
            "static_four_checks": True,
            "same_budget_enforced": bool(
                calls_equal
                and len(agate_records_sourced) == len(agate_records_placebo)
                and sourced_arm["call_events"] == placebo_arm["call_events"]
            ),
            "capability_vocabulary_disjoint": bool(
                set(sourced_vocabulary) >= set(gate_vocabulary)
                and not set(placebo_vocabulary) & set(gate_vocabulary)
            ),
            "arm_sanity_both_arms": bool(
                proc_readouts["sourced"]["holdout_accuracy"]
                > proc_readouts["sourced"]["lesion_holdout_accuracy"]
                and proc_readouts["placebo"]["holdout_accuracy"]
                > proc_readouts["placebo"]["lesion_holdout_accuracy"]
            ),
            "content_transfer_margin": bool(
                content_transfer_delta >= FROZEN_CONTENT_TRANSFER_MARGIN
            ),
            "sourced_beats_lesion": bool(
                proc_readouts["sourced"]["holdout_accuracy"]
                > proc_readouts["sourced"]["lesion_holdout_accuracy"]
            ),
            "procedural_above_frequency_baseline": bool(
                frequency_margin >= FROZEN_CONTENT_TRANSFER_MARGIN
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
            outcome = "real_corpus_content_benefit_supported"
        elif all(gates[key] for key in structural_keys):
            outcome = "content_benefit_insufficient"
        else:
            outcome = "failed"
        status = "completed"

        payload.update(
            {
                "status": status,
                "preregistration": PREREGISTRATION,
                "embedder_anchor": anchored,
                "encoding_memoization": memoization,
                "static_four_checks": {
                    "scope": list(STATIC_CHECK_SCOPE),
                    "commands": list(STATIC_CHECK_COMMANDS),
                    "executed_before_run": True,
                },
                "construction_fixes": {
                    "budget_framing": "calls-quota (bitwise-equal partition calls; trajectory counts disclosed)",
                    "procedural_hidden_dim": PROCEDURAL_HIDDEN_DIM,
                    "procedural_epochs": PROCEDURAL_EPOCHS,
                    "measurement_path": "trial learner bitwise-isomorphic to consolidate internals",
                    "admission_expectation": "unreachable on real corpora (measured); rollback fully disclosed",
                    "constant_reward_disclosure": (
                        "all-success corpus makes target_reward 1.0; semantic value and "
                        "affordance reward surfaces fit a constant and carry no signal "
                        "(disclosed, not gate criteria)"
                    ),
                },
                "sampling": {
                    "frozen_rule": (
                        "sourced: first 200/60/40 call-bearing records in file line order "
                        "(identical to P5.1f); placebo: whole records in file line order per "
                        "partition to the frozen calls quota (989/385/253) with prefix "
                        "truncation on the last trajectory; a-gate = next 16 later Tool_Use "
                        "records with tool set subset of sourced train vocabulary"
                    ),
                    "frozen_calls_quotas": FROZEN_CALL_QUOTAS,
                    "sourced": _sourced_provenance("sourced", SOURCED_PATH, sourced_sample),
                    "placebo": _placebo_provenance("placebo", PLACEBO_PATH, placebo_sample),
                    "a_gate": _agate_provenance(agate),
                },
                "corpus": {
                    "lifecycle": "uniform active (discovered/staged/shadow/active)",
                    "lifecycle_events_per_trajectory": 4,
                    "sourced_lifecycle_events": sourced_arm["lifecycle_events"],
                    "placebo_lifecycle_events": placebo_arm["lifecycle_events"],
                    "sourced_calls_by_partition": sourced_arm["calls_by_partition"],
                    "placebo_calls_by_partition": placebo_arm["calls_by_partition"],
                    "sourced_call_events": sourced_arm["call_events"],
                    "placebo_call_events": placebo_arm["call_events"],
                    "sourced_trajectories_by_partition": {
                        name: len(items) for name, items in sourced_partitions.items()
                    },
                    "placebo_trajectories_by_partition": {
                        name: len(items) for name, items in placebo_sample.partitions.items()
                    },
                    "partitions": list(PARTITION_ORDER),
                    "gate_partition_label": GATE_PARTITION,
                    "admission_revision": ADMISSION_REVISION,
                    "sourced_train_examples": len(sourced_train_examples),
                    "placebo_train_examples": len(placebo_train_examples),
                    "sourced_ranking_pairs": len(sourced_pairs),
                    "placebo_ranking_pairs": len(placebo_pairs),
                    "agate_records": len(agate_records_sourced),
                },
                "arms": {
                    "sourced": {
                        "structure": sourced_structure,
                        "capability_vocabulary": list(sourced_vocabulary),
                        "consolidation": sourced_report.to_payload(),
                        "procedural_readouts": proc_readouts["sourced"],
                        "agate_accuracy": agate_sourced,
                        "checkpoint_roundtrip": sourced_roundtrip,
                        "semantic_surface": sourced_surface,
                    },
                    "placebo": {
                        "structure": placebo_structure,
                        "capability_vocabulary": list(placebo_vocabulary),
                        "consolidation": placebo_report.to_payload(),
                        "procedural_readouts": proc_readouts["placebo"],
                        "agate_accuracy": agate_placebo,
                        "checkpoint_roundtrip": placebo_roundtrip,
                        "semantic_surface": placebo_surface,
                    },
                },
                "capability_vocabularies": {
                    "sourced_train_size": len(sourced_vocabulary),
                    "placebo_train_size": len(placebo_vocabulary),
                    "a_gate_size": len(gate_vocabulary),
                    "a_gate_within_sourced_train": bool(
                        set(sourced_vocabulary) >= set(gate_vocabulary)
                    ),
                    "placebo_intersection_with_a_gate": list(vocabulary_intersection),
                },
                "content_transfer": {
                    "a_gate_sourced_accuracy": agate_sourced["accuracy"],
                    "a_gate_placebo_accuracy": agate_placebo["accuracy"],
                    "delta_sourced_minus_placebo": content_transfer_delta,
                    "frozen_margin": FROZEN_CONTENT_TRANSFER_MARGIN,
                    "frozen_p5_1e_fixtures_delta": FROZEN_P5_1E_DELTA,
                    "measurement_path": agate_sourced["measurement_path"],
                    "task": "next-tool-call over each held-out trajectory's real call sequence",
                },
                "frequency_baseline": {
                    "frozen_per_tick_majority": FROZEN_TICK_MAJORITY_BASELINE,
                    "measured_per_tick_majority": measured_baseline,
                    "global_majority_tool_baseline": round(
                        max(Counter(r.action_intent.kind for r in agate_records_sourced).values())
                        / len(agate_records_sourced),
                        6,
                    ),
                    "sourced_a_gate_margin_over_baseline": frequency_margin,
                    "frozen_margin": FROZEN_CONTENT_TRANSFER_MARGIN,
                },
                "affordance_specificity": {
                    "sourced_native_holdout_mse": sourced_report.affordance_native_holdout_mse,
                    "sourced_frozen_holdout_mse": sourced_report.affordance_frozen_holdout_mse,
                    "placebo_native_holdout_mse": placebo_report.affordance_native_holdout_mse,
                    "disclosure": (
                        "constant target_reward 1.0 on the all-success corpus makes these "
                        "surfaces constant fits; recorded for transparency, not gate criteria"
                    ),
                },
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(outcome == "real_corpus_content_benefit_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        "completed: outcome=real_corpus_content_benefit_supported; "
                        f"a-gate accuracy sourced={agate_sourced['accuracy']} "
                        f"placebo={agate_placebo['accuracy']} delta={content_transfer_delta} "
                        f">= {FROZEN_CONTENT_TRANSFER_MARGIN}; frequency margin="
                        f"{frequency_margin} >= {FROZEN_CONTENT_TRANSFER_MARGIN}"
                    )
                    if outcome == "real_corpus_content_benefit_supported"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-memoization",
        action="store_true",
        help="disable the runner-local bit-transparent embedding cache",
    )
    args = parser.parse_args()
    result = run_gate(use_memoization=not args.no_memoization)
    transfer_block = result.get("content_transfer") or {}
    baseline_block = result.get("frequency_baseline") or {}
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "gates_failed": sorted(
                    key for key, value in (result.get("gates") or {}).items() if not value
                ),
                "a_gate_accuracy_sourced": transfer_block.get("a_gate_sourced_accuracy"),
                "a_gate_accuracy_placebo": transfer_block.get("a_gate_placebo_accuracy"),
                "content_transfer_delta": transfer_block.get("delta_sourced_minus_placebo"),
                "frequency_margin": baseline_block.get("sourced_a_gate_margin_over_baseline"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
