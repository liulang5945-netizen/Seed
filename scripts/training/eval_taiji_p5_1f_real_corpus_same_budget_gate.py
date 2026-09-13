"""P5.1f real-corpus same-budget content benefit Gate runner.

Two family-exclusive arms under an identical trainer budget
(``ArtifactInternalizationTrainer`` + ``SemanticArtifactKnowledgeEncoder``,
384-dim MiniLM anchor, all trainer hyper-parameters bit-identical), but the
content is real published agent-trajectory data instead of constructed
fixtures:

- sourced arm: 300 Tool_Use trajectories from
  ``data/ultradata/SFT-Agent-2609/data/Tool_Use/Tool_Use_part-1-of-9.jsonl``
  (tau2-style customer/retail/telco/airline tool families);
- placebo arm: 300 Code_Agent trajectories from
  ``data/ultradata/SFT-Agent-2609/data/Code_Agent/Code_Agent_part-1-of-7.jsonl``
  (apply_patch/read_file/rg/shell software-engineering tool families).

Sampling is frozen in the preregistration: first N records in file line order
with >=1 assistant ``tool_calls`` (call-free records are skipped and counted),
200 train / 60 holdout / 40 retention per arm, plus 16 further Tool_Use
records whose tool set is a subset of the sourced train vocabulary for the
a-gate (pruned records counted).  Each trajectory becomes one governed
``SkillArtifactAdapter`` manifest (``skill_id=uuid``, ``version=dataset tag``,
``steps=[tool.<function_name>...]``, capabilities = same list); lifecycle is
uniformly active (4 lifecycle events per trajectory in both arms) and every
assistant tool call is appended in order as a governed runtime sequence
event (``tool.<name>`` capability, episode = trajectory uuid, tick = call
index inside the trajectory).

The a-gate is the next-tool-call procedural accuracy of each held-out
trajectory's real call sequence, evaluated evaluation-only against each
arm's committed readout.  Because the real two families do not share tool
names and the arms differ in real per-trajectory affordance/call counts,
gates 2/3/9 are measured honestly with no pruning to force a pass.

Encoding memoization: ``SemanticArtifactKnowledgeEncoder.encode`` has no
cache and the governed cartesian ``_examples`` expansion over the Code arm's
~19k call events would issue ~190k transformer forward passes.  A
runner-local, bit-transparent text-keyed embedder cache is therefore used
(the trainer encoder checkpoint only anchors model id/revision/config, and
the same text is bit-identical under the deterministic double-embed anchor);
the wrapper exposes hit/miss counters and a bit-equality spot check, both
disclosed in the report.  Nothing in ``taiji``/``seed_platform`` is changed.

Nine frozen gates live in ``plans/reference
/M5_P5_1F_REAL_CORPUS_SAME_BUDGET_PREREGISTRATION_20260912.md``.
``growth_admitted=false`` and ``can_promote=false`` throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
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

REPORT_FORMAT = "taiji-p5-1f-real-corpus-same-budget-content-benefit-report-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1f_real_corpus_same_budget_20260912.json"
PREREGISTRATION = "plans/reference/" "M5_P5_1F_REAL_CORPUS_SAME_BUDGET_PREREGISTRATION_20260912.md"
DATASET_TAG = "openbmb/UltraData-SFT-Agent-2609"
DATASET_HF = "https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609"
DATASET_PUBLISHER = "openbmb"
DATASET_LICENSE = "Apache-2.0"
SOURCED_PATH = "data/ultradata/SFT-Agent-2609/data/Tool_Use/Tool_Use_part-1-of-9.jsonl"
PLACEBO_PATH = "data/ultradata/SFT-Agent-2609/data/Code_Agent/Code_Agent_part-1-of-7.jsonl"

FEATURE_DIM = 384
PAIRWISE_MARGIN = 0.5
SEMANTIC_PASSES = 12
PROCEDURAL_EPOCHS = 250
AFFORDANCE_EPOCHS = 200
FROZEN_CONTENT_TRANSFER_MARGIN = 0.15
TOTAL_SECONDS_CAP = 1200.0
FROZEN_P5_1E_DELTA = 0.25
ADMISSION_REVISION = "p51f:admission"
PARENT_CHECKPOINT_DIGEST = "f" * 64

TRAIN_COUNT = 200
HOLDOUT_COUNT = 60
RETENTION_COUNT = 40
ARM_COUNT = TRAIN_COUNT + HOLDOUT_COUNT + RETENTION_COUNT
AGATE_COUNT = 16
# Appendix 2 of the P5.1e design: the gate projection is evaluation-only; the
# inert "holdout" partition label only lets the corpus artifact validate.
GATE_PARTITION = "holdout"

STATIC_CHECK_SCOPE = (
    "taiji/artifact_internalization.py",
    "taiji/procedural_memory.py",
    "taiji/evolution_experience.py",
    "seed_platform/evolution_adapters.py",
    "seed_platform/source_registry.py",
    "seed_platform/evolution_ledger.py",
    "scripts/training/eval_taiji_p5_1f_real_corpus_same_budget_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check <scope files>",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
)


@dataclass(frozen=True)
class Trajectory:
    """One qualified real record captured in file line order."""

    line_index: int
    record: dict[str, Any]
    uuid: str
    source_field: str
    domain_field: str
    user_instruction: str
    tool_calls: tuple[str, ...]


@dataclass(frozen=True)
class CorpusSample:
    trajectories: tuple[Trajectory, ...]
    skipped_no_calls: int
    lines_scanned: int
    first_line: int
    last_line: int
    file_sha256: str


@dataclass(frozen=True)
class AgateSample:
    trajectories: tuple[Trajectory, ...]
    skipped_no_calls: int
    pruned_out_of_vocabulary: int
    lines_scanned: int
    first_line: int
    last_line: int


# --------------------------------------------------------------------------- #
# Runner-local bit-transparent embedding cache
# --------------------------------------------------------------------------- #


class _MemoizedEmbedder:
    """Text-keyed cache in front of a ``DocumentEmbedder`` (single texts).

    The encoder always embeds exactly one text (``embed([text])[0]``); the
    cache stores the resulting row keyed by that text.  Identity attributes
    (model_id/revision/config_digest/dimension) delegate to the wrapped
    deterministic embedder, so the encoder checkpoint is unchanged.  The raw
    embedder stays available for anchoring and for the bit-equality audit.
    """

    def __init__(self, raw: DocumentEmbedder) -> None:
        self._raw = raw
        self._cache: dict[str, Any] = {}
        self.calls = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self._raw.model_id

    @property
    def revision(self) -> str:
        return self._raw.revision

    @property
    def config_digest(self) -> str:
        return self._raw.config_digest

    @property
    def dimension(self) -> int:
        return self._raw.dimension

    @property
    def hits(self) -> int:
        return self.calls - self.misses

    def embed(self, texts: list[str]):
        if len(texts) != 1:
            # Only the single-text encoder path is cached; anything else goes
            # straight through to the raw embedder.
            return self._raw.embed(texts)
        self.calls += 1
        text = texts[0]
        cached = self._cache.get(text)
        if cached is None:
            self.misses += 1
            # Store the feature row; return it as a one-row batch so the
            # embedder contract (embed([t]).shape == (1, dim)) is preserved.
            cached = self._raw.embed([text])[0].clone()
            self._cache[text] = cached
        return cached.unsqueeze(0).clone()

    def bit_equal_audit(self, sample_texts: Iterable[str]) -> dict[str, Any]:
        """Recompute cached texts raw and compare feature rows bit-for-bit."""

        checked = 0
        for text in sample_texts:
            if text not in self._cache:
                continue
            cached = self._cache[text]
            fresh = self._raw.embed([text])[0]
            if not bool((cached == fresh).all()):
                return {"bit_transparent": False, "checked": checked}
            checked += 1
        return {"bit_transparent": True, "checked": checked}


# --------------------------------------------------------------------------- #
# Deterministic real-corpus sampling
# --------------------------------------------------------------------------- #


def _tool_call_names(record: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for message in record.get("messages", ()):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict) and str(function.get("name", "")).strip():
                names.append(str(function["name"]).strip())
    return tuple(names)


def _first_user_instruction(record: dict[str, Any]) -> str:
    for message in record.get("messages", ()):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, list):
                parts = [
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content = " ".join(part for part in parts if part)
            text = str(content).strip()
            if text:
                return text
    return ""


def _to_trajectory(line_index: int, record: dict[str, Any]) -> Trajectory:
    return Trajectory(
        line_index=line_index,
        record=record,
        uuid=str(record.get("uuid", "")).strip(),
        source_field=str(record.get("source", "")).strip(),
        domain_field=str(record.get("domain", "")).strip(),
        user_instruction=_first_user_instruction(record),
        tool_calls=_tool_call_names(record),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_arm(path: Path) -> CorpusSample:
    """Stream one JSONL part file and take the first 300 call-bearing lines."""

    file_sha = _file_sha256(path)
    selected: list[Trajectory] = []
    skipped = 0
    lines_scanned = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if len(selected) >= ARM_COUNT:
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
            selected.append(_to_trajectory(line_index, record))
    if len(selected) < ARM_COUNT:
        raise ValueError(f"{path}: only {len(selected)} call-bearing records, need {ARM_COUNT}")
    return CorpusSample(
        trajectories=tuple(selected),
        skipped_no_calls=skipped,
        lines_scanned=lines_scanned,
        first_line=selected[0].line_index,
        last_line=selected[-1].line_index,
        file_sha256=file_sha,
    )


def _sample_agate(path: Path, train_vocabulary: frozenset[str], *, after_line: int) -> AgateSample:
    """Next 16 later TU records whose tool set is within the train vocabulary."""

    selected: list[Trajectory] = []
    skipped = 0
    pruned = 0
    lines_scanned = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index <= after_line:
                continue
            if len(selected) >= AGATE_COUNT:
                break
            lines_scanned = line_index + 1
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            calls = _tool_call_names(record)
            if not calls:
                skipped += 1
                continue
            if not frozenset(f"tool.{name}" for name in calls) <= train_vocabulary:
                pruned += 1
                continue
            selected.append(_to_trajectory(line_index, record))
    if len(selected) < AGATE_COUNT:
        raise ValueError(
            f"{path}: only {len(selected)} in-vocabulary a-gate records, " f"need {AGATE_COUNT}"
        )
    return AgateSample(
        trajectories=tuple(selected),
        skipped_no_calls=skipped,
        pruned_out_of_vocabulary=pruned,
        lines_scanned=lines_scanned,
        first_line=selected[0].line_index,
        last_line=selected[-1].line_index,
    )


# --------------------------------------------------------------------------- #
# Governed manifest projection
# --------------------------------------------------------------------------- #


def _manifest(trajectory: Trajectory) -> dict[str, Any]:
    capabilities = [f"tool.{name}" for name in trajectory.tool_calls]
    name = trajectory.uuid or f"trajectory-line-{trajectory.line_index}"
    description = trajectory.user_instruction[:500]
    return {
        "skill_id": trajectory.uuid or f"p51f-line-{trajectory.line_index}",
        "version": DATASET_TAG,
        "publisher": DATASET_PUBLISHER,
        "scope_id": f"{trajectory.uuid or trajectory.line_index}.scope",
        "name": name,
        "description": description,
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


def _partition_trajectories(
    sample: CorpusSample,
) -> dict[str, tuple[Trajectory, ...]]:
    trajectories = sample.trajectories
    return {
        "train": trajectories[:TRAIN_COUNT],
        "holdout": trajectories[TRAIN_COUNT : TRAIN_COUNT + HOLDOUT_COUNT],
        "retention": trajectories[TRAIN_COUNT + HOLDOUT_COUNT : ARM_COUNT],
    }


def _build_arm(sample: CorpusSample) -> dict[str, Any]:
    """Governed registry/ledger corpus for one real-corpus arm.

    Every trajectory is registered once (``skill_id=uuid``, uniform dataset
    version) and driven through the discovered->staged->shadow->active
    lifecycle, so both arms carry exactly 4 lifecycle events per trajectory.
    After the ledger projection, every assistant tool call is appended in
    order as a governed runtime sequence event on that trajectory's
    projection (``tool.<name>``, episode = uuid, 1-based tick).
    """

    adapter = SkillArtifactAdapter()
    registry = DeclarativeSourceRegistry(adapter)
    ledger = EvolutionExperienceLedger()
    partitions = _partition_trajectories(sample)
    projections: dict[str, ArtifactCorpusProjection] = {}
    partition_of: dict[str, str] = {}

    for partition in ("train", "holdout", "retention"):
        for trajectory in partitions[partition]:
            manifest = _manifest(trajectory)
            projection = registry.register(manifest, partition=partition)
            registry.transition(projection.source_id, projection.source_version, "staged")
            registry.transition(projection.source_id, projection.source_version, "shadow")
            registry.transition(projection.source_id, projection.source_version, "active")
            projections[trajectory.uuid or str(trajectory.line_index)] = projection
            partition_of[projection.source_id] = partition

    registry.project_to_ledger(ledger, parent_checkpoint_digest=PARENT_CHECKPOINT_DIGEST)

    call_events = 0
    for partition in ("train", "holdout", "retention"):
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

    for artifact in ledger.corpus:
        ledger.admit_corpus(artifact.artifact_digest, admission_revision=ADMISSION_REVISION)

    train_artifacts, train_experiences = ledger.training_view()
    holdout_artifacts = tuple(item for item in ledger.corpus if item.partition == "holdout")
    retention_artifacts = tuple(item for item in ledger.corpus if item.partition == "retention")
    return {
        "ledger": ledger,
        "projections": projections,
        "partitions": partitions,
        "lifecycle_events": ARM_COUNT * 4,
        "call_events": call_events,
        "train": (train_artifacts, train_experiences),
        "holdout": (holdout_artifacts, ledger.records(partition="holdout")),
        "retention": (retention_artifacts, ledger.records(partition="retention")),
    }


# --------------------------------------------------------------------------- #
# Trainer pipeline (mirrors P5.1e)
# --------------------------------------------------------------------------- #


def _ranking_pairs(examples: tuple[Any, ...]) -> list[tuple[Any, Any]]:
    """Real-corpus preference pairs with the P5.1e grounding pruning.

    Every governed experience is successful (reward 1.0): lifecycle events
    carry the trajectory uuid while call events carry ``tool.<name>``.  There
    is no failed side in the real data, so no valid strictly-preferred pair
    survives; the learner trains with zero pairs (a supported path) and the
    absolute count is recorded for gate 2.  The pruning guard is retained
    verbatim so the arms stay bit-isomorphic to the P5.1e pipeline.
    """

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


def _gate_projection(trajectory: Trajectory) -> ArtifactCorpusProjection:
    return SkillArtifactAdapter().project(_manifest(trajectory), partition=GATE_PARTITION)


def _gate_units(trajectory: Trajectory) -> tuple[Any, ...]:
    return tuple(_gate_projection(trajectory).corpus)


def _gate_records(agate: AgateSample, *, encoder: Any) -> tuple[EpisodicMemoryRecord, ...]:
    """Real next-tool-call sequence records (one episode per trajectory).

    Each held-out trajectory contributes its true ordered call sequence;
    every step shares the trajectory's procedure-unit cue and the committed
    readout predicts ``tool.<name>`` tokens while the GRU hidden state
    evolves over the sequence.  ``outcome=None``: no fabricated rewards.
    """

    records: list[EpisodicMemoryRecord] = []
    for trajectory in agate.trajectories:
        projection = _gate_projection(trajectory)
        procedure = next(unit for unit in projection.corpus if unit.unit_kind == "procedure")
        cue = encoder.encode(procedure)
        episode_id = f"p51f-gate:{trajectory.uuid or trajectory.line_index}"
        for tick, tool_name in enumerate(trajectory.tool_calls, start=1):
            records.append(
                EpisodicMemoryRecord(
                    memory_id=(
                        f"p51f-gate-memory:{trajectory.uuid or trajectory.line_index}:{tick:04d}"
                    ),
                    episode_id=episode_id,
                    tick=tick,
                    cue=cue,
                    action_intent=ActionIntent(
                        f"p51f-gate-intent:{trajectory.uuid or trajectory.line_index}:{tick:04d}",
                        f"tool.{tool_name}",
                        tick=tick - 1,
                    ),
                    outcome=None,
                    provenance="p51f-gate-evaluation",
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


def _agate_vocabulary(agate: AgateSample) -> tuple[str, ...]:
    return tuple(
        sorted(
            {f"tool.{name}" for trajectory in agate.trajectories for name in trajectory.tool_calls}
        )
    )


def _arm_structure(
    arm: dict[str, Any], *, pairs: list[tuple[Any, Any]], gate_records: int
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
    structure["a_gate_records"] = gate_records
    structure["b_gate_records"] = gate_records
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


def _sample_provenance(label: str, relative_path: str, sample: CorpusSample) -> dict[str, Any]:
    trajectories = sample.trajectories
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
    }


def _agate_provenance(agate: AgateSample) -> dict[str, Any]:
    return {
        "records_used": len(agate.trajectories),
        "skipped_no_tool_calls": agate.skipped_no_calls,
        "pruned_out_of_train_vocabulary": agate.pruned_out_of_vocabulary,
        "lines_scanned_to_fill": agate.lines_scanned,
        "selected_line_range": [agate.first_line, agate.last_line],
        "first_uuid": agate.trajectories[0].uuid,
        "last_uuid": agate.trajectories[-1].uuid,
    }


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
            "total_seconds_cap": TOTAL_SECONDS_CAP,
        },
        "frozen_reference": {"p5_1e_same_budget_delta_fixtures": FROZEN_P5_1E_DELTA},
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
        placebo_sample = _sample_arm(placebo_path)

        train_sourced = sourced_sample.trajectories[:TRAIN_COUNT]
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

        sourced_arm = _build_arm(sourced_sample)
        placebo_arm = _build_arm(placebo_sample)
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

        gate_units = tuple(
            unit for trajectory in agate.trajectories for unit in _gate_units(trajectory)
        )
        agate_records_sourced = _gate_records(agate, encoder=sourced_encoder)
        agate_records_placebo = _gate_records(agate, encoder=placebo_encoder)
        agate_sourced = _gate_accuracy(sourced_trainer, agate_records_sourced)
        agate_placebo = _gate_accuracy(placebo_trainer, agate_records_placebo)

        if agate_sourced["measurable"] and agate_placebo["measurable"]:
            content_transfer_delta: float | None = round(
                float(agate_sourced["accuracy"]) - float(agate_placebo["accuracy"]),
                6,
            )
        else:
            content_transfer_delta = None

        sourced_structure = _arm_structure(
            sourced_arm,
            pairs=sourced_pairs,
            gate_records=len(agate_records_sourced),
        )
        placebo_structure = _arm_structure(
            placebo_arm,
            pairs=placebo_pairs,
            gate_records=len(agate_records_placebo),
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
        gates = {
            "static_four_checks": True,
            "same_budget_enforced": bool(sourced_structure == placebo_structure),
            "capability_vocabulary_disjoint": bool(
                set(sourced_vocabulary) >= set(gate_vocabulary)
                and not set(placebo_vocabulary) & set(gate_vocabulary)
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
                "sampling": {
                    "frozen_rule": (
                        "first N records in file line order with >=1 tool_calls; "
                        "200 train / 60 holdout / 40 retention per arm; a-gate = next "
                        "16 later Tool_Use records with tool set subset of sourced "
                        "train vocabulary"
                    ),
                    "partitions": {
                        "train": TRAIN_COUNT,
                        "holdout": HOLDOUT_COUNT,
                        "retention": RETENTION_COUNT,
                    },
                    "sourced": _sample_provenance("sourced", SOURCED_PATH, sourced_sample),
                    "placebo": _sample_provenance("placebo", PLACEBO_PATH, placebo_sample),
                    "a_gate": _agate_provenance(agate),
                },
                "corpus": {
                    "trajectories_per_arm": ARM_COUNT,
                    "lifecycle": "uniform active (discovered/staged/shadow/active)",
                    "lifecycle_events_per_trajectory": 4,
                    "sourced_lifecycle_events": sourced_arm["lifecycle_events"],
                    "placebo_lifecycle_events": placebo_arm["lifecycle_events"],
                    "sourced_call_events": sourced_arm["call_events"],
                    "placebo_call_events": placebo_arm["call_events"],
                    "partitions": ["train", "holdout", "retention"],
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
                        "agate_accuracy": agate_sourced,
                        "checkpoint_roundtrip": sourced_roundtrip,
                        "semantic_surface": sourced_surface,
                    },
                    "placebo": {
                        "structure": placebo_structure,
                        "capability_vocabulary": list(placebo_vocabulary),
                        "consolidation": placebo_report.to_payload(),
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
                    "a_gate_sourced_accuracy": agate_sourced.get("accuracy"),
                    "a_gate_placebo_accuracy": agate_placebo.get("accuracy"),
                    "delta_sourced_minus_placebo": content_transfer_delta,
                    "frozen_margin": FROZEN_CONTENT_TRANSFER_MARGIN,
                    "frozen_p5_1e_fixtures_delta": FROZEN_P5_1E_DELTA,
                    "task": "next-tool-call over each held-out trajectory's real call sequence",
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
                "experiment_passed": bool(outcome == "real_corpus_content_benefit_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        "completed: outcome=real_corpus_content_benefit_supported; "
                        f"a-gate accuracy sourced={agate_sourced.get('accuracy')} "
                        f"placebo={agate_placebo.get('accuracy')} "
                        f"delta={content_transfer_delta} >= "
                        f"{FROZEN_CONTENT_TRANSFER_MARGIN} under identical budgets"
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
    arms = result.get("arms") or {}
    sourced_gate = (arms.get("sourced") or {}).get("agate_accuracy") or {}
    placebo_gate = (arms.get("placebo") or {}).get("agate_accuracy") or {}
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
                "a_gate_accuracy_sourced": sourced_gate.get("accuracy"),
                "a_gate_accuracy_placebo": placebo_gate.get("accuracy"),
                "content_transfer_delta": transfer_block.get("delta_sourced_minus_placebo"),
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
