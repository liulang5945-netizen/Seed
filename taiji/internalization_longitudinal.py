"""R5A-S2 longitudinal Gate for grounded knowledge internalization.

The Gate is deliberately Taiji-owned and imports neither the Workbench nor a
provider.  ``SeedRuntime`` supplies already-verified ``GroundedOutcomeEvidence``
objects; this module converts them into bounded replay material, evaluates
unseen task choices, and creates only a *candidate* for external-description
removal.  It has no filesystem, MCP, tool, or deletion capability.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .internalization import (
    INTERNALIZATION_MANIFEST_REVISION,
    GroundedFeatureExample,
    GroundedOutcomeEvidence,
    InternalizationCausalGate,
    InternalizationLedger,
    content_digest,
)
from .internalization_learner import InternalizationLearningReport, InternalizedFeatureLearner

INTERNALIZATION_LONGITUDINAL_CHECKPOINT_FORMAT = "taiji-internalization-longitudinal-v1"
INTERNALIZATION_STABILITY_CHECKPOINT_FORMAT = "taiji-internalization-stability-v1"
_FORBIDDEN_DELETION_REVIEW_KEYS = frozenset(
    {"path", "disposer", "delete", "executor", "capability_id", "mcp"}
)


def _text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _examples(
    value: Iterable[GroundedFeatureExample], name: str
) -> tuple[GroundedFeatureExample, ...]:
    items = tuple(value)
    if not items:
        raise ValueError(f"{name} must contain at least one grounded example")
    if any(not isinstance(item, GroundedFeatureExample) for item in items):
        raise TypeError(f"{name} must contain GroundedFeatureExample values")
    seen: dict[str, GroundedFeatureExample] = {}
    for item in items:
        previous = seen.get(item.example_id)
        if previous is not None:
            if previous.content_digest != item.content_digest:
                raise ValueError(f"{name} binds one example_id to conflicting content")
            continue
        seen[item.example_id] = item
    return tuple(seen[key] for key in sorted(seen))


def _ordered_examples(
    value: Iterable[GroundedFeatureExample], name: str
) -> tuple[GroundedFeatureExample, ...]:
    """Deduplicate examples while retaining an evaluator's candidate order."""

    items = tuple(value)
    if not items:
        raise ValueError(f"{name} must contain at least one grounded example")
    if any(not isinstance(item, GroundedFeatureExample) for item in items):
        raise TypeError(f"{name} must contain GroundedFeatureExample values")
    seen: dict[str, GroundedFeatureExample] = {}
    for item in items:
        previous = seen.get(item.example_id)
        if previous is not None:
            if previous.content_digest != item.content_digest:
                raise ValueError(f"{name} binds one example_id to conflicting content")
            continue
        seen[item.example_id] = item
    return tuple(seen.values())


@dataclass(frozen=True)
class ExternalDescriptionArtifact:
    """An opaque, content-addressed external description used only as a control.

    The description's text or provider response is intentionally absent.  A
    task evaluator may use this reference to name its expected choice, but the
    learner can receive only grounded outcome feature examples.
    """

    artifact_id: str
    content_digest: str
    manifest_revision: str = INTERNALIZATION_MANIFEST_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "content_digest", _text(self.content_digest, "content_digest"))
        object.__setattr__(
            self,
            "manifest_revision",
            _text(self.manifest_revision, "manifest_revision"),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "content_digest": self.content_digest,
            "manifest_revision": self.manifest_revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalDescriptionArtifact:
        return cls(
            artifact_id=str(payload["artifact_id"]),
            content_digest=str(payload["content_digest"]),
            manifest_revision=str(payload["manifest_revision"]),
        )


@dataclass(frozen=True)
class GroundedSelectionTask:
    """A held-out or retention task with opaque external-choice supervision."""

    task_id: str
    candidates: tuple[GroundedFeatureExample, ...]
    external_choice_example_id: str
    expected_choice_example_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "selection task_id"))
        candidates = _ordered_examples(self.candidates, "selection task candidates")
        object.__setattr__(self, "candidates", candidates)
        candidate_ids = {item.example_id for item in candidates}
        for value, name in (
            (self.external_choice_example_id, "external_choice_example_id"),
            (self.expected_choice_example_id, "expected_choice_example_id"),
        ):
            normalized = _text(value, name)
            if normalized not in candidate_ids:
                raise ValueError(f"selection {name} is not one of its candidates")
            object.__setattr__(self, name, normalized)

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "candidates": [item.to_payload() for item in self.candidates],
            "external_choice_example_id": self.external_choice_example_id,
            "expected_choice_example_id": self.expected_choice_example_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> GroundedSelectionTask:
        return cls(
            task_id=str(payload["task_id"]),
            candidates=tuple(
                GroundedFeatureExample.from_payload(item) for item in payload["candidates"]
            ),
            external_choice_example_id=str(payload["external_choice_example_id"]),
            expected_choice_example_id=str(payload["expected_choice_example_id"]),
        )


@dataclass(frozen=True)
class ExternalDescriptionTombstoneCandidate:
    """Evidence that a description may be reviewed for a separate deletion action.

    This record intentionally does not contain a file path, a disposer, or an
    execution reference.  Producing it does not alter the referenced artifact.
    """

    artifact_id: str
    artifact_content_digest: str
    example_ids: tuple[str, ...]
    checkpoint_digest: str
    causal_gate: InternalizationCausalGate
    manifest_revision: str = INTERNALIZATION_MANIFEST_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "artifact_content_digest",
            _text(self.artifact_content_digest, "artifact_content_digest"),
        )
        example_ids = tuple(
            sorted(_text(item, "tombstone example_id") for item in self.example_ids)
        )
        if not example_ids or len(set(example_ids)) != len(example_ids):
            raise ValueError("tombstone candidate example_ids must be unique and non-empty")
        object.__setattr__(self, "example_ids", example_ids)
        object.__setattr__(
            self,
            "checkpoint_digest",
            _text(self.checkpoint_digest, "checkpoint_digest"),
        )
        if (
            not isinstance(self.causal_gate, InternalizationCausalGate)
            or not self.causal_gate.passed
        ):
            raise ValueError("tombstone candidate requires a passed causal gate")
        object.__setattr__(
            self,
            "manifest_revision",
            _text(self.manifest_revision, "manifest_revision"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_content_digest": self.artifact_content_digest,
            "example_ids": list(self.example_ids),
            "checkpoint_digest": self.checkpoint_digest,
            "causal_gate": self.causal_gate.to_payload(),
            "manifest_revision": self.manifest_revision,
            "disposition": "candidate_only_no_physical_deletion",
        }


@dataclass(frozen=True)
class InternalizationLongitudinalReport:
    """The measured outcome of one real-Outcome longitudinal Gate."""

    learning: InternalizationLearningReport
    external_description_quality: float
    external_removed_selection_quality: float
    internalized_lesion_selection_quality: float
    grounding_lesion_selection_quality: float
    retention_selection_quality: float
    restored_selection_quality: float
    checkpoint_digest: str
    checkpoint_recoverable: bool
    causal_gate: InternalizationCausalGate
    lifecycle_statuses: tuple[tuple[str, str], ...]
    tombstone_candidate: ExternalDescriptionTombstoneCandidate | None = None

    @property
    def passed(self) -> bool:
        return self.learning.passed and self.causal_gate.passed

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERNALIZATION_LONGITUDINAL_CHECKPOINT_FORMAT,
            "learning": self.learning.to_payload(),
            "selection": {
                "external_description_quality": self.external_description_quality,
                "external_removed_selection_quality": self.external_removed_selection_quality,
                "internalized_lesion_selection_quality": self.internalized_lesion_selection_quality,
                "grounding_lesion_selection_quality": self.grounding_lesion_selection_quality,
                "retention_selection_quality": self.retention_selection_quality,
                "restored_selection_quality": self.restored_selection_quality,
            },
            "checkpoint": {
                "digest": self.checkpoint_digest,
                "recoverable": self.checkpoint_recoverable,
            },
            "causal_gate": self.causal_gate.to_payload(),
            "lifecycle_statuses": dict(self.lifecycle_statuses),
            "tombstone_candidate": (
                None if self.tombstone_candidate is None else self.tombstone_candidate.to_payload()
            ),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class InternalizationStabilityTrial:
    """One independently seeded and task-sliced longitudinal result."""

    trial_id: str
    seed: int
    task_slice: str
    report: InternalizationLongitudinalReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_id", _text(self.trial_id, "stability trial_id"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("stability trial seed must be an integer")
        object.__setattr__(self, "task_slice", _text(self.task_slice, "stability task_slice"))
        if not isinstance(self.report, InternalizationLongitudinalReport):
            raise TypeError("stability trial report must be an InternalizationLongitudinalReport")
        for name, value in self.metrics:
            if not math.isfinite(value):
                raise ValueError(f"stability trial metric {name} must be finite")
        for name, value in self.resource_metrics:
            if value < 0:
                raise ValueError(f"stability trial resource metric {name} cannot be negative")

    @property
    def metrics(self) -> tuple[tuple[str, float], ...]:
        learning = self.report.learning
        values = {
            "holdout_gain": learning.holdout_gain,
            "internalization_drop": (
                self.report.external_removed_selection_quality
                - self.report.internalized_lesion_selection_quality
            ),
            "grounding_drop": (
                self.report.external_removed_selection_quality
                - self.report.grounding_lesion_selection_quality
            ),
            "retention_loss_delta": learning.retention_loss_after - learning.retention_loss_before,
        }
        return tuple((name, float(value)) for name, value in sorted(values.items()))

    @property
    def resource_metrics(self) -> tuple[tuple[str, int], ...]:
        learning = self.report.learning
        return (
            ("train_examples", learning.train_examples),
            ("holdout_examples", learning.holdout_examples),
            ("fit_updates", learning.fit_updates),
            ("ranking_updates", learning.ranking_updates),
            ("lineage_depth", learning.lineage_depth),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "seed": self.seed,
            "task_slice": self.task_slice,
            "report": self.report.to_payload(),
            "metrics": dict(self.metrics),
            "resource_metrics": dict(self.resource_metrics),
        }


@dataclass(frozen=True)
class IndependentDeletionReview:
    """A read-only review of candidate records before any external action."""

    passed: bool
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    reviewed_trial_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "failures": list(self.failures),
            "reviewed_trial_ids": list(self.reviewed_trial_ids),
            "disposition": "review_only_no_physical_deletion",
        }

    @classmethod
    def review(
        cls,
        external_description: ExternalDescriptionArtifact,
        trials: Sequence[InternalizationStabilityTrial],
    ) -> IndependentDeletionReview:
        if not isinstance(external_description, ExternalDescriptionArtifact):
            raise TypeError("external_description must be an ExternalDescriptionArtifact")
        checks: list[tuple[str, bool]] = []
        failures: list[str] = []

        def check(name: str, condition: bool) -> None:
            checks.append((name, bool(condition)))
            if not condition:
                failures.append(name)

        check(
            "external_artifact_has_opaque_content_digest", bool(external_description.content_digest)
        )
        check(
            "external_artifact_has_no_description_text",
            "description" not in external_description.__dict__,
        )
        for trial in trials:
            report = trial.report
            candidate = report.tombstone_candidate
            expected_candidate = report.passed
            check(
                f"{trial.trial_id}:candidate_presence",
                (candidate is not None) == expected_candidate,
            )
            if candidate is None:
                continue
            payload = candidate.to_payload()
            check(
                f"{trial.trial_id}:artifact_binding",
                candidate.artifact_id == external_description.artifact_id
                and candidate.artifact_content_digest == external_description.content_digest,
            )
            check(
                f"{trial.trial_id}:checkpoint_binding",
                candidate.checkpoint_digest == report.checkpoint_digest
                and bool(candidate.checkpoint_digest),
            )
            check(
                f"{trial.trial_id}:manifest_binding",
                candidate.manifest_revision == external_description.manifest_revision,
            )
            check(
                f"{trial.trial_id}:causal_gate_binding",
                candidate.causal_gate == report.causal_gate and candidate.causal_gate.passed,
            )
            status_map = dict(report.lifecycle_statuses)
            check(
                f"{trial.trial_id}:internalized_lifecycle",
                bool(candidate.example_ids)
                and all(
                    status_map.get(example_id) == "internalized"
                    for example_id in candidate.example_ids
                ),
            )
            check(
                f"{trial.trial_id}:content_addressed_example_ids",
                len(candidate.example_ids) == len(set(candidate.example_ids)),
            )
            check(
                f"{trial.trial_id}:no_physical_deletion_authority",
                not _FORBIDDEN_DELETION_REVIEW_KEYS.intersection(payload),
            )
            check(
                f"{trial.trial_id}:candidate_only_disposition",
                payload.get("disposition") == "candidate_only_no_physical_deletion",
            )
        return cls(
            passed=not failures,
            checks=tuple(checks),
            failures=tuple(failures),
            reviewed_trial_ids=tuple(trial.trial_id for trial in trials),
        )


@dataclass(frozen=True)
class InternalizationStabilityReport:
    """Cross-seed/task-slice evidence required after one longitudinal canary."""

    external_description: ExternalDescriptionArtifact
    trials: tuple[InternalizationStabilityTrial, ...]
    unique_seeds: tuple[int, ...]
    unique_task_slices: tuple[str, ...]
    minimum_holdout_gain: float
    minimum_internalization_drop: float
    minimum_grounding_drop: float
    maximum_retention_loss_delta: float
    maximum_metric_spread: float
    maximum_resource_metrics: tuple[tuple[str, int], ...]
    stability_passed: bool
    independent_deletion_review: IndependentDeletionReview

    @property
    def passed(self) -> bool:
        return self.stability_passed and self.independent_deletion_review.passed

    @property
    def evidence_digest(self) -> str:
        return content_digest(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERNALIZATION_STABILITY_CHECKPOINT_FORMAT,
            "external_description": self.external_description.to_payload(),
            "trials": [trial.to_payload() for trial in self.trials],
            "unique_seeds": list(self.unique_seeds),
            "unique_task_slices": list(self.unique_task_slices),
            "metrics": {
                "minimum_holdout_gain": self.minimum_holdout_gain,
                "minimum_internalization_drop": self.minimum_internalization_drop,
                "minimum_grounding_drop": self.minimum_grounding_drop,
                "maximum_retention_loss_delta": self.maximum_retention_loss_delta,
                "maximum_metric_spread": self.maximum_metric_spread,
            },
            "maximum_resource_metrics": dict(self.maximum_resource_metrics),
            "stability_passed": self.stability_passed,
            "independent_deletion_review": self.independent_deletion_review.to_payload(),
            "passed": self.passed,
        }


class InternalizationStabilityGate:
    """Require replication before any deletion review can be considered eligible."""

    def __init__(
        self,
        external_description: ExternalDescriptionArtifact,
        *,
        minimum_trials: int = 2,
        minimum_seeds: int = 2,
        minimum_task_slices: int = 2,
        minimum_holdout_gain: float = 0.05,
        minimum_lesion_drop: float = 0.5,
        maximum_retention_loss_delta: float = 0.05,
        maximum_metric_spread: float = 0.25,
    ) -> None:
        if not isinstance(external_description, ExternalDescriptionArtifact):
            raise TypeError("external_description must be an ExternalDescriptionArtifact")
        self.external_description = external_description
        self.minimum_trials = self._count(minimum_trials, "minimum_trials")
        self.minimum_seeds = self._count(minimum_seeds, "minimum_seeds")
        self.minimum_task_slices = self._count(minimum_task_slices, "minimum_task_slices")
        self.minimum_holdout_gain = self._finite(minimum_holdout_gain, "minimum_holdout_gain")
        self.minimum_lesion_drop = self._finite(minimum_lesion_drop, "minimum_lesion_drop")
        self.maximum_retention_loss_delta = self._finite(
            maximum_retention_loss_delta, "maximum_retention_loss_delta"
        )
        self.maximum_metric_spread = self._finite(maximum_metric_spread, "maximum_metric_spread")
        self.last_report: InternalizationStabilityReport | None = None

    @staticmethod
    def _count(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _finite(value: float, name: str) -> float:
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return normalized

    @staticmethod
    def _trials(
        value: Iterable[InternalizationStabilityTrial],
    ) -> tuple[InternalizationStabilityTrial, ...]:
        trials = tuple(value)
        if not trials:
            raise ValueError("stability Gate requires at least one trial")
        if any(not isinstance(item, InternalizationStabilityTrial) for item in trials):
            raise TypeError("stability Gate trials must be InternalizationStabilityTrial values")
        trial_ids = [item.trial_id for item in trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("stability Gate trial_id values must be unique")
        return trials

    def evaluate(
        self, trials: Iterable[InternalizationStabilityTrial]
    ) -> InternalizationStabilityReport:
        items = self._trials(trials)
        seeds = tuple(sorted({item.seed for item in items}))
        task_slices = tuple(sorted({item.task_slice for item in items}))
        metric_values = {
            name: tuple(dict(item.metrics)[name] for item in items) for name, _ in items[0].metrics
        }
        spreads = tuple(max(values) - min(values) for values in metric_values.values())
        minimum_holdout_gain = min(metric_values["holdout_gain"])
        minimum_internalization_drop = min(metric_values["internalization_drop"])
        minimum_grounding_drop = min(metric_values["grounding_drop"])
        maximum_retention_loss_delta = max(metric_values["retention_loss_delta"])
        maximum_metric_spread = max(spreads, default=0.0)
        resource_names = dict(items[0].resource_metrics).keys()
        maximum_resource_metrics = tuple(
            (name, max(dict(item.resource_metrics)[name] for item in items))
            for name in sorted(resource_names)
        )
        review = IndependentDeletionReview.review(self.external_description, items)
        stability_passed = bool(
            len(items) >= self.minimum_trials
            and len(seeds) >= self.minimum_seeds
            and len(task_slices) >= self.minimum_task_slices
            and all(item.report.passed for item in items)
            and minimum_holdout_gain >= self.minimum_holdout_gain
            and minimum_internalization_drop >= self.minimum_lesion_drop
            and minimum_grounding_drop >= self.minimum_lesion_drop
            and maximum_retention_loss_delta <= self.maximum_retention_loss_delta
            and maximum_metric_spread <= self.maximum_metric_spread
        )
        report = InternalizationStabilityReport(
            external_description=self.external_description,
            trials=items,
            unique_seeds=seeds,
            unique_task_slices=task_slices,
            minimum_holdout_gain=minimum_holdout_gain,
            minimum_internalization_drop=minimum_internalization_drop,
            minimum_grounding_drop=minimum_grounding_drop,
            maximum_retention_loss_delta=maximum_retention_loss_delta,
            maximum_metric_spread=maximum_metric_spread,
            maximum_resource_metrics=maximum_resource_metrics,
            stability_passed=stability_passed,
            independent_deletion_review=review,
        )
        self.last_report = report
        return report

    def checkpoint(self, trials: Iterable[InternalizationStabilityTrial]) -> dict[str, Any]:
        """Persist the evidence bundle without creating deletion authority."""

        items = self._trials(trials)
        return {
            "format": INTERNALIZATION_STABILITY_CHECKPOINT_FORMAT,
            "external_description": self.external_description.to_payload(),
            "thresholds": {
                "minimum_trials": self.minimum_trials,
                "minimum_seeds": self.minimum_seeds,
                "minimum_task_slices": self.minimum_task_slices,
                "minimum_holdout_gain": self.minimum_holdout_gain,
                "minimum_lesion_drop": self.minimum_lesion_drop,
                "maximum_retention_loss_delta": self.maximum_retention_loss_delta,
                "maximum_metric_spread": self.maximum_metric_spread,
            },
            "trials": [item.to_payload() for item in items],
        }


class InternalizationLongitudinalGate:
    """Evaluate grounded internalization without granting deletion authority."""

    def __init__(
        self,
        external_description: ExternalDescriptionArtifact,
        *,
        ledger: InternalizationLedger | None = None,
        learner: InternalizedFeatureLearner | None = None,
        minimum_quality: float = 1.0,
        minimum_lesion_drop: float = 0.5,
        pairwise_margin: float = 0.05,
    ) -> None:
        if not isinstance(external_description, ExternalDescriptionArtifact):
            raise TypeError("external_description must be an ExternalDescriptionArtifact")
        self.external_description = external_description
        self.ledger = ledger or InternalizationLedger()
        self.learner = learner
        self.minimum_quality = self._quality(minimum_quality, "minimum_quality")
        self.minimum_lesion_drop = self._quality(minimum_lesion_drop, "minimum_lesion_drop")
        self.pairwise_margin = float(pairwise_margin)
        if not 0.0 < self.pairwise_margin <= 1.0:
            raise ValueError("pairwise_margin must be within (0, 1]")
        if learner is not None and learner.pairwise_margin != self.pairwise_margin:
            raise ValueError("longitudinal learner pairwise_margin does not match the Gate")
        self.last_report: InternalizationLongitudinalReport | None = None

    @staticmethod
    def _quality(value: float, name: str) -> float:
        normalized = float(value)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
        return normalized

    def ingest_train(
        self, sources: Iterable[GroundedOutcomeEvidence]
    ) -> tuple[GroundedFeatureExample, ...]:
        """Convert real outcomes into the train-only replay ledger."""

        examples: list[GroundedFeatureExample] = []
        for source in tuple(sources):
            result = self.ledger.ingest(source)
            if not result.accepted or result.example is None:
                raise ValueError(f"grounded train outcome was rejected: {result.reason}")
            examples.append(result.example)
        return _examples(examples, "train outcomes")

    @staticmethod
    def _task_examples(
        tasks: Sequence[GroundedSelectionTask],
    ) -> tuple[GroundedFeatureExample, ...]:
        return _examples(
            (example for task in tasks for example in task.candidates),
            "selection task examples",
        )

    @staticmethod
    def _ranking_pairs(
        train: Sequence[GroundedFeatureExample],
    ) -> tuple[tuple[GroundedFeatureExample, GroundedFeatureExample], ...]:
        """Derive train-only preferences from strictly ordered experienced rewards."""

        pairs = []
        for preferred in train:
            for other in train:
                if preferred.target_reward > other.target_reward:
                    pairs.append((preferred, other))
        if not pairs:
            raise ValueError("longitudinal Gate requires distinct train outcome rewards")
        return tuple(pairs)

    @staticmethod
    def _select(
        learner: InternalizedFeatureLearner,
        task: GroundedSelectionTask,
        *,
        internalized_enabled: bool = True,
        grounding_enabled: bool = True,
    ) -> str:
        # Preserve explicit task candidate order for a deterministic no-signal
        # tie.  No action/capability vocabulary appears in this ordering.
        best_index, best_example = max(
            enumerate(task.candidates),
            key=lambda pair: (
                learner.score(
                    pair[1],
                    internalized_enabled=internalized_enabled,
                    grounding_enabled=grounding_enabled,
                ),
                -pair[0],
            ),
        )
        del best_index
        return best_example.example_id

    def _selection_quality(
        self,
        tasks: Sequence[GroundedSelectionTask],
        *,
        learner: InternalizedFeatureLearner | None = None,
        external_description_available: bool = False,
        internalized_enabled: bool = True,
        grounding_enabled: bool = True,
    ) -> float:
        if not tasks:
            raise ValueError("selection quality requires at least one task")
        active_learner = self.learner if learner is None else learner
        if not external_description_available and active_learner is None:
            raise RuntimeError("internalized selection requires a learner")
        correct = 0
        for task in tasks:
            selected = (
                task.external_choice_example_id
                if external_description_available
                else self._select(
                    active_learner,
                    task,
                    internalized_enabled=internalized_enabled,
                    grounding_enabled=grounding_enabled,
                )
            )
            correct += int(selected == task.expected_choice_example_id)
        return correct / len(tasks)

    def evaluate(
        self,
        train_examples: Iterable[GroundedFeatureExample],
        *,
        holdout_tasks: Sequence[GroundedSelectionTask],
        retention_tasks: Sequence[GroundedSelectionTask],
        passes: int = 8,
    ) -> InternalizationLongitudinalReport:
        """Run a parent-to-child consolidation trial and all five causal checks."""

        train = _examples(train_examples, "train examples")
        if any(self.ledger.lifecycle(item.example_id).status != "external" for item in train):
            raise ValueError(
                "longitudinal Gate train examples must be external before shadow learning"
            )
        holdout = self._task_examples(tuple(holdout_tasks))
        retention = self._task_examples(tuple(retention_tasks))
        train_ids = {item.example_id for item in train}
        if train_ids.intersection(item.example_id for item in holdout):
            raise ValueError("longitudinal holdout cannot reuse training examples")
        if self.learner is None:
            self.learner = InternalizedFeatureLearner(
                feature_dim=int(train[0].grounding.numel()),
                pairwise_margin=self.pairwise_margin,
                manifest_revision=self.external_description.manifest_revision,
            )
        if self.learner.feature_dim != int(train[0].grounding.numel()):
            raise ValueError("longitudinal learner feature dimension does not match train examples")

        learning = self.learner.consolidate(
            train,
            holdout_examples=holdout,
            retention_examples=retention,
            replay_digest=self.ledger.replay_digest,
            passes=passes,
            ranking_pairs=self._ranking_pairs(train),
        )
        for example in train:
            self.ledger.advance_status(example.example_id, "shadow")

        external_quality = self._selection_quality(
            holdout_tasks,
            external_description_available=True,
        )
        external_removed_quality = self._selection_quality(holdout_tasks)
        internalized_lesion_quality = self._selection_quality(
            holdout_tasks,
            internalized_enabled=False,
        )
        grounding_lesion_quality = self._selection_quality(
            holdout_tasks,
            grounding_enabled=False,
        )
        retention_quality = self._selection_quality(retention_tasks)

        checkpoint = self.checkpoint()
        checkpoint_digest = content_digest(checkpoint)
        restored = type(self).from_checkpoint(checkpoint)
        restored_quality = restored._selection_quality(holdout_tasks)
        checkpoint_recoverable = bool(
            content_digest(restored.checkpoint()) == checkpoint_digest
            and restored_quality == external_removed_quality
        )
        gate = InternalizationCausalGate(
            external_sufficiency=external_quality >= self.minimum_quality,
            internalization_necessity=(
                external_removed_quality >= self.minimum_quality
                and external_removed_quality - internalized_lesion_quality
                >= self.minimum_lesion_drop
            ),
            grounding_necessity=(
                external_removed_quality - grounding_lesion_quality >= self.minimum_lesion_drop
            ),
            checkpoint_recoverable=checkpoint_recoverable,
            old_task_retention=(
                retention_quality >= self.minimum_quality
                and learning.retention_loss_after <= learning.retention_loss_before + 0.05
            ),
        )
        if gate.passed:
            for example in train:
                self.ledger.advance_status(
                    example.example_id,
                    "internalized",
                    causal_gate=gate,
                )
        statuses = tuple(
            (item.example_id, self.ledger.lifecycle(item.example_id).status) for item in train
        )
        candidate = (
            None
            if not gate.passed
            else ExternalDescriptionTombstoneCandidate(
                artifact_id=self.external_description.artifact_id,
                artifact_content_digest=self.external_description.content_digest,
                example_ids=tuple(item.example_id for item in train),
                checkpoint_digest=checkpoint_digest,
                causal_gate=gate,
                manifest_revision=self.external_description.manifest_revision,
            )
        )
        report = InternalizationLongitudinalReport(
            learning=learning,
            external_description_quality=external_quality,
            external_removed_selection_quality=external_removed_quality,
            internalized_lesion_selection_quality=internalized_lesion_quality,
            grounding_lesion_selection_quality=grounding_lesion_quality,
            retention_selection_quality=retention_quality,
            restored_selection_quality=restored_quality,
            checkpoint_digest=checkpoint_digest,
            checkpoint_recoverable=checkpoint_recoverable,
            causal_gate=gate,
            lifecycle_statuses=statuses,
            tombstone_candidate=candidate,
        )
        self.last_report = report
        return report

    def checkpoint(self) -> dict[str, Any]:
        """Serialize all Taiji-owned state needed for a continued Gate run."""

        return {
            "format": INTERNALIZATION_LONGITUDINAL_CHECKPOINT_FORMAT,
            "external_description": self.external_description.to_payload(),
            "ledger": self.ledger.checkpoint(),
            "learner": None if self.learner is None else self.learner.checkpoint(),
            "minimum_quality": self.minimum_quality,
            "minimum_lesion_drop": self.minimum_lesion_drop,
            "pairwise_margin": self.pairwise_margin,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> InternalizationLongitudinalGate:
        if payload.get("format") != INTERNALIZATION_LONGITUDINAL_CHECKPOINT_FORMAT:
            raise ValueError("unsupported internalization longitudinal checkpoint format")
        learner_payload = payload.get("learner")
        gate = cls(
            ExternalDescriptionArtifact.from_payload(payload["external_description"]),
            ledger=InternalizationLedger.from_checkpoint(payload["ledger"]),
            learner=(
                None
                if learner_payload is None
                else InternalizedFeatureLearner.from_checkpoint(learner_payload)
            ),
            minimum_quality=float(payload["minimum_quality"]),
            minimum_lesion_drop=float(payload["minimum_lesion_drop"]),
            pairwise_margin=float(payload.get("pairwise_margin", 0.05)),
        )
        return gate


__all__ = [
    "ExternalDescriptionArtifact",
    "ExternalDescriptionTombstoneCandidate",
    "GroundedSelectionTask",
    "INTERNALIZATION_LONGITUDINAL_CHECKPOINT_FORMAT",
    "INTERNALIZATION_STABILITY_CHECKPOINT_FORMAT",
    "IndependentDeletionReview",
    "InternalizationLongitudinalGate",
    "InternalizationLongitudinalReport",
    "InternalizationStabilityGate",
    "InternalizationStabilityReport",
    "InternalizationStabilityTrial",
]
