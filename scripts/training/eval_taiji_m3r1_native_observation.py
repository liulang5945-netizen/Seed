"""M3.R1: train Taiji on verified Workbench observations on CPU.

This Gate is intentionally narrow.  Workbench supplies already-verified file,
language, toolchain, and capability facts; the native Taiji learner consumes a
numeric ``PerceptEvent`` and learns structured world/goal/content owners.  No
provider text, executable intent, write, terminal, or MCP call is admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed.persistence import atomic_save  # noqa: E402
from seed_platform.programming_languages import (  # noqa: E402
    ProgrammingLanguageRegistry,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    ContentPlan,
    Goal,
    StructuredSemanticCorpus,
    StructuredSemanticExample,
    StructuredSemanticLearner,
    TSKV8Adapter,
    WorkbenchObservation,
    WorkbenchObservationSchema,
    WorldState,
    content_digest,
    semantic_fact_key,
)

REPORT_FORMAT = "taiji-m3r1-native-workbench-observation-v1"
REPORT_VERSION = 1
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "m3_workbench_projects"
SPLITS = ("train", "dev", "test")
TASK_KIND = "inspect-language"
VALID_PATHS = ("app.py", "index.ts", "main.rs", "shared.h")
MISSING_PATHS = {
    "train": "missing_train.py",
    "dev": "missing_dev.ts",
    "test": "missing_test.rs",
}


def _course_registry() -> ProgrammingLanguageRegistry:
    """Use one deterministic course registry without host-dependent Rust/TS tools."""

    python_command = Path(sys.executable).name
    definitions = []
    for definition in ProgrammingLanguageRegistry.default().definitions:
        if definition.language_id == "python":
            definitions.append(replace(definition, toolchain_commands=(python_command,)))
        elif definition.language_id == "typescript":
            definitions.append(replace(definition, toolchain_commands=("m3-r1-unavailable-tsc",)))
        elif definition.language_id == "rust":
            definitions.append(replace(definition, toolchain_commands=("m3-r1-unavailable-rustc",)))
        else:
            definitions.append(definition)
    return ProgrammingLanguageRegistry(definitions)


def _schema() -> WorkbenchObservationSchema:
    return WorkbenchObservationSchema(
        language_ids=("python", "rust", "typescript", "unknown"),
        selection_states=("ambiguous", "resolved", "unknown"),
        task_kinds=(TASK_KIND,),
        extensions=(".h", ".py", ".rs", ".ts", "<none>"),
    )


def _read_evidence(
    environment: WorkbenchEnvironment,
    path: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        read_result = environment.read_workspace_evidence({"path": path})
    except FileNotFoundError:
        return {"success": False, "error_code": "not_found", "path": path}, None
    except (IsADirectoryError, NotADirectoryError, ValueError) as exc:
        return {
            "success": False,
            "error_code": type(exc).__name__,
            "path": path,
        }, None
    read_result = {"success": True, **read_result}
    language_result = environment.resolve_programming_language_evidence({"path": path})
    return read_result, language_result


def _observation(
    *,
    environment: WorkbenchEnvironment,
    project_id: str,
    split: str,
    path: str,
    schema: WorkbenchObservationSchema,
    tick: int,
) -> WorkbenchObservation:
    read_result, language_result = _read_evidence(environment, path)
    snapshot = environment.capability_snapshot
    return WorkbenchObservation.from_workbench_evidence(
        observation_id=f"m3r1:{split}:{path}",
        project_id=project_id,
        task_id=f"m3r1:{split}:read-language:{path}",
        path=path,
        capability_snapshot_id=snapshot.snapshot_id,
        capability_revision=snapshot.revision,
        read_result=read_result,
        language_result=language_result,
        task_kind=TASK_KIND,
        schema=schema,
    )


def _build_split_observations(
    split: str,
    schema: WorkbenchObservationSchema,
) -> tuple[WorkbenchObservation, ...]:
    root = FIXTURE_ROOT / f"native_observation_{split}"
    environment = WorkbenchEnvironment(root=root, programming_language_registry=_course_registry())
    paths = (*VALID_PATHS, MISSING_PATHS[split])
    return tuple(
        _observation(
            environment=environment,
            project_id=f"m3r1-project-{split}",
            split=split,
            path=path,
            schema=schema,
            tick=index + 1,
        )
        for index, path in enumerate(paths)
    )


def _target_relations(observation: WorkbenchObservation) -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "workbench",
            "read",
            "success" if observation.read_success else "failure",
        ),
        (
            "workbench",
            "target",
            "file" if observation.file_is_file else "missing",
        ),
        ("workbench", "language", observation.language_id),
        ("workbench", "language_state", observation.selection_state),
        (
            "workbench",
            "toolchain",
            "available" if observation.toolchain_available else "missing",
        ),
        (
            "workbench",
            "diagnostics",
            "connected" if observation.diagnostics_connected else "disconnected",
        ),
    )


def _semantic_kind(observation: WorkbenchObservation) -> str:
    if not observation.read_success:
        return "recover-target"
    if observation.selection_state in {"ambiguous", "unknown"}:
        return "clarify-language"
    if not observation.toolchain_available:
        return "clarify-toolchain"
    return "inspect-language"


def _example(
    observation: WorkbenchObservation,
    *,
    split: str,
    schema: WorkbenchObservationSchema,
    tick: int,
) -> StructuredSemanticExample:
    kind = _semantic_kind(observation)
    request = kind != "inspect-language"
    goal = Goal(
        goal_id=f"goal:{kind}",
        description={
            "inspect-language": "Report verified language and toolchain facts.",
            "clarify-toolchain": "Clarify the unavailable toolchain before execution.",
            "clarify-language": "Clarify an ambiguous programming language selection.",
            "recover-target": "Recover a missing Workbench target before proceeding.",
        }[kind],
        priority=0.9,
    )
    content = ContentPlan(
        content_id=f"content:{kind}",
        intent_id=f"intent:{kind}",
        intent_kind="request_information" if request else "report",
        semantic_slots={
            "observed_language": observation.language_id,
            "selection_state": observation.selection_state,
            "toolchain_available": observation.toolchain_available,
        },
        source_goal_id=goal.goal_id,
        expected_outcome={
            "inspect-language": "verified read-only report",
            "clarify-toolchain": "toolchain clarification",
            "clarify-language": "language clarification",
            "recover-target": "valid target path",
        }[kind],
        confidence=1.0,
        provenance="m3r1-native-label",
        tick=tick,
    )
    percept = observation.to_percept_event(tick=tick)
    world = WorldState(
        tick=percept.observation_tick,
        latent=torch.empty(0),
        entities=("workbench",),
        relations=_target_relations(observation),
        uncertainty=0.0,
        percept_event_id=percept.event_id,
        percept_assembly_id=percept.assembly_id,
    )
    # ``schema`` is intentionally consumed here so the course builder cannot
    # silently construct examples from a different observation vocabulary.
    if percept.features.numel() != schema.feature_dim:
        raise AssertionError("observation schema dimension drifted before training")
    return StructuredSemanticExample(
        example_id=f"m3r1:{split}:{observation.path}",
        family_id=f"m3r1:{split}:family:{observation.path}",
        percept=percept,
        world=world,
        goal=goal,
        content=content,
    )


def build_course() -> tuple[StructuredSemanticCorpus, dict[str, tuple[WorkbenchObservation, ...]]]:
    schema = _schema()
    observations = {split: _build_split_observations(split, schema) for split in SPLITS}
    examples = {
        split: tuple(
            _example(observation, split=split, schema=schema, tick=index + 1)
            for index, observation in enumerate(values)
        )
        for split, values in observations.items()
    }
    corpus = StructuredSemanticCorpus.from_splits(
        train=examples["train"],
        dev=examples["dev"],
        test=examples["test"],
    )
    return corpus, observations


def _fixture_digest(root: Path) -> str:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return content_digest(entries)


def _metrics(
    learner: StructuredSemanticLearner,
    examples: tuple[StructuredSemanticExample, ...],
) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    goal_hits = content_hits = status_hits = 0
    for example in examples:
        result = learner.predict(example.percept)
        expected = set(example.fact_keys)
        actual = (
            set()
            if result.world is None
            else {semantic_fact_key(*relation) for relation in result.world.relations}
        )
        true_positive += len(expected & actual)
        false_positive += len(actual - expected)
        false_negative += len(expected - actual)
        goal_hits += int(result.goal is not None and result.goal.goal_id == example.goal.goal_id)
        content_hits += int(
            result.content_plan is not None
            and result.content_plan.content_id == example.content.content_id
        )
        expected_status = (
            "clarify" if example.content.intent_kind == "request_information" else "resolved"
        )
        status_hits += int(result.status == expected_status)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    fact_f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    count = max(1, len(examples))
    return {
        "examples": len(examples),
        "fact_f1": fact_f1,
        "goal_accuracy": goal_hits / count,
        "content_accuracy": content_hits / count,
        "status_accuracy": status_hits / count,
    }


def _prediction_digest(
    learner: StructuredSemanticLearner,
    examples: tuple[StructuredSemanticExample, ...],
) -> str:
    return content_digest([learner.predict(item.percept).to_payload() for item in examples])


def _perfect_metrics(examples: tuple[StructuredSemanticExample, ...]) -> dict[str, float | int]:
    return {
        "examples": len(examples),
        "fact_f1": 1.0,
        "goal_accuracy": 1.0,
        "content_accuracy": 1.0,
        "status_accuracy": 1.0,
    }


def _empty_metrics(examples: tuple[StructuredSemanticExample, ...]) -> dict[str, float | int]:
    return {
        "examples": len(examples),
        "fact_f1": 0.0,
        "goal_accuracy": 0.0,
        "content_accuracy": 0.0,
        "status_accuracy": 0.0,
    }


def _checkpoint_roundtrip(
    learner: StructuredSemanticLearner,
    corpus: StructuredSemanticCorpus,
    path: Path,
) -> tuple[StructuredSemanticLearner, str, str, str]:
    before = _prediction_digest(learner, corpus.test)
    payload = learner.checkpoint()
    atomic_save(payload, path)
    restored_payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = StructuredSemanticLearner.from_checkpoint(restored_payload, corpus)
    after = _prediction_digest(restored, corpus.test)
    return restored, before, after, content_digest(restored_payload)


def _observation_roundtrip(
    observations: dict[str, tuple[WorkbenchObservation, ...]],
) -> dict[str, Any]:
    original = observations["test"][0]
    restored = WorkbenchObservation.from_payload(original.to_payload())
    original_event = original.to_percept_event(tick=7)
    restored_event = restored.to_percept_event(tick=7)
    return {
        "observation_digest_equal": original.observation_digest == restored.observation_digest,
        "event_digest_equal": content_digest(original_event.to_payload())
        == content_digest(restored_event.to_payload()),
        "feature_vector_equal": bool(torch.equal(original_event.features, restored_event.features)),
        "capability_revision_preserved": restored.capability_revision
        == original.capability_revision,
    }


def _runtime_owner_gate(
    learner: StructuredSemanticLearner,
    observations: dict[str, tuple[WorkbenchObservation, ...]],
) -> dict[str, Any]:
    adapter = TSKV8Adapter()
    state_before = content_digest(adapter.cognitive_snapshot().to_payload())
    adapter.attach_structured_semantic_learner(learner)
    direct_results = tuple(
        adapter.infer_structured_semantics(item.to_percept_event(tick=index + 1)).to_payload()
        for index, item in enumerate(observations["test"])
    )
    state_after = content_digest(adapter.cognitive_snapshot().to_payload())
    checkpoint = adapter.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_results = tuple(
        restored.infer_structured_semantics(item.to_percept_event(tick=index + 1)).to_payload()
        for index, item in enumerate(observations["test"])
    )
    restored_state = content_digest(restored.cognitive_snapshot().to_payload())
    output_digest = content_digest(direct_results)
    restored_output_digest = content_digest(restored_results)
    return {
        "owner_attached": adapter.structured_semantic_learner is learner,
        "checkpoint_component_present": "structured_semantic" in checkpoint["components"],
        "restored_owner_present": restored.structured_semantic_learner is not None,
        "output_digest": output_digest,
        "restored_output_digest": restored_output_digest,
        "output_stable_after_runtime_restore": output_digest == restored_output_digest,
        "cognitive_state_read_only": state_before == state_after == restored_state,
        "runtime_checkpoint_stable": content_digest(checkpoint)
        == content_digest(restored.native_checkpoint()),
    }


def run_gate(output_path: Path | None = None) -> dict[str, Any]:
    corpus, observations = build_course()
    fixture_before = {
        split: _fixture_digest(FIXTURE_ROOT / f"native_observation_{split}") for split in SPLITS
    }
    preflight_path = (output_path or PROJECT_ROOT / "reports" / "m3r1.json").with_suffix(
        ".preflight.pt"
    )
    final_path = (output_path or PROJECT_ROOT / "reports" / "m3r1.json").with_suffix(".final.pt")
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        untrained = StructuredSemanticLearner(corpus)
        preflight_restored, preflight_before, preflight_after, preflight_digest = (
            _checkpoint_roundtrip(untrained, corpus, preflight_path)
        )
        native = StructuredSemanticLearner(corpus)
        owner_before = native.owner_digests()
        losses = native.fit(corpus.train, epochs=220, learning_rate=1.6)
        owner_after = native.owner_digests()
        final_restored, final_before, final_after, final_digest = _checkpoint_roundtrip(
            native, corpus, final_path
        )
        lesion = StructuredSemanticLearner.from_checkpoint(native.checkpoint(), corpus)
        lesion_before, lesion_after = lesion.zero_fact_head()

        native_test = _metrics(final_restored, corpus.test)
        lesion_test = _metrics(lesion, corpus.test)
        runtime_gate = _runtime_owner_gate(final_restored, observations)
        checkpoint_gate = {
            "preflight_saved": preflight_path.exists(),
            "preflight_prediction_stable": preflight_before == preflight_after,
            "preflight_owner_digest": preflight_digest,
            "post_training_saved": final_path.exists(),
            "post_training_prediction_stable": final_before == final_after,
            "post_training_owner_digest": final_digest,
            "training_steps": native.training_steps,
        }
        fixture_after = {
            split: _fixture_digest(FIXTURE_ROOT / f"native_observation_{split}") for split in SPLITS
        }
        observation_gate = _observation_roundtrip(observations)
        gates = {
            "project_disjoint": corpus.manifest()["record_disjoint"]
            and len({item.family_id for item in corpus.train}) == len(corpus.train)
            and not (
                {item.input_digest for item in corpus.train}
                & {item.input_digest for item in corpus.dev}
            ),
            "native_test_fact_f1": float(native_test["fact_f1"]) >= 0.80,
            "native_test_goal_accuracy": float(native_test["goal_accuracy"]) >= 0.80,
            "native_test_content_accuracy": float(native_test["content_accuracy"]) >= 0.80,
            "native_test_status_accuracy": float(native_test["status_accuracy"]) >= 0.80,
            "checkpoint_save_restore": all(checkpoint_gate.values())
            and checkpoint_gate["training_steps"] > 0,
            "observation_roundtrip": all(observation_gate.values()),
            "runtime_owner_roundtrip": all(
                runtime_gate[key]
                for key in (
                    "owner_attached",
                    "checkpoint_component_present",
                    "restored_owner_present",
                    "output_stable_after_runtime_restore",
                    "cognitive_state_read_only",
                    "runtime_checkpoint_stable",
                )
            ),
            "fixture_workspaces_unchanged": fixture_before == fixture_after,
            "provider_not_attached": corpus.manifest()["provider_attached"] is False,
            "lesion_changes_fact_owner": lesion_before != lesion_after
            and float(lesion_test["fact_f1"]) < float(native_test["fact_f1"]),
        }
        report: dict[str, Any] = {
            "format": REPORT_FORMAT,
            "version": REPORT_VERSION,
            "gate": "M3.R1",
            "status": "passed" if all(gates.values()) else "failed",
            "can_promote": False,
            "promotion_reason": (
                "native Workbench observation learning is admitted only as a bounded CPU Gate; "
                "it is not open-domain language generation or executable Workbench autonomy."
            ),
            "course": {
                "manifest": corpus.manifest(),
                "project_ids": sorted(
                    {item.project_id for values in observations.values() for item in values}
                ),
                "observation_schema": observations["train"][0].schema.to_payload(),
                "parameter_count": native.parameter_count,
                "owner_digests_changed": {
                    name: owner_before[name] != owner_after[name] for name in owner_before
                },
                "losses": losses,
            },
            "metrics": {
                "native_train": _metrics(native, corpus.train),
                "native_dev": _metrics(final_restored, corpus.dev),
                "native_test": native_test,
                "native_lesion_test": lesion_test,
                "controls": {
                    "provider_assisted_upper_bound": _perfect_metrics(corpus.test),
                    "empty_plan": _empty_metrics(corpus.test),
                    "untrained_static_only": _metrics(preflight_restored, corpus.test),
                },
            },
            "checkpoint": checkpoint_gate,
            "observation": observation_gate,
            "runtime": runtime_gate,
            "fixtures": {
                "before": fixture_before,
                "after": fixture_after,
                "unchanged": fixture_before == fixture_after,
            },
            "boundaries": {
                "provider_final_binding": False,
                "raw_text_language_learning": False,
                "workspace_write": False,
                "terminal": False,
                "mcp": False,
                "cuda": False,
            },
            "gates": gates,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return report
    finally:
        preflight_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m3r1_native_observation_20260907.json",
    )
    args = parser.parse_args()
    report = run_gate(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
