"""M3.R3: verify native read-only Workbench intent planning on CPU.

M3.R0--R2 establish the evidence, static semantic, and temporal state
boundaries.  This Gate asks the next narrow question: can the learned
``WorldState + Goal + ContentPlan`` be rendered into a capability-bound,
read-only ``ActionIntent`` without executing it?

The content-to-capability routes are host policy, not model knowledge.  The
planner rejects stale observations, unknown routes, ambiguous language facts,
missing targets, and any capability outside the explicit read-only allowlist.
No Workbench executor, terminal, MCP, patch, or language-selection operation
is called by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m3r1_native_observation import (  # noqa: E402
    FIXTURE_ROOT,
    SPLITS,
    _course_registry,
    _fixture_digest,
    _semantic_kind,
)
from scripts.training.eval_taiji_m3r2_task_state_sequence import (  # noqa: E402
    _runtime_gate,
    _sequence_for_split,
    _world,
    build_transition_course,
)
from seed.persistence import atomic_save  # noqa: E402
from seed_platform.workbench import (  # noqa: E402
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji import (  # noqa: E402
    ContentPlan,
    NativeReadOnlyIntentPlanner,
    ReadOnlyIntentDecision,
    ReadOnlyIntentPolicy,
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionLearner,
    WorkbenchObservation,
    content_digest,
)

REPORT_FORMAT = "taiji-m3r3-native-read-only-intent-v1"
REPORT_VERSION = 1
GATE_NAME = "M3.R3"
LEARNING_EPOCHS = 280
LEARNING_RATE = 0.2
READ_ONLY_ROUTES = (
    ("content:inspect-language", "workspace.read"),
    ("content:clarify-toolchain", "workspace.programming_language.resolve"),
)


def _catalog_semantics(
    corpus: StructuredSemanticTransitionCorpus,
    observation: WorkbenchObservation,
    *,
    tick: int,
) -> tuple[Any, ContentPlan]:
    """Return the gold structured semantics for a control arm."""

    kind = _semantic_kind(observation)
    goal = next(item for item in corpus.goals if item.goal_id == f"goal:{kind}")
    base = next(item for item in corpus.content_plans if item.content_id == f"content:{kind}")
    content = ContentPlan.from_payload({**base.to_payload(), "tick": int(tick)})
    return goal, content


def _expected_capability(observation: WorkbenchObservation) -> str | None:
    """Return the only capability this Gate permits for one verified fact."""

    kind = _semantic_kind(observation)
    if kind == "inspect-language":
        return "workspace.read"
    if kind == "clarify-toolchain":
        return "workspace.programming_language.resolve"
    return None


def _decision_digest(decision: ReadOnlyIntentDecision | None) -> str:
    return content_digest(None if decision is None else decision.to_payload())


def _live_evidence_matches(
    environment: WorkbenchEnvironment,
    observation: WorkbenchObservation,
    capability_id: str,
) -> tuple[bool, dict[str, Any]]:
    """Probe only side-effect-free Workbench evidence for an accepted intent."""

    if capability_id == "workspace.read":
        evidence = environment.read_workspace_evidence({"path": observation.path})
        matches = (
            bool(evidence.get("digest")) == bool(observation.file_digest)
            and str(evidence.get("digest", "")) == observation.file_digest
            and str(evidence.get("path", "")) == observation.path
        )
        return matches, {
            "operation": "workspace.read.evidence",
            "digest": str(evidence.get("digest", "")),
            "path": str(evidence.get("path", "")),
            "matches_observation": matches,
        }
    if capability_id == "workspace.programming_language.resolve":
        evidence = environment.resolve_programming_language_evidence({"path": observation.path})
        matches = (
            str(evidence.get("file_digest", "")) == observation.file_digest
            and str(evidence.get("programming_language_id", "")) == observation.language_id
            and str(evidence.get("selection_state", "")) == observation.selection_state
        )
        return matches, {
            "operation": "workspace.programming_language.resolve.evidence",
            "file_digest": str(evidence.get("file_digest", "")),
            "programming_language_id": str(evidence.get("programming_language_id", "")),
            "selection_state": str(evidence.get("selection_state", "")),
            "matches_observation": matches,
        }
    raise AssertionError(f"unexpected accepted read-only capability: {capability_id}")


def _probe_decision(
    *,
    environment: WorkbenchEnvironment,
    observation: WorkbenchObservation,
    decision: ReadOnlyIntentDecision | None,
    expected_capability: str | None,
) -> dict[str, Any]:
    """Validate an intent and probe its evidence without executing it."""

    row: dict[str, Any] = {
        "path": observation.path,
        "expected_capability": expected_capability,
        "accepted": bool(decision is not None and decision.accepted),
        "reason_code": None if decision is None else decision.reason_code,
        "intent_capability": (
            None
            if decision is None or decision.action_intent is None
            else decision.action_intent.kind
        ),
        "policy": None,
        "evidence": None,
        "route_correct": False,
        "clarification_correct": expected_capability is None
        and (decision is None or not decision.accepted),
        "side_effect_free": True,
    }
    if decision is None or not decision.accepted or decision.action_intent is None:
        row["route_correct"] = expected_capability is None
        return row

    intent = decision.action_intent
    row["route_correct"] = intent.kind == expected_capability
    state_before = content_digest(environment.last_result)
    boundary_before = environment.active_task_boundary
    try:
        request = WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=environment.capability_snapshot.snapshot_id,
        )
        policy = environment.policy_for(request)
        row["policy"] = {
            "decision": policy.decision,
            "reason_code": policy.reason_code,
            "snapshot_id": policy.snapshot_id,
        }
        if policy.decision == "allow":
            evidence_matches, evidence = _live_evidence_matches(
                environment,
                observation,
                intent.kind,
            )
            row["evidence"] = evidence
            row["evidence_match"] = evidence_matches
        else:
            row["evidence_match"] = False
    finally:
        row["side_effect_free"] = (
            state_before == content_digest(environment.last_result)
            and boundary_before is environment.active_task_boundary
        )
    return row


def _run_arm(
    *,
    arm: str,
    learner: StructuredSemanticTransitionLearner | None,
    corpus: StructuredSemanticTransitionCorpus,
    sequence: tuple[WorkbenchObservation, ...],
    environment: WorkbenchEnvironment,
    planner: NativeReadOnlyIntentPlanner,
) -> dict[str, Any]:
    """Run native, static-only, provider, or empty intent planning."""

    if arm not in {"native", "static-only", "provider", "empty"}:
        raise ValueError(f"unsupported M3.R3 arm: {arm}")
    initial = _world(sequence[0], tick=0)
    current = initial
    rows: list[dict[str, Any]] = []
    accepted = 0
    route_correct = 0
    policy_allowed = 0
    evidence_matches = 0
    clarification_correct = 0
    side_effect_free = True
    for tick, observation in enumerate(sequence[1:], start=1):
        expected = _expected_capability(observation)
        decision: ReadOnlyIntentDecision | None = None
        if arm == "native":
            if learner is None:
                raise AssertionError("native arm requires a transition learner")
            result = learner.predict(
                current,
                observation.to_percept_event(tick=tick),
            )
            next_world = result.world or current
            if result.goal is not None and result.content_plan is not None:
                decision = planner.propose(
                    observation=observation,
                    world=next_world,
                    goal=result.goal,
                    content=result.content_plan,
                    capability_snapshot=environment.capability_snapshot,
                    tick=tick,
                )
            current = next_world
        elif arm == "static-only":
            # The static-only control has the gold semantic label but never
            # applies the preceding observation transition.  The planner must
            # reject it as stale instead of turning frozen state into an intent.
            goal, content = _catalog_semantics(corpus, observation, tick=tick)
            decision = planner.propose(
                observation=observation,
                world=initial,
                goal=goal,
                content=content,
                capability_snapshot=environment.capability_snapshot,
                tick=tick,
            )
        elif arm == "provider":
            goal, content = _catalog_semantics(corpus, observation, tick=tick)
            decision = planner.propose(
                observation=observation,
                world=_world(observation, tick=tick),
                goal=goal,
                content=content,
                capability_snapshot=environment.capability_snapshot,
                tick=tick,
            )

        row = _probe_decision(
            environment=environment,
            observation=observation,
            decision=decision,
            expected_capability=expected,
        )
        row["tick"] = tick
        row["arm"] = arm
        rows.append(row)
        accepted += int(row["accepted"])
        route_correct += int(row["route_correct"])
        policy_allowed += int(
            isinstance(row["policy"], dict) and row["policy"].get("decision") == "allow"
        )
        evidence_matches += int(row.get("evidence_match", False))
        clarification_correct += int(row["clarification_correct"])
        side_effect_free = side_effect_free and bool(row["side_effect_free"])
    count = max(1, len(rows))
    expected_count = sum(_expected_capability(item) is not None for item in sequence[1:])
    return {
        "arm": arm,
        "steps": len(rows),
        "expected_intent_steps": expected_count,
        "accepted_intents": accepted,
        "intent_coverage": accepted / max(1, expected_count),
        "route_accuracy": route_correct / count,
        "policy_allow_accuracy": policy_allowed / max(1, accepted),
        "evidence_accuracy": evidence_matches / max(1, accepted),
        "clarification_accuracy": clarification_correct / max(1, count - expected_count),
        "side_effect_free": side_effect_free,
        "rows": rows,
    }


def _stale_snapshot_gate(
    *,
    planner: NativeReadOnlyIntentPlanner,
    corpus: StructuredSemanticTransitionCorpus,
    observation: WorkbenchObservation,
    environment: WorkbenchEnvironment,
) -> dict[str, Any]:
    goal, content = _catalog_semantics(corpus, observation, tick=1)
    decision = planner.propose(
        observation=observation,
        world=_world(observation, tick=1),
        goal=goal,
        content=content,
        capability_snapshot=environment.capability_snapshot,
        tick=1,
    )
    if not decision.accepted or decision.action_intent is None:
        return {
            "accepted_source_intent": False,
            "stale_request_denied": False,
            "reason_code": "source_intent_not_available",
        }
    request = WorkbenchActionRequest.from_action_intent(
        decision.action_intent,
        snapshot_id="stale-capability-snapshot",
    )
    policy = environment.policy_for(request)
    return {
        "accepted_source_intent": True,
        "stale_request_denied": policy.decision == "deny"
        and policy.reason_code == "stale_capability_snapshot",
        "reason_code": policy.reason_code,
    }


def _planner_checkpoint_gate(
    planner: NativeReadOnlyIntentPlanner,
    path: Path,
) -> dict[str, Any]:
    payload = planner.checkpoint()
    atomic_save(payload, path)
    restored = NativeReadOnlyIntentPlanner.from_checkpoint(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    return {
        "saved": path.exists(),
        "payload_stable": content_digest(payload) == content_digest(restored.checkpoint()),
        "policy_stable": planner.policy.to_payload() == restored.policy.to_payload(),
        "routes": dict(restored.policy.routes),
    }


def _decision_roundtrip_gate(
    decision: ReadOnlyIntentDecision,
) -> dict[str, Any]:
    restored = ReadOnlyIntentDecision.from_payload(decision.to_payload())
    return {
        "digest_stable": _decision_digest(decision) == _decision_digest(restored),
        "accepted_preserved": restored.accepted == decision.accepted,
        "intent_kind_preserved": (
            restored.action_intent is not None
            and decision.action_intent is not None
            and restored.action_intent.kind == decision.action_intent.kind
        ),
    }


def _lesion_arm(
    learner: StructuredSemanticTransitionLearner,
    corpus: StructuredSemanticTransitionCorpus,
    sequence: tuple[WorkbenchObservation, ...],
    environment: WorkbenchEnvironment,
    planner: NativeReadOnlyIntentPlanner,
) -> tuple[dict[str, Any], str, str]:
    lesion = StructuredSemanticTransitionLearner.from_checkpoint(learner.checkpoint(), corpus)
    before, after = lesion.zero_transition_head()
    result = _run_arm(
        arm="native",
        learner=lesion,
        corpus=corpus,
        sequence=sequence,
        environment=environment,
        planner=planner,
    )
    return result, before, after


def run_gate(output_path: Path | None = None) -> dict[str, Any]:
    corpus, observations = build_transition_course()
    test_sequence = _sequence_for_split(observations["test"])
    environment = WorkbenchEnvironment(
        root=FIXTURE_ROOT / "native_observation_test",
        programming_language_registry=_course_registry(),
    )
    policy = ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES)
    planner = NativeReadOnlyIntentPlanner(policy)
    fixture_before = {
        split: _fixture_digest(FIXTURE_ROOT / f"native_observation_{split}") for split in SPLITS
    }
    base = output_path or PROJECT_ROOT / "reports" / "m3r3.json"
    preflight_path = base.with_suffix(".preflight.pt")
    final_path = base.with_suffix(".final.pt")
    planner_path = base.with_suffix(".planner.pt")
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        untrained = StructuredSemanticTransitionLearner(corpus)
        preflight_before = content_digest(
            [
                item.to_payload()
                for item in (
                    untrained.predict(
                        _world(test_sequence[0], tick=0),
                        test_sequence[1].to_percept_event(tick=1),
                    ),
                )
            ]
        )
        atomic_save(untrained.checkpoint(), preflight_path)
        preflight_restored = StructuredSemanticTransitionLearner.from_checkpoint(
            torch.load(preflight_path, map_location="cpu", weights_only=False),
            corpus,
        )
        preflight_result = preflight_restored.predict(
            _world(test_sequence[0], tick=0),
            test_sequence[1].to_percept_event(tick=1),
        )
        preflight_after = content_digest([preflight_result.to_payload()])

        native = StructuredSemanticTransitionLearner(corpus)
        losses = native.fit(
            corpus.train,
            epochs=LEARNING_EPOCHS,
            learning_rate=LEARNING_RATE,
        )
        atomic_save(native.checkpoint(), final_path)
        restored = StructuredSemanticTransitionLearner.from_checkpoint(
            torch.load(final_path, map_location="cpu", weights_only=False),
            corpus,
        )
        planner_checkpoint = _planner_checkpoint_gate(planner, planner_path)
        native_metrics = _run_arm(
            arm="native",
            learner=restored,
            corpus=corpus,
            sequence=test_sequence,
            environment=environment,
            planner=planner,
        )
        static_metrics = _run_arm(
            arm="static-only",
            learner=None,
            corpus=corpus,
            sequence=test_sequence,
            environment=environment,
            planner=planner,
        )
        provider_metrics = _run_arm(
            arm="provider",
            learner=None,
            corpus=corpus,
            sequence=test_sequence,
            environment=environment,
            planner=planner,
        )
        empty_metrics = _run_arm(
            arm="empty",
            learner=None,
            corpus=corpus,
            sequence=test_sequence,
            environment=environment,
            planner=planner,
        )
        accepted_decision = planner.propose(
            observation=test_sequence[1],
            world=_world(test_sequence[1], tick=1),
            goal=_catalog_semantics(corpus, test_sequence[1], tick=1)[0],
            content=_catalog_semantics(corpus, test_sequence[1], tick=1)[1],
            capability_snapshot=environment.capability_snapshot,
            tick=1,
        )
        decision_roundtrip = _decision_roundtrip_gate(accepted_decision)
        stale_snapshot = _stale_snapshot_gate(
            planner=planner,
            corpus=corpus,
            observation=test_sequence[1],
            environment=environment,
        )
        lesion_metrics, lesion_before, lesion_after = _lesion_arm(
            restored,
            corpus,
            test_sequence,
            environment,
            planner,
        )
        runtime = _runtime_gate(restored, test_sequence)
        fixture_after = {
            split: _fixture_digest(FIXTURE_ROOT / f"native_observation_{split}") for split in SPLITS
        }
        checkpoint = {
            "preflight_saved": preflight_path.exists(),
            "preflight_output_stable": preflight_before == preflight_after,
            "post_training_saved": final_path.exists(),
            "post_training_output_stable": content_digest(
                [
                    item.to_payload()
                    for item in (
                        native.predict(
                            _world(test_sequence[0], tick=0),
                            test_sequence[1].to_percept_event(tick=1),
                        ),
                    )
                ]
            )
            == content_digest(
                [
                    item.to_payload()
                    for item in (
                        restored.predict(
                            _world(test_sequence[0], tick=0),
                            test_sequence[1].to_percept_event(tick=1),
                        ),
                    )
                ]
            ),
            "planner": planner_checkpoint,
            "training_steps": native.training_steps,
        }
        gates = {
            "project_disjoint": corpus.manifest()["record_disjoint"]
            and not (
                {item.family_id for item in corpus.train} & {item.family_id for item in corpus.test}
            )
            and not (
                {item.input_digest for item in corpus.train}
                & {item.input_digest for item in corpus.test}
            ),
            "native_expected_intents": native_metrics["accepted_intents"] == 3,
            "native_route_accuracy": native_metrics["route_accuracy"] == 1.0,
            "native_policy_allow": native_metrics["policy_allow_accuracy"] == 1.0,
            "native_evidence_accuracy": native_metrics["evidence_accuracy"] == 1.0,
            "native_clarification_accuracy": native_metrics["clarification_accuracy"] == 1.0,
            "native_side_effect_free": native_metrics["side_effect_free"],
            "native_beats_static_only": native_metrics["accepted_intents"]
            > static_metrics["accepted_intents"],
            "provider_upper_bound": provider_metrics["accepted_intents"] == 3
            and provider_metrics["route_accuracy"] == 1.0
            and provider_metrics["evidence_accuracy"] == 1.0,
            "empty_arm_is_empty": empty_metrics["accepted_intents"] == 0,
            "stale_snapshot_denied": all(stale_snapshot.values())
            and stale_snapshot["stale_request_denied"],
            "decision_roundtrip": all(decision_roundtrip.values()),
            "checkpoint_save_restore": all(
                (
                    checkpoint["preflight_saved"],
                    checkpoint["preflight_output_stable"],
                    checkpoint["post_training_saved"],
                    checkpoint["post_training_output_stable"],
                    checkpoint["planner"]["saved"],
                    checkpoint["planner"]["payload_stable"],
                    checkpoint["planner"]["policy_stable"],
                    checkpoint["training_steps"] > 0,
                )
            ),
            "runtime_owner_roundtrip": all(
                runtime[key]
                for key in (
                    "owner_attached",
                    "checkpoint_component_present",
                    "restored_owner_present",
                    "output_stable_after_runtime_restore",
                    "cognitive_state_read_only",
                    "runtime_checkpoint_stable",
                )
            ),
            "transition_lesion_changes_intent": lesion_before != lesion_after
            and lesion_metrics["accepted_intents"] < native_metrics["accepted_intents"],
            "fixture_workspaces_unchanged": fixture_before == fixture_after,
            "provider_not_attached": corpus.manifest()["provider_attached"] is False,
        }
        report: dict[str, Any] = {
            "format": REPORT_FORMAT,
            "version": REPORT_VERSION,
            "gate": GATE_NAME,
            "status": "passed" if all(gates.values()) else "failed",
            "can_promote": False,
            "promotion_reason": (
                "native read-only ActionIntent planning is a bounded CPU canary; "
                "preview approval, writes, terminal, MCP, and language selection "
                "remain frozen for M3.R4."
            ),
            "course": {
                "manifest": corpus.manifest(),
                "parameter_count": native.parameter_count,
                "learning_epochs": LEARNING_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "losses": losses,
                "read_only_policy": policy.to_payload(),
                "static_routes_are_host_policy": True,
            },
            "metrics": {
                "native": native_metrics,
                "static_only": static_metrics,
                "provider_assisted_upper_bound": provider_metrics,
                "empty": empty_metrics,
                "lesion": lesion_metrics,
                "decision_roundtrip": decision_roundtrip,
                "stale_snapshot": stale_snapshot,
            },
            "checkpoint": checkpoint,
            "runtime": runtime,
            "fixtures": {
                "before": fixture_before,
                "after": fixture_after,
                "unchanged": fixture_before == fixture_after,
            },
            "boundaries": {
                "provider_final_binding": False,
                "raw_text_language_learning": False,
                "workspace_read_execution": False,
                "workspace_write": False,
                "terminal": False,
                "mcp": False,
                "editor_set_language": False,
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
        for path in (preflight_path, final_path, planner_path):
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m3r3_read_only_intent_20260907.json",
    )
    args = parser.parse_args()
    report = run_gate(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
