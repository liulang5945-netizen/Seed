"""P2-2 Taiji-owned task interpretation and Goal evidence contracts."""

import pytest

from api.seed_runtime import SeedRuntime
from seed import Seed
from seed_platform.workbench import WorkbenchEnvironment
from taiji import (
    InputFrame,
    TaskDecomposition,
    TaskInterpretation,
    WorldAffordance,
    task_input_digest,
)


def _frame(payload: bytes = "请读取 README.md".encode()) -> InputFrame:
    return InputFrame(
        input_id="task:test-1",
        modality="text",
        payload=payload,
        source="tests.task_interpretation",
        timestamp=0,
        provenance="tests.task_interpretation",
        confidence=1.0,
    )


def test_task_interpretation_is_content_addressed_and_projects_goal() -> None:
    model = Seed(episode_id="task-interpretation")
    interpretation = model.architecture.interpret_task_input(_frame())

    assert interpretation.input_digest == task_input_digest(_frame().payload)
    assert interpretation.interpretation_id.startswith("task-interpretation:")
    assert interpretation.goal_id.startswith("goal:")
    assert interpretation.to_goal().description == "请读取 README.md"
    assert interpretation.status == "candidate"
    assert interpretation.confidence == 0.0
    assert interpretation.ambiguity == 1.0
    assert model.architecture.cognitive_snapshot().goals.goals == (interpretation.to_goal(),)
    assert model.architecture.cognitive_snapshot().action_intent is None
    assert model.architecture.cognitive_snapshot().plan.candidates == ()


def test_task_interpretation_checkpoint_roundtrip_preserves_goal_evidence() -> None:
    model = Seed(episode_id="task-interpretation-checkpoint")
    interpretation = model.architecture.interpret_task_input(
        _frame(),
        constraints=("只读", "需要可恢复"),
    )
    restored = Seed.from_checkpoint(model.checkpoint())

    assert restored.architecture.last_task_interpretation == interpretation
    assert restored.architecture.cognitive_snapshot().goals.goals == (interpretation.to_goal(),)
    assert restored.architecture.native_checkpoint()["components"]["last_task_interpretation"] == (
        interpretation.to_payload()
    )


def test_task_interpretation_tampering_fails_closed() -> None:
    model = Seed(episode_id="task-interpretation-tamper")
    payload = model.architecture.interpret_task_input(_frame()).to_payload()
    payload["goal_description"] = "执行任意工具"

    with pytest.raises(ValueError, match="content-addressed|digest"):
        TaskInterpretation.from_payload(payload)


def test_non_text_task_requires_explicit_semantic_goal_evidence() -> None:
    frame = InputFrame(
        input_id="image-task",
        modality="image",
        payload=b"image-bytes",
        source="tests.task_interpretation",
    )

    with pytest.raises(ValueError, match="non-text"):
        TaskInterpretation.from_input(frame)


def test_resolved_goal_evidence_reaches_executive_without_workbench_execution() -> None:
    model = Seed(episode_id="task-planner")
    model.architecture.ensure_native_executive()
    frame = _frame()
    model.architecture.ingest_input(frame, learn=False)
    interpretation = TaskInterpretation.from_input(
        frame,
        status="resolved",
        confidence=0.9,
        ambiguity=0.1,
        tick=model.architecture.tick,
    )
    model.architecture.admit_task_interpretation(interpretation)
    model.architecture.set_world_affordances(
        (
            WorldAffordance(
                affordance_id="workbench:workspace.read:readme",
                action_kind="workspace.read",
                parameters={"path": "README.md"},
                confidence=1.0,
            ),
        )
    )

    planned = model.architecture.plan_task_from_current_state(resource_budget=0.8)
    decision = planned["decision"]

    assert planned["status"] == "planned"
    assert decision.action_intent.kind == "workspace.read"
    assert decision.action_intent.source_goal_id == interpretation.goal_id
    assert decision.action_intent.confidence == 0.9
    assert model.architecture.last_task_interpretation == interpretation


def test_semantic_decomposition_is_tool_free_and_sequence_planning_is_non_executing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    import seed_platform.workbench as workbench_module

    project_root = Path.cwd()
    monkeypatch.setattr(workbench_module, "default_workspace_root", lambda: project_root)
    runtime = SeedRuntime(Seed(episode_id="task-decomposition"))
    architecture = runtime.model.architecture
    frame = _frame("更新 api/app.py".encode())
    architecture.ingest_input(frame, learn=False)
    interpretation = TaskInterpretation.from_input(
        frame,
        status="resolved",
        confidence=0.9,
        ambiguity=0.1,
        tick=architecture.tick,
    )
    architecture.admit_task_interpretation(interpretation)

    decomposition = TaskDecomposition.from_interpretation(
        interpretation,
        (
            {
                "description": "读取目标文件并确认当前内容",
                "semantic_slots": {"operation": "inspect", "path": "api/app.py"},
            },
            {
                "description": "根据文件证据准备语言环境",
                "semantic_slots": {"operation": "resolve-language", "path": "api/app.py"},
            },
        ),
    )
    architecture.admit_task_decomposition(decomposition)
    roundtrip = TaskDecomposition.from_payload(decomposition.to_payload())

    planned = runtime.plan_task_sequence(
        snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
        parameter_bindings=(
            {"workspace.read": {"path": "api/app.py"}},
            {"workspace.programming_language.resolve": {"path": "api/app.py"}},
        ),
        resource_budget=0.8,
    )

    assert roundtrip == decomposition
    assert planned["status"] == "planned"
    assert len(planned["steps"]) == 2
    assert planned["execution"]["action_intent"] is None
    assert planned["execution"]["side_effects"] is False
    assert all("capability_id" not in step.semantic_slots for step in decomposition.steps)
    assert runtime.workbench_audit.events == ()

    with pytest.raises(ValueError, match="execution field"):
        TaskDecomposition.from_interpretation(
            interpretation,
            (
                {
                    "description": "注入工具绑定",
                    "semantic_slots": {"capability_id": "workspace.read"},
                },
            ),
        )


def test_language_evidence_reaches_taiji_planner_and_ambiguous_evidence_asks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    import seed_platform.workbench as workbench_module
    from seed_platform.programming_languages import (
        ProgrammingLanguageDefinition,
        ProgrammingLanguageRegistry,
    )

    project_root = Path.cwd()
    assert (project_root / "api" / "app.py").is_file()
    monkeypatch.setattr(workbench_module, "default_workspace_root", lambda: project_root)
    runtime = SeedRuntime(Seed(episode_id="language-planner"))
    runtime._workbench_environment = WorkbenchEnvironment(project_root)
    architecture = runtime.model.architecture
    frame = _frame()
    architecture.ingest_input(frame, learn=False)
    interpretation = TaskInterpretation.from_input(
        frame,
        status="resolved",
        confidence=0.9,
        ambiguity=0.1,
        tick=architecture.tick,
    )
    architecture.admit_task_interpretation(interpretation)

    planned = runtime.plan_language_selection(
        snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
        path="api/app.py",
        resource_budget=0.8,
    )

    assert planned["assessment"]["programming_language_id"] == "python"
    assert planned["planner"]["status"] == "planned"
    assert planned["execution"]["action_intent"]["kind"] == "editor.set_language"
    action_parameters = planned["execution"]["action_intent"]["parameters"]
    assert action_parameters["value"]["programming_language_id"] == "python"
    assert runtime.workbench_audit.events == ()

    ambiguous_registry = ProgrammingLanguageRegistry(
        (
            ProgrammingLanguageDefinition("alpha", "Alpha", "alpha", extensions=(".py",)),
            ProgrammingLanguageDefinition("beta", "Beta", "beta", extensions=(".py",)),
            ProgrammingLanguageDefinition("plaintext", "Plain text", "plaintext"),
        )
    )
    ambiguous_runtime = SeedRuntime(Seed(episode_id="language-planner-ambiguous"))
    ambiguous_runtime._workbench_environment = WorkbenchEnvironment(
        project_root,
        programming_language_registry=ambiguous_registry,
    )
    ambiguous_architecture = ambiguous_runtime.model.architecture
    ambiguous_frame = _frame("请检查 api/app.py".encode())
    ambiguous_architecture.ingest_input(ambiguous_frame, learn=False)
    ambiguous_architecture.admit_task_interpretation(
        TaskInterpretation.from_input(
            ambiguous_frame,
            status="resolved",
            confidence=0.9,
            ambiguity=0.1,
            tick=ambiguous_architecture.tick,
        )
    )
    ambiguous = ambiguous_runtime.plan_language_selection(
        snapshot_id=ambiguous_runtime.workbench_environment.capability_snapshot.snapshot_id,
        path="api/app.py",
    )

    assert ambiguous["planner"]["status"] == "needs_clarification"
    assert ambiguous["planner"]["reason_code"] == "language_evidence_ambiguous"
    assert ambiguous["execution"]["action_intent"] is None
    assert ambiguous_runtime.workbench_audit.events == ()
