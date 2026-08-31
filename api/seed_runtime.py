"""Seed 原生运行时：加载 / 卸载 / 热切换 / 字节级对话。

``neuroplex/core/app_state.py`` 的对等物：Cortex（neuroplex）保留为可切换的
冻结对照，Seed 是独立可切换的原生运行时。两者互斥：任一时刻聊天主路由
只走其中一个。本模块不导入 ``neuroplex``，切换语义由调用方编排。

对话口径与阶段 1 训练管线一致（``scripts/training/train_seed_corpus.py``）：
对话结构用 ``问：/答：`` 文本标记序列化，会话边界由基底的
``boundary_symbol`` 承担，全程不引入 tokenizer。
"""

from __future__ import annotations

import functools
import hashlib
import logging
import pickle
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from taiji import InputFrame

logger = logging.getLogger("ApiServer.SeedRuntime")

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "seed_corpus.pt"

_TURN_MARKERS = ("\n问：", "问：")

# 输入长度上限（字符）：基底逐字节处理前缀（实测 ≈430 字节/秒），十万级
# 提示会让单请求阻塞数分钟（压测实测 100K 字符 ≈ 309 秒），构成可用性风险；
# 2048 字符（约 6KB ≈ 14s 前缀成本）是病态输入的封顶，典型对话消息远低于此，
# 不影响首字节延迟门槛。超长输入截断处理而非拒绝，保持对话可用。
MAX_PROMPT_CHARS = 2048
MAX_WORKBENCH_LOOP_COMMITTED_REQUESTS = 128


def _workbench_synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize workbench mutations while allowing nested checkpoint saves."""

    @functools.wraps(method)
    def synchronized(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized


def _workbench_successor_synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize successor execution and publish its branch mutation atomically."""

    @functools.wraps(method)
    def synchronized(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            result = method(self, *args, **kwargs)
            return self._sync_recovery_portfolio_after_successor(result)

    return synchronized


class SeedRuntime:
    """单个 Seed 有机体 + 字节级对话接口（线程安全）。"""

    RUNTIME_TYPE = "seed"

    def __init__(
        self,
        model: Any,
        checkpoint_path: Path | None = None,
        provider_status: dict[str, str] | None = None,
        provider_runtime: Any | None = None,
        provider_config: Any | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        from taiji import LanguageOrgan, NativeReadableTextLanguageOrgan

        self.model.architecture.ensure_native_executive()
        # Internal drives (curiosity/fatigue/stress) are integrated by Taiji on
        # every observe/settle.  Attach idempotently so a controller restored
        # from the checkpoint keeps its accumulated state.
        self.model.architecture.ensure_homeostatic_controller()

        # Chat stays local by default.  An external organ can reach product
        # chat only through explicit config plus the realization/safety Gate.
        # Annotated as the LanguageOrgan protocol because the slot may hold
        # either the native realizer or an allowlisted external decoder organ.
        self._chat_organ: LanguageOrgan = NativeReadableTextLanguageOrgan()
        if provider_status is None:
            from seed.language_provider import attach_native_language_provider

            attach_native_language_provider(self.model.architecture)
            self._provider_status = self._native_status()
        else:
            self._provider_status = dict(provider_status)
            if self._provider_status.get("chat_enabled") == "true":
                candidate = self.model.architecture.language_organ
                if isinstance(candidate, LanguageOrgan):
                    self._chat_organ = candidate
        self._provider_runtime = provider_runtime
        self._provider_config = provider_config
        from seed_platform.workbench import WorkbenchAuditLog, WorkbenchEnvironment

        self._workbench_environment = WorkbenchEnvironment()
        self._workbench_audit = WorkbenchAuditLog()
        self._workbench_loop_state: dict[str, Any] = {}
        self._last_artifact_consumption_audit: dict[str, Any] | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _native_status() -> dict[str, str]:
        """Mirror ``LanguageProviderStatus.to_dict`` for the built-in organ.

        Returning the status object's own projection (not a hand-written dict)
        guarantees the native mode emits the same 14-key shape as every guarded
        mode, so consumers of ``language_provider_status`` never see a narrower
        native payload.
        """

        from seed.language_provider import (
            LanguageProviderStatus,
            NativeReadableTextLanguageOrgan,
        )

        return LanguageProviderStatus(
            mode="native",
            state="active",
            provider="native",
            backend_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
            artifact_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
            chat_enabled=False,
        ).to_dict()

    @property
    def name(self) -> str:
        if self.checkpoint_path is not None:
            return f"seed:{self.checkpoint_path.name}"
        return "seed:scratch"

    def rotate_language_provider(self, config: Any, registry: Any) -> Any:
        """Atomically rotate to an allowlisted provider after its canary passes."""

        from seed.language_provider import rotate_language_provider

        with self._lock:
            result = rotate_language_provider(
                self.model.architecture,
                registry,
                config,
                current_runtime=self._provider_runtime,
            )
            if result.committed:
                self._provider_status = result.status.to_dict()
                self._provider_runtime = result.runtime
                self._provider_config = config
                from taiji import LanguageOrgan, NativeReadableTextLanguageOrgan

                candidate = self.model.architecture.language_organ
                self._chat_organ = (
                    candidate
                    if result.status.chat_enabled and isinstance(candidate, LanguageOrgan)
                    else NativeReadableTextLanguageOrgan()
                )
            return result

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str | None = None,
        *,
        provider_config: Any | None = None,
    ) -> SeedRuntime:
        """从 seed-native-v1 检查点装配 Seed（与训练管线同一信封）。"""
        import torch

        from seed import Seed

        path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            logger.warning(
                "checkpoint %s 含自定义对象，以不安全模式（weights_only=False）"
                "加载受信 checkpoint",
                path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = Seed.from_checkpoint(checkpoint)
        from seed import LanguageProviderConfig
        from seed.language_provider import activate_language_provider

        selected_config = provider_config or LanguageProviderConfig.from_environment(
            model.config.language_provider
        )
        provider_status, provider_runtime = activate_language_provider(
            model.architecture,
            selected_config,
        )
        logger.info("Seed runtime loaded from %s", path)
        runtime = cls(model, path, provider_status.to_dict(), provider_runtime, selected_config)
        metadata = checkpoint.get("metadata")
        if isinstance(metadata, Mapping):
            runtime._restore_workbench_metadata(metadata.get("workbench"))
        return runtime

    @staticmethod
    def _serialize(prompt: str, history: Sequence[tuple[str, str]] | None) -> str:
        """沿用训练语料的 问：/答： 标记把多轮对话铺成一段文本。

        前缀总长受 ``MAX_PROMPT_CHARS`` 保护：超限时从最早的轮次开始丢弃，
        保留最近上下文（多轮引用的新鲜度优先级高于久远历史）。
        """
        parts: list[str] = []
        budget = MAX_PROMPT_CHARS - len(prompt) - len("问：\n答：")
        kept: list[str] = []
        for user, assistant in reversed(list(history or [])):
            if user and assistant:
                part = f"问：{user}\n答：{assistant}"
                budget -= len(part) + 1
                if budget < 0:
                    break
                kept.append(part)
        parts = list(reversed(kept))
        parts.append(f"问：{prompt}\n答：")
        return "\n".join(parts)

    def chat(
        self,
        prompt: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        max_length: int = 256,
        learn: bool = True,
    ) -> str:
        """生成回复并经 Taiji 语言器官形成可读表层。"""
        from taiji import ExpressionPlan

        prompt = (prompt or "")[:MAX_PROMPT_CHARS]
        text = self._serialize(prompt, history)
        with self._lock:
            frame = InputFrame(
                input_id=f"chat:{self.model.tick}",
                modality="text",
                payload=text.encode("utf-8"),
                source="seed.client.chat",
                timestamp=self.model.tick,
                provenance="external",
                confidence=1.0,
            )
            raw = self.model.generate_input(
                frame,
                max_length,
                stop_at_boundary=True,
                sample=False,
            )
            native_prediction = raw.decode("utf-8", errors="replace")
            for marker in _TURN_MARKERS:
                index = native_prediction.find(marker)
                if index >= 0:
                    native_prediction = native_prediction[:index]
            history_payload = [
                {"user": user, "assistant": assistant}
                for user, assistant in (history or [])
                if user and assistant
            ]
            expression = ExpressionPlan(
                expression_id=f"chat:{self.model.tick}:expression",
                content_id=f"chat:{self.model.tick}:content",
                modality="text",
                channel="message",
                fields={
                    "intent_kind": "chat_answer",
                    "semantic_slots": {
                        "prompt": prompt,
                        "history": history_payload,
                    },
                    "native_prediction": native_prediction,
                    "expected_outcome": "answer user in readable language",
                },
                provenance="seed.client.chat",
                tick=self.model.tick,
            )
            emission = self._chat_organ.emit(expression)
            answer = emission.text_bytes.decode("utf-8", errors="strict")
            self._observe_provider_health(expression, emission)
            if learn:
                # 多轮上下文由基底持久状态天然承担：整段会话文本一次写回，
                # 与 learn_bytes 的训练语义完全一致。
                self.model.learn_bytes(
                    (text + answer).encode("utf-8"),
                    include_boundary=True,
                )
        for marker in _TURN_MARKERS:
            index = answer.find(marker)
            if index >= 0:
                answer = answer[:index]
        return answer.strip()

    @staticmethod
    def _task_context_digest(history: Sequence[Sequence[str]] | None) -> str:
        from taiji import content_digest

        history_payload = []
        for item in history or ():
            if len(item) != 2:
                raise ValueError("task history entries must contain user and assistant text")
            history_payload.append({"user": str(item[0]), "assistant": str(item[1])})
        return content_digest(history_payload) if history_payload else ""

    def _task_frame(self, prompt: str) -> tuple[str, InputFrame]:
        prompt = (prompt or "")[:MAX_PROMPT_CHARS]
        if not prompt.strip():
            raise ValueError("task prompt cannot be empty")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return prompt, InputFrame(
            input_id=f"task:{self.model.tick}:{digest[:16]}",
            modality="text",
            payload=prompt.encode("utf-8"),
            source="seed.client.workbench.task",
            timestamp=self.model.tick,
            provenance="seed.client.workbench.task",
            confidence=1.0,
        )

    def interpret_task(
        self,
        prompt: str,
        *,
        history: Sequence[Sequence[str]] | None = None,
        constraints: Sequence[str] = (),
    ):
        """Admit natural-language task as Taiji goal evidence only.

        This method deliberately has no Workbench environment access and no
        execution path. History is content-addressed as context; the goal
        candidate preserves the current prompt and remains unresolved until a
        Taiji semantic interpreter/planner earns a decision.
        """

        from taiji import TaskInterpretation

        prompt, frame = self._task_frame(prompt)
        context_digest = self._task_context_digest(history)
        with self._lock:
            interpretation = self.model.architecture.interpret_task_input(
                frame,
                goal_description=prompt,
                constraints=constraints,
                context_digest=context_digest,
            )
        if not isinstance(interpretation, TaskInterpretation):
            raise TypeError("Taiji task interpretation did not return its contract")
        return interpretation

    def admit_semantic_provider_evidence(
        self,
        prompt: str,
        proposal: Any,
    ) -> dict[str, Any]:
        """Admit provider semantic evidence; never let the provider execute."""

        from taiji import SemanticEvidenceProposal

        prompt, frame = self._task_frame(prompt)
        if isinstance(proposal, Mapping):
            proposal = SemanticEvidenceProposal.from_payload(proposal)
        if not isinstance(proposal, SemanticEvidenceProposal):
            raise TypeError("semantic provider evidence must use its Taiji contract")
        with self._lock:
            interpretation, decomposition = (
                self.model.architecture.admit_semantic_provider_evidence(frame, proposal)
            )
        return {
            "format": "taiji-semantic-provider-admission-v1",
            "provider_evidence": proposal.to_payload(),
            "interpretation": interpretation.to_payload(),
            "goal": interpretation.to_goal().to_payload(),
            "decomposition": (
                None if decomposition is None else decomposition.to_payload()
            ),
            "status": interpretation.status,
            "execution": {
                "status": "not_planned",
                "action_intent": None,
                "tool_call": None,
                "side_effects": False,
                "next": "taiji_workbench_grounding" if decomposition else "taiji_planner",
            },
        }

    def decompose_task(
        self,
        semantic_steps: Sequence[Mapping[str, Any]],
        *,
        confidence: float | None = None,
        ambiguity: float | None = None,
        status: str = "resolved",
        provenance: str = "taiji.semantic",
    ) -> dict[str, Any]:
        """Admit bounded semantic steps without selecting or executing a tool."""

        from taiji import TaskDecomposition

        with self._lock:
            interpretation = self.model.architecture.last_task_interpretation
            if interpretation is None:
                raise RuntimeError("task decomposition requires task interpretation evidence")
            decomposition = TaskDecomposition.from_interpretation(
                interpretation,
                semantic_steps,
                confidence=confidence,
                ambiguity=ambiguity,
                status=status,
                provenance=provenance,
            )
            self.model.architecture.admit_task_decomposition(decomposition)
        return {
            "format": decomposition.to_payload()["format"],
            "decomposition": decomposition.to_payload(),
            "execution": {
                "status": "not_planned",
                "action_intent": None,
                "tool_call": None,
                "side_effects": False,
                "next": "taiji_workbench_grounding",
            },
        }

    def plan_task_sequence(
        self,
        *,
        snapshot_id: str,
        parameter_bindings: Sequence[Mapping[str, Mapping[str, Any]]],
        decomposition: Any | None = None,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> dict[str, Any]:
        """Ground semantic steps against live affordances without executing them.

        ``parameter_bindings`` is a backend/workbench projection, not language
        provider output. The decomposition itself cannot contain capability or
        tool identifiers; the Taiji planner resolves each step against the
        current capability snapshot and returns non-executing candidates.
        """

        from taiji import TASK_PLANNER_CONFIDENCE_FLOOR, TaskDecomposition

        with self._lock:
            architecture = self.model.architecture
            active = decomposition or architecture.last_task_decomposition
            if not isinstance(active, TaskDecomposition):
                raise RuntimeError("task sequence planning requires Taiji task decomposition")
            if architecture.last_task_decomposition != active:
                raise ValueError("task sequence decomposition is not the current Taiji evidence")
            if active.tick != architecture.tick:
                raise RuntimeError("task decomposition evidence is stale for sequence planning")
            environment = self._sync_workbench_root()
            if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
                raise ValueError("task sequence planning capability snapshot drifted")
            if isinstance(parameter_bindings, (str, bytes)) or not isinstance(
                parameter_bindings, Sequence
            ):
                raise TypeError("task sequence parameter bindings must be a sequence")
            if len(parameter_bindings) != len(active.steps):
                raise ValueError("task sequence bindings must match semantic step count")
            if active.status != "resolved" or active.confidence < TASK_PLANNER_CONFIDENCE_FLOOR:
                return {
                    "format": "taiji-task-sequence-planning-v1",
                    "decomposition": active.to_payload(),
                    "status": "needs_clarification",
                    "reason_code": "task_decomposition_low_confidence",
                    "steps": [],
                    "execution": {
                        "status": "not_executed",
                        "action_intent": None,
                        "tool_call": None,
                        "side_effects": False,
                    },
                }

            step_results: list[dict[str, Any]] = []
            for index, (step, bindings) in enumerate(
                zip(active.steps, parameter_bindings, strict=True)
            ):
                affordances = environment.capability_snapshot.to_taiji_affordances(bindings)
                architecture.set_world_affordances(affordances)
                planned = architecture.plan_task_from_current_state(
                    novelty=novelty,
                    resource_budget=resource_budget,
                )
                decision = planned.get("decision")
                step_results.append(
                    {
                        "index": index,
                        "step_id": step.step_id,
                        "semantic_evidence_digest": step.evidence_digest,
                        "grounding": [
                            self._taiji_workbench_affordance_payload(item)
                            for item in affordances
                        ],
                        "planner": {
                            "status": planned["status"],
                            "reason_code": planned["reason_code"],
                            "decision": (
                                None if decision is None else decision.to_payload()
                            ),
                        },
                    }
                )
                if planned["status"] != "planned":
                    break
            sequence_status = (
                "planned" if len(step_results) == len(active.steps) else "needs_clarification"
            )
            sequence_reason = (
                "taiji_sequence_grounded"
                if sequence_status == "planned"
                else str(step_results[-1]["planner"]["reason_code"])
            )
            decomposition_payload = active.to_payload()
        return {
            "format": "taiji-task-sequence-planning-v1",
            "decomposition": decomposition_payload,
            "status": sequence_status,
            "reason_code": sequence_reason,
            "steps": step_results,
            "execution": {
                "status": "not_executed",
                "action_intent": None,
                "tool_call": None,
                "side_effects": False,
            },
        }

    def plan_task(
        self,
        prompt: str,
        *,
        snapshot_id: str,
        parameter_bindings: Mapping[str, Mapping[str, Any]],
        history: Sequence[Sequence[str]] | None = None,
        constraints: Sequence[str] = (),
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> dict[str, Any]:
        """Run Taiji interpretation plus non-executing Workbench planning."""

        from taiji import TaskInterpretation

        prompt, frame = self._task_frame(prompt)
        context_digest = self._task_context_digest(history)
        with self._lock:
            architecture = self.model.architecture
            architecture.ingest_input(frame, learn=False)
            interpretation = TaskInterpretation.from_input(
                frame,
                goal_description=prompt,
                constraints=constraints,
                context_digest=context_digest,
                tick=architecture.tick,
            )
            architecture.admit_task_interpretation(interpretation)
            environment = self._sync_workbench_root()
            if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
                raise ValueError("Taiji task planning capability snapshot drifted")
            affordances = environment.capability_snapshot.to_taiji_affordances(
                parameter_bindings
            )
            architecture.set_world_affordances(affordances)
            planner_result = architecture.plan_task_from_current_state(
                novelty=novelty,
                resource_budget=resource_budget,
            )
        decision = planner_result.get("decision")
        return {
            "format": "taiji-task-planning-v1",
            "interpretation": interpretation.to_payload(),
            "goal": interpretation.to_goal().to_payload(),
            "affordances": [
                self._taiji_workbench_affordance_payload(item) for item in affordances
            ],
            "planner": {
                "status": planner_result["status"],
                "reason_code": planner_result["reason_code"],
                "decision": None
                if decision is None
                else decision.to_payload(),
            },
            "execution": {
                "status": "not_executed",
                "action_intent": None
                if decision is None
                else decision.action_intent.to_payload(),
                "tool_call": None,
                "side_effects": False,
            },
        }

    def _ground_natural_language_workbench_step(
        self,
        environment: Any,
        step: Any,
    ) -> tuple[
        dict[str, tuple[dict[str, Any], ...]],
        str,
        dict[str, Any] | None,
        str,
    ]:
        """Ground one semantic step, deriving language IDs from live evidence.

        Semantic provider evidence may describe the desired operation and file,
        but it cannot carry the final programming-language binding.  The
        language resolver is a Workbench sensor owned by Seed; its current
        assessment is therefore the only source allowed to populate
        ``programming_language_id`` before an editor action is planned.
        """

        grounded = environment.capability_snapshot.ground_semantic_step(
            step.semantic_slots,
            allow_reversible_ui=True,
        )
        if not grounded:
            return {}, "semantic_grounding_unresolved", None, ""
        if len(grounded) > 1:
            return {}, "semantic_grounding_ambiguous", None, ""

        normalized = {
            str(capability_id): tuple(dict(parameters) for parameters in bindings)
            for capability_id, bindings in grounded.items()
        }
        language_evidence: dict[str, Any] | None = None
        language_bindings = normalized.get("editor.set_language")
        if language_bindings is not None:
            rebound: list[dict[str, Any]] = []
            for binding in language_bindings:
                path = str(binding.get("path", "")).strip()
                if not path:
                    return {}, "semantic_grounding_unresolved", None, ""
                assessment = environment.resolve_programming_language_evidence({"path": path})
                selection_state = str(assessment.get("selection_state", "unknown"))
                if selection_state == "user_override" and not bool(
                    binding.get("user_override", False)
                ):
                    return {}, "user_override_has_priority", assessment, "language_evidence"
                if selection_state in {"ambiguous", "unknown"}:
                    return {}, "language_evidence_ambiguous", assessment, "language_evidence"
                language_id = str(assessment.get("programming_language_id", "")).strip()
                if not language_id:
                    return {}, "language_evidence_unresolved", assessment, "language_evidence"
                rebound.append(
                    {
                        **binding,
                        "programming_language_id": language_id,
                    }
                )
                language_evidence = assessment
            normalized["editor.set_language"] = tuple(rebound)
            return normalized, "", language_evidence, "language_evidence"

        patch_bindings = normalized.get("workspace.apply_patch")
        if patch_bindings is None:
            return normalized, "", None, ""
        if len(patch_bindings) != 1:
            return {}, "semantic_grounding_ambiguous", None, ""
        edit = step.semantic_slots.get("edit")
        if not isinstance(edit, Mapping):
            return {}, "edit_evidence_unresolved", None, ""
        if str(edit.get("kind", "")).strip() != "replace_text":
            return {}, "edit_kind_unsupported", None, ""
        find_text = edit.get("find")
        replacement_text = edit.get("replace", "")
        if not isinstance(find_text, str) or not find_text:
            return {}, "edit_target_unresolved", None, ""
        if not isinstance(replacement_text, str):
            return {}, "edit_replacement_invalid", None, ""
        path = str(patch_bindings[0].get("path", "")).strip()
        if not path:
            return {}, "semantic_grounding_unresolved", None, ""
        current = environment.read_workspace_evidence({"path": path})
        if bool(current.get("truncated", False)):
            return {}, "edit_source_truncated", current, "patch_evidence"
        content = current.get("content")
        if not isinstance(content, str):
            return {}, "edit_source_not_text", current, "patch_evidence"
        match_count = content.count(find_text)
        patch_evidence = {
            "path": path,
            "before_digest": str(current["digest"]),
            "match_count": match_count,
            "edit_kind": "replace_text",
        }
        if match_count == 0:
            return {}, "edit_target_not_found", patch_evidence, "patch_evidence"
        if match_count != 1:
            return {}, "edit_target_ambiguous", patch_evidence, "patch_evidence"
        start = content.index(find_text)
        updated = content[:start] + replacement_text + content[start + len(find_text) :]
        updated_raw = updated.encode("utf-8")
        expected_after_digest = hashlib.sha256(updated_raw).hexdigest()
        normalized["workspace.apply_patch"] = (
            {
                **patch_bindings[0],
                "before_digest": str(current["digest"]),
                "patch": {
                    "kind": "text_replace",
                    "operations": [
                        {
                            "start": start,
                            "end": start + len(find_text),
                            "text": replacement_text,
                        }
                    ],
                },
                "expected_after_digest": expected_after_digest,
            },
        )
        return (
            normalized,
            "",
            {
                **patch_evidence,
                "expected_after_digest": expected_after_digest,
            },
            "patch_evidence",
        )

    def execute_natural_language_workbench_task(
        self,
        prompt: str,
        semantic_evidence: Any,
        *,
        snapshot_id: str,
        parameter_bindings: Sequence[Mapping[str, Mapping[str, Any]]] | None = None,
        loop_id: str,
        max_steps: int = 1,
        max_budget_units: float = 1.0,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
        learn: bool = False,
    ) -> dict[str, Any]:
        """Run one bounded natural-language task without caller-supplied intent.

        The provider contributes only validated semantic evidence.  Taiji owns
        interpretation admission, live affordance grounding, executive intent
        creation, Workbench preflight, and execution.  The optional
        ``parameter_bindings`` argument is a legacy compatibility seam for the
        P2-8 canary; when omitted, the current Taiji-owned semantic contracts
        derive bindings from the decomposition's semantic slots.
        """

        from seed_platform.workbench import WorkbenchActionRequest
        from taiji import SemanticEvidenceProposal, TaskDecomposition

        if isinstance(semantic_evidence, Mapping):
            semantic_evidence = SemanticEvidenceProposal.from_payload(semantic_evidence)
        if not isinstance(semantic_evidence, SemanticEvidenceProposal):
            raise TypeError("natural-language Workbench execution requires semantic evidence")

        admission = self.admit_semantic_provider_evidence(prompt, semantic_evidence)
        base = {
            "format": "seed-natural-language-workbench-task-v1",
            "provider_evidence": admission["provider_evidence"],
            "interpretation": admission["interpretation"],
            "goal": admission["goal"],
            "decomposition": admission["decomposition"],
            "planning": None,
            "preflight": None,
            "execution": {
                "status": "not_executed",
                "action_intent": None,
                "side_effects": False,
            },
        }
        decomposition = self.model.architecture.last_task_decomposition
        if admission["status"] != "resolved" or not isinstance(decomposition, TaskDecomposition):
            base["status"] = "needs_clarification"
            base["reason_code"] = "semantic_evidence_not_resolved"
            return base

        # Provider evidence is first validated against the client frame above.
        # The planner, however, must only run with a current Taiji percept.  A
        # natural-language task therefore passes through Taiji perception
        # before its already-validated semantic content is rebound to the
        # resulting tick.  Rebinding changes only the frame lineage; it does
        # not allow provider evidence to add capabilities, tools, intents, or
        # execution policy.
        with self._lock:
            architecture = self.model.architecture
            _, input_frame = self._task_frame(prompt)
            architecture.ingest_input(input_frame, learn=False)
            _, grounded_frame = self._task_frame(prompt)
            semantic_evidence = SemanticEvidenceProposal.from_frame(
                grounded_frame,
                provider_id=semantic_evidence.provider_id,
                goal_description=semantic_evidence.goal_description,
                semantic_steps=semantic_evidence.semantic_steps,
                constraints=semantic_evidence.constraints,
                context_digest=semantic_evidence.context_digest,
                confidence=semantic_evidence.confidence,
                ambiguity=semantic_evidence.ambiguity,
                provenance=semantic_evidence.provenance,
                tick=grounded_frame.timestamp,
            )
            interpretation, decomposition = architecture.admit_semantic_provider_evidence(
                grounded_frame,
                semantic_evidence,
            )
            admission = {
                "provider_evidence": semantic_evidence.to_payload(),
                "interpretation": interpretation.to_payload(),
                "goal": interpretation.to_goal().to_payload(),
                "decomposition": decomposition.to_payload(),
                "status": interpretation.status,
            }
            base.update(
                provider_evidence=admission["provider_evidence"],
                interpretation=admission["interpretation"],
                goal=admission["goal"],
                decomposition=admission["decomposition"],
            )

        if parameter_bindings is not None:
            if isinstance(parameter_bindings, (str, bytes)) or not isinstance(
                parameter_bindings, Sequence
            ):
                raise TypeError("natural-language Workbench parameter bindings must be a sequence")
            if len(parameter_bindings) != len(decomposition.steps):
                raise ValueError(
                    "natural-language Workbench bindings must match semantic step count"
                )

        with self._lock:
            environment = self._sync_workbench_root()
            if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
                raise ValueError("natural-language Workbench capability snapshot drifted")
            architecture = self.model.architecture
            intents: list[Any] = []
            planning_steps: list[dict[str, Any]] = []
            steps = (
                zip(decomposition.steps, parameter_bindings, strict=True)
                if parameter_bindings is not None
                else ((step, None) for step in decomposition.steps)
            )
            for index, (step, bindings) in enumerate(steps):
                live_evidence = None
                live_evidence_key = ""
                if bindings is None:
                    (
                        grounded_bindings,
                        grounding_error,
                        live_evidence,
                        live_evidence_key,
                    ) = (
                        self._ground_natural_language_workbench_step(environment, step)
                    )
                    if grounding_error:
                        planning_steps.append(
                            {
                                "index": index,
                                "step_id": step.step_id,
                                "semantic_evidence_digest": step.evidence_digest,
                                "grounding": [],
                                "grounding_source": "taiji-semantic-contract",
                                "planner": {
                                    "status": "needs_clarification",
                                    "reason_code": grounding_error,
                                    "decision": None,
                                },
                                **(
                                    {live_evidence_key: live_evidence}
                                    if live_evidence is not None and live_evidence_key
                                    else {}
                                ),
                            }
                        )
                        base["status"] = "needs_clarification"
                        base["reason_code"] = grounding_error
                        base["planning"] = {
                            "status": "needs_clarification",
                            "steps": planning_steps,
                        }
                        return base
                    bindings = grounded_bindings
                    grounding_source = "taiji-semantic-contract"
                    if live_evidence is not None and live_evidence_key == "language_evidence":
                        grounding_source = (
                            "taiji-semantic-contract+workbench-language-evidence"
                        )
                    elif live_evidence is not None and live_evidence_key == "patch_evidence":
                        grounding_source = "taiji-semantic-contract+workbench-patch-evidence"
                else:
                    if not isinstance(bindings, Mapping):
                        raise TypeError(
                            "each natural-language Workbench binding must be a mapping"
                        )
                    grounding_source = "legacy-workbench-evidence"
                affordances = environment.capability_snapshot.to_taiji_affordances(
                    bindings,
                    allow_reversible_ui=True,
                )
                if not affordances:
                    raise ValueError("natural-language Workbench step has no live affordance")
                architecture.set_world_affordances(affordances)
                planned = architecture.plan_task_from_current_state(
                    novelty=novelty,
                    resource_budget=resource_budget,
                )
                decision = planned.get("decision")
                planning_steps.append(
                    {
                        "index": index,
                        "step_id": step.step_id,
                        "semantic_evidence_digest": step.evidence_digest,
                        "grounding_source": grounding_source,
                        "grounding": [
                            self._taiji_workbench_affordance_payload(item)
                            for item in affordances
                        ],
                        "planner": {
                            "status": planned["status"],
                            "reason_code": planned["reason_code"],
                            "decision": (
                                None if decision is None else decision.to_payload()
                            ),
                        },
                        **(
                            {live_evidence_key: live_evidence}
                            if live_evidence is not None and live_evidence_key
                            else {}
                        ),
                    }
                )
                if planned["status"] != "planned" or decision is None:
                    base["status"] = "needs_clarification"
                    base["reason_code"] = str(planned["reason_code"])
                    base["planning"] = {
                        "status": "needs_clarification",
                        "steps": planning_steps,
                    }
                    return base
                intents.append(decision.action_intent)

            requests = tuple(
                WorkbenchActionRequest.from_action_intent(
                    intent,
                    snapshot_id=environment.capability_snapshot.snapshot_id,
                    mcp_registry_snapshot_id=(
                        environment.mcp_registry.snapshot_id
                        if str(intent.kind).startswith("mcp.")
                        else ""
                    ),
                    capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
                )
                for intent in intents
            )
            preflight = self.preflight_workbench_loop(
                requests,
                loop_id=loop_id,
                max_steps=max_steps,
                max_budget_units=max_budget_units,
            )
            base["planning"] = {
                "status": "planned",
                "steps": planning_steps,
                "action_intents": [intent.to_payload() for intent in intents],
            }
            base["preflight"] = preflight
            if not preflight.get("accepted"):
                base["status"] = "rejected"
                base["reason_code"] = str(preflight.get("error_code", "preflight_rejected"))
                return base
            execution = self.execute_preflighted_workbench_loop(
                intents,
                requests,
                loop_id=loop_id,
                preflight_id=str(preflight["preflight_id"]),
                max_steps=max_steps,
                max_budget_units=max_budget_units,
                learn=learn,
            )
            side_effects = any(
                bool(step.get("success"))
                and (
                    (descriptor := environment.capability_snapshot.get(
                        str(step.get("capability_id", ""))
                    ))
                    is not None
                    and descriptor.risk != "read_only"
                )
                for step in execution.get("steps", ())
            )
            base["execution"] = {
                **execution,
                "side_effects": side_effects,
            }
            base["status"] = str(execution.get("status", "rejected"))
            base["reason_code"] = str(execution.get("error_code", ""))
            return base

    def plan_language_selection(
        self,
        *,
        snapshot_id: str,
        path: str,
        lsp_language_id: str | None = None,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> dict[str, Any]:
        """Plan editor language selection from evidence without changing the editor."""

        from taiji import TASK_PLANNER_CONFIDENCE_FLOOR

        with self._lock:
            architecture = self.model.architecture
            interpretation = architecture.last_task_interpretation
            if interpretation is None:
                raise RuntimeError("language planning requires Taiji task interpretation evidence")
            if interpretation.tick != architecture.tick:
                raise RuntimeError("task interpretation evidence is stale for language planning")
            environment = self._sync_workbench_root()
            if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
                raise ValueError("language planning capability snapshot drifted")
            assessment = environment.resolve_programming_language_evidence(
                {
                    "path": path,
                    "lsp_language_id": lsp_language_id,
                }
            )
            selection_state = str(assessment.get("selection_state", "unknown"))
            if selection_state in {"ambiguous", "unknown"}:
                planner = {
                    "status": "needs_clarification",
                    "reason_code": "language_evidence_ambiguous",
                    "decision": None,
                }
            elif interpretation.status != "resolved" or (
                interpretation.confidence < TASK_PLANNER_CONFIDENCE_FLOOR
            ):
                planner = {
                    "status": "needs_clarification",
                    "reason_code": "task_interpretation_low_confidence",
                    "decision": None,
                }
            elif selection_state == "user_override":
                planner = {
                    "status": "already_selected",
                    "reason_code": "user_override_has_priority",
                    "decision": None,
                }
            else:
                language_confidence = min(
                    interpretation.confidence,
                    float(assessment.get("confidence", 0.0)),
                )
                affordances = environment.capability_snapshot.to_taiji_affordances(
                    {
                        "editor.set_language": {
                            "path": str(path),
                            "programming_language_id": str(
                                assessment["programming_language_id"]
                            ),
                            "user_override": False,
                        }
                    },
                    allow_reversible_ui=True,
                )
                affordances = tuple(
                    replace(item, confidence=min(item.confidence, language_confidence))
                    for item in affordances
                )
                architecture.set_world_affordances(affordances)
                planned = architecture.plan_task_from_current_state(
                    novelty=novelty,
                    resource_budget=resource_budget,
                )
                planner = {
                    "status": planned["status"],
                    "reason_code": planned["reason_code"],
                    "decision": (
                        None
                        if planned["decision"] is None
                        else planned["decision"].to_payload()
                    ),
                }
            live_affordances = architecture.cognitive_snapshot().world.affordances
        decision_payload = planner["decision"]
        return {
            "format": "taiji-language-planning-v1",
            "interpretation": interpretation.to_payload(),
            "assessment": assessment,
            "affordances": [
                self._taiji_workbench_affordance_payload(item) for item in live_affordances
            ],
            "planner": planner,
            "execution": {
                "status": "not_executed",
                "action_intent": (
                    None
                    if decision_payload is None
                    else decision_payload.get("selected", {}).get("action_intent")
                ),
                "tool_call": None,
                "side_effects": False,
            },
        }

    def _observe_provider_health(self, expression: Any, emission: Any) -> None:
        """把一次真实发射折叠进健康记录，必要时自动回退（仅外部 provider 生效）。

        探针绝不允许让对话失败：任何异常都被吞掉并保留当前表层，
        因为一次探针误报的代价不该是一次用户可见的失败。
        未触发的名义探针也会把健康计数实时叠加进状态，让 status API
        反映真实健康负载；只有回退提交后才重挂表层并刷新 provider 状态。
        """
        if self._provider_config is None or self._provider_status.get("mode") in {
            "native",
            "structured",
        }:
            return
        try:
            from seed.language_provider import observe_language_provider

            result = observe_language_provider(
                self.model.architecture,
                self._provider_config,
                expression=expression,
                emission=emission,
                current_runtime=self._provider_runtime,
            )
        except Exception:  # 探针不可让对话失败
            logger.exception("language provider health probe failed; keeping current surface")
            return
        if not result.committed:
            # 名义探针：健康计数随真实发射增长，必须立刻可观测，但表层与队列不变。
            if result.health is not None:
                self._provider_status = _overlay_health(self._provider_status, result.health)
            return
        from taiji import LanguageOrgan, NativeReadableTextLanguageOrgan

        self._provider_status = result.status.to_dict()
        self._provider_runtime = result.runtime
        self._sync_provider_config()
        candidate = self.model.architecture.language_organ
        self._chat_organ = (
            candidate
            if result.status.chat_enabled and isinstance(candidate, LanguageOrgan)
            else NativeReadableTextLanguageOrgan()
        )
        logger.warning(
            "language provider health watchdog rolled back to %s (%s)",
            result.status.artifact_id,
            result.status.reason_code,
        )

    def _sync_provider_config(self) -> None:
        """把选择配置重锚到当前已挂载的 provider artifact。

        回退提交后，原先被降级的版本配置已经失联：单一事实来源是运行中
        adapter 上挂载的 artifact。原生/结构化回退没有外部 artifact，因此把
        配置清空，防止残留的降级配置被下一次观察或状态上报误用。
        """

        from seed.language_provider import project_language_provider_config

        artifact = self.model.architecture.language_provider_artifact
        if artifact is None:
            self._provider_config = None
        elif self._provider_config is not None:
            self._provider_config = project_language_provider_config(
                self._provider_config, artifact
            )

    def save(self, path: Path | str | None = None) -> Path:
        """落盘当前状态（默认写回来源检查点；原子写，崩溃不产生半写文件）。"""
        from seed.persistence import atomic_save, attach_metadata

        target = Path(path or self.checkpoint_path or DEFAULT_CHECKPOINT)
        with self._lock:
            workbench = self._sync_workbench_root()
            envelope = attach_metadata(
                self.model.checkpoint(),
                tick=self.model.tick,
                extra={
                    "trainer": "api_seed_runtime",
                    "workbench": {
                        "snapshot": workbench.capability_snapshot.to_payload(),
                        "mcp_registry": workbench.mcp_registry.to_payload(),
                        "audit": self._workbench_audit.to_payload(),
                        "language_state": workbench.language_state_checkpoint(),
                        "transaction_state": workbench.transaction_state_checkpoint(),
                        "loop_state": dict(self._workbench_loop_state),
                        "artifact_consumption_audit": self._last_artifact_consumption_audit,
                    },
                },
            )
            atomic_save(envelope, target)
        logger.info("Seed runtime saved to %s", target)
        return target

    def status(self) -> dict[str, Any]:
        workbench = self._sync_workbench_root()
        return {
            "runtime_type": self.RUNTIME_TYPE,
            "name": self.name,
            "tick": int(self.model.tick),
            "parameters": int(self.model.parameter_count()),
            "language_provider": dict(self._provider_status),
            "workbench": workbench.status(),
            "homeostasis": self.homeostasis_status(),
            "structural_maintenance": self.structural_maintenance_status(),
            "artifact_consumption": self.artifact_consumption_status(),
        }

    def artifact_consumption_status(self) -> dict[str, Any]:
        """Return the latest read-only external artifact consumption audit."""

        from taiji import ARTIFACT_CONSUMPTION_AUDIT_FORMAT

        with self._lock:
            policy = self.model.architecture.artifact_consumption_policy.to_payload()
            audit = (
                None
                if self._last_artifact_consumption_audit is None
                else dict(self._last_artifact_consumption_audit)
            )
        return {
            "policy": policy,
            "last_audit": audit,
            "audit_format": ARTIFACT_CONSUMPTION_AUDIT_FORMAT,
        }

    def structural_maintenance_status(self) -> dict[str, Any]:
        """Return a pure read-only projection of Taiji maintenance state.

        This method intentionally exposes the last persisted retention audit,
        if any; it never schedules maintenance or treats pressure as a growth
        decision.  The empty state is explicit so clients do not infer an
        audit from missing keys.
        """

        from taiji import STRUCTURAL_MAINTENANCE_STATUS_FORMAT

        with self._lock:
            architecture = self.model.architecture
            retention = architecture.structural_lineage_retention_result
            policy = architecture.structural_lineage_retention_policy
            migration = architecture.structural_lineage_retention_policy_migration
            tick = int(architecture.structural_runtime_tick)
            retention_payload = None if retention is None else retention.to_payload()
            policy_payload = None if policy is None else policy.to_payload()
            migration_payload = None if migration is None else migration.to_payload()
        return {
            "format": STRUCTURAL_MAINTENANCE_STATUS_FORMAT,
            "status": "audit_available" if retention is not None else "no_audit",
            "structural_runtime_tick": tick,
            "has_retention_audit": retention is not None,
            "last_retention_audit": retention_payload,
            "last_retention_policy": policy_payload,
            "last_retention_policy_migration": migration_payload,
            "retention_pressure": False if retention is None else retention.retention_pressure,
        }

    def homeostasis_status(self) -> dict[str, Any]:
        """Report the live internal drive state measured by Taiji.

        Only facts the architecture actually produced are emitted; when no
        controller is attached the payload stays empty rather than guessing.
        """

        architecture = self.model.architecture
        if not architecture.homeostatic_controller_attached:
            return {"attached": False, "needs": {}, "drives": {}, "mode": "", "tick": 0}
        with self._lock:
            state = architecture.homeostatic_state()
            drive = architecture.homeostatic_drive()
            mode = architecture.homeostatic_mode()
        return {
            "attached": True,
            "tick": int(state.tick),
            "mode": mode,
            # Native units are 0..1; clients scale for display.
            "needs": {
                "curiosity": float(state.curiosity),
                "fatigue": float(state.fatigue),
                "stress": float(state.stress),
            },
            "drives": {
                "exploration": float(drive.exploration),
                "replay": float(drive.replay),
                "rest": float(drive.rest),
                "play": float(drive.play),
            },
        }

    def _sync_workbench_root(self) -> Any:
        """Keep the native environment aligned with the active workspace setting."""

        from seed_platform.workbench import default_workspace_root

        current_root = default_workspace_root()
        if current_root != self._workbench_environment.root:
            from seed_platform.workbench import WorkbenchEnvironment

            self._workbench_environment = WorkbenchEnvironment(
                current_root,
                snapshot=self._workbench_environment.capability_snapshot,
                mcp_registry=self._workbench_environment.mcp_registry,
            )
        return self._workbench_environment

    @property
    def workbench_environment(self) -> Any:
        """Return the Seed-owned workbench execution environment."""

        return self._sync_workbench_root()

    @property
    def workbench_audit(self) -> Any:
        """Return the shared workbench event stream for UI/audit observers."""

        return self._workbench_audit

    def _restore_workbench_metadata(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        from seed_platform.mcp_registry import McpToolRegistry
        from seed_platform.workbench import (
            CapabilitySnapshot,
            WorkbenchAuditLog,
            WorkbenchEnvironment,
        )

        snapshot_payload = payload.get("snapshot")
        registry_payload = payload.get("mcp_registry")
        registry = self._workbench_environment.mcp_registry
        if isinstance(registry_payload, Mapping):
            registry = McpToolRegistry.from_payload(registry_payload)
        if isinstance(snapshot_payload, Mapping):
            snapshot = CapabilitySnapshot.from_payload(snapshot_payload)
            current = self._workbench_environment.capability_snapshot
            if (
                snapshot.snapshot_id != current.snapshot_id
                and snapshot.revision >= current.revision
            ):
                raise ValueError("workbench capability snapshot drifted during restore")
            if snapshot.snapshot_id == current.snapshot_id:
                self._workbench_environment = WorkbenchEnvironment(
                    self._workbench_environment.root,
                    snapshot=snapshot,
                    mcp_registry=registry,
                )
            else:
                logger.info(
                    "migrating workbench capability snapshot revision %s to %s",
                    snapshot.revision,
                    current.revision,
                )
        audit_payload = payload.get("audit")
        if isinstance(audit_payload, Mapping):
            self._workbench_audit = WorkbenchAuditLog.from_payload(audit_payload)
        language_payload = payload.get("language_state")
        if isinstance(language_payload, Mapping):
            self._workbench_environment.restore_language_state(language_payload)
        transaction_payload = payload.get("transaction_state")
        if isinstance(transaction_payload, Mapping):
            self._workbench_environment.restore_transaction_state(transaction_payload)
        artifact_consumption_payload = payload.get("artifact_consumption_audit")
        if artifact_consumption_payload is not None:
            from taiji import ArtifactConsumptionAudit

            if not isinstance(artifact_consumption_payload, Mapping):
                raise ValueError("artifact consumption audit checkpoint must be a mapping")
            self._last_artifact_consumption_audit = ArtifactConsumptionAudit.from_payload(
                artifact_consumption_payload
            ).to_payload()
        loop_payload = payload.get("loop_state")
        if isinstance(loop_payload, Mapping):
            committed = loop_payload.get("committed_request_ids", ())
            if isinstance(committed, (str, bytes)) or not isinstance(committed, Sequence):
                raise ValueError("workbench loop checkpoint request ids are invalid")
            restored_loop_state: dict[str, Any] = {
                "format": str(loop_payload.get("format", "")),
                "version": int(loop_payload.get("version", 0)),
                "loop_id": str(loop_payload.get("loop_id", "")),
                "preflight_id": str(loop_payload.get("preflight_id", "")),
                "committed_request_ids": [str(item) for item in committed],
                "status": str(loop_payload.get("status", "")),
            }
            successor_graph = loop_payload.get("successor_graph")
            if isinstance(successor_graph, Mapping):
                restored_loop_state["successor_graph"] = dict(successor_graph)
            recovery_portfolio = loop_payload.get("recovery_portfolio")
            if isinstance(recovery_portfolio, Mapping):
                restored_loop_state["recovery_portfolio"] = dict(recovery_portfolio)
            self._workbench_loop_state = restored_loop_state

    def preview_workbench_intent(
        self,
        intent: Any,
        *,
        snapshot_id: str,
        mcp_registry_snapshot_id: str = "",
    ) -> dict[str, Any]:
        """Validate an action and issue approval without executing its side effect."""

        from seed_platform.workbench import (
            WorkbenchActionRequest,
        )
        from taiji import ActionIntent

        if not isinstance(intent, ActionIntent):
            raise TypeError("workbench preview requires an ActionIntent")
        environment = self._sync_workbench_root()
        request = WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=snapshot_id,
            mcp_registry_snapshot_id=(
                mcp_registry_snapshot_id
                or (
                    environment.mcp_registry.snapshot_id
                    if str(intent.kind).startswith("mcp.")
                    else ""
                )
            ),
        )
        tick = int(self.model.tick)
        self._workbench_audit.append(
            "planned",
            request.request_id,
            tick=tick,
            payload={"request": request.to_payload(), "preview": True},
        )
        policy = environment.policy_for(request)
        self._workbench_audit.append(
            "policy",
            request.request_id,
            tick=tick,
            payload={"policy": policy.to_payload(), "preview": True},
        )
        result: dict[str, Any] = {
            "request": request.to_payload(),
            "policy": policy.to_payload(),
            "preview": None,
            "approval": None,
            "events": [event.to_payload() for event in self._workbench_audit.events],
        }
        if policy.reason_code == "capability_requires_approval":
            approval = environment.issue_approval(request)
            result["preview"] = approval["preview"]
            result["approval"] = {key: value for key, value in approval.items() if key != "preview"}
        return result

    def preflight_workbench_loop(
        self,
        requests: Sequence[Any],
        *,
        loop_id: str,
        max_steps: int = 8,
        max_budget_units: float = 32.0,
        on_failure: str = "stop",
        checkpoint_boundary: str = "after_each_step",
    ) -> dict[str, Any]:
        """Preflight a bounded request sequence without executing side effects."""

        environment = self._sync_workbench_root()
        result = environment.preflight_loop(
            requests,
            loop_id=loop_id,
            max_steps=max_steps,
            max_budget_units=max_budget_units,
            on_failure=on_failure,
            checkpoint_boundary=checkpoint_boundary,
        )
        result["runtime"] = {
            "tick": int(self.model.tick),
            "checkpoint_path": (
                None if self.checkpoint_path is None else str(self.checkpoint_path)
            ),
            "checkpoint_boundary": result.get("checkpoint", {}).get("boundary"),
        }
        return result

    def execute_preflighted_workbench_loop(
        self,
        intents: Sequence[Any],
        requests: Sequence[Any],
        *,
        loop_id: str,
        preflight_id: str,
        max_steps: int = 8,
        max_budget_units: float = 32.0,
        on_failure: str = "stop",
        checkpoint_boundary: str = "after_each_step",
        learn: bool = False,
    ) -> dict[str, Any]:
        """Execute only an unchanged preflight, checkpointing every attempted step."""

        from seed_platform.workbench import WorkbenchActionRequest
        from taiji import ActionIntent

        environment = self._sync_workbench_root()
        if isinstance(intents, (str, bytes)) or not isinstance(intents, Sequence):
            raise TypeError("loop intents must be a sequence")
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise TypeError("loop requests must be a sequence")
        if len(intents) != len(requests):
            raise ValueError("loop intents and requests must have the same length")
        typed_requests = tuple(requests)
        for intent, request in zip(intents, typed_requests, strict=True):
            if not isinstance(intent, ActionIntent):
                raise TypeError("loop intents must contain ActionIntent values")
            if not isinstance(request, WorkbenchActionRequest):
                raise TypeError("loop requests must contain WorkbenchActionRequest values")
            if (
                request.intent_id != intent.intent_id
                or request.capability_id != intent.kind
                or dict(request.parameters) != dict(intent.parameters)
            ):
                raise ValueError("loop intent and workbench request binding drifted")

        result: dict[str, Any] = {
            "format": "seed-workbench-loop-v1",
            "version": 1,
            "loop_id": str(loop_id),
            "preflight_id": str(preflight_id),
            "preflight": None,
            "steps": [],
            "completed_prefix": 0,
            "status": "rejected",
        }
        committed = {
            str(item) for item in self._workbench_loop_state.get("committed_request_ids", ())
        }
        replayed = [
            request.request_id for request in typed_requests if request.request_id in committed
        ]
        if replayed:
            result["error_code"] = "loop_request_already_committed"
            result["error"] = "loop request was already committed in a checkpoint"
            result["replayed_request_ids"] = replayed
            return result

        preflight = environment.preflight_loop(
            typed_requests,
            loop_id=loop_id,
            max_steps=max_steps,
            max_budget_units=max_budget_units,
            on_failure=on_failure,
            checkpoint_boundary=checkpoint_boundary,
        )
        result["preflight"] = preflight
        if not preflight.get("accepted"):
            result["error_code"] = str(preflight.get("error_code", "preflight_rejected"))
            result["error"] = str(preflight.get("error", "loop preflight was rejected"))
            return result
        if preflight.get("preflight_id") != preflight_id:
            result["error_code"] = "preflight_identity_mismatch"
            result["error"] = "provided preflight_id does not match current requests"
            return result

        self._workbench_loop_state = {
            **self._workbench_loop_state,
            "format": "seed-workbench-loop-v1",
            "version": 1,
            "loop_id": str(loop_id),
            "preflight_id": str(preflight_id),
            "committed_request_ids": [
                str(item) for item in self._workbench_loop_state.get("committed_request_ids", ())
            ][-MAX_WORKBENCH_LOOP_COMMITTED_REQUESTS:],
            "status": "running",
        }
        checkpoint_path: Path | None = None
        for index, (intent, request) in enumerate(zip(intents, typed_requests, strict=True)):
            step: dict[str, Any] = {
                "index": index,
                "request_id": request.request_id,
                "intent_id": request.intent_id,
                "capability_id": request.capability_id,
            }
            try:
                execution = self.execute_workbench_intent(
                    intent,
                    snapshot_id=request.snapshot_id,
                    approval_token=request.approval_token,
                    mcp_registry_snapshot_id=request.mcp_registry_snapshot_id,
                    learn=learn,
                )
                outcome = dict(execution.get("outcome") or {})
                step.update(
                    {
                        "status": str(outcome.get("status", "error")),
                        "success": bool(outcome.get("success", False)),
                        "policy": execution.get("policy"),
                        "outcome": outcome,
                        "tool_call": execution.get("tool_call"),
                    }
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                step.update(
                    {
                        "status": "error",
                        "success": False,
                        "error_code": "loop_step_error",
                        "error": str(exc),
                    }
                )

            committed_ids = [
                *self._workbench_loop_state.get("committed_request_ids", ()),
                request.request_id,
            ]
            self._workbench_loop_state = {
                **self._workbench_loop_state,
                "committed_request_ids": committed_ids[-MAX_WORKBENCH_LOOP_COMMITTED_REQUESTS:],
                "status": "running" if step.get("success") else "failed",
            }
            try:
                checkpoint_path = self.save()
            except (OSError, RuntimeError, ValueError) as exc:
                self._workbench_loop_state = {
                    **self._workbench_loop_state,
                    "status": "checkpoint_failed",
                }
                step["checkpoint"] = {
                    "committed": False,
                    "error_code": "checkpoint_failed",
                    "error": str(exc),
                }
                result["steps"].append(step)
                result["status"] = "checkpoint_failed"
                result["error_code"] = "checkpoint_failed"
                result["error"] = str(exc)
                result["stopped_at"] = index
                return result

            step["checkpoint"] = {
                "committed": True,
                "path": str(checkpoint_path),
            }
            result["steps"].append(step)
            if not step.get("success") or step.get("status") != "success":
                result["status"] = "failed"
                result["stopped_at"] = index
                result["completed_prefix"] = sum(
                    1 for item in result["steps"] if item.get("success")
                )
                return result
            result["completed_prefix"] = index + 1

        self._workbench_loop_state = {
            **self._workbench_loop_state,
            "status": "completed",
        }
        # Persist the terminal loop state as a final checkpoint with no new side effect.
        try:
            checkpoint_path = self.save()
        except (OSError, RuntimeError, ValueError) as exc:
            self._workbench_loop_state = {
                **self._workbench_loop_state,
                "status": "checkpoint_failed",
            }
            result["status"] = "checkpoint_failed"
            result["error_code"] = "checkpoint_failed"
            result["error"] = str(exc)
            return result
        result["status"] = "completed"
        result["checkpoint"] = {"committed": True, "path": str(checkpoint_path)}
        return result

    def _select_taiji_workbench_candidate(
        self,
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> Any:
        """Select one candidate from Taiji's current executive state."""

        architecture = self.model.architecture
        candidates = architecture.synthesize_executive_candidates()
        if not candidates:
            raise RuntimeError("Taiji task admission requires a current executive candidate")
        decision = architecture.select_executive(
            candidates,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        return decision

    @staticmethod
    def _taiji_workbench_decision_payload(decision: Any) -> dict[str, Any]:
        """Project an executive decision to JSON without leaking checkpoint tensors."""

        return {
            "selected_candidate_id": decision.selected.candidate_id,
            "scores": {
                str(candidate_id): float(score) for candidate_id, score in decision.scores.items()
            },
            "tick": int(decision.context.tick),
            "goal_id": decision.context.goal_id,
        }

    def admit_taiji_workbench_task(
        self,
        *,
        snapshot_id: str,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> dict[str, Any]:
        """Run the Taiji-owned read-only task admission Gate without execution."""

        decision = self._select_taiji_workbench_candidate(
            novelty=novelty,
            resource_budget=resource_budget,
        )
        environment = self._sync_workbench_root()
        world = self.model.architecture.cognitive_snapshot().world
        admission = environment.admit_taiji_candidate(
            decision.selected,
            snapshot_id=snapshot_id,
            current_tick=world.tick,
            current_affordance_ids=tuple(item.affordance_id for item in world.affordances),
            current_affordances=world.affordances,
        )
        return {
            "admission": admission.to_payload(),
            "decision": self._taiji_workbench_decision_payload(decision),
            "execution": None,
        }

    def project_workbench_affordances(
        self,
        *,
        snapshot_id: str,
        parameter_bindings: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Project explicit Workbench evidence into Taiji's current world state."""

        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("Taiji capability projection snapshot drifted")
        world = self.model.architecture.cognitive_snapshot().world
        from seed_platform.workbench import WORKBENCH_TAIJI_EVIDENCE_KIND

        latest_evidence = next(
            (item for item in reversed(world.events) if item.kind == WORKBENCH_TAIJI_EVIDENCE_KIND),
            None,
        )
        if latest_evidence is not None:
            if latest_evidence.tick != world.tick:
                raise ValueError(
                    "latest WorkBench evidence is stale; acquire fresh workspace evidence"
                )
            raise ValueError("latest WorkBench evidence requires /api/workbench/taiji/reproject")
        affordances = environment.capability_snapshot.to_taiji_affordances(parameter_bindings)
        world = self.model.architecture.set_world_affordances(affordances)
        return {
            "snapshot_id": environment.capability_snapshot.snapshot_id,
            "revision": environment.capability_snapshot.revision,
            "tick": int(world.tick),
            "affordances": [self._taiji_workbench_affordance_payload(item) for item in affordances],
        }

    def reproject_workbench_from_latest_evidence(
        self,
        *,
        snapshot_id: str,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        """Re-project the latest current-tick workspace evidence only."""

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_EVIDENCE_KIND,
            WorkbenchTaijiEvidence,
        )

        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("Taiji capability re-projection snapshot drifted")
        world = self.model.architecture.cognitive_snapshot().world
        event = next(
            (item for item in reversed(world.events) if item.kind == WORKBENCH_TAIJI_EVIDENCE_KIND),
            None,
        )
        if event is None:
            raise ValueError("Taiji re-projection requires current WorkBench evidence")
        if event.tick != world.tick:
            raise ValueError("latest WorkBench evidence is stale; acquire fresh workspace evidence")
        evidence = WorkbenchTaijiEvidence.from_taiji_event(event)
        affordances = evidence.to_taiji_affordances(environment.capability_snapshot)
        if not affordances and not allow_empty:
            raise ValueError("failed WorkBench evidence cannot produce a Taiji affordance")
        world = self.model.architecture.set_world_affordances(affordances)
        return {
            "snapshot_id": environment.capability_snapshot.snapshot_id,
            "revision": environment.capability_snapshot.revision,
            "tick": int(world.tick),
            "evidence": evidence.to_payload(),
            "affordances": [self._taiji_workbench_affordance_payload(item) for item in affordances],
        }

    def project_workbench_outcome_for_internalization(
        self,
        *,
        snapshot_id: str,
        affordance_id: str,
        reward: float,
        reward_terms: Mapping[str, float],
        parent_checkpoint_id: str,
        owner_id: str = "taiji:workbench-outcome",
    ) -> Any:
        """Project one current Workbench outcome into a Taiji learning DTO.

        This is intentionally an evidence boundary, not an internalization
        operation.  The runtime verifies that the selected successor
        affordance was re-projected from the latest, snapshot-bound read-only
        Workbench evidence, then returns a typed Taiji-owned input.  It cannot
        write replay, fit a learner, or advance a lifecycle status.

        ``reward`` and ``reward_terms`` are supplied by the task evaluator so
        the executor never invents a learning objective from a capability
        name.  The Taiji converter remains responsible for validating their
        bounds and deciding whether the DTO becomes learnable material.
        """

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_EVIDENCE_KIND,
            WorkbenchTaijiEvidence,
        )
        from taiji import GroundedOutcomeEvidence, Outcome, content_digest

        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("Taiji internalization projection snapshot drifted")
        world = self.model.architecture.cognitive_snapshot().world
        event = next(
            (item for item in reversed(world.events) if item.kind == WORKBENCH_TAIJI_EVIDENCE_KIND),
            None,
        )
        if event is None:
            raise ValueError("Taiji internalization projection requires Workbench evidence")
        if event.tick != world.tick:
            raise ValueError("Taiji internalization projection requires current Workbench evidence")
        evidence = WorkbenchTaijiEvidence.from_taiji_event(event)
        if evidence.snapshot_id != environment.capability_snapshot.snapshot_id:
            raise ValueError("Workbench evidence capability snapshot is stale")
        if not evidence.success:
            raise ValueError("failed Workbench evidence cannot enter internalization")

        requested_affordance_id = str(affordance_id).strip()
        if not requested_affordance_id:
            raise ValueError("Taiji internalization projection requires an affordance_id")
        affordance = next(
            (item for item in world.affordances if item.affordance_id == requested_affordance_id),
            None,
        )
        if affordance is None:
            raise ValueError("Taiji internalization affordance is not current")
        grounded_affordance_ids = {
            item.affordance_id
            for item in evidence.to_taiji_affordances(environment.capability_snapshot)
        }
        if affordance.affordance_id not in grounded_affordance_ids:
            raise ValueError("Taiji internalization affordance is not grounded by latest evidence")

        snapshot = self.model.architecture.cognitive_snapshot()
        percept_payload = None if snapshot.percept is None else snapshot.percept.to_payload()
        recovery_payload = self._workbench_loop_state.get("recovery_portfolio", {})
        return GroundedOutcomeEvidence(
            evidence_id=evidence.evidence_id,
            outcome_id="workbench-outcome:" + evidence.evidence_id,
            outcome=Outcome(
                intent_id=evidence.intent_id,
                reward=float(reward),
                success=evidence.success,
                provenance="workbench-observed",
                tick=int(event.tick),
            ),
            affordance=affordance,
            capability_snapshot_digest=environment.capability_snapshot.snapshot_id,
            parent_checkpoint_id=str(parent_checkpoint_id),
            owner_id=str(owner_id),
            reward_terms=dict(reward_terms),
            percept_digest="" if percept_payload is None else content_digest(percept_payload),
            world_digest=content_digest(world.to_payload()),
            recovery_digest=content_digest(recovery_payload),
            metadata={
                "after_state_digest": evidence.after_state_digest,
                "source": "workbench-observed",
                "workbench_evidence": evidence.evidence_id,
            },
        )

    @staticmethod
    def _taiji_workbench_affordance_payload(affordance: Any) -> dict[str, Any]:
        """Project a WorldAffordance checkpoint payload to JSON-safe values."""

        payload = dict(affordance.to_payload())
        features = payload.get("features")
        if hasattr(features, "detach"):
            payload["features"] = [float(value) for value in features.detach().flatten().tolist()]
        return payload

    def execute_taiji_workbench_task(
        self,
        *,
        snapshot_id: str,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
        learn: bool = False,
    ) -> dict[str, Any]:
        """Select, admit, and execute one Taiji-owned read-only task."""

        decision = self._select_taiji_workbench_candidate(
            novelty=novelty,
            resource_budget=resource_budget,
        )
        environment = self._sync_workbench_root()
        world = self.model.architecture.cognitive_snapshot().world
        admission = environment.admit_taiji_candidate(
            decision.selected,
            snapshot_id=snapshot_id,
            current_tick=world.tick,
            current_affordance_ids=tuple(item.affordance_id for item in world.affordances),
            current_affordances=world.affordances,
        )
        payload = {
            "admission": admission.to_payload(),
            "decision": self._taiji_workbench_decision_payload(decision),
            "execution": None,
        }
        if not admission.accepted:
            return payload
        if admission.request is None:
            raise RuntimeError("accepted Taiji task admission lost its Workbench request")
        execution = self.execute_workbench_intent(
            decision.action_intent,
            snapshot_id=admission.request.snapshot_id,
            learn=learn,
            # Only online learning needs the identity-bound decision context;
            # read-only execution must remain valid after admission alone.
            executive_decision=decision if learn else None,
        )
        payload["execution"] = execution
        return payload

    def _sync_recovery_portfolio_after_successor(self, result: Any) -> dict[str, Any]:
        """Publish a recovery branch's successor mutation as one portfolio revision."""

        if not isinstance(result, dict):
            return result
        state = self._workbench_loop_state.get("successor_graph")
        portfolio = self._workbench_loop_state.get("recovery_portfolio")
        if not isinstance(state, Mapping) or not isinstance(portfolio, Mapping):
            return result
        branch_id = str(state.get("recovery_branch_id", ""))
        if not branch_id:
            return result
        branches = portfolio.get("branches", ())
        if isinstance(branches, (str, bytes)) or not isinstance(branches, Sequence):
            result["recovery_portfolio_checkpoint"] = {
                "committed": False,
                "error_code": "portfolio_invalid",
            }
            return result
        try:
            revision = int(portfolio.get("revision", 0))
            ttl_ticks = int(portfolio.get("branch_ttl_ticks", 256))
        except (TypeError, ValueError) as exc:
            result["recovery_portfolio_checkpoint"] = {
                "committed": False,
                "error_code": "portfolio_invalid",
                "error": str(exc),
            }
            return result
        current_tick = int(self.model.architecture.cognitive_snapshot().world.tick)
        status = str(result.get("status", state.get("status", "running")))
        branch_status = (
            "completed"
            if status == "completed"
            else "failed" if status in {"recovery_needed", "checkpoint_failed"} else "active"
        )
        updated_branches: list[dict[str, Any]] = []
        found = False
        for raw_branch in branches:
            if not isinstance(raw_branch, Mapping):
                result["recovery_portfolio_checkpoint"] = {
                    "committed": False,
                    "error_code": "portfolio_invalid",
                }
                return result
            branch = dict(raw_branch)
            if str(branch.get("branch_id", "")) == branch_id:
                branch.update(
                    {
                        "loop_id": str(state.get("loop_id", branch.get("loop_id", ""))),
                        "status": branch_status,
                        "budget_units": float(state.get("budget_units", 0.0)),
                        "completed_steps": int(state.get("completed_steps", 0)),
                        "frontier_affordance_ids": list(state.get("frontier_affordance_ids", ())),
                        "committed_request_ids": [
                            str(value) for value in state.get("committed_request_ids", ())
                        ],
                        "consumed_affordance_ids": [
                            str(value) for value in state.get("consumed_affordance_ids", ())
                        ],
                        "event_ids": [str(value) for value in state.get("event_ids", ())],
                        "last_touched_tick": current_tick,
                        "expires_at_tick": (
                            current_tick + ttl_ticks if branch_status == "active" else current_tick
                        ),
                    }
                )
                found = True
            updated_branches.append(branch)
        if not found:
            return result
        updated_portfolio = dict(portfolio)
        updated_portfolio["branches"] = updated_branches
        updated_portfolio["last_maintenance_tick"] = current_tick
        updated_portfolio["revision"] = revision + 1
        self._workbench_loop_state["recovery_portfolio"] = updated_portfolio
        try:
            self.save()
        except (OSError, RuntimeError, ValueError) as exc:
            result["recovery_portfolio_checkpoint"] = {
                "committed": False,
                "error_code": "checkpoint_failed",
                "error": str(exc),
            }
        else:
            result["recovery_portfolio_checkpoint"] = {"committed": True}
        result["recovery_portfolio"] = updated_portfolio
        return result

    @_workbench_successor_synchronized
    def execute_taiji_workbench_successor_loop(
        self,
        *,
        snapshot_id: str,
        loop_id: str,
        max_steps: int = 8,
        max_budget_units: float = 32.0,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
        learn: bool = False,
        expected_portfolio_revision: int | None = None,
    ) -> dict[str, Any]:
        """Execute a bounded Taiji-owned successor graph with checkpoint continuation.

        Each successful read-only workspace event invalidates the complete old
        affordance frontier and deterministically re-projects only the latest
        evidence.  The local step limit bounds one invocation; the persisted
        budget bounds continuation across invocations and checkpoints.
        """

        import math

        from seed_platform.workbench import (
            WORKBENCH_MAX_LOOP_BUDGET_UNITS,
            WORKBENCH_MAX_LOOP_STEPS,
            WORKBENCH_TAIJI_EVIDENCE_KIND,
            WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
            WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
            WORKBENCH_TAIJI_SUCCESSOR_STEP_BUDGET_UNITS,
        )

        environment = self._sync_workbench_root()
        if not str(loop_id).strip():
            raise ValueError("successor loop_id cannot be empty")
        try:
            step_limit = int(max_steps)
            budget_limit = float(max_budget_units)
        except (TypeError, ValueError) as exc:
            raise ValueError("successor loop limits must be numeric") from exc
        if not 1 <= step_limit <= WORKBENCH_MAX_LOOP_STEPS:
            raise ValueError("successor max_steps must be between 1 and 8")
        if (
            not math.isfinite(budget_limit)
            or not 0.0 < budget_limit <= WORKBENCH_MAX_LOOP_BUDGET_UNITS
        ):
            raise ValueError("successor max_budget_units must be in (0, 32]")

        world = self.model.architecture.cognitive_snapshot().world
        existing = self._workbench_loop_state.get("successor_graph")
        same_loop = isinstance(existing, Mapping) and str(existing.get("loop_id", "")) == str(
            loop_id
        )
        if expected_portfolio_revision is not None:
            portfolio = self._workbench_loop_state.get("recovery_portfolio")
            if not isinstance(portfolio, Mapping):
                raise RuntimeError("expected recovery portfolio revision is not persisted")
            try:
                expected_revision = int(expected_portfolio_revision)
                actual_revision = int(portfolio.get("revision", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("recovery portfolio revision is invalid") from exc
            if expected_revision != actual_revision:
                raise RuntimeError("recovery portfolio revision is stale")
        if same_loop:
            successor_state = dict(existing)
            if successor_state.get("format") != WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT:
                raise ValueError("unsupported Taiji successor graph checkpoint format")
            if int(successor_state.get("version", 0)) != WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION:
                raise ValueError("unsupported Taiji successor graph checkpoint version")
            if str(successor_state.get("snapshot_id", "")) != str(snapshot_id):
                raise ValueError("Taiji successor graph capability snapshot drifted")
            if not math.isclose(
                float(successor_state.get("budget_limit", 0.0)),
                budget_limit,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("Taiji successor graph budget changed during continuation")
        else:
            retired_loop_ids = (
                {str(item) for item in existing.get("retired_loop_ids", ())}
                if isinstance(existing, Mapping)
                else set()
            )
            if str(loop_id) in retired_loop_ids:
                raise RuntimeError("Taiji successor graph loop identity was retired after recovery")
            if isinstance(existing, Mapping) and isinstance(existing.get("in_flight"), Mapping):
                raise RuntimeError(
                    "Taiji successor graph has an unresolved in-flight step; recovery is required"
                )
            successor_state = {
                "format": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
                "version": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
                "loop_id": str(loop_id),
                "snapshot_id": str(snapshot_id),
                "budget_limit": budget_limit,
                "budget_units": 0.0,
                "completed_steps": 0,
                "committed_request_ids": [],
                "consumed_affordance_ids": [],
                "event_ids": [],
                "frontier_affordance_ids": [item.affordance_id for item in world.affordances],
                "remaining_frontier_affordance_ids": [
                    item.affordance_id for item in world.affordances
                ],
                "latest_evidence_id": "",
                "failure": None,
                "retired_loop_ids": [],
                "status": "running",
            }

        if same_loop and isinstance(successor_state.get("in_flight"), Mapping):
            pending = dict(successor_state["in_flight"])
            current_world = self.model.architecture.cognitive_snapshot().world
            latest_evidence = next(
                (
                    item
                    for item in reversed(current_world.events)
                    if item.kind == WORKBENCH_TAIJI_EVIDENCE_KIND
                ),
                None,
            )
            failure = {
                "code": "in_flight_checkpoint_recovery_required",
                "reason": "a step was reserved before checkpoint completion; execution state is unknown",
                "step_index": int(pending.get("index", successor_state.get("completed_steps", 0))),
                "candidate_id": str(pending.get("candidate_id", "")),
                "capability_id": str(pending.get("capability_id", "")),
                "parameters": (
                    dict(pending.get("parameters", {}))
                    if isinstance(pending.get("parameters", {}), Mapping)
                    else {}
                ),
                "source_affordance_id": str(pending.get("source_affordance_id", "")),
                "latest_evidence_id": (
                    "" if latest_evidence is None else str(latest_evidence.event_id)
                ),
                "completed_prefix": int(successor_state.get("completed_steps", 0)),
                "remaining_frontier_affordance_ids": [
                    item.affordance_id for item in current_world.affordances
                ],
            }
            successor_state["status"] = "recovery_needed"
            successor_state["failure"] = failure
            successor_state["latest_evidence_id"] = failure["latest_evidence_id"]
            successor_state["remaining_frontier_affordance_ids"] = failure[
                "remaining_frontier_affordance_ids"
            ]
            self._workbench_loop_state = {
                **self._workbench_loop_state,
                "successor_graph": successor_state,
            }
            checkpoint: dict[str, Any] = {"committed": False, "error_code": "checkpoint_failed"}
            try:
                checkpoint = {"committed": True, "path": str(self.save())}
            except (OSError, RuntimeError, ValueError) as exc:
                checkpoint["error"] = str(exc)
            return {
                "format": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
                "version": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
                "loop_id": str(loop_id),
                "snapshot_id": str(snapshot_id),
                "status": "recovery_needed",
                "recovery_needed": True,
                "steps": [],
                "completed_prefix": int(successor_state.get("completed_steps", 0)),
                "budget_units": float(successor_state.get("budget_units", 0.0)),
                "frontier_affordance_ids": [
                    item.affordance_id for item in current_world.affordances
                ],
                "failure": failure,
                "checkpoint": checkpoint,
            }

        terminal_statuses = {
            "completed",
            "failed",
            "recovery_needed",
            "budget_exhausted",
            "checkpoint_failed",
        }
        if same_loop and str(successor_state.get("status", "")) in terminal_statuses:
            payload = {
                "format": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
                "version": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
                "loop_id": str(loop_id),
                "snapshot_id": str(snapshot_id),
                "status": str(successor_state["status"]),
                "steps": [],
                "completed_prefix": int(successor_state.get("completed_steps", 0)),
                "budget_units": float(successor_state.get("budget_units", 0.0)),
                "frontier_affordance_ids": list(successor_state.get("frontier_affordance_ids", ())),
                "error_code": "successor_loop_terminal",
            }
            if str(successor_state["status"]) in {"recovery_needed", "checkpoint_failed"}:
                payload["recovery_needed"] = True
            failure = successor_state.get("failure")
            if isinstance(failure, Mapping):
                payload["failure"] = dict(failure)
            return payload

        current_frontier = tuple(item.affordance_id for item in world.affordances)
        expected_frontier = tuple(
            str(item) for item in successor_state.get("frontier_affordance_ids", ())
        )
        if (
            same_loop
            and int(successor_state.get("completed_steps", 0)) > 0
            and not successor_state.get("recovery_branch_id")
        ):
            # Do not trust an arbitrary restored frontier: reconstruct it from
            # the latest typed evidence, then compare content identity.
            self.reproject_workbench_from_latest_evidence(
                snapshot_id=snapshot_id,
                allow_empty=True,
            )
            world = self.model.architecture.cognitive_snapshot().world
            current_frontier = tuple(item.affordance_id for item in world.affordances)
        if current_frontier != expected_frontier:
            raise ValueError("Taiji successor graph frontier drifted during continuation")
        if same_loop and successor_state.get("recovery_branch_id"):
            branch_source_id = str(successor_state.get("recovery_branch_source_evidence_id", ""))
            if not branch_source_id or not any(
                item.event_id == branch_source_id and item.tick == world.tick
                for item in world.events
            ):
                raise ValueError("selected recovery branch evidence expired")
        if not current_frontier:
            if same_loop:
                successor_state["status"] = "completed"
                self._workbench_loop_state = {
                    **self._workbench_loop_state,
                    "successor_graph": successor_state,
                }
                return {
                    "format": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
                    "version": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
                    "loop_id": str(loop_id),
                    "snapshot_id": str(snapshot_id),
                    "status": "completed",
                    "steps": [],
                    "completed_prefix": int(successor_state.get("completed_steps", 0)),
                    "budget_units": float(successor_state.get("budget_units", 0.0)),
                    "frontier_affordance_ids": [],
                }
            raise RuntimeError("Taiji successor graph requires a current world affordance")

        self._workbench_loop_state = {
            **self._workbench_loop_state,
            "successor_graph": successor_state,
        }
        result: dict[str, Any] = {
            "format": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
            "version": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
            "loop_id": str(loop_id),
            "snapshot_id": str(snapshot_id),
            "status": "running",
            "steps": [],
            "completed_prefix": int(successor_state.get("completed_steps", 0)),
            "budget_units": float(successor_state.get("budget_units", 0.0)),
            "frontier_affordance_ids": list(current_frontier),
        }

        def current_evidence_id() -> str:
            current_world = self.model.architecture.cognitive_snapshot().world
            event = next(
                (
                    item
                    for item in reversed(current_world.events)
                    if item.kind == WORKBENCH_TAIJI_EVIDENCE_KIND
                ),
                None,
            )
            return "" if event is None else str(event.event_id)

        def mark_recovery_needed(
            step: dict[str, Any],
            code: str,
            reason: str,
            *,
            evidence_id: str = "",
        ) -> dict[str, Any]:
            current_world = self.model.architecture.cognitive_snapshot().world
            remaining = [item.affordance_id for item in current_world.affordances]
            latest_evidence_id = evidence_id or current_evidence_id()
            failure = {
                "code": str(code),
                "reason": str(reason),
                "step_index": int(step.get("index", successor_state.get("completed_steps", 0))),
                "candidate_id": str(step.get("candidate_id", "")),
                "capability_id": str(step.get("capability_id", "")),
                "parameters": (
                    dict(step.get("parameters", {}))
                    if isinstance(step.get("parameters", {}), Mapping)
                    else {}
                ),
                "source_affordance_id": str(step.get("source_affordance_id", "")),
                "latest_evidence_id": latest_evidence_id,
                "completed_prefix": int(successor_state.get("completed_steps", 0)),
                "remaining_frontier_affordance_ids": remaining,
            }
            successor_state["status"] = "recovery_needed"
            successor_state["latest_evidence_id"] = latest_evidence_id
            successor_state["frontier_affordance_ids"] = remaining
            successor_state["remaining_frontier_affordance_ids"] = remaining
            successor_state["failure"] = failure
            step.update(
                {
                    "status": "recovery_needed",
                    "success": False,
                    "error_code": str(code),
                    "error": str(reason),
                    "failure": failure,
                }
            )
            result.update(
                {
                    "status": "recovery_needed",
                    "error_code": str(code),
                    "error": str(reason),
                    "recovery_needed": True,
                    "failure": failure,
                }
            )
            return failure

        def commit_checkpoint(
            step: dict[str, Any] | None = None,
            *,
            phase: str = "after_step",
        ) -> bool:
            try:
                path = self.save()
            except (OSError, RuntimeError, ValueError) as exc:
                successor_state["status"] = "checkpoint_failed"
                failure = {
                    "code": "checkpoint_failed",
                    "reason": str(exc),
                    "step_index": int(
                        successor_state.get("completed_steps", 0)
                        if step is None
                        else step.get("index", successor_state.get("completed_steps", 0))
                    ),
                    "candidate_id": "" if step is None else str(step.get("candidate_id", "")),
                    "capability_id": "" if step is None else str(step.get("capability_id", "")),
                    "parameters": (
                        dict(step.get("parameters", {}))
                        if step is not None and isinstance(step.get("parameters", {}), Mapping)
                        else {}
                    ),
                    "source_affordance_id": (
                        "" if step is None else str(step.get("source_affordance_id", ""))
                    ),
                    "latest_evidence_id": current_evidence_id(),
                    "completed_prefix": int(successor_state.get("completed_steps", 0)),
                    "remaining_frontier_affordance_ids": [
                        item.affordance_id
                        for item in self.model.architecture.cognitive_snapshot().world.affordances
                    ],
                }
                successor_state["failure"] = failure
                successor_state["latest_evidence_id"] = failure["latest_evidence_id"]
                successor_state["remaining_frontier_affordance_ids"] = failure[
                    "remaining_frontier_affordance_ids"
                ]
                if step is not None:
                    step["checkpoint"] = {
                        "phase": phase,
                        "committed": False,
                        "error_code": "checkpoint_failed",
                        "error": str(exc),
                    }
                result.update(
                    {
                        "status": "checkpoint_failed",
                        "error_code": "checkpoint_failed",
                        "error": str(exc),
                        "recovery_needed": True,
                        "failure": failure,
                    }
                )
                return False
            if step is not None:
                step["checkpoint"] = {"phase": phase, "committed": True, "path": str(path)}
            result["checkpoint"] = {"committed": True, "path": str(path)}
            return True

        for _ in range(step_limit):
            budget_used = float(successor_state.get("budget_units", 0.0))
            if budget_used + WORKBENCH_TAIJI_SUCCESSOR_STEP_BUDGET_UNITS > budget_limit:
                successor_state["status"] = "budget_exhausted"
                result["status"] = "budget_exhausted"
                result["error_code"] = "successor_loop_budget_limit"
                break

            world = self.model.architecture.cognitive_snapshot().world
            frontier_before = tuple(item.affordance_id for item in world.affordances)
            if frontier_before != tuple(
                str(item) for item in successor_state.get("frontier_affordance_ids", ())
            ):
                raise ValueError("Taiji successor graph frontier changed before execution")
            decision = self._select_taiji_workbench_candidate(
                novelty=novelty,
                resource_budget=resource_budget,
            )
            source_affordance_id = str(decision.selected.source_affordance_id or "")
            step: dict[str, Any] = {
                "index": int(successor_state.get("completed_steps", 0)),
                "candidate_id": decision.selected.candidate_id,
                "capability_id": decision.selected.action_intent.kind,
                "parameters": dict(decision.selected.action_intent.parameters),
                "source_affordance_id": source_affordance_id,
                "frontier_before_affordance_ids": list(frontier_before),
                "decision": self._taiji_workbench_decision_payload(decision),
            }
            if source_affordance_id in {
                str(item) for item in successor_state.get("consumed_affordance_ids", ())
            }:
                step.update(
                    {
                        "status": "error",
                        "success": False,
                        "error_code": "reused_successor_affordance",
                        "error": "Taiji attempted to reuse a consumed successor affordance",
                    }
                )
                mark_recovery_needed(
                    step,
                    "reused_successor_affordance",
                    "Taiji attempted to reuse a consumed successor affordance",
                )
                result["steps"].append(step)
                if not commit_checkpoint(step):
                    return result
                return result

            admission = environment.admit_taiji_candidate(
                decision.selected,
                snapshot_id=snapshot_id,
                current_tick=world.tick,
                current_affordance_ids=frontier_before,
                current_affordances=world.affordances,
            )
            step["admission"] = admission.to_payload()
            if not admission.accepted or admission.request is None:
                step.update(
                    {
                        "status": "rejected",
                        "success": False,
                        "error_code": admission.reason_code,
                        "error": admission.reason,
                    }
                )
                mark_recovery_needed(step, admission.reason_code, admission.reason)
                result["steps"].append(step)
                if not commit_checkpoint(step):
                    return result
                return result

            successor_state["in_flight"] = {
                "index": int(step["index"]),
                "candidate_id": str(step["candidate_id"]),
                "source_affordance_id": source_affordance_id,
                "request_id": str(admission.request.request_id),
                "capability_id": str(step["capability_id"]),
                "parameters": dict(step["parameters"]),
                "frontier_before_affordance_ids": list(frontier_before),
            }
            if not commit_checkpoint(step, phase="before_execution"):
                result["steps"].append(step)
                return result

            try:
                execution = self.execute_workbench_intent(
                    decision.action_intent,
                    snapshot_id=admission.request.snapshot_id,
                    learn=learn,
                    # Keep non-learning successor steps independent of the
                    # architecture's mutable learning cursor.
                    executive_decision=decision if learn else None,
                )
                outcome = dict(execution.get("outcome") or {})
                step.update(
                    {
                        "status": str(outcome.get("status", "error")),
                        "success": bool(outcome.get("success", False)),
                        "outcome": outcome,
                        "tool_call": execution.get("tool_call"),
                    }
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                step.update(
                    {
                        "status": "error",
                        "success": False,
                        "error_code": "successor_step_error",
                        "error": str(exc),
                    }
                )
            successor_state.pop("in_flight", None)

            if not step.get("success") or step.get("status") != "success":
                outcome_payload = step.get("outcome")
                outcome_error_code = (
                    str(outcome_payload.get("error_code", ""))
                    if isinstance(outcome_payload, Mapping)
                    else ""
                )
                outcome_error = (
                    str(outcome_payload.get("error", ""))
                    if isinstance(outcome_payload, Mapping)
                    else ""
                )
                mark_recovery_needed(
                    step,
                    str(step.get("error_code") or outcome_error_code or "successor_step_failed"),
                    str(step.get("error") or outcome_error or "successor step failed"),
                )
                result["steps"].append(step)
                if not commit_checkpoint(step):
                    return result
                return result

            event_payload = execution.get("taiji_world_event")
            if not isinstance(event_payload, Mapping):
                mark_recovery_needed(
                    step,
                    "missing_taiji_world_event",
                    "successful successor step produced no Taiji world event",
                )
                result["steps"].append(step)
                if not commit_checkpoint(step):
                    return result
                return result

            event_id = str(event_payload.get("event_id", ""))
            if not event_id:
                mark_recovery_needed(
                    step,
                    "missing_taiji_world_event_id",
                    "successful successor step world event has no event_id",
                )
                result["steps"].append(step)
                if not commit_checkpoint(step):
                    return result
                return result
            try:
                projection = self.reproject_workbench_from_latest_evidence(
                    snapshot_id=snapshot_id,
                    allow_empty=True,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                mark_recovery_needed(
                    step, "successor_reprojection_error", str(exc), evidence_id=event_id
                )
                result["steps"].append(step)
                if not commit_checkpoint(step):
                    return result
                return result
            frontier_after = tuple(
                str(item.get("affordance_id", ""))
                for item in projection.get("affordances", ())
                if isinstance(item, Mapping)
            )
            invalidated = [item for item in frontier_before if item not in frontier_after]
            step.update(
                {
                    "event_id": event_id,
                    "frontier_after_affordance_ids": list(frontier_after),
                    "invalidated_affordance_ids": invalidated,
                    "successor_count": len(frontier_after),
                }
            )
            successor_state["budget_units"] = (
                budget_used + WORKBENCH_TAIJI_SUCCESSOR_STEP_BUDGET_UNITS
            )
            successor_state["completed_steps"] = int(successor_state.get("completed_steps", 0)) + 1
            successor_state["committed_request_ids"] = [
                *successor_state.get("committed_request_ids", ()),
                str(admission.request.request_id),
            ][-MAX_WORKBENCH_LOOP_COMMITTED_REQUESTS:]
            successor_state["consumed_affordance_ids"] = [
                *successor_state.get("consumed_affordance_ids", ()),
                source_affordance_id,
            ][-MAX_WORKBENCH_LOOP_COMMITTED_REQUESTS:]
            successor_state["event_ids"] = [
                *successor_state.get("event_ids", ()),
                event_id,
            ][-MAX_WORKBENCH_LOOP_COMMITTED_REQUESTS:]
            successor_state["frontier_affordance_ids"] = list(frontier_after)
            successor_state["remaining_frontier_affordance_ids"] = list(frontier_after)
            successor_state["latest_evidence_id"] = event_id
            successor_state["status"] = "completed" if not frontier_after else "running"
            result["steps"].append(step)
            result["completed_prefix"] = int(successor_state["completed_steps"])
            result["budget_units"] = float(successor_state["budget_units"])
            result["frontier_affordance_ids"] = list(frontier_after)
            if not commit_checkpoint(step):
                return result
            if successor_state["status"] == "completed":
                result["status"] = "completed"
                return result

        if successor_state["status"] == "running":
            result["status"] = "paused"
        elif successor_state["status"] == "budget_exhausted" and not commit_checkpoint():
            return result
        result["completed_prefix"] = int(successor_state.get("completed_steps", 0))
        result["budget_units"] = float(successor_state.get("budget_units", 0.0))
        result["frontier_affordance_ids"] = list(successor_state.get("frontier_affordance_ids", ()))
        return result

    @_workbench_synchronized
    def handoff_taiji_workbench_recovery(
        self,
        *,
        parent_loop_id: str,
        recovery_loop_id: str,
        snapshot_id: str,
        max_steps: int = 8,
        max_budget_units: float = 32.0,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
        learn: bool = False,
    ) -> dict[str, Any]:
        """Start a new bounded loop only from fresh external workspace evidence."""

        import math

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_EVIDENCE_KIND,
            WORKBENCH_TAIJI_RECOVERY_BRANCH_TTL_TICKS,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_MAX_BRANCHES,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
            WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
            WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
        )

        if not str(parent_loop_id).strip() or not str(recovery_loop_id).strip():
            raise ValueError("recovery parent and loop ids cannot be empty")
        if str(parent_loop_id) == str(recovery_loop_id):
            raise ValueError("recovery loop must have a new identity")
        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("recovery capability snapshot is not current")
        existing = self._workbench_loop_state.get("successor_graph")
        if not isinstance(existing, Mapping):
            raise RuntimeError("recovery requires a persisted successor graph failure")
        if str(existing.get("loop_id", "")) != str(parent_loop_id):
            raise ValueError("recovery parent loop does not match the persisted graph")
        if str(existing.get("status", "")) not in {"recovery_needed", "checkpoint_failed"}:
            raise RuntimeError("recovery requires a successor graph in recovery-needed state")
        failure = existing.get("failure")
        if not isinstance(failure, Mapping):
            raise RuntimeError("recovery state has no auditable failure record")
        try:
            budget_limit = float(existing.get("budget_limit", 0.0))
            budget_used = float(existing.get("budget_units", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("recovery budget state is invalid") from exc
        if not math.isfinite(budget_limit) or not math.isfinite(budget_used):
            raise ValueError("recovery budget state must be finite")
        if not math.isclose(
            budget_limit,
            float(max_budget_units),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("recovery cannot reset or change the parent budget")
        if budget_used >= budget_limit:
            raise RuntimeError("recovery has no remaining budget")

        world = self.model.architecture.cognitive_snapshot().world
        latest_event = next(
            (item for item in reversed(world.events) if item.kind == WORKBENCH_TAIJI_EVIDENCE_KIND),
            None,
        )
        failure_event_id = str(failure.get("latest_evidence_id", ""))
        if latest_event is None or latest_event.event_id == failure_event_id:
            raise RuntimeError("recovery requires a newer workspace evidence event")
        if latest_event.tick != world.tick:
            raise RuntimeError("recovery evidence is stale")

        from seed_platform.workbench import WorkbenchTaijiEvidence

        evidence = WorkbenchTaijiEvidence.from_taiji_event(latest_event)
        if not evidence.success:
            raise RuntimeError("recovery evidence must be successful")
        if evidence.snapshot_id != environment.capability_snapshot.snapshot_id:
            raise RuntimeError("recovery evidence capability snapshot drifted")
        failed_capability_id = str(failure.get("capability_id", ""))
        failed_parameters = failure.get("parameters")
        if not failed_capability_id or not isinstance(failed_parameters, Mapping):
            raise RuntimeError("recovery failure has no capability context")
        if evidence.capability_id != failed_capability_id:
            raise RuntimeError("recovery evidence is incompatible with the failed capability")
        if dict(evidence.parameters) != dict(failed_parameters):
            raise RuntimeError("recovery evidence parameters do not match the failed context")
        affordances = evidence.to_taiji_affordances(environment.capability_snapshot)
        if not affordances:
            raise RuntimeError("recovery evidence produced no successor affordance")
        consumed = {str(item) for item in existing.get("consumed_affordance_ids", ())}
        if any(item.affordance_id in consumed for item in affordances):
            raise RuntimeError("recovery evidence reintroduced a consumed affordance")
        world = self.model.architecture.set_world_affordances(affordances)
        branch_id = (
            "recovery-branch:"
            + hashlib.sha256(
                f"{parent_loop_id}:{evidence.evidence_id}:{evidence.after_state_digest}".encode()
            ).hexdigest()[:32]
        )
        retired_loop_ids = [
            *existing.get("retired_loop_ids", ()),
            str(parent_loop_id),
        ]
        branch = {
            "branch_id": branch_id,
            "loop_id": str(recovery_loop_id),
            "parent_loop_id": str(parent_loop_id),
            "source_evidence_id": evidence.evidence_id,
            "source_after_state_digest": evidence.after_state_digest,
            "capability_id": evidence.capability_id,
            "parameters": dict(evidence.parameters),
            "status": "selected",
            "budget_limit": budget_limit,
            "budget_units": budget_used,
            "completed_steps": int(existing.get("completed_steps", 0)),
            "committed_request_ids": [
                str(item) for item in existing.get("committed_request_ids", ())
            ],
            "consumed_affordance_ids": [
                str(item) for item in existing.get("consumed_affordance_ids", ())
            ],
            "event_ids": [str(item) for item in existing.get("event_ids", ())],
            "evidence": evidence.to_payload(),
            "frontier_affordance_ids": [item.affordance_id for item in affordances],
            "created_tick": int(world.tick),
            "last_touched_tick": int(world.tick),
            "expires_at_tick": int(world.tick) + WORKBENCH_TAIJI_RECOVERY_BRANCH_TTL_TICKS,
        }
        portfolio = {
            "format": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            "version": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
            "parent_loop_id": str(parent_loop_id),
            "snapshot_id": str(snapshot_id),
            "parent_failure": dict(failure),
            "parent_budget_limit": budget_limit,
            "parent_budget_units": budget_used,
            "parent_completed_steps": int(existing.get("completed_steps", 0)),
            "parent_committed_request_ids": [
                str(item) for item in existing.get("committed_request_ids", ())
            ],
            "parent_consumed_affordance_ids": [
                str(item) for item in existing.get("consumed_affordance_ids", ())
            ],
            "parent_event_ids": [str(item) for item in existing.get("event_ids", ())],
            "branches": [branch],
            "retired_loop_ids": retired_loop_ids,
            "max_branches": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_MAX_BRANCHES,
            "branch_ttl_ticks": WORKBENCH_TAIJI_RECOVERY_BRANCH_TTL_TICKS,
            "last_maintenance_tick": int(world.tick),
            "evicted_branches": [],
            "revision": 1,
        }
        recovery_state = {
            "format": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
            "version": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
            "loop_id": str(recovery_loop_id),
            "parent_loop_id": str(parent_loop_id),
            "recovery_branch_id": branch_id,
            "recovery_branch_source_evidence_id": evidence.evidence_id,
            "snapshot_id": str(snapshot_id),
            "budget_limit": budget_limit,
            "budget_units": budget_used,
            "completed_steps": int(existing.get("completed_steps", 0)),
            "committed_request_ids": [
                str(item) for item in existing.get("committed_request_ids", ())
            ],
            "consumed_affordance_ids": [
                str(item) for item in existing.get("consumed_affordance_ids", ())
            ],
            "event_ids": [str(item) for item in existing.get("event_ids", ())],
            "frontier_affordance_ids": [item.affordance_id for item in affordances],
            "remaining_frontier_affordance_ids": [item.affordance_id for item in affordances],
            "latest_evidence_id": evidence.evidence_id,
            "failure": None,
            "retired_loop_ids": retired_loop_ids,
            "recovery": {
                "parent_loop_id": str(parent_loop_id),
                "parent_failure": dict(failure),
                "branch_id": branch_id,
                "source_evidence_id": evidence.evidence_id,
                "source_after_state_digest": evidence.after_state_digest,
            },
            "status": "running",
        }
        self._workbench_loop_state = {
            **self._workbench_loop_state,
            "successor_graph": recovery_state,
            "recovery_portfolio": portfolio,
        }
        result = self.execute_taiji_workbench_successor_loop(
            snapshot_id=snapshot_id,
            loop_id=recovery_loop_id,
            max_steps=max_steps,
            max_budget_units=max_budget_units,
            novelty=novelty,
            resource_budget=resource_budget,
            learn=learn,
        )
        result["recovery"] = dict(recovery_state["recovery"])
        active_state = self._workbench_loop_state.get("successor_graph")
        if isinstance(active_state, Mapping):
            branch.update(
                {
                    "status": (
                        "completed"
                        if active_state.get("status") == "completed"
                        else (
                            "failed"
                            if active_state.get("status")
                            in {"recovery_needed", "checkpoint_failed"}
                            else "active"
                        )
                    ),
                    "budget_units": float(active_state.get("budget_units", budget_used)),
                    "completed_steps": int(
                        active_state.get("completed_steps", branch["completed_steps"])
                    ),
                    "frontier_affordance_ids": list(
                        active_state.get(
                            "frontier_affordance_ids", branch["frontier_affordance_ids"]
                        )
                    ),
                    "committed_request_ids": [
                        str(item) for item in active_state.get("committed_request_ids", ())
                    ],
                    "consumed_affordance_ids": [
                        str(item) for item in active_state.get("consumed_affordance_ids", ())
                    ],
                    "event_ids": [str(item) for item in active_state.get("event_ids", ())],
                }
            )
            branch_tick = int(self.model.architecture.cognitive_snapshot().world.tick)
            branch["last_touched_tick"] = branch_tick
            branch["expires_at_tick"] = (
                branch_tick + int(portfolio["branch_ttl_ticks"])
                if branch["status"] == "active"
                else branch_tick
            )
        portfolio["branches"] = [branch]
        portfolio["revision"] = int(portfolio.get("revision", 0)) + 1
        self._workbench_loop_state["recovery_portfolio"] = portfolio
        try:
            self.save()
        except (OSError, RuntimeError, ValueError) as exc:
            result["recovery"]["portfolio_checkpoint"] = {
                "committed": False,
                "error_code": "checkpoint_failed",
                "error": str(exc),
            }
        else:
            result["recovery"]["portfolio_checkpoint"] = {"committed": True}
        result["recovery"]["branch_id"] = branch_id
        result["recovery"]["portfolio"] = dict(portfolio)
        return result

    @staticmethod
    def _validate_recovery_source_evidence(
        event: Any,
        *,
        failure: Mapping[str, Any],
        environment: Any,
        current_tick: int,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Validate one fresh evidence event against its parent failure context."""

        from seed_platform.workbench import WorkbenchTaijiEvidence

        if event is None:
            raise RuntimeError("recovery requires a newer workspace evidence event")
        if event.tick != int(current_tick):
            raise RuntimeError("recovery evidence is stale")
        evidence = WorkbenchTaijiEvidence.from_taiji_event(event)
        if not evidence.success:
            raise RuntimeError("recovery evidence must be successful")
        if evidence.snapshot_id != environment.capability_snapshot.snapshot_id:
            raise RuntimeError("recovery evidence capability snapshot drifted")
        failed_capability_id = str(failure.get("capability_id", ""))
        failed_parameters = failure.get("parameters")
        if not failed_capability_id or not isinstance(failed_parameters, Mapping):
            raise RuntimeError("recovery failure has no capability context")
        if evidence.capability_id != failed_capability_id:
            raise RuntimeError("recovery evidence is incompatible with the failed capability")
        if dict(evidence.parameters) != dict(failed_parameters):
            raise RuntimeError("recovery evidence parameters do not match the failed context")
        affordances = evidence.to_taiji_affordances(environment.capability_snapshot)
        if not affordances:
            raise RuntimeError("recovery evidence produced no successor affordance")
        return evidence, affordances

    def _maintain_taiji_workbench_recovery_portfolio(
        self,
        portfolio: Mapping[str, Any],
        *,
        current_tick: int,
        reserve_slots: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply deterministic branch liveness and capacity rules."""

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_RECOVERY_BRANCH_TTL_TICKS,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_MAX_BRANCHES,
        )

        if int(current_tick) < 0:
            raise ValueError("recovery portfolio tick cannot be negative")
        if not 0 <= int(reserve_slots) <= 1:
            raise ValueError("recovery portfolio reserve_slots must be 0 or 1")
        try:
            max_branches = int(
                portfolio.get("max_branches", WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_MAX_BRANCHES)
            )
            ttl_ticks = int(
                portfolio.get("branch_ttl_ticks", WORKBENCH_TAIJI_RECOVERY_BRANCH_TTL_TICKS)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("recovery portfolio capacity policy is invalid") from exc
        if not 1 <= max_branches <= WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_MAX_BRANCHES:
            raise ValueError("recovery portfolio max_branches is outside the native limit")
        if ttl_ticks < 1:
            raise ValueError("recovery portfolio branch_ttl_ticks must be positive")

        raw_branches = portfolio.get("branches", ())
        if isinstance(raw_branches, (str, bytes)) or not isinstance(raw_branches, Sequence):
            raise ValueError("recovery portfolio branches are invalid")
        raw_tombstones = portfolio.get("evicted_branches", ())
        if isinstance(raw_tombstones, (str, bytes)) or not isinstance(raw_tombstones, Sequence):
            raise ValueError("recovery portfolio evicted branches are invalid")
        raw_retired = portfolio.get("retired_loop_ids", ())
        if isinstance(raw_retired, (str, bytes)) or not isinstance(raw_retired, Sequence):
            raise ValueError("recovery portfolio retired loop ids are invalid")

        branches: list[dict[str, Any]] = []
        expired_branch_ids: list[str] = []
        retired_loop_ids = list(dict.fromkeys(str(item) for item in raw_retired))
        try:
            stored_max_branches = int(portfolio.get("max_branches", max_branches))
            stored_ttl_ticks = int(portfolio.get("branch_ttl_ticks", ttl_ticks))
            revision = int(portfolio.get("revision", 0))
            stored_maintenance_tick = int(portfolio.get("last_maintenance_tick", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("recovery portfolio revision or maintenance state is invalid") from exc
        if revision < 0:
            raise ValueError("recovery portfolio revision cannot be negative")
        changed = (
            stored_max_branches != max_branches
            or stored_ttl_ticks != ttl_ticks
            or "revision" not in portfolio
        )
        for raw_branch in raw_branches:
            if not isinstance(raw_branch, Mapping):
                raise ValueError("recovery portfolio branch is invalid")
            branch = dict(raw_branch)
            branch_id = str(branch.get("branch_id", ""))
            if not branch_id:
                raise ValueError("recovery portfolio branch id is empty")
            status = str(branch.get("status", ""))
            if status not in {"active", "selected", "completed", "failed", "expired"}:
                raise ValueError("recovery portfolio branch status is invalid")
            try:
                created_tick = int(branch.get("created_tick", current_tick))
                last_touched_tick = int(branch.get("last_touched_tick", created_tick))
                expires_at_tick = int(branch.get("expires_at_tick", last_touched_tick + ttl_ticks))
            except (TypeError, ValueError) as exc:
                raise ValueError("recovery portfolio branch liveness is invalid") from exc
            if (
                created_tick < 0
                or last_touched_tick < created_tick
                or expires_at_tick < last_touched_tick
            ):
                raise ValueError("recovery portfolio branch liveness is inconsistent")
            if "created_tick" not in branch:
                changed = True
            if "last_touched_tick" not in branch:
                changed = True
            if "expires_at_tick" not in branch:
                changed = True
            branch.update(
                {
                    "created_tick": created_tick,
                    "last_touched_tick": last_touched_tick,
                    "expires_at_tick": expires_at_tick,
                }
            )
            if status in {"active", "selected"} and int(current_tick) > expires_at_tick:
                branch.update(
                    {
                        "status": "expired",
                        "expired_tick": int(current_tick),
                        "terminal_reason": "liveness_ttl",
                    }
                )
                expired_branch_ids.append(branch_id)
                loop_id = str(branch.get("loop_id", ""))
                if loop_id:
                    retired_loop_ids.append(loop_id)
                changed = True
            branches.append(branch)

        evicted_branch_ids: list[str] = []
        evicted_branches: list[dict[str, Any]] = []
        for item in raw_tombstones:
            if not isinstance(item, Mapping):
                raise ValueError("recovery portfolio evicted branch is invalid")
            evicted_branches.append(dict(item))
        while len(branches) + int(reserve_slots) > max_branches:
            candidates = [
                (index, item)
                for index, item in enumerate(branches)
                if item.get("status") in {"expired", "completed", "failed"}
            ]
            if not candidates:
                break
            index, branch = min(
                candidates,
                key=lambda item: (
                    int(item[1].get("last_touched_tick", 0)),
                    str(item[1].get("branch_id", "")),
                ),
            )
            removed = branches.pop(index)
            branch_id = str(removed.get("branch_id", ""))
            evicted_branch_ids.append(branch_id)
            evicted_branches.append(
                {
                    "branch_id": branch_id,
                    "loop_id": str(removed.get("loop_id", "")),
                    "source_evidence_id": str(removed.get("source_evidence_id", "")),
                    "source_after_state_digest": str(removed.get("source_after_state_digest", "")),
                    "status": "evicted",
                    "evicted_tick": int(current_tick),
                    "reason": "portfolio_capacity",
                }
            )
            loop_id = str(removed.get("loop_id", ""))
            if loop_id:
                retired_loop_ids.append(loop_id)
            changed = True

        updated = dict(portfolio)
        updated.update(
            {
                "max_branches": max_branches,
                "branch_ttl_ticks": ttl_ticks,
                "last_maintenance_tick": int(current_tick),
                "branches": branches,
                "retired_loop_ids": list(dict.fromkeys(retired_loop_ids)),
                "evicted_branches": evicted_branches,
                "revision": revision + 1 if changed else revision,
            }
        )
        if stored_maintenance_tick != int(current_tick):
            changed = True
            updated["revision"] = revision + 1
        return updated, {
            "changed": changed,
            "current_tick": int(current_tick),
            "max_branches": max_branches,
            "branch_ttl_ticks": ttl_ticks,
            "expired_branch_ids": expired_branch_ids,
            "evicted_branch_ids": evicted_branch_ids,
            "live_branch_count": sum(
                1 for item in branches if item.get("status") in {"active", "selected"}
            ),
            "revision": int(updated["revision"]),
        }

    @staticmethod
    def _require_recovery_portfolio_revision(
        portfolio: Mapping[str, Any], expected_revision: int | None
    ) -> None:
        if expected_revision is None:
            return
        try:
            expected = int(expected_revision)
            actual = int(portfolio.get("revision", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("recovery portfolio revision is invalid") from exc
        if expected != actual:
            raise RuntimeError("recovery portfolio revision is stale")

    @_workbench_synchronized
    def maintain_taiji_workbench_recovery_portfolio(
        self,
        *,
        parent_loop_id: str,
        snapshot_id: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Persist lifecycle maintenance without selecting or executing a branch."""

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
        )

        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("recovery portfolio capability snapshot is not current")
        portfolio = self._workbench_loop_state.get("recovery_portfolio")
        if not isinstance(portfolio, Mapping):
            raise RuntimeError("recovery portfolio is not persisted")
        if str(portfolio.get("parent_loop_id", "")) != str(parent_loop_id):
            raise ValueError("recovery portfolio parent does not match")
        self._require_recovery_portfolio_revision(portfolio, expected_revision)
        world = self.model.architecture.cognitive_snapshot().world
        updated, maintenance = self._maintain_taiji_workbench_recovery_portfolio(
            portfolio,
            current_tick=world.tick,
        )
        previous = self._workbench_loop_state.get("recovery_portfolio")
        self._workbench_loop_state["recovery_portfolio"] = updated
        try:
            checkpoint = {"committed": True, "path": str(self.save())}
        except (OSError, RuntimeError, ValueError) as exc:
            if previous is None:
                self._workbench_loop_state.pop("recovery_portfolio", None)
            else:
                self._workbench_loop_state["recovery_portfolio"] = previous
            raise RuntimeError("recovery portfolio maintenance checkpoint failed") from exc
        return {
            "format": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            "version": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
            "status": "portfolio_maintained",
            "parent_loop_id": str(parent_loop_id),
            "maintenance": maintenance,
            "portfolio": updated,
            "checkpoint": checkpoint,
        }

    @_workbench_synchronized
    def taiji_workbench_recovery_portfolio_snapshot(
        self,
        *,
        parent_loop_id: str,
        snapshot_id: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Return a non-executable recovery portfolio read model."""

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_RECOVERY_BRANCH_TTL_TICKS,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_MAX_BRANCHES,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
        )

        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("recovery portfolio capability snapshot is not current")
        portfolio = self._workbench_loop_state.get("recovery_portfolio")
        if not isinstance(portfolio, Mapping):
            raise RuntimeError("recovery portfolio is not persisted")
        if str(portfolio.get("parent_loop_id", "")) != str(parent_loop_id):
            raise ValueError("recovery portfolio parent does not match")
        self._require_recovery_portfolio_revision(portfolio, expected_revision)
        try:
            revision = int(portfolio.get("revision", 0))
            max_branches = int(
                portfolio.get("max_branches", WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_MAX_BRANCHES)
            )
            ttl_ticks = int(
                portfolio.get("branch_ttl_ticks", WORKBENCH_TAIJI_RECOVERY_BRANCH_TTL_TICKS)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("recovery portfolio snapshot metadata is invalid") from exc
        raw_branches = portfolio.get("branches", ())
        if isinstance(raw_branches, (str, bytes)) or not isinstance(raw_branches, Sequence):
            raise ValueError("recovery portfolio branches are invalid")
        raw_evicted = portfolio.get("evicted_branches", ())
        if isinstance(raw_evicted, (str, bytes)) or not isinstance(raw_evicted, Sequence):
            raise ValueError("recovery portfolio evicted branches are invalid")

        current_tick = int(self.model.architecture.cognitive_snapshot().world.tick)
        branches: list[dict[str, Any]] = []
        counts = {"active": 0, "selected": 0, "completed": 0, "failed": 0, "expired": 0}
        liveness_due: list[str] = []
        for raw_branch in raw_branches:
            if not isinstance(raw_branch, Mapping):
                raise ValueError("recovery portfolio branch is invalid")
            status = str(raw_branch.get("status", ""))
            if status not in counts:
                raise ValueError("recovery portfolio branch status is invalid")
            try:
                created_tick = int(raw_branch.get("created_tick", 0))
                last_touched_tick = int(raw_branch.get("last_touched_tick", created_tick))
                expires_at_tick = int(raw_branch.get("expires_at_tick", last_touched_tick))
            except (TypeError, ValueError) as exc:
                raise ValueError("recovery portfolio branch liveness is invalid") from exc
            effective_status = status
            if status in {"active", "selected"} and current_tick > expires_at_tick:
                effective_status = "expired"
                liveness_due.append(str(raw_branch.get("branch_id", "")))
            counts[effective_status] += 1
            branches.append(
                {
                    "branch_id": str(raw_branch.get("branch_id", "")),
                    "loop_id": str(raw_branch.get("loop_id", "")),
                    "parent_loop_id": str(raw_branch.get("parent_loop_id", "")),
                    "status": effective_status,
                    "persisted_status": status,
                    "capability_id": str(raw_branch.get("capability_id", "")),
                    "source_evidence_id": str(raw_branch.get("source_evidence_id", "")),
                    "source_after_state_digest": str(
                        raw_branch.get("source_after_state_digest", "")
                    ),
                    "budget_limit": float(raw_branch.get("budget_limit", 0.0)),
                    "budget_units": float(raw_branch.get("budget_units", 0.0)),
                    "completed_steps": int(raw_branch.get("completed_steps", 0)),
                    "frontier_affordance_ids": [
                        str(value) for value in raw_branch.get("frontier_affordance_ids", ())
                    ],
                    "created_tick": created_tick,
                    "last_touched_tick": last_touched_tick,
                    "expires_at_tick": expires_at_tick,
                    "terminal_reason": str(raw_branch.get("terminal_reason", "")),
                }
            )

        evicted: list[dict[str, Any]] = []
        for raw_branch in raw_evicted:
            if not isinstance(raw_branch, Mapping):
                raise ValueError("recovery portfolio evicted branch is invalid")
            evicted.append(
                {
                    "branch_id": str(raw_branch.get("branch_id", "")),
                    "loop_id": str(raw_branch.get("loop_id", "")),
                    "source_evidence_id": str(raw_branch.get("source_evidence_id", "")),
                    "source_after_state_digest": str(
                        raw_branch.get("source_after_state_digest", "")
                    ),
                    "status": "evicted",
                    "evicted_tick": int(raw_branch.get("evicted_tick", 0)),
                    "reason": str(raw_branch.get("reason", "")),
                }
            )
        successor_graph = self._workbench_loop_state.get("successor_graph")
        selected_branch_id = ""
        if isinstance(successor_graph, Mapping) and str(
            successor_graph.get("parent_loop_id", "")
        ) == str(parent_loop_id):
            selected_branch_id = str(successor_graph.get("recovery_branch_id", ""))
        return {
            "format": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            "version": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
            "status": "portfolio_snapshot",
            "parent_loop_id": str(parent_loop_id),
            "snapshot_id": str(snapshot_id),
            "revision": revision,
            "current_tick": current_tick,
            "max_branches": max_branches,
            "branch_ttl_ticks": ttl_ticks,
            "last_maintenance_tick": int(portfolio.get("last_maintenance_tick", 0)),
            "selected_branch_id": selected_branch_id,
            "counts": {**counts, "evicted": len(evicted)},
            "liveness_due_branch_ids": liveness_due,
            "branches": branches,
            "evicted_branches": evicted,
        }

    def taiji_workbench_recovery_portfolio_context(self) -> dict[str, Any]:
        """Return the read-only lineage binding key for the client audit view.

        plans/active/roadmap/04_EXECUTION_PLAN.md §2.1：客户端不得用输入框、
        固定 loop id 或「最近一次」猜测去绑定 recovery ledger；这里把持久化的
        parent loop / snapshot / revision 以只读投影形式作为唯一绑定来源。
        无 portfolio 时返回结构化空态，而不是错误。
        """

        portfolio = self._workbench_loop_state.get("recovery_portfolio")
        if not isinstance(portfolio, Mapping):
            return {"status": "portfolio_context", "has_portfolio": False}
        successor_graph = self._workbench_loop_state.get("successor_graph")
        selected_branch_id = ""
        if isinstance(successor_graph, Mapping):
            selected_branch_id = str(successor_graph.get("recovery_branch_id", ""))
        return {
            "status": "portfolio_context",
            "has_portfolio": True,
            "parent_loop_id": str(portfolio.get("parent_loop_id", "")),
            "snapshot_id": str(portfolio.get("snapshot_id", "")),
            "revision": int(portfolio.get("revision", 0)),
            "selected_branch_id": selected_branch_id,
        }

    @_workbench_synchronized
    def register_taiji_workbench_recovery_branch(
        self,
        *,
        parent_loop_id: str,
        recovery_loop_id: str,
        snapshot_id: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Persist another compatible recovery evidence branch without executing it."""

        import hashlib

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_EVIDENCE_KIND,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
        )

        if not str(parent_loop_id).strip() or not str(recovery_loop_id).strip():
            raise ValueError("recovery branch parent and loop ids cannot be empty")
        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("recovery branch capability snapshot is not current")
        portfolio = self._workbench_loop_state.get("recovery_portfolio")
        if not isinstance(portfolio, Mapping):
            raise RuntimeError("recovery branch registration requires a recovery portfolio")
        if str(portfolio.get("parent_loop_id", "")) != str(parent_loop_id):
            raise ValueError("recovery branch parent does not match the portfolio")
        self._require_recovery_portfolio_revision(portfolio, expected_revision)
        world = self.model.architecture.cognitive_snapshot().world
        portfolio, maintenance = self._maintain_taiji_workbench_recovery_portfolio(
            portfolio,
            current_tick=world.tick,
            reserve_slots=1,
        )
        self._workbench_loop_state["recovery_portfolio"] = portfolio
        if maintenance["changed"]:
            try:
                self.save()
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError("recovery portfolio capacity checkpoint failed") from exc
        max_branches = int(portfolio["max_branches"])
        branches = portfolio.get("branches", ())
        if len(branches) >= max_branches:
            raise RuntimeError("recovery portfolio capacity exhausted")
        known_loop_ids = {str(item) for item in portfolio.get("retired_loop_ids", ())}
        known_event_ids: set[str] = set()
        for item in branches:
            if not isinstance(item, Mapping):
                raise ValueError("recovery portfolio branch is invalid")
            known_loop_ids.add(str(item.get("loop_id", "")))
            known_loop_ids.update(str(loop) for loop in item.get("retired_loop_ids", ()))
            known_event_ids.add(str(item.get("source_evidence_id", "")))
            known_event_ids.update(str(event) for event in item.get("event_ids", ()))
        if str(recovery_loop_id) in known_loop_ids:
            raise ValueError("recovery branch loop identity already exists")

        failure = portfolio.get("parent_failure")
        if not isinstance(failure, Mapping):
            raise RuntimeError("recovery portfolio has no parent failure context")
        event = next(
            (
                item
                for item in reversed(world.events)
                if item.kind == WORKBENCH_TAIJI_EVIDENCE_KIND
                and item.event_id not in known_event_ids
            ),
            None,
        )
        evidence, affordances = self._validate_recovery_source_evidence(
            event,
            failure=failure,
            environment=environment,
            current_tick=world.tick,
        )
        branch_id = (
            "recovery-branch:"
            + hashlib.sha256(
                f"{parent_loop_id}:{evidence.evidence_id}:{evidence.after_state_digest}".encode()
            ).hexdigest()[:32]
        )
        if any(
            isinstance(item, Mapping) and str(item.get("branch_id", "")) == branch_id
            for item in branches
        ):
            raise ValueError("recovery branch evidence already exists in the portfolio")
        branch = {
            "branch_id": branch_id,
            "loop_id": str(recovery_loop_id),
            "parent_loop_id": str(parent_loop_id),
            "source_evidence_id": evidence.evidence_id,
            "source_after_state_digest": evidence.after_state_digest,
            "capability_id": evidence.capability_id,
            "parameters": dict(evidence.parameters),
            "status": "active",
            "budget_limit": float(portfolio.get("parent_budget_limit", 0.0)),
            "budget_units": float(portfolio.get("parent_budget_units", 0.0)),
            "completed_steps": int(portfolio.get("parent_completed_steps", 0)),
            "committed_request_ids": [
                str(item) for item in portfolio.get("parent_committed_request_ids", ())
            ],
            "consumed_affordance_ids": [
                str(item) for item in portfolio.get("parent_consumed_affordance_ids", ())
            ],
            "event_ids": [str(item) for item in portfolio.get("parent_event_ids", ())],
            "evidence": evidence.to_payload(),
            "frontier_affordance_ids": [item.affordance_id for item in affordances],
            "created_tick": int(world.tick),
            "last_touched_tick": int(world.tick),
            "expires_at_tick": int(world.tick) + int(portfolio["branch_ttl_ticks"]),
        }
        updated_portfolio = dict(portfolio)
        updated_portfolio["format"] = WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT
        updated_portfolio["version"] = WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION
        updated_portfolio["branches"] = [*branches, branch]
        updated_portfolio["last_maintenance_tick"] = int(world.tick)
        updated_portfolio["revision"] = int(portfolio.get("revision", 0)) + 1
        self._workbench_loop_state["recovery_portfolio"] = updated_portfolio
        checkpoint = {"committed": True, "path": str(self.save())}
        return {
            "format": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            "version": WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
            "status": "branch_registered",
            "parent_loop_id": str(parent_loop_id),
            "branch": branch,
            "portfolio": updated_portfolio,
            "maintenance": maintenance,
            "checkpoint": checkpoint,
        }

    @_workbench_synchronized
    def select_taiji_workbench_recovery_branch(
        self,
        *,
        parent_loop_id: str,
        branch_id: str,
        recovery_loop_id: str,
        snapshot_id: str,
        max_steps: int = 8,
        max_budget_units: float = 32.0,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
        learn: bool = False,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Select one active recovery branch and execute it as a new loop."""

        from seed_platform.workbench import (
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT,
            WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION,
            WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
            WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
        )

        if not str(branch_id).strip() or not str(recovery_loop_id).strip():
            raise ValueError("recovery branch and loop ids cannot be empty")
        environment = self._sync_workbench_root()
        if str(snapshot_id) != environment.capability_snapshot.snapshot_id:
            raise ValueError("recovery branch capability snapshot is not current")
        portfolio = self._workbench_loop_state.get("recovery_portfolio")
        if not isinstance(portfolio, Mapping):
            raise RuntimeError("recovery branch selection requires a recovery portfolio")
        if str(portfolio.get("parent_loop_id", "")) != str(parent_loop_id):
            raise ValueError("recovery branch parent does not match the portfolio")
        self._require_recovery_portfolio_revision(portfolio, expected_revision)
        world = self.model.architecture.cognitive_snapshot().world
        portfolio, maintenance = self._maintain_taiji_workbench_recovery_portfolio(
            portfolio,
            current_tick=world.tick,
        )
        self._workbench_loop_state["recovery_portfolio"] = portfolio
        if maintenance["changed"]:
            try:
                self.save()
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError("recovery portfolio liveness checkpoint failed") from exc
        branches = portfolio.get("branches", ())
        if isinstance(branches, (str, bytes)) or not isinstance(branches, Sequence):
            raise ValueError("recovery portfolio branches are invalid")
        branch = next(
            (
                dict(item)
                for item in branches
                if isinstance(item, Mapping) and str(item.get("branch_id", "")) == str(branch_id)
            ),
            None,
        )
        if branch is None:
            raise ValueError("recovery branch is not in the portfolio")
        if branch.get("status") != "active":
            raise RuntimeError("recovery branch is not active")
        if any(
            str(item.get("loop_id", "")) == str(recovery_loop_id)
            for item in branches
            if isinstance(item, Mapping)
        ) or str(recovery_loop_id) in {str(item) for item in portfolio.get("retired_loop_ids", ())}:
            raise ValueError("recovery branch selection loop identity already exists")
        failure = portfolio.get("parent_failure")
        if not isinstance(failure, Mapping):
            raise RuntimeError("recovery portfolio has no parent failure context")
        event = next(
            (
                item
                for item in world.events
                if item.event_id == str(branch.get("source_evidence_id", ""))
            ),
            None,
        )
        evidence, affordances = self._validate_recovery_source_evidence(
            event,
            failure=failure,
            environment=environment,
            current_tick=world.tick,
        )
        if evidence.after_state_digest != str(branch.get("source_after_state_digest", "")):
            raise ValueError("recovery branch evidence content drifted")
        consumed = {str(item) for item in portfolio.get("parent_consumed_affordance_ids", ())}
        if any(item.affordance_id in consumed for item in affordances):
            raise RuntimeError("recovery branch reintroduced a consumed affordance")
        current_state = self._workbench_loop_state.get("successor_graph")
        if isinstance(current_state, Mapping) and isinstance(
            current_state.get("in_flight"), Mapping
        ):
            raise RuntimeError("cannot select a recovery branch while another step is in flight")
        retired_loop_ids = [str(item) for item in portfolio.get("retired_loop_ids", ())]
        if isinstance(current_state, Mapping):
            current_loop_id = str(current_state.get("loop_id", ""))
            if current_loop_id and current_loop_id != str(parent_loop_id):
                retired_loop_ids.append(current_loop_id)
        retired_loop_ids = list(dict.fromkeys([*retired_loop_ids, str(parent_loop_id)]))
        recovery_state = {
            "format": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT,
            "version": WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION,
            "loop_id": str(recovery_loop_id),
            "parent_loop_id": str(parent_loop_id),
            "recovery_branch_id": str(branch_id),
            "recovery_branch_source_evidence_id": evidence.evidence_id,
            "snapshot_id": str(snapshot_id),
            "budget_limit": float(branch.get("budget_limit", 0.0)),
            "budget_units": float(branch.get("budget_units", 0.0)),
            "completed_steps": int(branch.get("completed_steps", 0)),
            "committed_request_ids": [
                str(item) for item in branch.get("committed_request_ids", ())
            ],
            "consumed_affordance_ids": [
                str(item) for item in branch.get("consumed_affordance_ids", ())
            ],
            "event_ids": [str(item) for item in branch.get("event_ids", ())],
            "frontier_affordance_ids": [item.affordance_id for item in affordances],
            "remaining_frontier_affordance_ids": [item.affordance_id for item in affordances],
            "latest_evidence_id": evidence.evidence_id,
            "failure": None,
            "retired_loop_ids": retired_loop_ids,
            "recovery": {
                "parent_loop_id": str(parent_loop_id),
                "parent_failure": dict(failure),
                "branch_id": str(branch_id),
                "source_evidence_id": evidence.evidence_id,
                "source_after_state_digest": evidence.after_state_digest,
            },
            "status": "running",
        }
        world = self.model.architecture.set_world_affordances(affordances)
        updated_portfolio = dict(portfolio)
        updated_portfolio["format"] = WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT
        updated_portfolio["version"] = WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION
        updated_portfolio["retired_loop_ids"] = retired_loop_ids
        updated_branches = []
        for item in branches:
            if isinstance(item, Mapping) and str(item.get("branch_id", "")) == str(branch_id):
                updated = dict(branch)
                updated["loop_id"] = str(recovery_loop_id)
                updated["status"] = "selected"
                updated["last_touched_tick"] = int(world.tick)
                updated["expires_at_tick"] = int(world.tick) + int(
                    updated_portfolio["branch_ttl_ticks"]
                )
                updated_branches.append(updated)
            else:
                updated_branches.append(dict(item))
        updated_portfolio["branches"] = updated_branches
        self._workbench_loop_state = {
            **self._workbench_loop_state,
            "successor_graph": recovery_state,
            "recovery_portfolio": updated_portfolio,
        }
        result = self.execute_taiji_workbench_successor_loop(
            snapshot_id=snapshot_id,
            loop_id=recovery_loop_id,
            max_steps=max_steps,
            max_budget_units=max_budget_units,
            novelty=novelty,
            resource_budget=resource_budget,
            learn=learn,
        )
        active_state = self._workbench_loop_state.get("successor_graph")
        for item in updated_branches:
            if str(item.get("branch_id", "")) == str(branch_id):
                item["status"] = (
                    "completed"
                    if isinstance(active_state, Mapping)
                    and active_state.get("status") == "completed"
                    else (
                        "failed"
                        if isinstance(active_state, Mapping)
                        and active_state.get("status") in {"recovery_needed", "checkpoint_failed"}
                        else "active"
                    )
                )
                if isinstance(active_state, Mapping):
                    item["budget_units"] = float(active_state.get("budget_units", 0.0))
                    item["completed_steps"] = int(active_state.get("completed_steps", 0))
                    item["frontier_affordance_ids"] = list(
                        active_state.get("frontier_affordance_ids", ())
                    )
                    item["committed_request_ids"] = [
                        str(value) for value in active_state.get("committed_request_ids", ())
                    ]
                    item["consumed_affordance_ids"] = [
                        str(value) for value in active_state.get("consumed_affordance_ids", ())
                    ]
                    item["event_ids"] = [str(value) for value in active_state.get("event_ids", ())]
                    branch_tick = int(self.model.architecture.cognitive_snapshot().world.tick)
                    item["last_touched_tick"] = branch_tick
                    item["expires_at_tick"] = (
                        branch_tick + int(updated_portfolio["branch_ttl_ticks"])
                        if item["status"] == "active"
                        else branch_tick
                    )
        updated_portfolio["branches"] = updated_branches
        updated_portfolio["revision"] = int(portfolio.get("revision", 0)) + 1
        self._workbench_loop_state["recovery_portfolio"] = updated_portfolio
        try:
            self.save()
        except (OSError, RuntimeError, ValueError) as exc:
            result["recovery_portfolio_checkpoint"] = {
                "committed": False,
                "error_code": "checkpoint_failed",
                "error": str(exc),
            }
        else:
            result["recovery_portfolio_checkpoint"] = {"committed": True}
        result["recovery_portfolio"] = updated_portfolio
        result["recovery"] = dict(recovery_state["recovery"])
        return result

    def _ingest_workbench_structural_evidence(
        self,
        outcome: Any,
        payload: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Bind explicit evaluator metrics to the real executed Workbench outcome."""

        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("structural_evidence must be a mapping")
        required = (
            "network_id",
            "region_id",
            "task_slice_id",
            "partition",
            "usage",
            "resource_pressure",
            "prediction_error",
            "learning_gain",
            "holdout_transfer",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"structural_evidence is missing fields: {missing}")
        from seed_platform.workbench import WorkbenchOutcome, WorkbenchStructuralEvidence

        if not isinstance(outcome, WorkbenchOutcome):
            raise TypeError("structural evidence must bind to a WorkbenchOutcome")
        evidence = WorkbenchStructuralEvidence.from_outcome(
            outcome,
            task_slice_id=str(payload["task_slice_id"]),
            partition=str(payload["partition"]),
            usage=float(payload["usage"]),
            resource_pressure=float(payload["resource_pressure"]),
            prediction_error=(
                None
                if payload["prediction_error"] is None
                else float(payload["prediction_error"])
            ),
            learning_gain=float(payload["learning_gain"]),
            holdout_transfer=float(payload["holdout_transfer"]),
        )
        observation = evidence.to_structural_observation(
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            tick=int(self.model.architecture.structural_runtime_tick) + 1,
        )
        append = self.model.architecture.record_structural_runtime_observation(observation)
        return {
            "evidence": evidence.to_payload(),
            "observation": observation.to_payload(),
            "append": {
                "evidence_id": append.evidence_id,
                "status": append.status,
                "window_id": append.window_id,
                "sealed_window_digest": append.sealed_window_digest,
            },
        }

    def schedule_structural_growth_from_workbench_evidence(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the bounded scheduler over already sealed Workbench evidence."""

        result = self.model.architecture.schedule_structural_growth_from_evidence(**kwargs)
        return result.to_payload()

    def schedule_structural_candidate_batch_from_workbench_evidence(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Create one deterministic structural batch from multiple evidence regions."""

        result = self.model.architecture.schedule_structural_candidate_batch_from_workbench_evidence(
            requests
        )
        return result.to_payload()

    def continue_structural_candidate(
        self,
        candidate_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Continue one scheduler candidate through the existing Taiji lifecycle."""

        return self.model.architecture.continue_structural_candidate(candidate_id, **kwargs)

    @_workbench_synchronized
    def continue_structural_candidate_from_validation_artifact(
        self,
        artifact: Mapping[str, Any] | Any,
        *,
        holdout_inputs: Sequence[Any],
        expected_activities: Sequence[Any],
    ) -> dict[str, Any]:
        """Consume a replay-bound validation artifact through Taiji's lifecycle."""

        from taiji import WorkbenchStructuralValidationArtifact

        resolved = (
            WorkbenchStructuralValidationArtifact.from_payload(artifact)
            if isinstance(artifact, Mapping)
            else artifact
        )
        return self.model.architecture.continue_structural_candidate_from_validation_artifact(
            resolved,
            holdout_inputs=holdout_inputs,
            expected_activities=expected_activities,
        )

    def arbitrate_structural_candidate_batch(
        self,
        candidate_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Create a deterministic, checkpointable batch reservation for candidates."""

        batch = self.model.architecture.arbitrate_structural_candidate_batch(candidate_ids)
        return batch.to_payload()

    def continue_structural_candidate_batch(
        self,
        batch_id: str,
        *,
        continuations_by_candidate: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Continue a reserved candidate batch without owning structural policy."""

        return self.model.architecture.continue_structural_candidate_batch(
            batch_id,
            continuations_by_candidate=continuations_by_candidate,
        )

    @_workbench_synchronized
    def continue_structural_candidate_batch_from_validation_artifacts(
        self,
        batch_id: str,
        *,
        artifacts_by_candidate: Mapping[str, Any],
        replays_by_candidate: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Continue a batch using replay-bound artifacts without manual metrics."""

        return self.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
            batch_id,
            artifacts_by_candidate=artifacts_by_candidate,
            replays_by_candidate=replays_by_candidate,
        )

    @_workbench_synchronized
    def continue_structural_candidate_batch_from_artifact_store(
        self,
        batch_id: str,
        *,
        artifact_store: Any,
        artifact_digests_by_candidate: Mapping[str, str],
        replays_by_candidate: Mapping[str, Mapping[str, Any]],
        artifact_consumption_policy: Any | None = None,
        require_verified_measurements: bool | None = None,
    ) -> dict[str, Any]:
        """Resolve external artifacts under one explicit policy before consumption.

        ``require_verified_measurements`` remains only as a compatibility
        shim for S51 callers.  It is converted immediately into a
        reason-bearing :class:`ArtifactConsumptionPolicy`; new callers should
        pass ``artifact_consumption_policy`` or configure the checkpointed
        Taiji default.
        """

        from taiji import (
            ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY,
            ArtifactConsumptionAudit,
            StructuralValidationArtifactStore,
        )

        if not isinstance(artifact_store, StructuralValidationArtifactStore):
            raise TypeError("artifact_store must be a StructuralValidationArtifactStore")
        if not isinstance(artifact_digests_by_candidate, Mapping):
            raise TypeError("artifact digests must be a mapping")
        if not isinstance(replays_by_candidate, Mapping):
            raise TypeError("validation replays must be a mapping")
        policy = self.model.architecture.resolve_artifact_consumption_policy(
            artifact_consumption_policy,
            require_verified_measurements=require_verified_measurements,
        )
        batch = next(
            (
                item
                for item in self.model.architecture.structural_candidate_batches
                if item.batch_id == str(batch_id)
            ),
            None,
        )
        if batch is None:
            raise ValueError(f"unknown structural candidate batch: {batch_id}")
        statuses: dict[str, str] = {}

        def record_audit(*, result: str, error_code: str = "") -> dict[str, Any]:
            audit = ArtifactConsumptionAudit.create(
                str(batch_id),
                policy,
                statuses,
                result=result,
                error_code=error_code,
            )
            self._last_artifact_consumption_audit = audit.to_payload()
            return audit.to_payload()

        selected_candidate_ids = set(batch.selected_candidate_ids)
        unknown_keys = tuple(
            sorted(
                {
                    str(candidate_id)
                    for candidate_id in (
                        *artifact_digests_by_candidate.keys(),
                        *replays_by_candidate.keys(),
                    )
                    if str(candidate_id) not in selected_candidate_ids
                }
            )
        )
        if unknown_keys:
            for candidate_id in unknown_keys:
                statuses[candidate_id] = "rejected"
            record_audit(result="rejected", error_code="unknown_candidate")
            raise ValueError(
                "artifact store bridge contains candidates outside the selected batch: "
                f"{unknown_keys}"
            )
        artifacts_by_candidate: dict[str, Any] = {}
        loaded_artifacts: dict[str, Any] = {}
        try:
            for candidate_id, artifact_digest in artifact_digests_by_candidate.items():
                candidate_key = str(candidate_id)
                if not isinstance(artifact_digest, str):
                    statuses[candidate_key] = "rejected"
                    raise TypeError("artifact digest references must be strings")
                artifact = artifact_store.load(artifact_digest)
                loaded_artifacts[candidate_key] = artifact
                if policy.mode == ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY:
                    if not artifact.measurement_digest:
                        statuses[candidate_key] = "legacy_unverified"
                        raise ValueError(
                            "verified-only policy rejects an artifact without a measurement sidecar"
                        )
                    artifact = artifact_store.load_verified_artifact(artifact_digest)
                    statuses[candidate_key] = "verified"
                else:
                    artifact = artifact_store.load(artifact_digest)
                    if not artifact.measurement_digest:
                        statuses[candidate_key] = "legacy_unverified"
                    elif artifact_store.contains_measurement(artifact.measurement_digest):
                        artifact_store.load_measurements(artifact.measurement_digest)
                        statuses[candidate_key] = "verified"
                    else:
                        statuses[candidate_key] = "legacy_unverified"
                artifacts_by_candidate[candidate_key] = artifact
        except FileNotFoundError:
            failed_key = next(
                (
                    str(candidate_id)
                    for candidate_id in artifact_digests_by_candidate
                    if str(candidate_id) not in artifacts_by_candidate
                ),
                "unknown",
            )
            statuses[failed_key] = (
                "legacy_unverified"
                if failed_key in loaded_artifacts
                and loaded_artifacts[failed_key].measurement_digest
                else "missing"
            )
            record_audit(result="rejected", error_code="artifact_missing")
            raise
        except (KeyError, TypeError, ValueError):
            failed_key = next(
                (
                    str(candidate_id)
                    for candidate_id in artifact_digests_by_candidate
                    if str(candidate_id) not in artifacts_by_candidate
                ),
                "unknown",
            )
            statuses.setdefault(failed_key, "tampered")
            record_audit(result="rejected", error_code="artifact_rejected")
            raise
        try:
            result = self.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
                str(batch_id),
                artifacts_by_candidate=artifacts_by_candidate,
                replays_by_candidate=replays_by_candidate,
            )
        except (TypeError, ValueError, RuntimeError):
            audit = record_audit(result="rejected", error_code="native_consumption_rejected")
            raise
        audit = record_audit(result="consumed")
        return {**result, "artifact_consumption": audit}

    @_workbench_synchronized
    def project_structural_artifact_store_audit(
        self,
        *,
        artifact_store: Any,
    ) -> dict[str, Any]:
        """Project external artifact facts without registering or consuming them."""

        from taiji import (
            STRUCTURAL_ARTIFACT_STORE_FORMAT,
            STRUCTURAL_ARTIFACT_STORE_PROJECTION_FORMAT,
            StructuralValidationArtifactStore,
            structural_artifact_store_audit_digest,
        )

        if not isinstance(artifact_store, StructuralValidationArtifactStore):
            raise TypeError("artifact_store must be a StructuralValidationArtifactStore")

        architecture = self.model.architecture
        runtime_artifact_digests = {
            artifact.artifact_digest
            for artifact in architecture.structural_validation_artifacts
        }
        batch_ids_by_digest: dict[str, set[str]] = {}
        for artifact_batch in architecture.structural_validation_artifact_batches:
            for _, artifact_digest in artifact_batch.artifact_digests_by_candidate:
                batch_ids_by_digest.setdefault(str(artifact_digest), set()).add(
                    artifact_batch.batch_id
                )

        inventory = tuple(artifact_store.audit())
        external_digests = {str(item["artifact_digest"]) for item in inventory}
        runtime_batch_artifact_digests = set(batch_ids_by_digest)
        entries: list[dict[str, Any]] = []
        for item in inventory:
            artifact_digest = str(item["artifact_digest"])
            batch_ids = tuple(sorted(batch_ids_by_digest.get(artifact_digest, ())))
            if artifact_digest in runtime_artifact_digests:
                visibility = "runtime_recorded"
            elif batch_ids:
                visibility = "runtime_batch_referenced"
            else:
                visibility = "external_orphan"
            entries.append(
                {
                    **item,
                    "runtime_visibility": visibility,
                    "runtime_batch_ids": list(batch_ids),
                }
            )

        payload = {
            "format": STRUCTURAL_ARTIFACT_STORE_PROJECTION_FORMAT,
            "store_format": STRUCTURAL_ARTIFACT_STORE_FORMAT,
            "entries": entries,
            "runtime_artifact_digests": sorted(runtime_artifact_digests),
            "runtime_batch_artifact_digests": sorted(runtime_batch_artifact_digests),
            "missing_runtime_artifact_digests": sorted(
                runtime_artifact_digests - external_digests
            ),
            "missing_runtime_batch_artifact_digests": sorted(
                runtime_batch_artifact_digests - external_digests
            ),
        }
        return {
            **payload,
            "audit_digest": structural_artifact_store_audit_digest(payload),
        }

    @_workbench_synchronized
    def run_structural_maintenance_cycle(
        self,
        *,
        holdout_inputs_by_candidate: Mapping[str, Sequence[Any]],
        expected_activities_by_candidate: Mapping[str, Sequence[Any]],
        candidate_ids: Sequence[str] | None = None,
        lineage_retention_max_batches: int | None = None,
        lineage_retention_policy: Any | None = None,
    ) -> dict[str, Any]:
        """Expose one explicit, content-addressed structural maintenance audit.

        SeedRuntime only projects the result.  Taiji retains ownership of the
        candidate lifecycle, retention policy, topology, and structural budget.
        No background maintenance is started by this method.
        """

        from taiji import (
            STRUCTURAL_MAINTENANCE_AUDIT_FORMAT,
            StructuralMaintenanceAudit,
            structural_maintenance_audit_digest,
        )

        architecture = self.model.architecture
        previous_retention = architecture.structural_lineage_retention_result
        previous_policy = architecture.structural_lineage_retention_policy
        results = architecture.run_structural_maintenance_cycle(
            holdout_inputs_by_candidate=holdout_inputs_by_candidate,
            expected_activities_by_candidate=expected_activities_by_candidate,
            candidate_ids=candidate_ids,
            lineage_retention_max_batches=lineage_retention_max_batches,
            lineage_retention_policy=lineage_retention_policy,
        )
        current_retention = architecture.structural_lineage_retention_result
        current_policy = architecture.structural_lineage_retention_policy
        # Project only the audit produced by this call.  A default candidate
        # maintenance call must not masquerade as a fresh retention action by
        # replaying an older audit already restored from checkpoint.
        retention = (
            current_retention
            if current_retention is not previous_retention
            else None
        )
        policy = current_policy if current_policy is not previous_policy else None
        payload = {
            "format": STRUCTURAL_MAINTENANCE_AUDIT_FORMAT,
            "maintenance_results": [item.to_payload() for item in results],
            "lineage_retention": None if retention is None else retention.to_payload(),
            "retention_policy": None if policy is None else policy.to_payload(),
            "structural_runtime_tick": architecture.structural_runtime_tick,
        }
        audit = StructuralMaintenanceAudit(
            maintenance_results=results,
            lineage_retention=retention,
            structural_runtime_tick=architecture.structural_runtime_tick,
            audit_digest=structural_maintenance_audit_digest(payload),
            retention_policy=policy,
        )
        return audit.to_payload()

    @_workbench_synchronized
    def migrate_structural_lineage_retention_policy(
        self,
        target_policy: Any,
    ) -> dict[str, Any]:
        """Commit one explicit adjacent retention-policy migration."""

        migration = self.model.architecture.migrate_structural_lineage_retention_policy(
            target_policy
        )
        return migration.to_payload()

    @_workbench_synchronized
    def rollback_structural_lineage_retention_policy_migration(
        self,
        migration: Any,
    ) -> dict[str, Any]:
        """Roll back one explicit committed retention-policy migration."""

        rolled_back = self.model.architecture.rollback_structural_lineage_retention_policy_migration(
            migration
        )
        return rolled_back.to_payload()

    def measure_structural_capacity_pressure(
        self,
        region_id: str,
        *,
        capacity_limit: int,
    ) -> dict[str, Any]:
        """Expose an explicit read-only capacity pressure measurement."""

        return self.model.architecture.measure_structural_capacity_pressure(
            region_id,
            capacity_limit=capacity_limit,
        ).to_payload()

    def rollback_structural_candidate_batch(
        self,
        batch_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        """Reverse one admitted batch candidate through Taiji's rollback ledger."""

        return self.model.architecture.rollback_structural_candidate_batch(
            batch_id,
            candidate_id,
        )

    def execute_workbench_intent(
        self,
        intent: Any,
        *,
        snapshot_id: str,
        approval_token: str = "",
        mcp_registry_snapshot_id: str = "",
        capability_registry_snapshot_id: str = "",
        structural_evidence: Mapping[str, Any] | None = None,
        learn: bool = False,
        event_sink: Callable[[Any], None] | None = None,
        executive_decision: Any | None = None,
    ) -> dict[str, Any]:
        """Execute one Taiji-owned intent through Seed's workbench."""

        from seed_platform.workbench import (
            ExecutionPolicyDecision,
            WorkbenchActionRequest,
            WorkbenchOutcome,
            WorkbenchTransaction,
        )
        from taiji import ActionIntent

        if not isinstance(intent, ActionIntent):
            raise TypeError("workbench execution requires an ActionIntent")
        environment = self._sync_workbench_root()
        architecture = self.model.architecture
        source_affordance = None
        affordance_context = None
        if executive_decision is not None:
            if architecture.last_executive_decision is not executive_decision:
                raise ValueError("workbench learning requires the current executive decision")
            if executive_decision.action_intent.intent_id != intent.intent_id:
                raise ValueError("workbench learning intent does not match the selected decision")
            source_affordance_id = str(executive_decision.selected.source_affordance_id or "")
            if not source_affordance_id:
                raise ValueError("workbench learning requires a selected source affordance")
            world_before_execution = architecture.cognitive_snapshot().world
            source_affordance = next(
                (
                    item
                    for item in world_before_execution.affordances
                    if item.affordance_id == source_affordance_id
                ),
                None,
            )
            if source_affordance is None:
                raise ValueError(
                    "workbench learning source affordance does not match the selected decision"
                )
            affordance_context = architecture._affordance_context()

        def append_event(
            phase: str,
            request_id: str,
            *,
            tick: int,
            payload: Mapping[str, Any],
        ) -> None:
            event = self._workbench_audit.append(
                phase,
                request_id,
                tick=tick,
                payload=payload,
            )
            if event_sink is not None:
                try:
                    event_sink(event)
                except Exception:  # pragma: no cover - observer must not break execution
                    logger.exception("workbench event sink failed; execution continues")

        request = WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=snapshot_id,
            approval_token=approval_token,
            mcp_registry_snapshot_id=(
                mcp_registry_snapshot_id
                or (
                    environment.mcp_registry.snapshot_id
                    if str(intent.kind).startswith("mcp.")
                    else ""
                )
            ),
            capability_registry_snapshot_id=(
                capability_registry_snapshot_id or environment.capability_registry.snapshot_id
            ),
        )
        tick = int(self.model.tick)
        append_event(
            "planned",
            request.request_id,
            tick=tick,
            payload={"request": request.to_payload()},
        )
        policy = environment.policy_for(request)
        append_event(
            "policy",
            request.request_id,
            tick=tick,
            payload={"policy": policy.to_payload()},
        )
        if policy.decision != "allow":
            workbench_outcome = WorkbenchOutcome(
                request_id=request.request_id,
                intent_id=request.intent_id,
                call_id="",
                capability_id=request.capability_id,
                snapshot_id=environment.capability_snapshot.snapshot_id,
                status="rejected",
                success=False,
                error_code=policy.reason_code,
                error="workbench action was not admitted",
                tick=tick,
                mcp_registry_snapshot_id=request.mcp_registry_snapshot_id,
            )
            append_event(
                "outcome",
                request.request_id,
                tick=tick,
                payload={"outcome": workbench_outcome.to_payload()},
            )
            return {
                "request": request.to_payload(),
                "policy": policy.to_payload(),
                "outcome": workbench_outcome.to_payload(),
                "taiji_outcome": None,
                "events": [event.to_payload() for event in self._workbench_audit.events],
            }

        try:
            environment.consume_approval(request)
        except ValueError as exc:
            policy = ExecutionPolicyDecision(
                request_id=request.request_id,
                capability_id=request.capability_id,
                snapshot_id=request.snapshot_id,
                decision="ask_user",
                reason_code="approval_invalid",
            )
            workbench_outcome = WorkbenchOutcome(
                request_id=request.request_id,
                intent_id=request.intent_id,
                call_id="",
                capability_id=request.capability_id,
                snapshot_id=environment.capability_snapshot.snapshot_id,
                status="rejected",
                success=False,
                error_code="approval_invalid",
                error=str(exc),
                tick=tick,
                mcp_registry_snapshot_id=request.mcp_registry_snapshot_id,
            )
            append_event(
                "outcome",
                request.request_id,
                tick=tick,
                payload={"outcome": workbench_outcome.to_payload()},
            )
            return {
                "request": request.to_payload(),
                "policy": policy.to_payload(),
                "outcome": workbench_outcome.to_payload(),
                "taiji_outcome": None,
                "events": [event.to_payload() for event in self._workbench_audit.events],
            }

        append_event(
            "executing",
            request.request_id,
            tick=tick,
            payload={
                "capability_id": request.capability_id,
                "approval_granted": bool(request.approval_token),
            },
        )
        with environment.request_context(request.request_id):
            try:
                call, taiji_outcome = self.model.architecture.execute_tool_intent(
                    intent,
                    environment,
                    learn=learn,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                workbench_outcome = WorkbenchOutcome(
                    request_id=request.request_id,
                    intent_id=request.intent_id,
                    call_id="",
                    capability_id=request.capability_id,
                    snapshot_id=environment.capability_snapshot.snapshot_id,
                    status="error",
                    success=False,
                    error_code="taiji_execution_error",
                    error=str(exc),
                    tick=int(self.model.tick),
                    mcp_registry_snapshot_id=request.mcp_registry_snapshot_id,
                )
                append_event(
                    "outcome",
                    request.request_id,
                    tick=int(self.model.tick),
                    payload={"outcome": workbench_outcome.to_payload()},
                )
                raise

        result = environment.last_result
        transaction_payload = result.get("transaction")
        if isinstance(transaction_payload, Mapping):
            transaction = WorkbenchTransaction.from_payload(transaction_payload)
        else:
            # Preserve the legacy read-only outcome shape for capabilities
            # that do not produce a real file transaction.
            transaction = WorkbenchTransaction(
                operation=request.capability_id,
                path=str(result.get("path", request.parameters.get("path", "."))),
                before_digest=str(result.get("digest", "")),
                after_digest=str(result.get("digest", "")),
                reversible=True,
            )
        workbench_outcome = WorkbenchOutcome(
            request_id=request.request_id,
            intent_id=request.intent_id,
            call_id=call.call_id,
            capability_id=request.capability_id,
            snapshot_id=environment.capability_snapshot.snapshot_id,
            status="success" if taiji_outcome.success is not False else "error",
            success=taiji_outcome.success is not False,
            result=result,
            error_code=str(result.get("error_code", "")),
            error=str(result.get("error", "")),
            transaction=transaction,
            tick=int(taiji_outcome.tick),
            mcp_registry_snapshot_id=request.mcp_registry_snapshot_id,
        )
        structural_evidence_result = self._ingest_workbench_structural_evidence(
            workbench_outcome,
            structural_evidence,
        )
        taiji_world_event = None
        descriptor = environment.capability_snapshot.get(request.capability_id)
        if (
            descriptor is not None
            and descriptor.category == "workspace"
            and descriptor.risk == "read_only"
        ):
            from seed_platform.workbench import WorkbenchTaijiEvidence

            world = self.model.architecture.cognitive_snapshot().world
            evidence = WorkbenchTaijiEvidence(
                request_id=request.request_id,
                intent_id=request.intent_id,
                call_id=call.call_id,
                capability_id=request.capability_id,
                snapshot_id=environment.capability_snapshot.snapshot_id,
                tick=int(world.tick),
                status=workbench_outcome.status,
                success=workbench_outcome.success,
                parameters=request.parameters,
                result=result,
            )
            taiji_world_event = evidence.to_taiji_event()
            invalidated_affordances = tuple(
                item.affordance_id
                for item in world.affordances
                if item.affordance_id.startswith("workbench:")
            )
            self.model.architecture.record_world_event(
                taiji_world_event,
                invalidate_affordance_ids=invalidated_affordances,
            )
            if executive_decision is not None:
                if taiji_outcome.intent_id != executive_decision.action_intent.intent_id:
                    raise ValueError("workbench outcome does not match the selected decision")
                if learn:
                    architecture.record_executive_outcome(
                        taiji_outcome,
                        learn=True,
                        source_affordance=source_affordance,
                        affordance_context=affordance_context,
                    )
        append_event(
            "outcome",
            request.request_id,
            tick=int(taiji_outcome.tick),
            payload={"outcome": workbench_outcome.to_payload()},
        )
        return {
            "request": request.to_payload(),
            "policy": policy.to_payload(),
            "outcome": workbench_outcome.to_payload(),
            "taiji_outcome": taiji_outcome.to_payload(),
            "structural_evidence": structural_evidence_result,
            "taiji_world_event": (
                None if taiji_world_event is None else taiji_world_event.to_payload()
            ),
            "tool_call": {
                **call.to_payload(),
                "workbench_binding": request.binding_payload(),
            },
            "events": [event.to_payload() for event in self._workbench_audit.events],
        }

    @property
    def language_provider_status(self) -> dict[str, str]:
        return dict(self._provider_status)

    @property
    def chat_language_backend(self) -> str:
        """Return the language surface used for product chat output."""

        return self._chat_organ.backend_id


# ---------------- 进程级单例与热切换 ----------------

_runtime: SeedRuntime | None = None
_runtime_lock = threading.Lock()


def _overlay_health(status: dict[str, str], health: Any) -> dict[str, str]:
    """叠加健康记录的观测位到一份 provider 状态，不改变模式/状态语义。

    名义探针返回的 result.status 携带 state="unchanged" 与最新健康位；
    这里只取健康字段，保留调用方当前的 mode/state/provider 不变，从而
    让 status API 实时反映健康负载而不翻转 provider 的角色语义。
    """

    payload = dict(status)
    payload.update(
        {
            "health_probes": str(int(health.probe_count)),
            "health_accepted_rate": f"{float(health.accepted_rate):.6f}",
            "health_degraded": "true" if health.degraded else "false",
            "health_rollback_count": str(int(health.rollback_count)),
            "health_cooldown_until": f"{float(health.cooldown_until):.3f}",
        }
    )
    return payload


def is_seed_active() -> bool:
    return _runtime is not None


def get_seed_runtime() -> SeedRuntime | None:
    return _runtime


def activate_seed(
    checkpoint_path: Path | str | None = None,
    *,
    provider_config: Any | None = None,
) -> SeedRuntime:
    """加载并激活 Seed 运行时（替换既有实例）。"""
    global _runtime
    with _runtime_lock:
        runtime = SeedRuntime.load(checkpoint_path, provider_config=provider_config)
        _runtime = runtime
        return runtime


def deactivate_seed() -> None:
    """卸载 Seed 运行时（切回 Cortex 主路径时调用）。"""
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            logger.info("Seed runtime deactivated (%s)", _runtime.name)
        _runtime = None


def seed_status() -> dict[str, Any]:
    runtime = _runtime
    if runtime is None:
        return {"runtime_type": "seed", "active": False}
    payload = runtime.status()
    payload["active"] = True
    return payload
