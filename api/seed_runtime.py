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
            executive_decision=decision,
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
                    executive_decision=decision,
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
        if isinstance(successor_graph, Mapping):
            if str(successor_graph.get("parent_loop_id", "")) == str(parent_loop_id):
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

    def execute_workbench_intent(
        self,
        intent: Any,
        *,
        snapshot_id: str,
        approval_token: str = "",
        mcp_registry_snapshot_id: str = "",
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
