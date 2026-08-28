"""Seed 原生运行时：加载 / 卸载 / 热切换 / 字节级对话。

``neuroplex/core/app_state.py`` 的对等物：Cortex（neuroplex）保留为可切换的
冻结对照，Seed 是独立可切换的原生运行时。两者互斥：任一时刻聊天主路由
只走其中一个。本模块不导入 ``neuroplex``，切换语义由调用方编排。

对话口径与阶段 1 训练管线一致（``scripts/training/train_seed_corpus.py``）：
对话结构用 ``问：/答：`` 文本标记序列化，会话边界由基底的
``boundary_symbol`` 承担，全程不引入 tokenizer。
"""

from __future__ import annotations

import logging
import pickle
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from taiji import InputFrame

logger = logging.getLogger("ApiServer.SeedRuntime")

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "checkpoints" / "seed_corpus.pt"
)

_TURN_MARKERS = ("\n问：", "问：")

# 输入长度上限（字符）：基底逐字节处理前缀（实测 ≈430 字节/秒），十万级
# 提示会让单请求阻塞数分钟（压测实测 100K 字符 ≈ 309 秒），构成可用性风险；
# 2048 字符（约 6KB ≈ 14s 前缀成本）是病态输入的封顶，典型对话消息远低于此，
# 不影响首字节延迟门槛。超长输入截断处理而非拒绝，保持对话可用。
MAX_PROMPT_CHARS = 2048


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
        self._lock = threading.Lock()

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
                    if result.status.chat_enabled
                    and isinstance(candidate, LanguageOrgan)
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
        runtime = cls(
            model, path, provider_status.to_dict(), provider_runtime, selected_config
        )
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
            logger.exception(
                "language provider health probe failed; keeping current surface"
            )
            return
        if not result.committed:
            # 名义探针：健康计数随真实发射增长，必须立刻可观测，但表层与队列不变。
            if result.health is not None:
                self._provider_status = _overlay_health(
                    self._provider_status, result.health
                )
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
            result["approval"] = {
                key: value for key, value in approval.items() if key != "preview"
            }
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

    def execute_workbench_intent(
        self,
        intent: Any,
        *,
        snapshot_id: str,
        approval_token: str = "",
        mcp_registry_snapshot_id: str = "",
        learn: bool = False,
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
        self._workbench_audit.append(
            "planned",
            request.request_id,
            tick=tick,
            payload={"request": request.to_payload()},
        )
        policy = environment.policy_for(request)
        self._workbench_audit.append(
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
            self._workbench_audit.append(
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
                "events": [
                    event.to_payload() for event in self._workbench_audit.events
                ],
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
            self._workbench_audit.append(
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
                "events": [
                    event.to_payload() for event in self._workbench_audit.events
                ],
            }

        self._workbench_audit.append(
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
                self._workbench_audit.append(
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
        self._workbench_audit.append(
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
