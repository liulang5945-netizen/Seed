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
from collections.abc import Sequence
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


class SeedRuntime:
    """单个 Seed 有机体 + 字节级对话接口（线程安全）。"""

    RUNTIME_TYPE = "seed"

    def __init__(
        self,
        model: Any,
        checkpoint_path: Path | None = None,
        provider_status: dict[str, str] | None = None,
        provider_runtime: Any | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        from taiji import LanguageOrgan, NativeReadableTextLanguageOrgan

        # Chat stays local by default.  An external organ can reach product
        # chat only through explicit config plus the realization/safety Gate.
        self._chat_organ = NativeReadableTextLanguageOrgan()
        if provider_status is None:
            from seed.language_provider import attach_native_language_provider

            attach_native_language_provider(self.model.architecture)
            self._provider_status = {
                "mode": "native",
                "state": "active",
                "provider": "native",
                "backend_id": "native-readable",
                "artifact_id": "native-readable",
                "reason_code": "",
                "reason": "",
                "rollback": "native-readable",
            }
        else:
            self._provider_status = dict(provider_status)
            if self._provider_status.get("chat_enabled") == "true":
                candidate = self.model.architecture.language_organ
                if isinstance(candidate, LanguageOrgan):
                    self._chat_organ = candidate
        self._provider_runtime = provider_runtime
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        if self.checkpoint_path is not None:
            return f"seed:{self.checkpoint_path.name}"
        return "seed:scratch"

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
        return cls(model, path, provider_status.to_dict(), provider_runtime)

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

    def save(self, path: Path | str | None = None) -> Path:
        """落盘当前状态（默认写回来源检查点；原子写，崩溃不产生半写文件）。"""
        from seed.persistence import atomic_save, attach_metadata

        target = Path(path or self.checkpoint_path or DEFAULT_CHECKPOINT)
        with self._lock:
            envelope = attach_metadata(
                self.model.checkpoint(),
                tick=self.model.tick,
                extra={"trainer": "api_seed_runtime"},
            )
            atomic_save(envelope, target)
        logger.info("Seed runtime saved to %s", target)
        return target

    def status(self) -> dict[str, Any]:
        return {
            "runtime_type": self.RUNTIME_TYPE,
            "name": self.name,
            "tick": int(self.model.tick),
            "parameters": int(self.model.parameter_count()),
            "language_provider": dict(self._provider_status),
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
