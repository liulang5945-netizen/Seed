"""Optional model-backed semantic provider at the Seed integration edge.

This module is deliberately outside ``taiji/``.  A local Qwen decoder can
propose semantic evidence, but Taiji still owns admission, grounding,
planning, policy, memory, and execution.  The decoder never receives a tool
catalog and never returns an ActionIntent or parameter binding.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from taiji import (
    SemanticEvidenceProposal,
    SemanticProviderRequest,
    content_digest,
    language_provider_content_digest,
)

SEMANTIC_PROVIDER_ARTIFACT_FORMAT = "taiji-semantic-provider-artifact-v1"
SEMANTIC_PROVIDER_ARTIFACT_VERSION = 1
SEMANTIC_PROVIDER_PROMPT_FORMAT = "taiji-qwen-semantic-provider-prompt-v1"
SEMANTIC_PROVIDER_MODEL_DIR_ENV = "TAIJI_SEMANTIC_PROVIDER_MODEL_DIR"
SEMANTIC_PROVIDER_MODEL_DIGEST_ENV = "TAIJI_SEMANTIC_PROVIDER_MODEL_DIGEST"

_ARTIFACT_DIGEST_PREFIX = "semantic-artifact:"
_PROPOSAL_FIELDS = frozenset(
    {"goal_description", "constraints", "semantic_steps", "confidence", "ambiguity"}
)
_BOUNDED_COMPATIBILITY_FIELDS = frozenset({"constraint"})
_OPERATION_ALIASES = {
    "resolve_language": "resolve-language",
    "set_language": "set-language",
}


class TextGenerationBackend(Protocol):
    """Small integration-edge contract used by the semantic adapter."""

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        """Generate untrusted text for parsing into semantic evidence."""


@dataclass(frozen=True)
class SemanticProviderArtifact:
    """Content-addressed model selection for one semantic provider."""

    artifact_id: str
    provider_id: str
    backend_id: str
    model_dir: str
    model_digest: str
    artifact_digest: str = ""
    version: int = SEMANTIC_PROVIDER_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.artifact_id, "artifact_id"),
            (self.provider_id, "provider_id"),
            (self.backend_id, "backend_id"),
            (self.model_dir, "model_dir"),
        ):
            if not str(value).strip():
                raise ValueError(f"semantic provider artifact {name} cannot be empty")
        if int(self.version) != SEMANTIC_PROVIDER_ARTIFACT_VERSION:
            raise ValueError("unsupported semantic provider artifact version")
        model_digest = str(self.model_digest).strip().lower()
        if len(model_digest) != 64 or any(char not in "0123456789abcdef" for char in model_digest):
            raise ValueError("semantic provider artifact model_digest must be SHA-256 hex")
        identity = {
            "format": SEMANTIC_PROVIDER_ARTIFACT_FORMAT,
            "version": SEMANTIC_PROVIDER_ARTIFACT_VERSION,
            "artifact_id": str(self.artifact_id),
            "provider_id": str(self.provider_id),
            "backend_id": str(self.backend_id),
            "model_digest": model_digest,
        }
        expected_digest = f"{_ARTIFACT_DIGEST_PREFIX}{content_digest(identity)[:24]}"
        if self.artifact_digest and str(self.artifact_digest) != expected_digest:
            raise ValueError("semantic provider artifact digest does not match its manifest")
        object.__setattr__(self, "model_digest", model_digest)
        object.__setattr__(self, "artifact_digest", expected_digest)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        expected_model_digest: str,
        backend_id: str = "qwen2.5-0.5b-instruct",
        artifact_id: str | None = None,
        provider_id: str | None = None,
    ) -> SemanticProviderArtifact:
        """Build an artifact only when the local model matches an allowlist.

        The expected digest is intentionally mandatory.  A discovered local
        directory is not trusted merely because it looks like a model.
        """

        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"semantic provider model directory not found: {path}")
        actual_digest = language_provider_content_digest(path)
        expected = str(expected_model_digest).strip().lower()
        if actual_digest != expected:
            raise ValueError(
                "semantic provider model digest is not allowlisted: "
                f"expected {expected}, got {actual_digest}"
            )
        stable_artifact_id = artifact_id or f"qwen-semantic-{actual_digest[:16]}"
        stable_provider_id = provider_id or f"{stable_artifact_id}:provider"
        return cls(
            artifact_id=stable_artifact_id,
            provider_id=stable_provider_id,
            backend_id=backend_id,
            model_dir=str(path),
            model_digest=actual_digest,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": SEMANTIC_PROVIDER_ARTIFACT_FORMAT,
            "version": self.version,
            "artifact_id": self.artifact_id,
            "provider_id": self.provider_id,
            "backend_id": self.backend_id,
            "model_dir": self.model_dir,
            "model_digest": self.model_digest,
            "artifact_digest": self.artifact_digest,
        }


class QwenSemanticEvidenceProvider:
    """Use a Qwen decoder as a bounded semantic evidence organ."""

    def __init__(
        self,
        generator: TextGenerationBackend,
        artifact: SemanticProviderArtifact,
        *,
        max_tokens: int = 192,
        temperature: float = 0.0,
    ) -> None:
        if not hasattr(generator, "generate"):
            raise TypeError("semantic provider generator must expose generate")
        if not isinstance(artifact, SemanticProviderArtifact):
            raise TypeError("semantic provider requires a SemanticProviderArtifact")
        if int(max_tokens) <= 0:
            raise ValueError("semantic provider max_tokens must be positive")
        if float(temperature) < 0.0:
            raise ValueError("semantic provider temperature cannot be negative")
        self._generator = generator
        self._artifact = artifact
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        expected_model_digest: str,
        backend_id: str = "qwen2.5-0.5b-instruct",
        artifact_id: str | None = None,
        provider_id: str | None = None,
        max_tokens: int = 192,
        temperature: float = 0.0,
    ) -> QwenSemanticEvidenceProvider:
        artifact = SemanticProviderArtifact.from_model_dir(
            model_dir,
            expected_model_digest=expected_model_digest,
            backend_id=backend_id,
            artifact_id=artifact_id,
            provider_id=provider_id,
        )
        return cls(
            _QwenTextGenerationBackend(Path(model_dir)),
            artifact,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @property
    def provider_id(self) -> str:
        return self._artifact.provider_id

    @property
    def artifact(self) -> SemanticProviderArtifact:
        return self._artifact

    def checkpoint(self) -> Mapping[str, Any]:
        """Return rebinding metadata, never model weights or cognitive state."""

        return {
            "format": "taiji-semantic-provider-checkpoint-v1",
            "provider_id": self.provider_id,
            "backend_id": self._artifact.backend_id,
            "artifact_id": self._artifact.artifact_id,
            "artifact_digest": self._artifact.artifact_digest,
            "model_digest": self._artifact.model_digest,
            "model_dir": self._artifact.model_dir,
        }

    def propose(self, request: SemanticProviderRequest) -> SemanticEvidenceProposal:
        if not isinstance(request, SemanticProviderRequest):
            raise TypeError("Qwen semantic provider requires a SemanticProviderRequest")
        if request.frame.modality not in {"text", "text-utf8"}:
            return SemanticEvidenceProposal.from_frame(
                request.frame,
                provider_id=self.provider_id,
                goal_description="当前输入不是可解析的文本任务",
                constraints=request.constraints,
                context_digest=request.context_digest,
                confidence=0.0,
                ambiguity=1.0,
                provenance=f"qwen:{self._artifact.artifact_id}",
                tick=request.frame.timestamp,
            )

        prompt = _semantic_prompt(request)
        generated = self._generator.generate(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        payload = _parse_semantic_json(generated)
        unknown = set(payload) - _PROPOSAL_FIELDS - _BOUNDED_COMPATIBILITY_FIELDS
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ValueError(f"Qwen semantic proposal has unsupported fields: {names}")
        model_constraints = payload.get("constraints", ())
        if not isinstance(model_constraints, (list, tuple)):
            raise TypeError("Qwen semantic proposal constraints must be a list")
        singular_constraint = payload.get("constraint", ())
        if isinstance(singular_constraint, str):
            singular_constraints = (singular_constraint,) if singular_constraint else ()
        elif isinstance(singular_constraint, (list, tuple)):
            singular_constraints = tuple(str(item) for item in singular_constraint)
        else:
            raise TypeError("Qwen semantic proposal constraint must be a string or list")
        constraints = (
            tuple(request.constraints)
            + tuple(str(item) for item in model_constraints)
            + singular_constraints
        )
        semantic_steps = payload.get("semantic_steps", ())
        if not isinstance(semantic_steps, (list, tuple)):
            raise TypeError("Qwen semantic proposal semantic_steps must be a list")
        normalized_steps = []
        for step in semantic_steps:
            if not isinstance(step, Mapping):
                normalized_steps.append(step)
                continue
            normalized_step = dict(step)
            slots = normalized_step.get("semantic_slots", {})
            if isinstance(slots, Mapping):
                normalized_slots = dict(slots)
                operation = str(normalized_slots.get("operation", "")).strip()
                if operation in _OPERATION_ALIASES:
                    normalized_slots["operation"] = _OPERATION_ALIASES[operation]
                normalized_step["semantic_slots"] = normalized_slots
            normalized_steps.append(normalized_step)
        return SemanticEvidenceProposal.from_frame(
            request.frame,
            provider_id=self.provider_id,
            goal_description=str(payload.get("goal_description", "")),
            constraints=constraints,
            context_digest=request.context_digest,
            semantic_steps=tuple(normalized_steps),
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity=float(payload.get("ambiguity", 1.0)),
            provenance=f"qwen:{self._artifact.artifact_id}",
            tick=request.frame.timestamp,
        )


def load_qwen_semantic_provider_from_environment() -> QwenSemanticEvidenceProvider | None:
    """Load the semantic provider only from a complete explicit binding.

    The product default remains provider-free.  A client may opt in by
    supplying both the model directory and its content digest; the digest is
    checked before Transformers loads any model weights.  No local cache path
    or model choice is embedded in the runtime.
    """

    model_dir = os.environ.get(SEMANTIC_PROVIDER_MODEL_DIR_ENV, "").strip()
    expected_digest = os.environ.get(SEMANTIC_PROVIDER_MODEL_DIGEST_ENV, "").strip()
    if not model_dir and not expected_digest:
        return None
    if not model_dir or not expected_digest:
        raise ValueError(
            "semantic provider opt-in requires both "
            f"{SEMANTIC_PROVIDER_MODEL_DIR_ENV} and {SEMANTIC_PROVIDER_MODEL_DIGEST_ENV}"
        )
    return QwenSemanticEvidenceProvider.from_model_dir(
        model_dir,
        expected_model_digest=expected_digest,
    )


def _semantic_prompt(request: SemanticProviderRequest) -> str:
    input_text = bytes(request.frame.payload).decode("utf-8")
    output_template = {
        "goal_description": "替换为用户目标",
        "constraints": ["仅保留用户明确提出的约束"],
        "semantic_steps": [
            {
                "description": "替换为一个语义步骤",
                "semantic_slots": {"operation": "用户明确表达的操作", "path": "用户明确表达的路径"},
                "expected_outcome": "替换为预期结果",
                "confidence": 0.0,
                "ambiguity": 1.0,
            }
        ],
        "confidence": 0.0,
        "ambiguity": 1.0,
    }
    example = {
        "goal_description": "读取指定文件并确认内容",
        "constraints": ["只读"],
        "semantic_steps": [
            {
                "description": "读取 README.md",
                "semantic_slots": {"operation": "read", "path": "README.md"},
                "expected_outcome": "获得 README.md 内容",
                "confidence": 0.9,
                "ambiguity": 0.1,
            }
        ],
        "confidence": 0.9,
        "ambiguity": 0.1,
    }
    return (
        f"格式：{SEMANTIC_PROVIDER_PROMPT_FORMAT}\n"
        "你是 Taiji 的外部语义器官，只负责从用户文本提取语义证据。"
        "你不能选择工具、能力、参数绑定、ActionIntent，也不能执行任务。"
        "只能输出一个 JSON 对象，不要输出 Markdown、解释或 JSON 之外的文字。"
        "必须把模板中的中文提示替换为输入中的事实；不能把 string、number、替换为用户目标、"
        "仅保留用户明确提出的约束等提示词当作值；confidence 和 ambiguity 必须是 0 到 1 的 JSON 数字。"
        "一次清晰请求只输出一个 semantic_steps 元素；如果请求缺少必要对象或动作，"
        "semantic_steps 必须是空数组、confidence 不得高于 0.4、ambiguity 不得低于 0.6，不能自行猜测。"
        "semantic_slots 只能描述用户目标中的语义事实，例如 operation、path、query、language；"
        "operation 必须使用规范值 list、read、stat、search、open、resolve-language、"
        "set-language 或 apply_patch；用户未指定目录的 list 操作使用 path 为 .。"
        "search 必须保留用户明确给出的 query；resolve-language 必须保留 path；"
        "set-language 必须保留 path 和用户明确给出的 language。"
        "不要把 list 写成 get，不要把 stat 写成 read，不要把 resolve-language 拆成普通 read。"
        "每个 semantic step 的 description 必须是非空的简短中文句子。"
        "规范映射示例：列出当前目录=list/path=.；查看文件信息=stat；"
        "在 README.md 搜索 Taiji=search/path=README.md/query=Taiji；"
        "判断 README.md 的编程语言=resolve-language/path=README.md；"
        "将 README.md 设置为 Markdown=set-language/path=README.md/language=markdown。"
        "不要输出 capability、tool、intent、parameter_bindings、patch、command 或 executor。\n"
        f"输出模板（只复制字段结构，不要复制提示词）：{json.dumps(output_template, ensure_ascii=False)}\n"
        f"参考示例（不要照抄内容，只学习格式）：{json.dumps(example, ensure_ascii=False)}\n"
        f"请求约束：{json.dumps(list(request.constraints), ensure_ascii=False)}\n"
        f"输入文本：\n{input_text}"
    )


def _parse_semantic_json(value: str) -> dict[str, Any]:
    text = str(value).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.lower().startswith("json\n"):
            text = text[5:]
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if text[end:].strip():
            continue
        if not isinstance(parsed, dict):
            raise ValueError("Qwen semantic response must be a JSON object")
        return parsed
    raise ValueError("Qwen semantic response is not one strict JSON object")


class _QwenTextGenerationBackend:
    """Lazy optional Transformers loader kept outside Taiji's native package."""

    def __init__(self, model_dir: Path) -> None:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "Qwen semantic provider requires the optional transformers integration"
            ) from exc
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float32,
        )
        self.model.eval()

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        del temperature
        formatted_prompt = prompt
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=int(max_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()


__all__ = [
    "SEMANTIC_PROVIDER_ARTIFACT_FORMAT",
    "SEMANTIC_PROVIDER_ARTIFACT_VERSION",
    "SEMANTIC_PROVIDER_PROMPT_FORMAT",
    "SEMANTIC_PROVIDER_MODEL_DIR_ENV",
    "SEMANTIC_PROVIDER_MODEL_DIGEST_ENV",
    "QwenSemanticEvidenceProvider",
    "load_qwen_semantic_provider_from_environment",
    "SemanticProviderArtifact",
    "TextGenerationBackend",
]
