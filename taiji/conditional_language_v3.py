"""Opt-in V3 language candidate with independent input and output positions.

V2 remains byte-for-byte compatible. V3 increases both windows under an
explicit, separately versioned state and checkpoint contract.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .conditional_language_v2 import (
    ConditionalLanguageCandidateV2,
    ConditionalLanguageTrainerV2,
    _EvidenceCache,
)
from .internalization import content_digest
from .task_core_v2 import TaskCoreConfigV2, TaskCoreV2
from .task_state_v2 import TaskStateV2


@dataclass(frozen=True)
class TaskCoreConfigV3:
    width: int = 256
    embedding_width: int = 64
    key_width: int = 64
    slots: int = 16
    chunk_bytes: int = 32
    max_observation_bytes: int = 4096
    max_input_evidence_bytes: int = 4096
    max_output_position: int = 2048
    max_generation_bytes: int = 2048
    seed: int = 169
    position_extension_seed: int = 170

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.width,
                self.embedding_width,
                self.key_width,
                self.slots,
                self.chunk_bytes,
                self.max_observation_bytes,
                self.max_input_evidence_bytes,
                self.max_output_position,
                self.max_generation_bytes,
                self.position_extension_seed,
            )
        ):
            raise ValueError("V3 dimensions and byte bounds must be positive")
        if self.max_generation_bytes > self.max_output_position:
            raise ValueError("generation bound exceeds independent output-position table")
        if self.max_input_evidence_bytes > self.max_observation_bytes:
            raise ValueError("input evidence bound exceeds observation storage")
        if self.max_input_evidence_bytes > MAX_EVIDENCE_BYTES_V3:
            raise ValueError("input evidence exceeds the V3 task-state format")
        if self.max_output_position > MAX_OUTPUT_POSITION_V3:
            raise ValueError("output positions exceed the V3 format")

    @property
    def max_evidence_bytes(self) -> int:
        return self.max_input_evidence_bytes


MAX_EVIDENCE_BYTES_V3 = 4096
MAX_OUTPUT_POSITION_V3 = 2048


def _max_evidence_bytes(config: TaskCoreConfigV3) -> int:
    """Name expected by the shared recurrent chunk encoder."""
    return config.max_input_evidence_bytes


@dataclass(frozen=True)
class TaskStateV3(TaskStateV2):
    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.bundle_id, self.scope_id, self.session_id, self.native_state_ref)
        ):
            raise ValueError("task state requires explicit bundle/scope/session/native identity")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (self.hidden, self.workspace, self.byte_memory)
        ):
            raise ValueError("task state requires tensor representations")
        if (
            self.hidden.ndim != 1
            or self.workspace.ndim != 2
            or self.byte_memory.ndim != 2
            or self.workspace.shape[1] != len(self.hidden)
            or self.byte_memory.shape[1] != len(self.hidden)
        ):
            raise ValueError("task hidden/workspace/byte dimensions disagree")
        if len(self.workspace) != len(self.evidence_refs):
            raise ValueError("workspace and raw evidence references disagree")
        if (
            len(self.byte_memory) != len(self.byte_refs)
            or len(self.byte_memory) > MAX_EVIDENCE_BYTES_V3
        ):
            raise ValueError("byte memory and references disagree or exceed V3 evidence bound")
        tensors = [self.hidden, self.workspace, self.byte_memory] + (
            [] if self.decoder_hidden is None else [self.decoder_hidden]
        )
        if any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError("task state tensors must be finite float32")
        if self.decoder_hidden is not None and self.decoder_hidden.shape != self.hidden.shape:
            raise ValueError("decoder state dimension differs")
        if self.parameter_revision < 0 or self.revision < 0 or not 0 <= self.previous_byte <= 256:
            raise ValueError("task revision or previous byte is invalid")
        if not 0 <= self.utf8_remaining <= 3 or not 0 <= self.utf8_lead <= 255:
            raise ValueError("invalid UTF-8 continuation state")
        if self.decoder_hidden is None and (self.generated or self.previous_byte != 256):
            raise ValueError("response cursor lacks its decoder state")
        if self.generated and self.previous_byte != self.generated[-1]:
            raise ValueError("response cursor and generated bytes disagree")
        for source, start, end in (*self.evidence_refs, *self.byte_refs):
            if not 0 <= source < len(self.observations) or not 0 <= start < end <= len(
                self.observations[source][1]
            ):
                raise ValueError("evidence reference is outside its observation")

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["format"] = "taiji-task-state-v3"
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskStateV3:
        if payload.get("format") != "taiji-task-state-v3" or set(payload) != set(
            cls.__dataclass_fields__
        ) | {"format"}:
            raise ValueError("unsupported or incomplete V3 task state schema")
        values = {key: value for key, value in payload.items() if key != "format"}
        values["observations"] = tuple(
            (source, bytes.fromhex(raw)) for source, raw in values["observations"]
        )
        values["generated"] = bytes.fromhex(values["generated"])
        for name in ("hidden", "workspace", "byte_memory", "decoder_hidden"):
            if values[name] is not None:
                values[name] = values[name].detach().cpu().clone()
        return cls(**values)


class TaskCoreV3(TaskCoreV2):
    """V2 recurrent core with an independently versioned 4096-byte state."""

    def empty(
        self,
        *,
        bundle_id: str,
        scope_id: str,
        session_id: str,
        native_state_ref: str,
        parameter_revision: int,
    ) -> TaskStateV3:
        device = self.embedding.weight.device
        return TaskStateV3(
            bundle_id,
            scope_id,
            session_id,
            native_state_ref,
            parameter_revision,
            0,
            torch.zeros(self.config.width, device=device, dtype=torch.float32),
            torch.zeros((0, self.config.width), device=device, dtype=torch.float32),
            torch.zeros((0, self.config.width), device=device, dtype=torch.float32),
        )


class ConditionalLanguageCandidateV3(ConditionalLanguageCandidateV2):
    # core/config 类级注解：V3 在 __init__ 中以独立版本号的 V3 core 与 V3 配置替换了
    # V2 父类构造所建的同名属性，此处仅向类型检查器声明真实类型，运行时零改动。
    core: TaskCoreV3
    # V2/V3 配置是平行版本类（非继承），V3 有意在子类把 config 收窄为 V3 配置
    config: TaskCoreConfigV3  # type: ignore[assignment]

    def __init__(
        self, config: TaskCoreConfigV3 | None = None, *, bundle_id: str = "task-language-v3"
    ):
        config = TaskCoreConfigV3() if config is None else config
        legacy_shell = TaskCoreConfigV3(
            width=config.width,
            embedding_width=config.embedding_width,
            key_width=config.key_width,
            slots=config.slots,
            chunk_bytes=config.chunk_bytes,
            max_observation_bytes=config.max_observation_bytes,
            max_input_evidence_bytes=min(config.max_input_evidence_bytes, 512),
            max_output_position=min(config.max_output_position, 512),
            max_generation_bytes=min(config.max_generation_bytes, 512),
            seed=config.seed,
        )
        # legacy_shell 只供给 V2 构造所需的同名字段（鸭子兼容），V3 配置并非其子类属有意组合
        super().__init__(legacy_shell, bundle_id=bundle_id)  # type: ignore[arg-type]
        self.config = config
        del self.position
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config.position_extension_seed)
            # TaskCoreV3 未重写 __init__，继承签名按 V2 配置标注；运行时读取的是传入 V3 配置的同名字段
            self.core = TaskCoreV3(config)  # type: ignore[arg-type]
            self.input_position = nn.Embedding(
                config.max_input_evidence_bytes, config.key_width, dtype=torch.float32
            )
            self.output_position = nn.Embedding(
                config.max_output_position, config.key_width, dtype=torch.float32
            )
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def new_state(self, *, scope_id: str, session_id: str, native_state_ref: str) -> TaskStateV3:
        return self.core.empty(
            bundle_id=self.bundle_id,
            scope_id=scope_id,
            session_id=session_id,
            native_state_ref=native_state_ref,
            parameter_revision=self.parameter_revision,
        )

    # 以下三个 override 有意收窄参数至独立版本状态的 TaskStateV3（LSP 收窄），
    # 版本隔离由 validate() 运行时 isinstance 强制，放宽签名反而会放过跨版本状态。
    def validate(self, state: TaskStateV3) -> None:  # type: ignore[override]
        if not isinstance(state, TaskStateV3):
            raise ValueError("V3 requires an independently versioned task state")
        if state.bundle_id != self.bundle_id or state.parameter_revision != self.parameter_revision:
            raise ValueError("task state belongs to another bundle or parameter revision")
        if len(state.hidden) != self.config.width or state.workspace.shape[1] != self.config.width:
            raise ValueError("task state width differs from V3 candidate")
        if len(state.byte_memory) > self.config.max_input_evidence_bytes:
            raise ValueError("V3 input evidence exceeds its configured bound")

    def evidence_cache(self, state: TaskStateV3) -> _EvidenceCache:  # type: ignore[override]
        self.validate(state)
        if not len(state.byte_memory):
            raise ValueError("language generation requires actual per-byte task evidence")
        positions = torch.arange(len(state.byte_memory), device=state.byte_memory.device)
        return _EvidenceCache(
            state.bundle_id,
            state.scope_id,
            state.session_id,
            state.parameter_revision,
            state.byte_memory,
            state.hidden,
            self.core.keys(state.byte_memory) + self.input_position(positions),
            self.core.values(state.byte_memory),
        )

    def attention_weights(  # type: ignore[override]
        self, state: TaskStateV3, *, cache: _EvidenceCache | None = None
    ) -> torch.Tensor:
        self.validate(state)
        cache = self.evidence_cache(state) if cache is None else cache
        step = len(state.generated)
        if step >= self.config.max_output_position:
            raise ValueError("generation left the independent V3 output-position table")
        hidden = state.hidden if state.decoder_hidden is None else state.decoder_hidden
        query = self.query(hidden) + self.output_position.weight[step]
        return torch.softmax(cache.keys @ query / (self.config.key_width**0.5), dim=0)


class ConditionalLanguageTrainerV3(ConditionalLanguageTrainerV2):
    FORMAT = "taiji-conditional-language-trainer-v3"

    # lineage 仅在迁移（from_v2_checkpoint）与加载 checkpoint（from_checkpoint）路径赋值，
    # 构造时不初始化；只声明不赋值，保持 :457 处 getattr 默认值语义与运行时零改动。
    lineage: dict[str, Any]

    def update(self, examples: Sequence[tuple[bytes, bytes]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("update requires a declared nonempty batch")
        self.optimizer.zero_grad(set_to_none=True)
        byte_count = sum(len(target) for _, target in examples)
        if any(
            not target or len(target) > self.model.config.max_generation_bytes
            for _, target in examples
        ):
            raise ValueError("batch target is empty or exceeds the declared V3 generation bound")
        losses = []
        for prompt, target in examples:
            logits = self.model.teacher_forced(prompt, target)
            loss = (
                F.cross_entropy(
                    logits,
                    torch.tensor(list(target), dtype=torch.long),
                    reduction="sum",
                )
                / byte_count
            )
            losses.append(float(loss.detach()))
            loss.backward()
            del logits, loss
        gradients: dict[str, float] = {}
        for name, parameter in self.model.named_parameters():
            group = (
                ".".join(name.split(".")[:2]) if name.startswith("core.") else name.split(".")[0]
            )
            if parameter.grad is not None:
                gradients[group] = gradients.get(group, 0.0) + float(
                    parameter.grad.detach().square().sum()
                )
        norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        )
        self.optimizer.step()
        self.model.parameter_revision += 1
        self.steps += 1
        self.target_byte_exposures += byte_count
        mean_loss = sum(losses)
        return {
            "step": self.steps,
            "loss_nats": mean_loss,
            "loss_bits": mean_loss / math.log(2),
            "gradient_norm_before_clip": norm,
            "gradient_group_norms": {name: value**0.5 for name, value in gradients.items()},
            "target_byte_exposures": self.target_byte_exposures,
        }

    @classmethod
    def from_v2_checkpoint(
        cls,
        payload: dict[str, Any],
        *,
        config: TaskCoreConfigV3,
        parent_checkpoint_sha256: str,
    ) -> ConditionalLanguageTrainerV3:
        """Migrate a V2 parent without resetting shared weights or Adam state."""
        if (
            payload.get("format") != ConditionalLanguageTrainerV2.FORMAT
            or payload.get("version") != 2
        ):
            raise ValueError("migration requires a V2 trainer checkpoint")
        if content_digest(
            {k: v for k, v in payload.items() if k != "checkpoint_digest"}
        ) != payload.get("checkpoint_digest"):
            raise ValueError("V2 parent checkpoint digest differs")
        if len(parent_checkpoint_sha256) != 64:
            raise ValueError("migration requires the parent's file SHA-256")
        old_config = payload["config"]
        for name in ("width", "embedding_width", "key_width", "slots", "chunk_bytes", "seed"):
            if old_config[name] != getattr(config, name):
                raise ValueError(f"V2 parent {name} differs from the V3 migration config")
        if old_config["max_evidence_bytes"] > config.max_input_evidence_bytes:
            raise ValueError("V3 input evidence cannot be shorter than the V2 parent window")
        if old_config["max_generation_bytes"] > config.max_output_position:
            raise ValueError("V3 output positions cannot be shorter than the V2 parent window")

        old_model = ConditionalLanguageCandidateV2(
            TaskCoreConfigV2(**old_config),
            bundle_id=payload["bundle_id"],
        )
        old_model.load_state_dict(payload["parameters"], strict=True)
        model = ConditionalLanguageCandidateV3(config, bundle_id=payload["bundle_id"])
        old_parameters = dict(old_model.named_parameters())
        new_parameters = dict(model.named_parameters())
        mapped = set()
        with torch.no_grad():
            for name, target in new_parameters.items():
                source_name = (
                    "position.weight"
                    if name in {"input_position.weight", "output_position.weight"}
                    else name
                )
                if source_name not in old_parameters:
                    raise ValueError(f"V2 parent lacks parameter {source_name}")
                source = old_parameters[source_name]
                if name in {"input_position.weight", "output_position.weight"}:
                    if (
                        source.ndim != 2
                        or source.shape[1] != target.shape[1]
                        or source.shape[0] > target.shape[0]
                    ):
                        raise ValueError("V2 position table cannot be embedded into V3 table")
                    target[: source.shape[0]].copy_(source)
                elif target.shape != source.shape:
                    raise ValueError(f"shared parameter shape differs for {name}")
                else:
                    target.copy_(source)
                mapped.add(source_name)
        if mapped != set(old_parameters):
            raise ValueError("V2 parent contains unmapped parameters")
        model.parameter_revision = payload["parameter_revision"]

        trainer = cls(model, learning_rate=payload["learning_rate"])
        old_optimizer = payload["optimizer"]
        new_optimizer = trainer.optimizer.state_dict()
        if len(old_optimizer["param_groups"]) != 1 or len(new_optimizer["param_groups"]) != 1:
            raise ValueError("V3 migration currently requires one AdamW parameter group")
        old_ids = old_optimizer["param_groups"][0]["params"]
        new_ids = new_optimizer["param_groups"][0]["params"]
        old_names = list(old_parameters)
        new_names = list(new_parameters)
        if len(old_ids) != len(old_names) or len(new_ids) != len(new_names):
            raise ValueError("optimizer parameter ordering cannot be reconciled")
        old_id_by_name = dict(zip(old_names, old_ids, strict=True))
        migrated_state = {}
        for new_name, new_id in zip(new_names, new_ids, strict=True):
            old_name = (
                "position.weight"
                if new_name in {"input_position.weight", "output_position.weight"}
                else new_name
            )
            old_id = old_id_by_name[old_name]
            previous = old_optimizer["state"].get(old_id)
            if previous is None:
                continue
            state = {}
            for key, value in previous.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape == old_parameters[old_name].shape
                ):
                    expanded = torch.zeros_like(
                        new_parameters[new_name], memory_format=torch.preserve_format
                    )
                    if new_name in {"input_position.weight", "output_position.weight"}:
                        expanded[: value.shape[0]].copy_(value)
                    else:
                        expanded.copy_(value)
                    state[key] = expanded
                else:
                    state[key] = (
                        value.detach().clone()
                        if isinstance(value, torch.Tensor)
                        else copy.deepcopy(value)
                    )
            migrated_state[new_id] = state
        group = copy.deepcopy(old_optimizer["param_groups"][0])
        group["params"] = new_ids
        trainer.optimizer.load_state_dict({"state": migrated_state, "param_groups": [group]})
        trainer.steps = payload["steps"]
        trainer.target_byte_exposures = payload["target_byte_exposures"]
        model.generator.set_state(payload["rng_state"])
        trainer.lineage = {
            "migration": "conditional-language-v2-to-v3-v1",
            "parent_bundle_id": payload["bundle_id"],
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_checkpoint_digest": payload["checkpoint_digest"],
            "copied_parameters": sorted(old_parameters),
            "position_source": "v2.position.weight copied to the leading rows of both V3 tables; extension rows retain V3 deterministic initialization",
            "position_rows_copied": int(old_parameters["position.weight"].shape[0]),
            "position_extension_seed": config.position_extension_seed,
            "optimizer_state": "AdamW moments copied by parameter name; copied position moments placed in first 512 rows and extension rows zeroed",
        }
        return trainer

    # 同上：checkpoint 参数收窄至 TaskStateV3 属有意版本隔离（运行时先 validate）
    def checkpoint(  # type: ignore[override]
        self, state: TaskStateV3, *, cursor: dict[str, Any]
    ) -> dict[str, Any]:
        self.model.validate(state)
        payload = {
            "format": self.FORMAT,
            "version": 3,
            "config": asdict(self.model.config),
            "bundle_id": self.model.bundle_id,
            "parameter_revision": self.model.parameter_revision,
            "parameters": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "target_byte_exposures": self.target_byte_exposures,
            "lineage": copy.deepcopy(getattr(self, "lineage", {})),
            "rng_state": self.model.generator.get_state().clone(),
            "task_state": state.to_payload(),
            "cursor": copy.deepcopy(cursor),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]):
        if payload.get("format") != cls.FORMAT or payload.get("version") != 3:
            raise ValueError("unsupported conditional language checkpoint")
        if content_digest(
            {name: value for name, value in payload.items() if name != "checkpoint_digest"}
        ) != payload.get("checkpoint_digest"):
            raise ValueError("conditional language checkpoint digest differs")
        model = ConditionalLanguageCandidateV3(
            TaskCoreConfigV3(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = payload["parameter_revision"]
        trainer = cls(model, learning_rate=payload["learning_rate"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.steps = payload["steps"]
        trainer.target_byte_exposures = payload["target_byte_exposures"]
        trainer.lineage = copy.deepcopy(payload.get("lineage", {}))
        model.generator.set_state(payload["rng_state"])
        state = TaskStateV3.from_payload(payload["task_state"])
        model.validate(state)
        return trainer, state, copy.deepcopy(payload["cursor"])
