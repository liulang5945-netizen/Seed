"""R2 复制回路（A2.1 语言写入门 + A2.2 F1 内容直读 copy 通道）。

立项：`plans/reference/M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925.md`。
前件：A0 重判（`reports/taiji_r2_recall_split_v2_20260925.json`）——真实召回反馈改变动力学
但解不出内容字节 ⇒ (c) 真有罪：F1 与召回内容之间不存在发射通路。

设计约束（冻结，违一条即红）：

* **惰性挂载**：`Taiji` 默认不持有本器官（`_copy_circuit is None`），旧 checkpoint 直载行为
  零变化；本模块不 import `model.py`，无环。
* **gate 线性零初始化 ⇒ 位级不变**：发射证据 `evidence = gate · copy_dist` 以加性项进
  `episodic_evidence`（logit 空间，与既有 `memory_read_gain` 同族）。`gate_state/gate_content/
  gate_bias` 全部零初始化 ⇒ 挂载即 `evidence ≡ 0` ⇒ 输出逐位不变。刻意不用
  `sigmoid` 门（workspace 的凸混合形态）：`sigmoid(0)=0.5` 破坏位级不变性，加性 logit 形态
  允许 gate 越界为负（反复制），其取值制度留给 A2.3 预注册。
* **内容存储是字节序列，不是激活回归**（G3 的正主修正）：`ToldContentStore` 存
  「告知字节 + 段末皮质 cue」，与 `EpisodicField` 的三元组权重完全分离。provenance 语义
  复用 `external`（不扩枚举、不动 one-hot 维度）。
* copy 数学照抄本仓经 R2 训练的 `sequence_content_workspace._copy_weights/_copy_distribution`
  （query·key 打分 → 位置 softmax → index_add 到字节码），从 codepoint 域换到字节域。
* `address_override` / `gate_override` 是**诊断专用**显式属性（None＝生产路径）：
  §4.1 结构存在性判据用 oracle 寻址验证「内容能驱动输出」的接线，不依赖未训练的 query/key。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .config import TaijiConfig

#: 与 predictive_readout 同款纪律：独立 generator，器官的引入不得重播既有拓扑的随机流。
COPY_CIRCUIT_SEED_OFFSET = 0x2C0C_5017


@dataclass(frozen=True)
class ToldEvent:
    """一条被告知的事件：原始字节序列 + 段末皮质 cue（寻址键）。"""

    event_id: int
    content: bytes
    cue: torch.Tensor

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "content": list(self.content),
            "cue": self.cue.detach().cpu().clone(),
        }

    @staticmethod
    def from_payload(payload: Mapping[str, Any]) -> ToldEvent:
        content = bytes(int(value) for value in payload["content"])
        cue = payload["cue"]
        if not isinstance(cue, torch.Tensor) or cue.ndim != 1:
            raise ValueError("told event cue must be a 1-D tensor")
        return ToldEvent(event_id=int(payload["event_id"]), content=content, cue=cue.float())


class ToldContentStore:
    """A2.1：语言回路的合法内容写入门。容量为最近 `max_events` 条（FIFO 淘汰）。

    寻址＝归一化点积（余弦）top-1，不设阈值——(b) 侧的匹配质量制度属于 A2.3/A2.4，
    本件只保证「内容进得来、取得出」。
    """

    def __init__(self, *, cue_dim: int, max_events: int = 8) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.cue_dim = int(cue_dim)
        self.max_events = int(max_events)
        self._events: list[ToldEvent] = []
        self._next_event_id = 0

    def record(self, content: bytes, cue: torch.Tensor) -> int:
        if not content:
            raise ValueError("told content cannot be empty")
        if cue.shape != (self.cue_dim,):
            raise ValueError("told content cue dimension mismatch")
        event_id = self._next_event_id
        self._next_event_id += 1
        self._events.append(
            ToldEvent(event_id=event_id, content=bytes(content), cue=cue.detach().cpu().clone())
        )
        while len(self._events) > self.max_events:
            self._events.pop(0)
        return event_id

    @property
    def count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        """诊断/训练单事件体制用：清空存储（不回收 event_id 序列）。"""
        self._events = []

    def events(self) -> tuple[ToldEvent, ...]:
        return tuple(self._events)

    def best_match(self, cue: torch.Tensor) -> ToldEvent | None:
        if not self._events:
            return None
        if cue.shape != (self.cue_dim,):
            raise ValueError("told content cue dimension mismatch")
        query = torch.nn.functional.normalize(cue.detach().cpu().float(), dim=0)
        best: tuple[float, ToldEvent] | None = None
        for event in self._events:
            key = torch.nn.functional.normalize(event.cue, dim=0)
            score = float(torch.dot(query, key))
            if best is None or score > best[0]:
                best = (score, event)
        return best[1] if best is not None else None

    def to_payload(self) -> dict[str, Any]:
        return {
            "cue_dim": self.cue_dim,
            "max_events": self.max_events,
            "next_event_id": self._next_event_id,
            "events": [event.to_payload() for event in self._events],
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if int(payload["cue_dim"]) != self.cue_dim:
            raise ValueError("told content store cue_dim mismatch")
        self.max_events = int(payload["max_events"])
        self._next_event_id = int(payload["next_event_id"])
        self._events = [ToldEvent.from_payload(item) for item in payload["events"]]
        if len(self._events) > self.max_events:
            raise ValueError("told content store payload exceeds its capacity")


class CopyCircuit:
    """A2.2：F1 的内容直读通道。`evidence(cue, f1_context)` 返回 257 维加性 logit 证据。

    参数面（6 个张量，全部可训练）：
    `content_embed (alphabet×ew)`、`query_state (mcd×ew)`、`query_content (ew×ew)` 随机初始化；
    `gate_state (mcd,)`、`gate_content (ew,)`、`gate_bias (1,)` **零初始化**。
    """

    PAYLOAD_FORMAT = "taiji-copy-circuit-v1"
    PARAMETER_ORDER = (
        "content_embed",
        "query_state",
        "query_content",
        "gate_state",
        "gate_content",
        "gate_bias",
        # rev3（A2.3 预注册 §6）：后继归纳——前一字节＝刚发出字节的行加分
        # （D6 语义，数学照抄 sequence_content_workspace._copy_weights）。零初始化。
        "copy_induce_bias",
    )

    def __init__(
        self,
        config: TaijiConfig,
        *,
        max_events: int = 8,
        device: torch.device | str = "cpu",
        init_seed: int | None = None,
    ) -> None:
        """`init_seed=None` ⇒ 用 `config.seed`（**今天的行为，逐位不变**）。

        为什么要这个旋钮：A2.3b 之后"第二次独立取数"要求**两个独立训练出来的电路**，
        而训练器的 `--seed` 只喂语料取样流——电路的随机初始化当时只能跟着基座 config 走，
        换 `--seed` 重跑得到的是同一套初始 query/key 投影。没有这个参数，
        "独立电路"四个字在这条线上是造不出来的（`SPEC-A-17` §8）。
        它**只**影响三个非零初始化的投影张量；门参数仍零初始化 ⇒ 不训练时发射证据仍恒零。
        """
        self.config = config
        self.device = torch.device(device)
        self.evidence_width = int(config.motor_context_dim)
        self.store = ToldContentStore(
            cue_dim=int(config.cortical_context_dim), max_events=max_events
        )
        generator = torch.Generator(device="cpu")
        base_seed = int(config.seed) if init_seed is None else int(init_seed)
        generator.manual_seed(base_seed + COPY_CIRCUIT_SEED_OFFSET)
        width = self.evidence_width
        alphabet = int(config.alphabet_size)
        context_dim = int(config.motor_context_dim)

        def make(shape: tuple[int, ...], scale: float) -> torch.Tensor:
            return torch.randn(*shape, generator=generator) * scale

        self._parameters: dict[str, torch.Tensor] = {
            "content_embed": make((alphabet, width), 1.0 / math.sqrt(width)),
            "query_state": make((context_dim, width), 1.0 / math.sqrt(context_dim)),
            "query_content": make((width, width), 1.0 / math.sqrt(width)),
            # Zero-init emission gate: mounting the organ must not move a
            # single logit until A2.3 training opens it.
            "gate_state": torch.zeros(context_dim),
            "gate_content": torch.zeros(width),
            "gate_bias": torch.zeros(1),
            # rev3: zero-init successor bonus — inert until trained, exactly
            # like the gate (bit-identical mounting preserved).
            "copy_induce_bias": torch.zeros(1),
        }
        self._parameters = {
            name: tensor.to(self.device) for name, tensor in self._parameters.items()
        }
        # 诊断专用显式覆写（生产恒 None）。见模块 docstring。
        self.address_override: torch.Tensor | None = None
        self.gate_override: float | None = None

    @property
    def mounted(self) -> bool:
        return True

    def parameters(self) -> Mapping[str, torch.Tensor]:
        return dict(self._parameters)

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._parameters[name] for name in self.PARAMETER_ORDER)

    def _successor_bonus(self, codes: torch.Tensor, prev_byte: int | None) -> torch.Tensor:
        """rev3 后继归纳（D6 语义，照抄 workspace `_copy_weights`）：「前一字节＝刚
        发出的字节」的行加 `copy_induce_bias`。零初始化 ⇒ 未训练时逐位无影响。"""
        bonus = torch.zeros_like(codes, dtype=torch.float32)
        if prev_byte is None or int(codes.numel()) < 2:
            return bonus
        successors = [
            i for i in range(1, int(codes.numel())) if int(codes[i - 1]) == int(prev_byte)
        ]
        if successors:
            bonus[torch.tensor(successors, device=codes.device)] = self._parameters[
                "copy_induce_bias"
            ]
        return bonus

    def _position_weights(
        self, event: ToldEvent, f1_context: torch.Tensor, prev_byte: int | None = None
    ) -> torch.Tensor:
        if self.address_override is not None:
            weights = self.address_override.to(self.device).float()
            if weights.shape != (len(event.content),):
                raise ValueError("copy address override must match the event length")
            total = float(weights.sum())
            if total <= 0.0 or not math.isfinite(total):
                raise ValueError("copy address override must be a positive finite weighting")
            return weights / total
        codes = torch.tensor(list(event.content), device=self.device, dtype=torch.long)
        keys = self._parameters["content_embed"][codes] @ self._parameters["query_content"]
        query = f1_context @ self._parameters["query_state"]
        scale = math.sqrt(float(self.evidence_width))
        scores = query @ keys.T / scale + self._successor_bonus(codes, prev_byte)
        return torch.softmax(scores, dim=0)

    def evidence(
        self, *, cue: torch.Tensor, f1_context: torch.Tensor, prev_byte: int | None = None
    ) -> torch.Tensor:
        """257 维加性 logit 证据。无事件/未开闸时为精确零向量。"""
        distribution = torch.zeros(
            self.config.alphabet_size, dtype=torch.float32, device=self.device
        )
        event = self.store.best_match(cue)
        if event is None:
            return distribution
        weights = self._position_weights(event, f1_context, prev_byte)
        codes = torch.tensor(list(event.content), device=self.device, dtype=torch.long)
        distribution = distribution.index_add(0, codes, weights)
        keys = self._parameters["content_embed"][codes] @ self._parameters["query_content"]
        pooled = weights @ keys
        gate = (
            f1_context @ self._parameters["gate_state"]
            + pooled @ self._parameters["gate_content"]
            + self._parameters["gate_bias"]
        )
        if self.gate_override is not None:
            gate = torch.full_like(gate, float(self.gate_override))
        return gate * distribution

    def addressing(
        self, *, cue: torch.Tensor, f1_context: torch.Tensor, prev_byte: int | None = None
    ) -> dict[str, Any] | None:
        """训练器用的只读寻址快照（A2.3 预注册 §2）；store 无事件时返回 None。

        不消费诊断覆写——生产发射（``evidence``）与训练寻址是两个面。
        """
        event = self.store.best_match(cue)
        if event is None:
            return None
        codes = torch.tensor(list(event.content), dtype=torch.long, device=self.device)
        embed = self._parameters["content_embed"][codes]
        keys = embed @ self._parameters["query_content"]
        query = f1_context @ self._parameters["query_state"]
        scale = math.sqrt(float(self.evidence_width))
        scores = query @ keys.T / scale + self._successor_bonus(codes, prev_byte)
        weights = torch.softmax(scores, dim=0)
        distribution = torch.zeros(
            self.config.alphabet_size, dtype=torch.float32, device=self.device
        )
        distribution = distribution.index_add(0, codes, weights)
        pooled = weights @ keys
        gate = (
            f1_context @ self._parameters["gate_state"]
            + pooled @ self._parameters["gate_content"]
            + self._parameters["gate_bias"]
        )
        return {
            "event": event,
            "prev_byte": prev_byte,
            "codes": codes,
            "embed": embed,
            "keys": keys,
            "query": query,
            "scores": scores,
            "pooled": pooled,
            "copy_distribution": distribution,
            "gate_value": float(gate),
        }

    @torch.no_grad()
    def learn(
        self,
        state: Mapping[str, Any],
        *,
        f1_context: torch.Tensor,
        target_position: int,
        advantage: float,
        lr_address: float,
        lr_gate: float,
    ) -> None:
        """A2.3 预注册 §2 的局部更新：寻址交叉熵＋反事实优势 gate。无 autograd。

        ``content_embed`` 保持固定随机基（永不更新）；更新后统一 clamp 到
        ``max_weight_norm``。``advantage`` 由训练器计算并 clamp（±2）。
        """
        for name, value in (("lr_address", lr_address), ("lr_gate", lr_gate)):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        codes = state["codes"]
        if not 0 <= int(target_position) < int(codes.numel()):
            raise ValueError("copy target position outside the event")
        scale = math.sqrt(float(self.evidence_width))
        delta = torch.nn.functional.one_hot(
            torch.tensor(int(target_position), device=self.device),
            num_classes=int(codes.numel()),
        ).float() - torch.softmax(state["scores"], dim=0)
        grad_query = delta @ state["keys"] / scale
        self._parameters["query_state"].add_(lr_address * torch.outer(f1_context, grad_query))
        self._parameters["query_content"].add_(
            lr_address
            * (delta[:, None] * state["embed"]).T
            @ state["query"].expand_as(state["embed"])
            / scale
        )
        step = float(advantage) * lr_gate
        self._parameters["gate_state"].add_(step * f1_context)
        self._parameters["gate_content"].add_(step * state["pooled"])
        self._parameters["gate_bias"].add_(step)
        # rev3: the successor bonus learns through the same addressing delta —
        # positions whose predecessor was just emitted get pushed up/down.
        prev_byte = state.get("prev_byte")
        if prev_byte is not None and int(codes.numel()) > 1:
            successors = [
                i for i in range(1, int(codes.numel())) if int(codes[i - 1]) == int(prev_byte)
            ]
            if successors:
                successor_mask = torch.zeros_like(delta)
                successor_mask[torch.tensor(successors, device=self.device, dtype=torch.long)] = 1.0
                self._parameters["copy_induce_bias"].add_(
                    lr_address * float((delta * successor_mask).sum())
                )
        bound = float(self.config.max_weight_norm)
        for name in (
            "query_state",
            "query_content",
            "gate_state",
            "gate_content",
            "gate_bias",
            "copy_induce_bias",
        ):
            self._parameters[name].clamp_(-bound, bound)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.PAYLOAD_FORMAT,
            "evidence_width": self.evidence_width,
            "parameters": {
                name: tensor.detach().cpu().clone() for name, tensor in self._parameters.items()
            },
            "store": self.store.to_payload(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if str(payload.get("format")) != self.PAYLOAD_FORMAT:
            raise ValueError("copy circuit payload format mismatch")
        if int(payload["evidence_width"]) != self.evidence_width:
            raise ValueError("copy circuit evidence_width mismatch")
        values = payload["parameters"]
        missing = [name for name in self.PARAMETER_ORDER if name not in values]
        if missing:
            raise ValueError(f"copy circuit payload missed parameters: {missing}")
        for name in self.PARAMETER_ORDER:
            tensor = values[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"copy circuit parameter {name} must be a tensor")
            self._parameters[name] = tensor.detach().to(self.device).float().clone()
        self.store.load_payload(payload["store"])
