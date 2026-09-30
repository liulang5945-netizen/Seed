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
from .utf8_state import utf8_allowed

#: 与 predictive_readout 同款纪律：独立 generator，器官的引入不得重播既有拓扑的随机流。
COPY_CIRCUIT_SEED_OFFSET = 0x2C0C_5017

#: `_serialize` 的轮次标记（与 `api.seed_runtime` 同一份文本形状；此处不 import api，防环）。
_QUESTION_MARKER = "问：".encode()
_ANSWER_MARKER = "\n答：".encode()

#: 锁规则档（PLAN-A-27 §2.6.5，owner 裁定 (d)）。`cue_only`＝旧缺省；`byte_overlap`＝新默认。
LOCK_RULES = ("cue_only", "byte_overlap")


def last_question_bytes(serialized: bytes) -> bytes:
    """从 `_serialize` 铺出的整段对话文本里取**最后一个提问轮**的字节。

    PLAN-A-27 §2.6.3 的口径修正：`generate` 原来把**整段序列化文本**当
    `query_bytes` 传给 `lock_selection`，于是"与提问共享字符"实际是"与整段对话
    共享字符"——两条告知都在整段里 ⇒ 内容侧特征**无区分度**（定价实测的头号嫌疑）。
    本函数取最后一个 ``问：`` 之后、其配对 ``答：`` 之前的段（没有标记时原样返回）。
    """

    cut = serialized.rfind(_QUESTION_MARKER)
    if cut < 0:
        return serialized
    start = cut + len(_QUESTION_MARKER)
    end = serialized.find(_ANSWER_MARKER, start)
    return serialized[start : end if end >= 0 else len(serialized)]


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

    参数面（9 个张量，全部可训练）：
    `content_embed (alphabet×ew)`、`query_state (mcd×ew)`、`query_content (ew×ew)` 随机初始化；
    `gate_state (mcd,)`、`gate_content (ew,)`、`gate_bias (1,)`、`copy_induce_bias (1,)`、
    `selector_weight (5,)`、`selector_bias (1,)` **零初始化**。
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
        # rev5（A2.5 预注册 §6）：提问条件化选择头。零初始化 ⇒ 分数退化成第 1 个特征
        # （皮质 cue 余弦），也就是 `ToldContentStore.best_match` 的原样行为。
        "selector_weight",
        "selector_bias",
    )
    #: 旧 payload（A2.3/A2.3b 的判决件里那些电路）不含 rev5 的两项：缺它们时按零初始化补，
    #: 而不是抛错——否则已入判决件再也载不回来，对照就废了。新 payload 仍写全。
    OPTIONAL_ZERO_PARAMETERS = ("selector_weight", "selector_bias")

    #: 选择头的 5 个特征（`SPEC-A-22` §6 冻结表）：全部无参数可算，头只学权重。
    SELECTOR_FEATURE_NAMES = (
        "cue_cosine",
        "content_cosine",
        "byte_overlap",
        "recency",
        "length_log",
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
            # rev5（A2.5 §6）：提问条件化选择头。零初始化 ⇒ 未训时分数＝cue 余弦＝现状。
            "selector_weight": torch.zeros(len(self.SELECTOR_FEATURE_NAMES)),
            "selector_bias": torch.zeros(1),
        }
        self._parameters = {
            name: tensor.to(self.device) for name, tensor in self._parameters.items()
        }
        # 诊断专用显式覆写（生产恒 None）。见模块 docstring。
        self.address_override: torch.Tensor | None = None
        self.gate_override: float | None = None
        #: A2.5 §1.2 的漂移消死结：提问喂完时锁定的那条告知（None＝现状，逐步重挑）。
        self._locked_event_id: int | None = None
        #: 锁指向的事件被 FIFO 淘汰而退回逐步余弦的次数——**必须恒 0**，
        #: 否则"整轮固定"这个判据是在另一条路径上量的（同 `_pool_into` 的计数器纪律）。
        self.selection_lock_dropped: int = 0
        #: 诊断专用：字节聚合方式（`"sum"`＝现状＝生产；`"max"` 只在 §22 那类归因实验里显式设）。
        #: 动机：`index_add` 把同一字节在句中**所有位置**的质量相加，于是"位置多"会被读成"更相关"
        #: ——实测首字节输家系统性地输给 E5/E6/E8（中文最常见首字节），怀疑就出在这儿。
        self.pool_override: str = "sum"

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

    def _pool_into(
        self, distribution: torch.Tensor, codes: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """把位置质量聚合成字节质量：默认 `index_add`（现状＝生产路径），诊断可切 `amax`。

        两种聚合的差别只在一个字节**在这条告知里出现几次**：求和会把它的位置数变成质量，
        取最大只看最强的那个位置。见 `pool_override` 的注释与 `SPEC-A-17` §22。
        """
        if self.pool_override == "max":
            #: 不用 `index_reduce_`（torch 标为 beta、语义将来可能变）：诊断路径每条告知只有几十
            #: 个位置，显式循环更贵不了多少，但不会因为 torch 升级而静默换语义。
            pooled = distribution.clone()
            for code, weight in zip(codes.tolist(), weights.tolist(), strict=False):
                if float(weight) > float(pooled[int(code)]):
                    pooled[int(code)] = float(weight)
            return pooled
        if self.pool_override == "mean":
            #: 这才是"位置数被当成质量"这一假设的**公平检验**：保留重数、只去掉计数放大。
            #: `max` 把重数整个丢掉（实测指对率 17.8%→8.9%，方向已负），
            #: 用它否证"求和虚增"会把"重数确实有用"这半边一起切掉。
            #: 注意这里显式走 `index_add` 而不是 `self._pool_into`——后者会按当前
            #: `pool_override` 再绕回本分支，变成无限递归。
            counts = torch.zeros_like(distribution).index_add(0, codes, torch.ones_like(weights))
            summed = torch.zeros_like(distribution).index_add(0, codes, weights)
            return summed / counts.clamp(min=1.0)
        if self.pool_override != "sum":
            raise ValueError(f"unknown pool_override {self.pool_override!r}")
        return distribution.index_add(0, codes, weights)

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        """与 `ToldContentStore.best_match` **同一套算式**（先 detach→cpu→float 再归一化点积）。

        必须逐位相同：选择头零初始化时分数＝本项，argmax 的平手裁决也照搬（严格大于才换，
        所以同分取库里更前面的那条）——这两点一起构成 §5 守卫②"逐位回退现状"。
        """
        query = torch.nn.functional.normalize(left.detach().cpu().float(), dim=0)
        key = torch.nn.functional.normalize(right.detach().cpu().float(), dim=0)
        return float(torch.dot(query, key))

    def _content_key(self, codes: torch.Tensor) -> torch.Tensor:
        """事件的内容表征：固定随机基查表 → `query_content` 投影 → 按位置平均。

        刻意**不新建第二个 embedding 面**，也不问 `content_embed` 可不可训（那是需签字的
        内容表征案，`SPEC-A-22` §7）。平均而非求和：特征 5 已经单独带了长度，别让长度在这里再混一次。
        """
        keys = self._parameters["content_embed"][codes] @ self._parameters["query_content"]
        return keys.mean(dim=0)

    def selection(
        self,
        *,
        cue: torch.Tensor,
        f1_context: torch.Tensor,
        query_bytes: bytes = b"",
    ) -> Mapping[str, Any] | None:
        """A2.5 §2/§6 的选择快照：每条告知一行特征＋一个分数，`picked` 是 argmax。

        只读（`@torch.no_grad` 不适用：全程 float 标量与 clone）。无事件时返回 None，
        与 `evidence()` 的"无事件⇒零向量"同口径。
        """
        events = self.store.events()
        if not events:
            return None
        query_chars = {ch for ch in query_bytes.decode("utf-8", "ignore") if not ch.isspace()}
        longest = max(len(event.content) for event in events)
        rows: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            codes = torch.tensor(list(event.content), dtype=torch.long, device=self.device)
            content_chars = {
                ch for ch in event.content.decode("utf-8", "ignore") if not ch.isspace()
            }
            overlap = (
                0.0
                if not query_chars
                else len(query_chars & content_chars) / float(len(query_chars))
            )
            rows.append(
                {
                    "event_id": int(event.event_id),
                    "features": torch.tensor(
                        [
                            self._cosine(cue, event.cue),
                            self._cosine(f1_context, self._content_key(codes)),
                            overlap,
                            (index + 1) / float(len(events)),
                            math.log1p(float(len(event.content)))
                            / math.log1p(float(max(longest, 1))),
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                }
            )
        matrix = torch.stack([row["features"] for row in rows])
        head = matrix @ self._parameters["selector_weight"] + self._parameters["selector_bias"][0]
        rule = self.config.lock_selection_rule
        if rule == "byte_overlap":
            #: PLAN-A-27 §2.6.5（owner 裁定 (d)）：纯"与提问共享字符"列。无参数、
            #: 两电路逐位相同；真跑 30/104 对 cue_only 的"位置尺子"形态。
            scores = matrix[:, 2]
        elif rule == "cue_only":
            #: 旧缺省（cue 余弦＋学习头）——实测 ≈90% 选最早那条告知。
            scores = matrix[:, 0] + head
        else:
            raise ValueError(f"unknown lock_selection_rule {rule!r}; known: {LOCK_RULES}")
        picked = 0
        for index in range(1, int(scores.numel())):
            if float(scores[index]) > float(scores[picked]):
                picked = index
        return {
            "query_bytes": bytes(query_bytes),
            "rule": rule,
            "event_ids": [row["event_id"] for row in rows],
            "features": matrix,
            "head_values": head,
            "scores": scores,
            "picked": picked,
            "event": events[picked],
        }

    def lock_selection(
        self,
        *,
        cue: torch.Tensor,
        f1_context: torch.Tensor,
        query_bytes: bytes = b"",
    ) -> Mapping[str, Any] | None:
        """算一次并**锁到本轮生成结束**：之后的发射与训练寻址都用这一条，不再逐步换。"""
        state = self.selection(cue=cue, f1_context=f1_context, query_bytes=query_bytes)
        self._locked_event_id = None if state is None else int(state["event"].event_id)
        return state

    def drop_selection_lock(self) -> None:
        self._locked_event_id = None

    @property
    def locked_event_id(self) -> int | None:
        return self._locked_event_id

    def _chosen_event(self, cue: torch.Tensor) -> ToldEvent | None:
        """生产发射与训练寻址的**唯一**取事件入口（两处必须同源，否则轨迹件静默走原样）。"""
        if self._locked_event_id is None:
            return self.store.best_match(cue)
        for event in self.store.events():
            if int(event.event_id) == self._locked_event_id:
                return event
        #: 锁指向的事件已被 FIFO 淘汰：计数 + 就地弃锁。静默退回逐步余弦会让
        #: "整轮固定"那条判据在另一条路径上量（同 `_pool_into` 的"被走到"计数器纪律）。
        self.selection_lock_dropped += 1
        self._locked_event_id = None
        return self.store.best_match(cue)

    def legal_suffix_mask(self, utf8_state: tuple[int, int]) -> torch.Tensor:
        """当前位置合法后继字节的 0/1 掩码（PLAN-A-25）。

        合法性判定**取自共享状态机** `utf8_allowed`（不另写一份，这是 `taiji/utf8_state.py`
        的设计纪律）。`alphabet_size` 比 256 多一个边界符，而掩码只覆盖字面字节；
        内容存储里本来就不会出现边界符，所以掩码对它是零操作、不改变停止语义。
        """

        remaining, lead = int(utf8_state[0]), int(utf8_state[1])
        mask = torch.zeros(self.config.alphabet_size, device=self.device)
        legal = torch.tensor(utf8_allowed(remaining, lead), device=self.device, dtype=torch.long)
        mask[legal] = 1.0
        return mask

    def evidence(
        self,
        *,
        cue: torch.Tensor,
        f1_context: torch.Tensor,
        prev_byte: int | None = None,
        utf8_state: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """257 维加性 logit 证据。无事件/未开闸时为精确零向量。

        `utf8_state = (remaining, lead)`（PLAN-A-25）：给了就**只提议当前位置合法的后继字节**
        ——复制的内容本身是合法字节串，逐字复述在语义上就该按位置接得上。不传 ⇒ 逐位不变。
        """

        distribution = torch.zeros(
            self.config.alphabet_size, dtype=torch.float32, device=self.device
        )
        event = self._chosen_event(cue)
        if event is None:
            return distribution
        weights = self._position_weights(event, f1_context, prev_byte)
        codes = torch.tensor(list(event.content), device=self.device, dtype=torch.long)
        distribution = self._pool_into(distribution, codes, weights)
        if utf8_state is not None:
            distribution = distribution * self.legal_suffix_mask(utf8_state)
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

        不消费诊断覆写（`address_override`/`gate_override`/`pool_override`）——生产发射
        （``evidence``）与训练寻址是两个面。**但消费选择锁**：锁是生产语义（整轮固定一条
        告知），不是诊断开关，两处都走 `_chosen_event` 才能保证"训练学的就是发射用的那条"。
        """
        event = self._chosen_event(cue)
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
        distribution = self._pool_into(distribution, codes, weights)
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
        lr_embed: float = 0.0,
    ) -> None:
        """A2.3 预注册 §2 的局部更新：寻址交叉熵＋反事实优势 gate。无 autograd。

        ``content_embed`` 默认仍是**固定随机基**（永不更新）；``lr_embed>0`` 是 SPEC-A-23 的
        丁 臂，唯一被允许的例外——见下方分支的注释。更新后统一 clamp 到 ``max_weight_norm``。
        ``advantage`` 由训练器计算并 clamp（±2）。
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
        bound = float(self.config.max_weight_norm)
        if lr_embed > 0.0:
            #: SPEC-A-23（丁 臂）：把字节表征本身交给训练。`score_i = ⟨q, embed[c_i] @ Qc⟩ / scale`
            #: ⇒ `∂score_i/∂embed[c_i] = Qcᵀ q / scale`，交叉熵给每行的梯度就是
            #: `delta_i · (Qcᵀ q)/scale`——同一字节在多个位置被指对/指错时按 `index_add_` 累加，
            #: 与发射侧"重数有用"的实测一致（`SPEC-A-17` §23：`max` 丢掉重数方向为负）。
            #: 默认 `lr_embed=0` ⇒ 这个分支一次都不走，`content_embed` 与旧 payload 逐位相同；
            #: clamp 也放在分支内——随机初始化的行本就可能落在 ±2.5 之外，
            #: 未训练时把它夹一下会**静默改掉键**，那不是"惰性"而是另一份表征。
            projection = (self._parameters["query_content"].T @ state["query"]) / scale
            self._parameters["content_embed"].index_add_(
                0, codes, delta[:, None] * projection.unsqueeze(0)
            )
            self._parameters["content_embed"].clamp_(-bound, bound)
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
        for name in (
            "query_state",
            "query_content",
            "gate_state",
            "gate_content",
            "gate_bias",
            "copy_induce_bias",
        ):
            self._parameters[name].clamp_(-bound, bound)

    @torch.no_grad()
    def learn_selection(
        self,
        state: Mapping[str, Any],
        *,
        target_event_id: int,
        advantage: float,
        lr_selector: float,
    ) -> None:
        """A2.5 §2 的局部更新：cross-entropy over 该库全部事件，无 autograd。

        标签 `target_event_id` **来自题面**（哪条告知逐字含答案词），不来自模型自己的答复
        ——否则就把 §13 那种"自己挑错的分布"当成目标写进去。`advantage` 沿用 A2.3 的 ±2 clamp，
        取 0 即不更新（训练器用它把"已经挑对的那一步"挡在外面）。
        """
        if not math.isfinite(float(lr_selector)) or float(lr_selector) < 0.0:
            raise ValueError("lr_selector must be finite and non-negative")
        ids = [int(value) for value in state["event_ids"]]
        if int(target_event_id) not in ids:
            raise ValueError("selector target event is not in the store snapshot")
        delta = torch.zeros_like(state["scores"])
        delta[ids.index(int(target_event_id))] = 1.0
        delta -= torch.softmax(state["scores"], dim=0)
        step = float(lr_selector) * max(-2.0, min(2.0, float(advantage)))
        weight = self._parameters["selector_weight"]
        weight.add_(step * (delta[:, None] * state["features"]).sum(dim=0))
        self._parameters["selector_bias"].add_(step * delta.sum())
        bound = float(self.config.max_weight_norm)
        weight.clamp_(-bound, bound)
        self._parameters["selector_bias"].clamp_(-bound, bound)

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
        missing = [
            name
            for name in self.PARAMETER_ORDER
            if name not in values and name not in self.OPTIONAL_ZERO_PARAMETERS
        ]
        if missing:
            raise ValueError(f"copy circuit payload missed parameters: {missing}")
        for name in self.PARAMETER_ORDER:
            if name not in values:
                #: rev5 之前的判决电路没有选择头——保留构造时的零初始化，即"逐位回退现状"，
                #: 而不是抛错把已入判决件变成载不回来的死档。
                continue
            tensor = values[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"copy circuit parameter {name} must be a tensor")
            self._parameters[name] = tensor.detach().to(self.device).float().clone()
        self.store.load_payload(payload["store"])
        #: store 整个被换掉，锁指向的 event_id 未必还在——一律清锁，宁可退回现状路径。
        self._locked_event_id = None
