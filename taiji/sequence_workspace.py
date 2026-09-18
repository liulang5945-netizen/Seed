"""R2-H3.8 isolated joint-sequence-credit prototype.

Contract: ``plans/reference/M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md``
(section 7 revises the workspace graph to a single prefix channel).  This module
is a self-contained research prototype.  It does not touch the default Seed
entrypoint, does not read or write any existing Taiji checkpoint, and claims no
capability.  It exists so the joint-sequence-credit computation graph can be
implemented, gated, and later evaluated as its own package.

Workspace arm computation graph (v2, section 7; every detach / mask / reset /
truncation point annotated):

    prefix bytes ──> prefix scan (causal, last state) ──> h0
                       │
                       ├──> workspace keys   W_key   (slots, slot_width)   [frozen after begin]
                       └──> workspace values W_value (slots, slot_width)   [frozen after begin]

    start_vector (learned constant, prefix-independent) ──> r_0

    response y[t-1] ──> byte embedding ──> recurrent update r_t
                                              │        │
                                              │        └──> content addressing query -> softmax over
                                              │             W_key rows -> read = sum(a_i * W_value_i)
                                              └──> decode [r_t ; read] -> next-byte logits

The prefix reaches the answer stream through exactly one path: the
content-addressed read.  Zeroing the workspace parameters therefore makes the
logits prefix-invariant, which is the structural gate of contract section 7.2.
The no-workspace baseline arm keeps the v1 vanilla encoder (the prefix scan
state initializes the renderer) and is widened by config for the parameter
account of contract section 7.3.

Annotations required by the contract:

* **detach points**: none on the default training path.  The whole graph from
  the loss back to ``prefix_embedding`` is differentiable, which is the point
  of the contract's backprop clause.  ``detach_workspace_read`` exists only so
  the wiring gate can prove that disabling the connection removes its gradient;
  it is never used for training runs.
* **mask points**: the prefix scan and the renderer are strictly causal; the
  teacher-forced fold only ever sees ``y[:t]`` when producing position ``t``.
* **reset points**: ``begin_episode`` builds a fresh workspace and start state
  per episode.  Nothing carries between episodes.
* **truncation points**: none.  Version 1 expands short sequences in full; a
  sequence longer than ``config.max_sequence_bytes`` raises instead of silently
  truncating (long-sequence truncation needs its own contract).

The workspace is frozen *within* an episode: it is computed once from the
prefix and never written back during rendering.  Runtime adaptation (writing
to the workspace during generation) is explicitly out of scope here.

Version 3 (R2-D2, contract
``plans/reference/M5_R2_D2_PER_POSITION_EVIDENCE_PREREGISTRATION_FROZEN_20260918.md``)
adds a second evidence provenance while keeping every other edge of the graph:
instead of deriving all evidence rows once from the final scan state, the
prefix scan keeps every position state and projects one key/value evidence
entry per position; the renderer addresses those L entries with the same
content-read operator.  ``evidence_source`` selects between v2 rows
(``final_state_slots``), per-position rows (``per_position``) and the
parameter-matched broadcast control (``broadcast_final``, every position
carries the final state).  Evaluation-only lesions (value/key row rotation,
zeroed read) exist for the pre-registered misbind and zero-read probes; they
never participate in training.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .internalization import content_digest

SEQUENCE_WORKSPACE_FORMAT = "taiji-sequence-workspace-v1"
#: Version 6 (R2-D3 H-G, 2026-09-18): multi-head position-aware readout via
#: ``readout_heads`` and ``positional_keys``.  Versions 2-5 remain restorable;
#: version-1 payloads are refused.
SEQUENCE_WORKSPACE_VERSION = 6
#: Checkpoint payload versions this build is allowed to restore.
SEQUENCE_WORKSPACE_SUPPORTED_VERSIONS = frozenset({2, 3, 4, 5, 6})
SEQUENCE_WORKSPACE_TRAINER_FORMAT = "taiji-sequence-workspace-trainer-v1"
SEQUENCE_WORKSPACE_ALPHABET = 257
SEQUENCE_WORKSPACE_BOUNDARY = 256
SEQUENCE_WORKSPACE_SERIALIZATION = "torch.save/atomic"

#: Evidence provenance for the workspace arm (R2-D2 graph v3).
EVIDENCE_FINAL_STATE_SLOTS = "final_state_slots"
EVIDENCE_PER_POSITION = "per_position"
EVIDENCE_BROADCAST_FINAL = "broadcast_final"
EVIDENCE_SOURCES = frozenset(
    {EVIDENCE_FINAL_STATE_SLOTS, EVIDENCE_PER_POSITION, EVIDENCE_BROADCAST_FINAL}
)

#: Material-clause markers (byte form): the scan state immediately before the
#: earliest marker is the question-conditioned renderer start state (graph v5),
#: which by causal construction cannot yet contain material content.
MATERIAL_MARKERS: tuple[bytes, ...] = tuple(
    marker.encode("utf-8") for marker in ("背景：", "线索：", "已知：")
)

#: Canonical order used for per-name deterministic initialization.  Both arms'
#: inventories are subsequences of this tuple, so tensors that exist in both
#: arms share initial values under the same seed.
_CANONICAL_PARAMETER_ORDER: tuple[str, ...] = (
    "prefix_embedding",
    "prefix_input",
    "prefix_recur",
    "prefix_bias",
    "renderer_start",
    "start_vector",
    "renderer_embedding",
    "renderer_input",
    "renderer_recur",
    "renderer_bias",
    "address_query",
    "workspace_key",
    "workspace_value",
    "evidence_key",
    "evidence_value",
    "head_query_1",
    "head_query_2",
    "head_query_3",
    "head_query_4",
    "copy_gate_weight",
    "copy_gate_bias",
    "answer_start_weight",
    "answer_start_bias",
    "decoder",
    "decoder_bias",
)

#: Declared trainable parameter inventory for the workspace arm (v2 graph,
#: contract section 7): the renderer starts from a learned constant
#: ``start_vector`` and the prefix reaches the answer stream only through the
#: content-addressed workspace read.  The implementation gate asserts this list
#: against the live tensors.
SEQUENCE_WORKSPACE_PARAMETERS: tuple[str, ...] = tuple(
    name
    for name in _CANONICAL_PARAMETER_ORDER
    if name
    not in {
        "renderer_start",
        "evidence_key",
        "evidence_value",
        "head_query_1",
        "head_query_2",
        "head_query_3",
        "head_query_4",
        "copy_gate_weight",
        "copy_gate_bias",
        "answer_start_weight",
        "answer_start_bias",
    }
)

#: Declared inventory of the graph-v3 evidence arm (R2-D2): the per-position
#: (or broadcast) projections ``evidence_key/evidence_value`` replace the v2
#: ``workspace_key/workspace_value`` matrices; every other shared tensor keeps
#: its name and initialization.  A and C1 share this exact inventory.
SEQUENCE_WORKSPACE_EVIDENCE_PARAMETERS: tuple[str, ...] = tuple(
    name
    for name in _CANONICAL_PARAMETER_ORDER
    if name
    not in {
        "renderer_start",
        "workspace_key",
        "workspace_value",
        "head_query_1",
        "head_query_2",
        "head_query_3",
        "head_query_4",
        "copy_gate_weight",
        "copy_gate_bias",
        "answer_start_weight",
        "answer_start_bias",
    }
)

#: Graph-v4 copy-mixture arm (R2-D2 H-A2): v3 evidence inventory plus the
#: per-byte generate/copy gate (``64 + 1 = 65`` parameters; 82,658 total).
SEQUENCE_WORKSPACE_COPY_PARAMETERS: tuple[str, ...] = tuple(
    name
    for name in _CANONICAL_PARAMETER_ORDER
    if name
    not in {
        "renderer_start",
        "workspace_key",
        "workspace_value",
        "head_query_1",
        "head_query_2",
        "head_query_3",
        "head_query_4",
        "answer_start_weight",
        "answer_start_bias",
    }
)

#: Graph-v5 arm (R2-D2 H-A3): v4 copy mixture plus the question-conditioned
#: renderer start (``rw*rw + rw = 4,160`` parameters; 86,818 total).
SEQUENCE_WORKSPACE_QUESTION_START_PARAMETERS: tuple[str, ...] = tuple(
    name
    for name in _CANONICAL_PARAMETER_ORDER
    if name
    not in {
        "renderer_start",
        "workspace_key",
        "workspace_value",
        "head_query_1",
        "head_query_2",
        "head_query_3",
        "head_query_4",
    }
)

#: Graph-v6 multi-head arm (R2-D3 H-G): four head queries replace the single
#: address_query (``4*rw*sw`` vs ``rw*sw``); positional encoding carries no
#: parameters.  96,034 with copy mixture and question-conditioned start.
SEQUENCE_WORKSPACE_MULTIHEAD_PARAMETERS: tuple[str, ...] = tuple(
    name
    for name in _CANONICAL_PARAMETER_ORDER
    if name
    not in {
        "renderer_start",
        "address_query",
        "workspace_key",
        "workspace_value",
    }
)

#: Declared inventory of the no-workspace baseline arm.  The graph is the v1
#: vanilla encoder (prefix scan state initializes the renderer); the arm is
#: widened by config (renderer width 96, contract section 7.3) to align
#: parameter counts with the workspace arm rather than hiding the difference.
SEQUENCE_WORKSPACE_BASELINE_PARAMETERS: tuple[str, ...] = tuple(
    name
    for name in _CANONICAL_PARAMETER_ORDER
    if name
    not in {
        "start_vector",
        "address_query",
        "workspace_key",
        "workspace_value",
        "evidence_key",
        "evidence_value",
        "head_query_1",
        "head_query_2",
        "head_query_3",
        "head_query_4",
        "copy_gate_weight",
        "copy_gate_bias",
        "answer_start_weight",
        "answer_start_bias",
    }
)

#: Baseline arm renderer width frozen by contract section 7.3 (parameter
#: alignment: 103,601 versus the workspace arm's 101,025).
SEQUENCE_WORKSPACE_BASELINE_RENDERER_WIDTH = 96


def _sinusoidal_positional_encoding(length: int, width: int) -> torch.Tensor:
    """Fixed sin/cos positional encoding (graph v6); not trainable."""

    positions = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
    )
    angles = positions * frequencies
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class SequenceWorkspaceConfig:
    """Prototype geometry.  Small on purpose: the gates are structural."""

    prefix_width: int = 48
    slots: int = 4
    slot_width: int = 48
    renderer_width: int = 64
    seed: int = 20260917
    max_sequence_bytes: int = 512
    alphabet_size: int = SEQUENCE_WORKSPACE_ALPHABET
    boundary_symbol: int = SEQUENCE_WORKSPACE_BOUNDARY
    workspace_enabled: bool = True
    #: R2-D2 graph-v3 evidence provenance.  ``final_state_slots`` is the v2
    #: graph exactly; the other two select the per-position evidence arm and
    #: its parameter-matched broadcast control.
    evidence_source: str = EVIDENCE_FINAL_STATE_SLOTS
    #: R2-D2 H-A2 graph v4: mix the native vocabulary distribution with a
    #: copy distribution over visible prefix byte positions.  Only valid for
    #: the position-aligned evidence arms (per_position / broadcast_final).
    copy_mixture: bool = False
    #: R2-D2 H-A3 graph v5: condition the renderer start on the question-stem
    #: scan state (the state immediately before the material marker) instead
    #: of the prefix-independent learned constant.
    question_conditioned_start: bool = False
    #: R2-D3 H-G graph v6: number of independent readout heads.  ``1`` keeps
    #: the single ``address_query`` head (bit-identical to graphs v3-v5);
    #: ``4`` replaces it with four learnable head queries.
    readout_heads: int = 1
    #: Add fixed sinusoidal positional encoding to evidence keys (v6).
    positional_keys: bool = False

    def __post_init__(self) -> None:
        for name in ("prefix_width", "slots", "slot_width", "renderer_width"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"sequence workspace {name} must be positive")
        if int(self.max_sequence_bytes) <= 0:
            raise ValueError("sequence workspace max_sequence_bytes must be positive")
        if int(self.alphabet_size) != SEQUENCE_WORKSPACE_ALPHABET:
            raise ValueError("sequence workspace alphabet is fixed at 257 native bytes")
        if int(self.boundary_symbol) != SEQUENCE_WORKSPACE_BOUNDARY:
            raise ValueError("sequence workspace boundary symbol is fixed at 256")
        if str(self.evidence_source) not in EVIDENCE_SOURCES:
            raise ValueError(
                "evidence_source must be one of "
                f"{sorted(EVIDENCE_SOURCES)}, got {self.evidence_source!r}"
            )
        if not self.workspace_enabled and self.evidence_source != EVIDENCE_FINAL_STATE_SLOTS:
            raise ValueError(
                "evidence_source variants require the workspace arm " "(workspace_enabled=True)"
            )
        if self.copy_mixture and not self.workspace_enabled:
            raise ValueError("copy mixture requires the workspace arm")
        if self.copy_mixture and self.evidence_source == EVIDENCE_FINAL_STATE_SLOTS:
            raise ValueError(
                "copy mixture needs position-aligned evidence rows "
                "(per_position or broadcast_final), not the v2 slots"
            )
        if self.question_conditioned_start and not self.workspace_enabled:
            raise ValueError("question-conditioned start requires the workspace arm")
        if self.question_conditioned_start and self.evidence_source == EVIDENCE_FINAL_STATE_SLOTS:
            raise ValueError(
                "question-conditioned start is defined on the position-aligned "
                "evidence arms, not the v2 final-state slots"
            )
        if int(self.readout_heads) < 1:
            raise ValueError("readout_heads must be >= 1")
        if int(self.readout_heads) > 4:
            raise ValueError("readout_heads above 4 is outside the frozen contract")
        if int(self.readout_heads) > 1 and not self.workspace_enabled:
            raise ValueError("multi-head readout requires the workspace arm")
        if int(self.readout_heads) > 1 and self.evidence_source == EVIDENCE_FINAL_STATE_SLOTS:
            raise ValueError(
                "multi-head readout needs position-aligned evidence rows, not v2 slots"
            )
        if self.positional_keys and self.evidence_source == EVIDENCE_FINAL_STATE_SLOTS:
            raise ValueError("positional keys require position-aligned evidence rows")

    def to_payload(self) -> dict[str, Any]:
        return {
            "prefix_width": int(self.prefix_width),
            "slots": int(self.slots),
            "slot_width": int(self.slot_width),
            "renderer_width": int(self.renderer_width),
            "seed": int(self.seed),
            "max_sequence_bytes": int(self.max_sequence_bytes),
            "alphabet_size": int(self.alphabet_size),
            "boundary_symbol": int(self.boundary_symbol),
            "workspace_enabled": bool(self.workspace_enabled),
            "evidence_source": str(self.evidence_source),
            "copy_mixture": bool(self.copy_mixture),
            "question_conditioned_start": bool(self.question_conditioned_start),
            "readout_heads": int(self.readout_heads),
            "positional_keys": bool(self.positional_keys),
        }


@dataclass(frozen=True)
class WorkspaceState:
    """Renderer state carried across one response.  Immutable on purpose.

    The workspace tensors are ``None`` in the no-workspace baseline arm
    (contract section 3); they are frozen after ``begin_episode`` in the
    workspace arm.
    """

    workspace_key: torch.Tensor | None  # (entries, slot_width) -- v2 slots or v3 positions
    workspace_value: torch.Tensor | None  # (entries, slot_width) -- frozen after begin
    renderer_state: torch.Tensor  # (renderer_width,)
    entry_bytes: tuple[int, ...] | None = None  # prefix byte per evidence row (v3/v4)


@dataclass(frozen=True)
class GenerationResult:
    bytes_out: bytes
    stopped_on_boundary: bool
    steps: int


def _validate_bytes(value: bytes, name: str) -> bytes:
    if isinstance(value, (bytearray, memoryview)):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    return value


class SequenceWorkspacePrototype:
    """Trainable causal prefix encoder + content workspace + byte renderer."""

    def __init__(self, config: SequenceWorkspaceConfig | None = None) -> None:
        self.config = config or SequenceWorkspaceConfig()
        self._parameters: dict[str, torch.nn.Parameter] = {}
        pw = int(self.config.prefix_width)
        rw = int(self.config.renderer_width)
        slots = int(self.config.slots)
        sw = int(self.config.slot_width)

        def make(name: str, shape: tuple[int, ...], scale: float) -> None:
            # Per-name determinism: the two contract arms share the same seed,
            # and every tensor that exists in both arms must also share its
            # initial values, so the comparison is attributable to the
            # workspace and not to initialization order.
            index = _CANONICAL_PARAMETER_ORDER.index(name)
            generator = torch.Generator().manual_seed(int(self.config.seed) * 1_000_003 + index)
            tensor = (
                torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
            ) * scale
            self._parameters[name] = torch.nn.Parameter(tensor)

        make("prefix_embedding", (self.config.alphabet_size, pw), 0.5)
        make("prefix_input", (pw, rw), 1.0 / math.sqrt(pw))
        make("prefix_recur", (rw, rw), 1.0 / math.sqrt(rw))
        make("prefix_bias", (rw,), 0.1)
        decoder_columns = rw + sw if self.config.workspace_enabled else rw
        if self.config.workspace_enabled:
            # v2 graph (contract section 7): a learned constant start state;
            # the prefix enters the answer stream only through the workspace
            # rows generated here.  v3 (R2-D2) swaps the two row matrices for
            # shared per-position projections; the renderer side is identical.
            # v4 (H-A2) adds the per-byte generate/copy gate.
            make("start_vector", (rw,), 0.5)
            if int(self.config.readout_heads) == 1:
                make("address_query", (rw, sw), 1.0 / math.sqrt(rw))
            else:
                for head in range(1, int(self.config.readout_heads) + 1):
                    make(f"head_query_{head}", (rw, sw), 1.0 / math.sqrt(rw))
            if self.config.evidence_source == EVIDENCE_FINAL_STATE_SLOTS:
                make("workspace_key", (rw, slots * sw), 1.0 / math.sqrt(rw))
                make("workspace_value", (rw, slots * sw), 1.0 / math.sqrt(rw))
            else:
                make("evidence_key", (rw, sw), 1.0 / math.sqrt(rw))
                make("evidence_value", (rw, sw), 1.0 / math.sqrt(rw))
            if self.config.copy_mixture:
                make("copy_gate_weight", (rw,), 1.0 / math.sqrt(rw))
                make("copy_gate_bias", (1,), 0.0)
            if self.config.question_conditioned_start:
                make("answer_start_weight", (rw, rw), 1.0 / math.sqrt(rw))
                make("answer_start_bias", (rw,), 0.0)
        else:
            # baseline arm keeps the v1 vanilla encoder: the prefix scan state
            # initializes the renderer directly.
            make("renderer_start", (rw, rw), 1.0 / math.sqrt(rw))
        make("renderer_embedding", (self.config.alphabet_size, rw), 0.5)
        make("renderer_input", (rw, rw), 1.0 / math.sqrt(rw))
        make("renderer_recur", (rw, rw), 1.0 / math.sqrt(rw))
        make("renderer_bias", (rw,), 0.1)
        make(
            "decoder",
            (decoder_columns, self.config.alphabet_size),
            1.0 / math.sqrt(decoder_columns),
        )
        make("decoder_bias", (self.config.alphabet_size,), 0.1)

    # ---------------------------------------------------------------- inventory

    def declared_parameter_names(self) -> tuple[str, ...]:
        if not self.config.workspace_enabled:
            return SEQUENCE_WORKSPACE_BASELINE_PARAMETERS
        excluded = {"renderer_start"}
        if self.config.evidence_source == EVIDENCE_FINAL_STATE_SLOTS:
            excluded |= {
                "evidence_key",
                "evidence_value",
                "head_query_1",
                "head_query_2",
                "head_query_3",
                "head_query_4",
                "copy_gate_weight",
                "copy_gate_bias",
                "answer_start_weight",
                "answer_start_bias",
            }
        else:
            excluded |= {"workspace_key", "workspace_value"}
            if int(self.config.readout_heads) > 1:
                excluded.add("address_query")
            else:
                excluded |= {
                    "head_query_1",
                    "head_query_2",
                    "head_query_3",
                    "head_query_4",
                }
            if not self.config.copy_mixture:
                excluded |= {"copy_gate_weight", "copy_gate_bias"}
            if not self.config.question_conditioned_start:
                excluded |= {"answer_start_weight", "answer_start_bias"}
        return tuple(name for name in _CANONICAL_PARAMETER_ORDER if name not in excluded)

    def named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        return tuple((name, self._parameters[name]) for name in self.declared_parameter_names())

    def named_parameter(self, name: str) -> torch.nn.Parameter:
        if name not in self._parameters:
            raise KeyError(f"undeclared sequence workspace parameter: {name}")
        return self._parameters[name]

    def zero_grads_for_test(self) -> None:
        """Test helper: clear gradients without an optimizer instance."""

        for parameter in self.parameters():
            parameter.grad = None

    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self._parameters[name] for name in self.declared_parameter_names())

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    # ------------------------------------------------------------ core forward

    def _scan_prefix(self, prefix: bytes) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Causal scan over prefix bytes.

        Returns the final state h_L and every post-byte state (h_1, …, h_L).
        The v2 graph consumes only h_L; graph v3 consumes the full tuple so
        evidence rows are written at distinct input positions.
        """

        state = torch.zeros(int(self.config.renderer_width), dtype=torch.float32)
        embedding = self._parameters["prefix_embedding"]
        states: list[torch.Tensor] = []
        for symbol in prefix:
            state = torch.tanh(
                embedding[int(symbol)] @ self._parameters["prefix_input"]
                + state @ self._parameters["prefix_recur"]
                + self._parameters["prefix_bias"]
            )
            states.append(state)
        return state, tuple(states)

    def _embed_prefix(self, prefix: bytes) -> torch.Tensor:
        """Causal scan over prefix bytes; returns the last state h0."""

        return self._scan_prefix(prefix)[0]

    def _question_state(
        self,
        prefix: bytes,
        states: tuple[torch.Tensor, ...],
        h0: torch.Tensor,
    ) -> torch.Tensor:
        """Scan state at the end of the question stem, before material content.

        Graph v5: the earliest material marker (``背景：/线索：/已知：``) splits
        question from evidence.  Because the scan is causal, the state at the
        last stem byte cannot contain marker or material information.  Prefixes
        without a marker (synthetic tests) fall back to the final state.
        """

        if not self.config.question_conditioned_start:
            return self._parameters["start_vector"]
        marker_positions = [
            prefix.find(marker) for marker in MATERIAL_MARKERS if prefix.find(marker) >= 0
        ]
        if not marker_positions:
            return torch.tanh(
                h0 @ self._parameters["answer_start_weight"] + self._parameters["answer_start_bias"]
            )
        stem_end = min(marker_positions) - 1
        if stem_end < 0:
            raise ValueError("question stem is empty before the material marker")
        return torch.tanh(
            states[stem_end] @ self._parameters["answer_start_weight"]
            + self._parameters["answer_start_bias"]
        )

    def begin_episode(
        self,
        prefix: bytes,
        *,
        value_rotation: int = 0,
        entry_rotation: int = 0,
    ) -> WorkspaceState:
        """Build the per-episode workspace and start state.  Nothing carries over.

        ``value_rotation`` is the evaluation-only R2-D2 misbind lesion: value
        rows are cyclically shifted against their keys while every row's
        content is preserved.  ``entry_rotation`` (graph v4) cyclically shifts
        the prefix byte aligned to each evidence row, so the copy pathway
        copies from misaligned positions; keys and generative values stay put.
        Both must stay zero on every training path.
        """

        prefix = _validate_bytes(prefix, "prefix")
        if len(prefix) > int(self.config.max_sequence_bytes):
            raise ValueError("prefix exceeds max_sequence_bytes; truncation needs its own contract")
        rotation = int(value_rotation)
        if rotation < 0:
            raise ValueError("value_rotation must be non-negative")
        byte_rotation = int(entry_rotation)
        if byte_rotation < 0:
            raise ValueError("entry_rotation must be non-negative")
        h0, states = self._scan_prefix(prefix)
        slots = int(self.config.slots)
        sw = int(self.config.slot_width)
        entry_bytes: tuple[int, ...] | None = None
        if self.config.workspace_enabled:
            # v2 single prefix channel: h0 only produces the workspace rows;
            # the renderer starts from a learned constant (contract section 7).
            if self.config.evidence_source == EVIDENCE_FINAL_STATE_SLOTS:
                workspace_key = (h0 @ self._parameters["workspace_key"]).reshape(slots, sw)
                workspace_value = (h0 @ self._parameters["workspace_value"]).reshape(slots, sw)
            else:
                # Graph v3: one evidence entry per scan position (A), or L
                # identical entries projected from the final state (C1 control,
                # same machinery and parameters, no positional information).
                if not states:
                    raise ValueError(
                        "graph v3 evidence requires a non-empty prefix "
                        "(no scan positions to write)"
                    )
                if self.config.evidence_source == EVIDENCE_PER_POSITION:
                    positions = torch.stack(states, dim=0)
                    workspace_key = positions @ self._parameters["evidence_key"]
                    workspace_value = positions @ self._parameters["evidence_value"]
                else:
                    # One computed row replicated L times (shared storage):
                    # every entry is bit-identical, so the control differs
                    # from A only in provenance, never in arithmetic.
                    workspace_key = (h0 @ self._parameters["evidence_key"]).expand(len(states), -1)
                    workspace_value = (h0 @ self._parameters["evidence_value"]).expand(
                        len(states), -1
                    )
                entry_bytes = tuple(int(symbol) for symbol in prefix)
                if self.config.positional_keys:
                    workspace_key = workspace_key + _sinusoidal_positional_encoding(
                        workspace_key.shape[0], int(self.config.slot_width)
                    )
            if rotation:
                workspace_value = torch.roll(
                    workspace_value, shifts=rotation % workspace_value.shape[0], dims=0
                )
            if byte_rotation and entry_bytes is not None:
                shift = byte_rotation % len(entry_bytes)
                entry_bytes = tuple(
                    entry_bytes[(i - shift) % len(entry_bytes)] for i in range(len(entry_bytes))
                )
            renderer_state: torch.Tensor = self._question_state(prefix, states, h0)
        else:
            if rotation or byte_rotation:
                raise ValueError("the baseline arm has no evidence rows to rotate")
            workspace_key = None
            workspace_value = None
            renderer_state = torch.tanh(h0 @ self._parameters["renderer_start"])
        return WorkspaceState(
            workspace_key, workspace_value, renderer_state, entry_bytes=entry_bytes
        )

    def step(
        self,
        state: WorkspaceState,
        previous_symbol: int,
        *,
        detach_workspace_read: bool = False,
        zero_read: bool = False,
    ) -> tuple[WorkspaceState, torch.Tensor]:
        """One recurrent renderer step.  Pure function of (state, previous_symbol).

        ``zero_read`` is the evaluation-only R2-D2 read-ablation lesion: the
        decoder receives a zero evidence vector instead of the addressed read.
        """

        symbol = int(previous_symbol)
        if not 0 <= symbol < self.config.alphabet_size:
            raise ValueError("renderer symbol is outside the native byte alphabet")
        renderer_state = torch.tanh(
            self._parameters["renderer_embedding"][symbol] @ self._parameters["renderer_input"]
            + state.renderer_state @ self._parameters["renderer_recur"]
            + self._parameters["renderer_bias"]
        )
        if self.config.workspace_enabled:
            if zero_read:
                read = torch.zeros(int(self.config.slot_width), dtype=renderer_state.dtype)
            else:
                read = self._content_read(state, detach=detach_workspace_read)
            logits = (
                torch.cat((renderer_state, read), dim=0) @ self._parameters["decoder"]
                + self._parameters["decoder_bias"]
            )
        elif detach_workspace_read or zero_read:
            raise ValueError("the no-workspace baseline arm has no workspace read to ablate")
        else:
            logits = renderer_state @ self._parameters["decoder"] + self._parameters["decoder_bias"]
        return (
            WorkspaceState(
                state.workspace_key,
                state.workspace_value,
                renderer_state,
                entry_bytes=state.entry_bytes,
            ),
            logits,
        )

    def _readout_parameters(self) -> torch.Tensor:
        """Stacked readout query projections: shape (heads, rw, sw)."""

        heads = int(self.config.readout_heads)
        if heads == 1:
            return self._parameters["address_query"].unsqueeze(0)
        return torch.stack(
            [self._parameters[f"head_query_{h}"] for h in range(1, heads + 1)], dim=0
        )

    def _head_weights(self, state: WorkspaceState, *, detach: bool) -> torch.Tensor:
        """Per-head softmax weights with shape (heads, entries)."""

        key = state.workspace_key
        if key is None:
            raise ValueError("content addressing requires the workspace arm")
        if detach:
            key = key.detach()
        renderer = state.renderer_state
        queries = torch.einsum("r,hrw->hw", renderer, self._readout_parameters())
        scores = torch.einsum("nw,hw->hn", key, queries)
        scores = scores / math.sqrt(float(self.config.slot_width))
        return torch.softmax(scores, dim=-1)

    def _content_read(self, state: WorkspaceState, *, detach: bool) -> torch.Tensor:
        """Mean softmax content addressing over all heads (no positional pick)."""

        weights = self._head_weights(state, detach=detach)
        value = state.workspace_value
        if value is None:
            raise ValueError("content addressing requires the workspace arm")
        if detach:
            value = value.detach()
        return (weights @ value).mean(dim=0)

    def addressing_weights(self, state: WorkspaceState, *, detach: bool) -> torch.Tensor:
        """Read-only introspection: addressing weights per evidence row.

        Returns a 1-D tensor for the single-head graphs v3-v5 and a
        ``(heads, entries)`` tensor for graph v6 multi-head readout.
        """

        weights = self._head_weights(state, detach=detach)
        return weights.squeeze(0) if int(self.config.readout_heads) == 1 else weights

    def _copy_distribution(self, state: WorkspaceState, weights: torch.Tensor) -> torch.Tensor:
        """Mean p_copy across heads: p_copy[b] = sum of weights on byte-b rows.

        Normalized over visible prefix positions only; the boundary symbol
        (256) gets exactly zero mass, so stopping always goes through the
        generative vocabulary branch.
        """

        if state.entry_bytes is None:
            raise ValueError("copy distribution requires position-aligned evidence rows")
        index = torch.tensor(state.entry_bytes, dtype=torch.long)
        stacked = weights if weights.ndim == 2 else weights.unsqueeze(0)
        distribution = torch.zeros(int(self.config.alphabet_size), dtype=stacked.dtype)
        for head_weights in stacked:
            distribution = distribution.index_add(0, index, head_weights)
        return distribution / stacked.shape[0]

    def step_distribution(
        self, state: WorkspaceState, previous_symbol: int, *, zero_read: bool = False
    ) -> tuple[WorkspaceState, torch.Tensor]:
        """One renderer step returning the graph-v4 mixture distribution.

        ``p = pi * p_vocab + (1 - pi) * p_copy``; under ``zero_read`` the
        evidence vector is zeroed and the copy term removed (``p = p_vocab``).
        """

        new_state, probability, _ = self._step_mixture_and_copy(
            state, previous_symbol, zero_read=zero_read
        )
        return new_state, probability

    def _step_mixture_and_copy(
        self, state: WorkspaceState, previous_symbol: int, *, zero_read: bool = False
    ) -> tuple[WorkspaceState, torch.Tensor, torch.Tensor | None]:
        """``step_distribution`` plus the raw copy-component distribution.

        Returns ``(new_state, mixture, p_copy)``; ``p_copy is None`` exactly when
        the copy term is absent (``zero_read`` or no position-aligned evidence),
        in which case ``mixture is p_vocab``.  R2-D4 value supervision consumes
        ``p_copy``; the mixture path is byte-for-byte the legacy computation.
        """

        if not self.config.copy_mixture:
            raise ValueError("step_distribution requires copy_mixture=True")
        new_state, vocab_logits = self.step(state, previous_symbol, zero_read=zero_read)
        p_vocab = torch.softmax(vocab_logits, dim=0)
        if zero_read or new_state.entry_bytes is None:
            return new_state, p_vocab, None
        weights = self.addressing_weights(new_state, detach=False)
        p_copy = self._copy_distribution(new_state, weights)
        renderer_state = new_state.renderer_state
        gate = torch.sigmoid(
            renderer_state @ self._parameters["copy_gate_weight"]
            + self._parameters["copy_gate_bias"]
        )
        return new_state, gate * p_vocab + (1.0 - gate) * p_copy, p_copy

    def teacher_forced_distributions(self, prefix: bytes, response: bytes) -> torch.Tensor:
        """Mixture distributions at positions 0..len(response) (teacher-forced)."""

        mixtures, _ = self.teacher_forced_mixture_and_copy(prefix, response)
        return mixtures

    def teacher_forced_mixture_and_copy(
        self,
        prefix: bytes,
        response: bytes,
        *,
        value_rotation: int = 0,
        entry_rotation: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixture **and** copy-component distributions, positions 0..len(response).

        The rotation kwargs are the R2-D2 evaluation lesions, forwarded to
        :meth:`begin_episode`; the R2-D4 misbind probe reads the copy component
        under ``entry_rotation`` while training always uses the intact graph.
        """

        prefix = _validate_bytes(prefix, "prefix")
        response = _validate_bytes(response, "response")
        total = len(prefix) + len(response) + 1
        if total > int(self.config.max_sequence_bytes):
            raise ValueError(
                "sequence exceeds max_sequence_bytes; truncation needs its own contract"
            )
        state = self.begin_episode(
            prefix, value_rotation=value_rotation, entry_rotation=entry_rotation
        )
        mixtures: list[torch.Tensor] = []
        copies: list[torch.Tensor | None] = []
        previous = self.config.boundary_symbol
        for symbol in response:
            state, probability, copy_probability = self._step_mixture_and_copy(state, previous)
            mixtures.append(probability)
            copies.append(copy_probability)
            previous = int(symbol)
        state, probability, copy_probability = self._step_mixture_and_copy(state, previous)
        mixtures.append(probability)
        copies.append(copy_probability)
        if any(copy is None for copy in copies):
            raise ValueError(
                "copy component unavailable; teacher-forced copy readout requires "
                "position-aligned evidence (per_position/broadcast evidence source)"
            )
        return torch.stack(mixtures, dim=0), torch.stack(copies, dim=0)  # type: ignore[arg-type]

    def teacher_forced_logits(self, prefix: bytes, response: bytes) -> torch.Tensor:
        """Logits for positions 0..len(response); last position predicts the end marker."""

        response = _validate_bytes(response, "response")
        total = len(prefix) + len(response) + 1
        if total > int(self.config.max_sequence_bytes):
            raise ValueError(
                "sequence exceeds max_sequence_bytes; truncation needs its own contract"
            )
        state = self.begin_episode(prefix)
        logits: list[torch.Tensor] = []
        previous = self.config.boundary_symbol
        for symbol in response:
            state, step_logits = self.step(state, previous)
            logits.append(step_logits)
            previous = int(symbol)
        state, step_logits = self.step(state, previous)
        logits.append(step_logits)
        return torch.stack(logits, dim=0)

    def _response_logits(
        self,
        prefix: bytes,
        response: bytes,
        *,
        generation_state_ratio: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Logits for the response positions, optionally under **scheduled sampling**.

        ``generation_state_ratio == 0`` delegates to :meth:`teacher_forced_logits`
        so the teacher-forced path stays bit-identical.  Above zero, the symbol fed
        into the next step is the model's own ``argmax`` with that probability
        (the *targets* are unaffected -- the caller still scores the true symbols).
        """

        if generation_state_ratio <= 0.0:
            return self.teacher_forced_logits(prefix, response)
        response = _validate_bytes(response, "response")
        total = len(prefix) + len(response) + 1
        if total > int(self.config.max_sequence_bytes):
            raise ValueError(
                "sequence exceeds max_sequence_bytes; truncation needs its own contract"
            )
        state = self.begin_episode(prefix)
        collected: list[torch.Tensor] = []
        previous = self.config.boundary_symbol
        for symbol in response:
            state, step_logits = self.step(state, previous)
            collected.append(step_logits)
            draw = float(torch.rand((), generator=generator).item())
            if draw < generation_state_ratio:
                # Scheduled sampling: condition the next step on what the model
                # itself predicts (detached -- the sampling decision carries no
                # gradient; the cross-entropy targets do).
                previous = int(step_logits.detach().argmax(dim=0))
            else:
                previous = int(symbol)
        state, step_logits = self.step(state, previous)
        collected.append(step_logits)
        return torch.stack(collected, dim=0)

    def sequence_loss(
        self,
        prefix: bytes,
        response: bytes,
        *,
        generation_state_ratio: float = 0.0,
        generator: torch.Generator | None = None,
        contrastive_prefix: bytes | None = None,
        contrastive_margin: float = 1.0,
        contrastive_weight: float = 0.0,
        first_byte_weight: float = 1.0,
        copy_value_weight: float = 0.0,
        value_mask: Sequence[bool] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Real next-byte cross-entropy over the response plus the end marker.

        The **targets are always the true symbols**; ``generation_state_ratio`` only
        changes the prefix the model conditions on (scheduled sampling).  ``0.0``
        reproduces the teacher-forced path exactly, which is what the frozen H3.8
        records rely on.

        ``first_byte_weight`` scales the per-position loss of the **first response
        byte** (the readout-diagnosis failure point: 16/16 dev episodes err there).
        ``1.0`` keeps the plain mean — the legacy behaviour, byte-for-byte.

        ``contrastive_weight > 0`` adds the H-OBJ context-contrastive term: the same
        response scored under the true prefix must beat its score under a shuffled
        prefix by ``contrastive_margin`` nats, where the score is the **sequence**
        log-likelihood (``-sum`` cross-entropy, not the mean).  ``0.0`` disables it
        and reproduces the previous behaviour bit-for-bit.
        """

        ratio = float(generation_state_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("generation_state_ratio must be in [0, 1]")
        weight = float(contrastive_weight)
        if weight < 0.0:
            raise ValueError("contrastive_weight cannot be negative")
        first_weight = float(first_byte_weight)
        if first_weight <= 0.0:
            raise ValueError("first_byte_weight must be positive")
        targets = torch.tensor(
            [int(symbol) for symbol in response] + [int(self.config.boundary_symbol)],
            dtype=torch.long,
        )
        if self.config.copy_mixture:
            if ratio > 0.0 or weight > 0.0:
                raise ValueError(
                    "scheduled sampling and contrastive terms are not defined "
                    "on the copy-mixture distribution (R2-D2 H-A2)"
                )
            copy_weight = float(copy_value_weight)
            if copy_weight < 0.0:
                raise ValueError("copy_value_weight cannot be negative")
            if copy_weight > 0.0 and value_mask is None:
                raise ValueError("copy_value_weight requires value_mask")
            if copy_weight > 0.0:
                distributions, copy_distributions = self.teacher_forced_mixture_and_copy(
                    prefix, response
                )
            else:
                distributions = self.teacher_forced_distributions(prefix, response)
            target_probs = distributions.gather(1, targets.unsqueeze(1)).squeeze(1)
            per_position = -target_probs.clamp_min(1e-12).log()
            logits = distributions
        else:
            if copy_value_weight != 0.0:
                raise ValueError("copy_value_weight requires copy_mixture=True")
            logits = self._response_logits(
                prefix, response, generation_state_ratio=ratio, generator=generator
            )
            per_position = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        position_weights = torch.ones_like(per_position)
        position_weights[0] = first_weight
        loss = (per_position * position_weights).sum() / position_weights.sum()
        margin_gap: torch.Tensor | float = 0.0
        copy_value_prob_mean = 0.0
        if weight > 0.0:
            if contrastive_prefix is None:
                raise ValueError("contrastive_weight requires a contrastive_prefix")
            shuffled_logits = self._response_logits(
                contrastive_prefix, response, generation_state_ratio=0.0
            )
            # Sequence log-likelihood difference: the *sum* form, because the
            # criterion is about the whole answer, not per-position accuracy.
            correct_nll = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            shuffled_nll = torch.nn.functional.cross_entropy(
                shuffled_logits, targets, reduction="sum"
            )
            margin_gap = shuffled_nll - correct_nll
            margin = torch.as_tensor(float(contrastive_margin), dtype=logits.dtype)
            loss = loss + weight * torch.relu(margin - margin_gap)
        if self.config.copy_mixture and copy_value_weight > 0.0 and value_mask is not None:
            mask = torch.tensor([bool(flag) for flag in value_mask], dtype=torch.bool)
            if mask.numel() != len(response):
                raise ValueError("value_mask must align with the response bytes")
            if mask.any():
                copy_probs = copy_distributions[:-1][mask]
                copy_targets = targets[:-1][mask]
                copy_hits = copy_probs.gather(1, copy_targets.unsqueeze(1)).squeeze(1)
                copy_loss = -copy_hits.clamp_min(1e-12).log().mean()
                loss = loss + copy_weight * copy_loss
                copy_value_prob_mean = float(copy_hits.detach().mean())
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            correct = int((predictions == targets).sum())
            metrics = {
                "positions": int(targets.numel()),
                "correct": correct,
                "accuracy": correct / max(1, int(targets.numel())),
                "mean_surprise": float(loss.detach()),
                "contrastive_margin_gap": float(margin_gap),
                "first_byte_hit": int(predictions[0].item() == targets[0].item()),
                "copy_value_prob_mean": copy_value_prob_mean,
            }
        return loss, metrics

    @torch.no_grad()
    def generate(
        self,
        prefix: bytes,
        *,
        max_bytes: int | None = None,
        value_rotation: int = 0,
        entry_rotation: int = 0,
        zero_read: bool = False,
    ) -> GenerationResult:
        """Greedy native generation for later inference checks (no parameter update).

        ``value_rotation`` / ``entry_rotation`` / ``zero_read`` are the R2-D2
        evaluation-only generative-misbind, copy-misbind and read-ablation
        lesions; all default to the intact graph.  With copy mixture enabled
        the greedy symbol is the argmax of the mixture distribution.
        """

        limit = int(max_bytes if max_bytes is not None else self.config.max_sequence_bytes)
        state = self.begin_episode(
            prefix, value_rotation=value_rotation, entry_rotation=entry_rotation
        )
        previous = int(self.config.boundary_symbol)
        produced = bytearray()
        stopped = False
        steps = 0
        while steps < limit:
            if self.config.copy_mixture:
                state, distribution = self.step_distribution(state, previous, zero_read=zero_read)
                symbol = int(distribution.argmax(dim=0))
            else:
                state, logits = self.step(state, previous, zero_read=zero_read)
                symbol = int(logits.argmax(dim=0))
            steps += 1
            if symbol == int(self.config.boundary_symbol):
                stopped = True
                break
            produced.append(symbol)
            previous = symbol
        return GenerationResult(bytes(produced), stopped, steps)

    # ------------------------------------------------------------- checkpoint

    def parameter_payload(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters()
        }

    def load_parameter_payload(self, payload: Mapping[str, Any]) -> None:
        declared = self.declared_parameter_names()
        missing = [name for name in declared if name not in payload]
        if missing:
            raise ValueError(f"sequence workspace checkpoint is missing parameters: {missing}")
        for name, parameter in self.named_parameters():
            tensor = payload[name].detach().to(dtype=parameter.dtype).clone()
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(f"sequence workspace parameter shape mismatch: {name}")
            with torch.no_grad():
                parameter.copy_(tensor)

    @classmethod
    def from_parameter_payload(
        cls, config: SequenceWorkspaceConfig, payload: Mapping[str, Any]
    ) -> SequenceWorkspacePrototype:
        prototype = cls(config)
        prototype.load_parameter_payload(payload)
        return prototype


def _clone_optimizer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-clone optimizer state tensors (CPU) so checkpoints cannot alias.

    Keys are preserved verbatim: ``torch.optim`` keys ``state`` by integer
    parameter index, and rewriting them would silently break
    ``load_state_dict``.  ``content_digest`` stringifies mapping keys itself,
    so the digest stays canonical.
    """

    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    result: dict[str, Any] = convert(dict(state))
    return result


class SequenceWorkspaceTrainer:
    """Deterministic backprop trainer for the prototype (its own optimizer state)."""

    def __init__(
        self,
        prototype: SequenceWorkspacePrototype,
        *,
        learning_rate: float = 0.05,
        code_revision: str = "",
    ) -> None:
        self.prototype = prototype
        self.learning_rate = _finite_positive(learning_rate, "learning_rate")
        self.code_revision = str(code_revision)
        self.optimizer = torch.optim.Adam(prototype.parameters(), lr=self.learning_rate)
        self.episodes: tuple[tuple[bytes, bytes], ...] = ()
        self.data_order: tuple[int, ...] = ()
        self.cursor = 0
        self.global_step = 0
        # Scheduled-sampling state.  ``phi_max == 0`` (the default) keeps the
        # teacher-forced path exactly as before.  It is deliberately **not** part
        # of the checkpoint: it is a training-time schedule, not model state, so a
        # restored trainer must re-enable it explicitly (see the H-GEN spec).
        self.generation_state_ratio_max = 0.0
        self.generation_state_total_steps = 0
        self.sampling_generator: torch.Generator | None = None
        # H-OBJ context-contrastive term.  ``weight == 0`` (the default) keeps the
        # previous objective bit-for-bit; like the φ schedule it is a training-time
        # setting and is deliberately not part of the checkpoint.
        self.contrastive_weight = 0.0
        self.contrastive_margin = 1.0
        # Readout-diagnosis countermeasure: up-weight the loss on the **first**
        # response byte (16/16 dev episodes err there).  1.0 = legacy behaviour.
        self.first_byte_weight = 1.0
        # R2-D4 copy-value supervision (H-T): auxiliary NLL of the true answer
        # value bytes under the **copy component** at those positions.  0.0 keeps
        # the objective bit-for-bit; training-time only, not in the checkpoint.
        self.copy_value_weight = 0.0
        # Per-epoch data-order shuffle (course-scale sweep).  ``None`` keeps the
        # fixed data order — legacy behaviour, byte-for-byte.
        self.epoch_shuffle_seed: int | None = None
        self._shuffle_epoch_counter = 0

    def enable_epoch_shuffle(self, seed: int) -> None:
        """Reshuffle the episode order once per ``train_epoch`` (deterministic)."""

        self.epoch_shuffle_seed = int(seed)

    def enable_first_byte_weight(self, weight: float) -> None:
        value = float(weight)
        if value <= 0.0:
            raise ValueError("first_byte_weight must be positive")
        self.first_byte_weight = value

    def enable_context_contrastive(self, *, weight: float, margin: float = 1.0) -> None:
        """Turn on the H-OBJ term: the true prefix must beat a shuffled one by margin."""

        value = float(weight)
        if value < 0.0:
            raise ValueError("contrastive weight cannot be negative")
        if not math.isfinite(float(margin)):
            raise ValueError("contrastive margin must be finite")
        self.contrastive_weight = value
        self.contrastive_margin = float(margin)

    def enable_copy_value_supervision(self, weight: float) -> None:
        """Turn on the R2-D4 (H-T) auxiliary copy-component loss on value bytes."""

        value = float(weight)
        if value < 0.0:
            raise ValueError("copy_value_weight cannot be negative")
        self.copy_value_weight = value

    def enable_scheduled_sampling(
        self,
        *,
        phi_max: float,
        total_steps: int,
        generator: torch.Generator | None = None,
    ) -> None:
        """Freeze ``phi_max`` and the linear-schedule horizon (H-GEN spec §5)."""

        value = float(phi_max)
        if not 0.0 <= value <= 1.0:
            raise ValueError("phi_max must be in [0, 1]")
        if int(total_steps) <= 0:
            raise ValueError("total_steps must be positive")
        self.generation_state_ratio_max = value
        self.generation_state_total_steps = int(total_steps)
        self.sampling_generator = generator

    def current_generation_state_ratio(self) -> float:
        """φ(t) = φ_max · global_step / total_steps, capped at φ_max."""

        if self.generation_state_ratio_max <= 0.0:
            return 0.0
        horizon = max(1, self.generation_state_total_steps)
        fraction = min(1.0, self.global_step / horizon)
        return self.generation_state_ratio_max * fraction

    def set_episodes(self, episodes: Sequence[tuple[bytes, bytes]]) -> None:
        order = tuple(range(len(episodes)))
        self.episodes = tuple(
            (_validate_bytes(p, "prefix"), _validate_bytes(r, "response")) for p, r in episodes
        )
        self.data_order = order
        self.cursor = 0

    def train_step(
        self,
        batch: Sequence[tuple[bytes, bytes]],
        *,
        contrastive_prefixes: Sequence[bytes] | None = None,
        value_masks: Sequence[Sequence[bool]] | None = None,
    ) -> dict[str, Any]:
        """One optimizer step over an explicit batch.  Deterministic.

        ``contrastive_prefixes`` supplies, per batch item, the **shuffled** prefix used
        by the H-OBJ term (deterministically taken from the next episode in the frozen
        data order by :meth:`train_epoch`).  ``None`` leaves the objective unchanged.

        ``value_masks`` supplies, per batch item, the response-byte mask selecting the
        answer **value** positions for the R2-D4 copy supervision (H-T).  ``None``
        leaves the objective unchanged; required when ``copy_value_weight > 0``.
        """

        if not batch:
            raise ValueError("sequence workspace batch cannot be empty")
        if contrastive_prefixes is not None and len(contrastive_prefixes) != len(batch):
            raise ValueError("contrastive_prefixes must align with the batch")
        if value_masks is not None and len(value_masks) != len(batch):
            raise ValueError("value_masks must align with the batch")
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        positions = 0
        correct = 0
        margin_gaps: list[float] = []
        copy_value_probs: list[float] = []
        ratio = self.current_generation_state_ratio()
        for position, (prefix, response) in enumerate(batch):
            loss, metrics = self.prototype.sequence_loss(
                prefix,
                response,
                generation_state_ratio=ratio,
                generator=self.sampling_generator,
                contrastive_prefix=(
                    None if contrastive_prefixes is None else contrastive_prefixes[position]
                ),
                contrastive_margin=self.contrastive_margin,
                contrastive_weight=self.contrastive_weight,
                first_byte_weight=self.first_byte_weight,
                copy_value_weight=self.copy_value_weight,
                value_mask=None if value_masks is None else value_masks[position],
            )
            losses.append(loss)
            positions += int(metrics["positions"])
            correct += int(metrics["correct"])
            margin_gaps.append(float(metrics["contrastive_margin_gap"]))
            copy_value_probs.append(float(metrics["copy_value_prob_mean"]))
        total = torch.stack(losses).mean()
        total.backward()
        self.optimizer.step()
        self.global_step += 1
        return {
            "loss": float(total.detach()),
            "positions": positions,
            "accuracy": correct / max(1, positions),
            "global_step": self.global_step,
            "generation_state_ratio": ratio,
            "contrastive_margin_gap": sum(margin_gaps) / max(1, len(margin_gaps)),
            "copy_value_prob_mean": (
                sum(copy_value_probs) / max(1, len(copy_value_probs))
                if copy_value_probs
                else 0.0
            ),
        }

    def train_epoch(self, *, max_episodes: int | None = None) -> dict[str, Any]:
        """Walk the frozen data order from the cursor; one batch per episode."""

        if not self.episodes:
            raise ValueError("sequence workspace trainer has no episodes")
        limit = len(self.episodes) if max_episodes is None else int(max_episodes)
        if self.epoch_shuffle_seed is not None:
            generator = torch.Generator().manual_seed(
                self.epoch_shuffle_seed + self._shuffle_epoch_counter
            )
            order = torch.randperm(len(self.episodes), generator=generator).tolist()
            self._shuffle_epoch_counter += 1
        else:
            order = list(self.data_order)
        records: list[dict[str, Any]] = []
        for step_index in range(limit):
            index = order[self.cursor % len(order)]
            # Deterministic shuffle source for the H-OBJ term: the **next** episode in
            # the (possibly shuffled) order — no extra random source.
            next_index = order[(self.cursor + 1) % len(order)]
            record = self.train_step(
                [self.episodes[index]],
                contrastive_prefixes=[self.episodes[next_index][0]],
            )
            record["episode_index"] = index
            record["contrastive_prefix_index"] = next_index
            records.append(record)
            self.cursor = (self.cursor + 1) % len(self.data_order)
        return {
            "status": "completed",
            "episodes": len(records),
            "cursor": self.cursor,
            "mean_loss": sum(item["loss"] for item in records) / max(1, len(records)),
            "mean_contrastive_margin_gap": sum(item["contrastive_margin_gap"] for item in records)
            / max(1, len(records)),
            "records": records,
        }

    # ------------------------------------------------------------- checkpoint

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": SEQUENCE_WORKSPACE_TRAINER_FORMAT,
            "version": SEQUENCE_WORKSPACE_VERSION,
            "serialization": SEQUENCE_WORKSPACE_SERIALIZATION,
            "code_revision": self.code_revision,
            "config": self.prototype.config.to_payload(),
            "parameters": self.prototype.parameter_payload(),
            "parameter_names": list(self.prototype.declared_parameter_names()),
            "optimizer": _clone_optimizer_state(self.optimizer.state_dict()),
            "rng_state": torch.get_rng_state().clone(),
            "episodes": [
                {"prefix": prefix, "response": response} for prefix, response in self.episodes
            ],
            "data_order": list(self.data_order),
            "cursor": int(self.cursor),
            "global_step": int(self.global_step),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        from .persistence import atomic_save

        return atomic_save(self.checkpoint(), path)

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> SequenceWorkspaceTrainer:
        if payload.get("format") != SEQUENCE_WORKSPACE_TRAINER_FORMAT:
            raise ValueError("unsupported sequence workspace trainer format")
        # Dual read path (R2-D2): current writers emit version 3, version-2
        # single-channel checkpoints stay restorable on their own graph, and
        # anything older is refused rather than silently reinterpreted.
        if int(payload.get("version", -1)) not in SEQUENCE_WORKSPACE_SUPPORTED_VERSIONS:
            raise ValueError("unsupported sequence workspace trainer version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if payload.get("checkpoint_digest") != expected:
            raise ValueError("sequence workspace checkpoint digest mismatch")
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("sequence workspace checkpoint is missing its config")
        config = SequenceWorkspaceConfig(**dict(config_payload))
        parameters = payload.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("sequence workspace checkpoint is missing its parameters")
        prototype = SequenceWorkspacePrototype.from_parameter_payload(config, parameters)
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("sequence workspace checkpoint is missing optimizer state")
        groups = optimizer_state.get("param_groups")
        if not isinstance(groups, Sequence) or not groups:
            raise ValueError("sequence workspace checkpoint has no optimizer parameter groups")
        trainer = cls(
            prototype,
            learning_rate=float(groups[0].get("lr", 0.05)),
            code_revision=str(payload.get("code_revision", "")),
        )
        trainer.optimizer.load_state_dict(dict(optimizer_state))
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, torch.Tensor):
            raise ValueError("sequence workspace checkpoint is missing rng state")
        torch.set_rng_state(rng_state.detach().cpu().to(dtype=torch.uint8).clone())
        episodes_payload = payload.get("episodes")
        if not isinstance(episodes_payload, Sequence):
            raise ValueError("sequence workspace checkpoint is missing episodes")
        trainer.set_episodes(tuple((item["prefix"], item["response"]) for item in episodes_payload))
        trainer.data_order = tuple(int(index) for index in payload.get("data_order", ()))
        trainer.cursor = int(payload.get("cursor", 0))
        trainer.global_step = int(payload.get("global_step", 0))
        return trainer
