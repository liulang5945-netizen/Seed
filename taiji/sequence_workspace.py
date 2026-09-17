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
#: Version 2 (contract section 7, 2026-09-17): the workspace arm's only prefix
#: channel is the content-addressed read; ``start_vector`` replaces the
#: ``renderer_start`` direct path.  Version-1 payloads are refused, never
#: silently reinterpreted.
SEQUENCE_WORKSPACE_VERSION = 2
SEQUENCE_WORKSPACE_TRAINER_FORMAT = "taiji-sequence-workspace-trainer-v1"
SEQUENCE_WORKSPACE_ALPHABET = 257
SEQUENCE_WORKSPACE_BOUNDARY = 256
SEQUENCE_WORKSPACE_SERIALIZATION = "torch.save/atomic"

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
    "decoder",
    "decoder_bias",
)

#: Declared trainable parameter inventory for the workspace arm (v2 graph,
#: contract section 7): the renderer starts from a learned constant
#: ``start_vector`` and the prefix reaches the answer stream only through the
#: content-addressed workspace read.  The implementation gate asserts this list
#: against the live tensors.
SEQUENCE_WORKSPACE_PARAMETERS: tuple[str, ...] = tuple(
    name for name in _CANONICAL_PARAMETER_ORDER if name != "renderer_start"
)

#: Declared inventory of the no-workspace baseline arm.  The graph is the v1
#: vanilla encoder (prefix scan state initializes the renderer); the arm is
#: widened by config (renderer width 96, contract section 7.3) to align
#: parameter counts with the workspace arm rather than hiding the difference.
SEQUENCE_WORKSPACE_BASELINE_PARAMETERS: tuple[str, ...] = tuple(
    name
    for name in _CANONICAL_PARAMETER_ORDER
    if name not in {"start_vector", "address_query", "workspace_key", "workspace_value"}
)

#: Baseline arm renderer width frozen by contract section 7.3 (parameter
#: alignment: 103,601 versus the workspace arm's 101,025).
SEQUENCE_WORKSPACE_BASELINE_RENDERER_WIDTH = 96


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
        }


@dataclass(frozen=True)
class WorkspaceState:
    """Renderer state carried across one response.  Immutable on purpose.

    The workspace tensors are ``None`` in the no-workspace baseline arm
    (contract section 3); they are frozen after ``begin_episode`` in the
    workspace arm.
    """

    workspace_key: torch.Tensor | None  # (slots, slot_width) -- frozen after begin
    workspace_value: torch.Tensor | None  # (slots, slot_width) -- frozen after begin
    renderer_state: torch.Tensor  # (renderer_width,)


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
            # rows generated here.
            make("start_vector", (rw,), 0.5)
            make("address_query", (rw, sw), 1.0 / math.sqrt(rw))
            make("workspace_key", (rw, slots * sw), 1.0 / math.sqrt(rw))
            make("workspace_value", (rw, slots * sw), 1.0 / math.sqrt(rw))
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
        return (
            SEQUENCE_WORKSPACE_PARAMETERS
            if self.config.workspace_enabled
            else SEQUENCE_WORKSPACE_BASELINE_PARAMETERS
        )

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

    def _embed_prefix(self, prefix: bytes) -> torch.Tensor:
        """Causal scan over prefix bytes; returns the last state h0."""

        state = torch.zeros(int(self.config.renderer_width), dtype=torch.float32)
        embedding = self._parameters["prefix_embedding"]
        for symbol in prefix:
            state = torch.tanh(
                embedding[int(symbol)] @ self._parameters["prefix_input"]
                + state @ self._parameters["prefix_recur"]
                + self._parameters["prefix_bias"]
            )
        return state

    def begin_episode(self, prefix: bytes) -> WorkspaceState:
        """Build the per-episode workspace and start state.  Nothing carries over."""

        prefix = _validate_bytes(prefix, "prefix")
        if len(prefix) > int(self.config.max_sequence_bytes):
            raise ValueError("prefix exceeds max_sequence_bytes; truncation needs its own contract")
        h0 = self._embed_prefix(prefix)
        slots = int(self.config.slots)
        sw = int(self.config.slot_width)
        if self.config.workspace_enabled:
            # v2 single prefix channel: h0 only produces the workspace rows;
            # the renderer starts from a learned constant (contract section 7).
            workspace_key = (h0 @ self._parameters["workspace_key"]).reshape(slots, sw)
            workspace_value = (h0 @ self._parameters["workspace_value"]).reshape(slots, sw)
            renderer_state: torch.Tensor = self._parameters["start_vector"]
        else:
            workspace_key = None
            workspace_value = None
            renderer_state = torch.tanh(h0 @ self._parameters["renderer_start"])
        return WorkspaceState(workspace_key, workspace_value, renderer_state)

    def step(
        self,
        state: WorkspaceState,
        previous_symbol: int,
        *,
        detach_workspace_read: bool = False,
    ) -> tuple[WorkspaceState, torch.Tensor]:
        """One recurrent renderer step.  Pure function of (state, previous_symbol)."""

        symbol = int(previous_symbol)
        if not 0 <= symbol < self.config.alphabet_size:
            raise ValueError("renderer symbol is outside the native byte alphabet")
        renderer_state = torch.tanh(
            self._parameters["renderer_embedding"][symbol] @ self._parameters["renderer_input"]
            + state.renderer_state @ self._parameters["renderer_recur"]
            + self._parameters["renderer_bias"]
        )
        if self.config.workspace_enabled:
            read = self._content_read(state, detach=detach_workspace_read)
            logits = (
                torch.cat((renderer_state, read), dim=0) @ self._parameters["decoder"]
                + self._parameters["decoder_bias"]
            )
        elif detach_workspace_read:
            raise ValueError("the no-workspace baseline arm has no workspace read to detach")
        else:
            logits = renderer_state @ self._parameters["decoder"] + self._parameters["decoder_bias"]
        return (
            WorkspaceState(state.workspace_key, state.workspace_value, renderer_state),
            logits,
        )

    def _content_read(self, state: WorkspaceState, *, detach: bool) -> torch.Tensor:
        """Softmax content addressing over workspace slots (no positional pick)."""

        key = state.workspace_key
        value = state.workspace_value
        if key is None or value is None:
            raise ValueError("content addressing requires the workspace arm")
        if detach:
            key = key.detach()
            value = value.detach()
        query = state.renderer_state @ self._parameters["address_query"]
        scores = (key @ query) / math.sqrt(float(self.config.slot_width))
        weights = torch.softmax(scores, dim=0)
        return weights @ value

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
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Real next-byte cross-entropy over the response plus the end marker.

        The **targets are always the true symbols**; ``generation_state_ratio`` only
        changes the prefix the model conditions on (scheduled sampling).  ``0.0``
        reproduces the teacher-forced path exactly, which is what the frozen H3.8
        records rely on.
        """

        ratio = float(generation_state_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("generation_state_ratio must be in [0, 1]")
        logits = self._response_logits(
            prefix, response, generation_state_ratio=ratio, generator=generator
        )
        targets = torch.tensor(
            [int(symbol) for symbol in response] + [int(self.config.boundary_symbol)],
            dtype=torch.long,
        )
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            correct = int((predictions == targets).sum())
            metrics = {
                "positions": int(targets.numel()),
                "correct": correct,
                "accuracy": correct / max(1, int(targets.numel())),
                "mean_surprise": float(loss.detach()),
            }
        return loss, metrics

    @torch.no_grad()
    def generate(self, prefix: bytes, *, max_bytes: int | None = None) -> GenerationResult:
        """Greedy native generation for later inference checks (no parameter update)."""

        limit = int(max_bytes if max_bytes is not None else self.config.max_sequence_bytes)
        state = self.begin_episode(prefix)
        previous = int(self.config.boundary_symbol)
        produced = bytearray()
        stopped = False
        steps = 0
        while steps < limit:
            state, logits = self.step(state, previous)
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

    def train_step(self, batch: Sequence[tuple[bytes, bytes]]) -> dict[str, Any]:
        """One optimizer step over an explicit batch.  Deterministic."""

        if not batch:
            raise ValueError("sequence workspace batch cannot be empty")
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        positions = 0
        correct = 0
        ratio = self.current_generation_state_ratio()
        for prefix, response in batch:
            loss, metrics = self.prototype.sequence_loss(
                prefix,
                response,
                generation_state_ratio=ratio,
                generator=self.sampling_generator,
            )
            losses.append(loss)
            positions += int(metrics["positions"])
            correct += int(metrics["correct"])
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
        }

    def train_epoch(self, *, max_episodes: int | None = None) -> dict[str, Any]:
        """Walk the frozen data order from the cursor; one batch per episode."""

        if not self.episodes:
            raise ValueError("sequence workspace trainer has no episodes")
        limit = len(self.episodes) if max_episodes is None else int(max_episodes)
        records: list[dict[str, Any]] = []
        for _ in range(limit):
            index = self.data_order[self.cursor % len(self.data_order)]
            record = self.train_step([self.episodes[index]])
            record["episode_index"] = index
            records.append(record)
            self.cursor = (self.cursor + 1) % len(self.data_order)
        return {
            "status": "completed",
            "episodes": len(records),
            "cursor": self.cursor,
            "mean_loss": sum(item["loss"] for item in records) / max(1, len(records)),
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
        if int(payload.get("version", -1)) != SEQUENCE_WORKSPACE_VERSION:
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
