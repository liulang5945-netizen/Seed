"""R2-D7 H-Tok: the character-token sibling of the byte sequence workspace.

Module: taiji/sequence_char_workspace.py
Contract: plans/reference/M5_R2_D7_CHAR_TOKEN_CONTRACT_FROZEN_20260918.md

The byte graphs (v2-v8) were adjudicated failed on representational grounds:
a UTF-8 byte is not an addressable word unit, so every per-word copy needs
3-6 independent addressing decisions whose match sets are diluted by shared
lead bytes (E4-E9).  This module re-runs the SAME mechanisms — per-position
evidence, sinusoidal keys, four readout heads, question-conditioned start,
the copy mixture, the D4 value supervision and the D6 emitted-token induction
bonus — with one variable changed: every step, row and copy slot is a
CHARACTER (Unicode codepoint), one decision per word glyph.

Design facts frozen by the contract:
- Training vocabulary = characters of the v2 train split ONLY (plus boundary
  and unk); dev/final-exclusive characters must never enter the table
  (implementation gate 1 asserts this); unknown input characters collapse to
  a shared unk embedding.
- The generation candidate set per episode is vocab characters plus the
  material's unseen codepoints (a pointer-network output space); copy mass on
  duplicate characters merges into one slot, so an unseen dev glyph exits
  through the copy identity channel without ever needing an embedding table.
- The D1 instrument is untouched: strings cross the boundary at encode/
  decode only; M1/M3/M4 and every lesion keep the exact string semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .internalization import content_digest

#: Independent version space from the byte graphs (contract section 2.5).
SEQUENCE_CHAR_WORKSPACE_FORMAT = "taiji-sequence-char-workspace-v1"
SEQUENCE_CHAR_WORKSPACE_VERSION = 1
SEQUENCE_CHAR_WORKSPACE_SUPPORTED_VERSIONS = frozenset({1})
SEQUENCE_CHAR_WORKSPACE_TRAINER_FORMAT = "taiji-sequence-char-workspace-trainer-v1"

#: Slot identities (contract section 2.2).
CHAR_BOUNDARY_SLOT = 0
CHAR_UNK_SLOT = 1

#: Evaluation-only marker returned by :meth:`decode` for the unk slot.
CHAR_UNK_GLYPH = "\ufffd"  # U+FFFD replacement: never legal as an answer glyph


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _sinusoidal_positional_encoding(length: int, width: int) -> torch.Tensor:
    """Fixed sin/cos positional encoding over character rows (same spec as v6)."""

    positions = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
    )
    angles = positions * frequencies
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding


def _find_all(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = text.find(needle)
    while start != -1:
        positions.append(start)
        start = text.find(needle, start + 1)
    return positions


class CharVocab:
    """Train-only character vocabulary; codepoint <-> slot with unk collapse.

    Slots: 0 = boundary, 1 = unk, 2.. = sorted distinct train characters.
    Material characters absent from the table receive *dynamic* candidate slots
    (see :class:`SequenceCharWorkspaceState`) so the copy channel can emit
    unseen glyphs without them entering the embedding/decoder matrices.
    """

    def __init__(self, train_text: str) -> None:
        seen = sorted({character for character in train_text if character != "\x00"})
        self.char_to_slot: dict[str, int] = {character: index + 2 for index, character in enumerate(seen)}
        self.slot_to_char: dict[int, str] = {slot: character for character, slot in self.char_to_slot.items()}
        self.size = len(seen) + 2  # boundary + unk + characters

    def encode(self, text: str) -> list[int]:
        return [self.char_to_slot.get(character, CHAR_UNK_SLOT) for character in text]

    def slot_of(self, character: str) -> int:
        return self.char_to_slot.get(character, CHAR_UNK_SLOT)

    def glyph_of(self, slot: int) -> str:
        if slot == CHAR_BOUNDARY_SLOT:
            return ""
        if slot == CHAR_UNK_SLOT:
            return CHAR_UNK_GLYPH
        return self.slot_to_char[slot]


@dataclass(frozen=True)
class SequenceCharConfig:
    """Geometry mirrors the byte v8 arm at the character unit."""

    prefix_width: int = 48
    slot_width: int = 48
    renderer_width: int = 64
    seed: int = 20260917
    max_sequence_chars: int = 256
    readout_heads: int = 4
    copy_induction: bool = True

    def __post_init__(self) -> None:
        for name in ("prefix_width", "slot_width", "renderer_width"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"char workspace {name} must be positive")
        if int(self.max_sequence_chars) <= 0:
            raise ValueError("char workspace max_sequence_chars must be positive")
        if not 1 <= int(self.readout_heads) <= 4:
            raise ValueError("char workspace readout_heads must be in [1, 4]")

    def to_payload(self) -> dict[str, Any]:
        return {
            "prefix_width": int(self.prefix_width),
            "slot_width": int(self.slot_width),
            "renderer_width": int(self.renderer_width),
            "seed": int(self.seed),
            "max_sequence_chars": int(self.max_sequence_chars),
            "readout_heads": int(self.readout_heads),
            "copy_induction": bool(self.copy_induction),
        }


@dataclass(frozen=True)
class SequenceCharWorkspaceState:
    """Renderer state at the character unit (frozen per contract style)."""

    workspace_key: torch.Tensor  # (rows, slot_width), rows = prefix characters
    workspace_value: torch.Tensor  # (rows, slot_width)
    renderer_state: torch.Tensor  # (renderer_width,)
    #: codepoint of each material row (identity channel for induction/copy).
    entry_codepoints: tuple[int, ...]
    #: candidate slot of each material row (vocab slot or dynamic extra slot).
    entry_slots: tuple[int, ...]
    #: codepoint -> candidate slot for this episode (vocab chars + extras).
    char2slot: Mapping[str, int]
    #: slot -> codepoint for the extra (non-vocab) candidates.
    extra_slot_codepoints: Mapping[int, int]
    candidate_size: int
    #: v8-style induction pointer: the codepoint emitted at the previous step.
    last_emitted_codepoint: int | None = None


@dataclass(frozen=True)
class CharGenerationResult:
    text: str
    stopped_on_boundary: bool
    steps: int


#: Canonical parameter order; new names must be APPENDED at the tail
#: (the D5 canonical-index lesson applies from module inception).
_CHAR_CANONICAL_ORDER: tuple[str, ...] = (
    "prefix_embedding",
    "prefix_input",
    "prefix_recur",
    "prefix_bias",
    "start_vector",
    "head_query_1",
    "head_query_2",
    "head_query_3",
    "head_query_4",
    "evidence_key",
    "evidence_value",
    "copy_gate_weight",
    "copy_gate_bias",
    "answer_start_weight",
    "answer_start_bias",
    "renderer_embedding",
    "renderer_input",
    "renderer_recur",
    "renderer_bias",
    "decoder",
    "decoder_bias",
    "copy_induce_bias",
)


class SequenceCharWorkspace:
    """Character-unit twin of the byte workspace (contract section 2)."""

    def __init__(
        self,
        vocab: CharVocab,
        config: SequenceCharConfig | None = None,
        *,
        value_rotation: int = 0,
        entry_rotation: int = 0,
    ) -> None:
        self.vocab = vocab
        self.config = config or SequenceCharConfig()
        self.value_rotation = int(value_rotation)
        self.entry_rotation = int(entry_rotation)
        pw = int(self.config.prefix_width)
        sw = int(self.config.slot_width)
        rw = int(self.config.renderer_width)
        heads = int(self.config.readout_heads)
        vocab_size = int(vocab.size)
        self._parameters: dict[str, torch.nn.Parameter] = {}

        def make(name: str, shape: tuple[int, ...], scale: float) -> None:
            index = _CHAR_CANONICAL_ORDER.index(name)
            generator = torch.Generator().manual_seed(int(self.config.seed) * 1_000_003 + index)
            tensor = (
                torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
            ) * scale
            self._parameters[name] = torch.nn.Parameter(tensor)

        make("prefix_embedding", (vocab_size, pw), 0.5)
        make("prefix_input", (pw, rw), 1.0 / math.sqrt(pw))
        make("prefix_recur", (rw, rw), 1.0 / math.sqrt(rw))
        make("prefix_bias", (rw,), 0.1)
        make("start_vector", (rw,), 0.5)
        for head in range(1, heads + 1):
            make(f"head_query_{head}", (rw, sw), 1.0 / math.sqrt(rw))
        make("evidence_key", (rw, sw), 1.0 / math.sqrt(rw))
        make("evidence_value", (rw, sw), 1.0 / math.sqrt(rw))
        make("copy_gate_weight", (rw,), 1.0 / math.sqrt(rw))
        make("copy_gate_bias", (1,), 0.0)
        make("answer_start_weight", (rw, rw), 1.0 / math.sqrt(rw))
        make("answer_start_bias", (rw,), 0.0)
        make("renderer_embedding", (vocab_size, rw), 0.5)
        make("renderer_input", (rw, rw), 1.0 / math.sqrt(rw))
        make("renderer_recur", (rw, rw), 1.0 / math.sqrt(rw))
        make("renderer_bias", (rw,), 0.1)
        make("decoder", (rw + sw, vocab_size), 1.0 / math.sqrt(rw + sw))
        make("decoder_bias", (vocab_size,), 0.1)
        # Induction scalar, fixed zero start (v8 rule).
        self._parameters["copy_induce_bias"] = torch.nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ setup

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self._parameters.values()))

    def named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        return tuple((name, self._parameters[name]) for name in _CHAR_CANONICAL_ORDER)

    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self._parameters[name] for name in _CHAR_CANONICAL_ORDER)

    def _readout_parameters(self) -> torch.Tensor:
        heads = int(self.config.readout_heads)
        if heads == 1:
            return self._parameters["head_query_1"].unsqueeze(0)
        return torch.stack(
            [self._parameters[f"head_query_{h}"] for h in range(1, heads + 1)], dim=0
        )

    # ------------------------------------------------------------- tokenizing

    def _episode_slots(self, prefix: str) -> tuple[tuple[int, ...], dict[str, int], dict[int, int]]:
        """Per-row candidate slots plus the episode's dynamic extra slots."""

        char2slot: dict[str, int] = {}
        extra_slot_codepoints: dict[int, int] = {}
        next_extra = int(self.vocab.size)
        for character in prefix:
            if character in char2slot:
                continue
            known = self.vocab.slot_of(character)
            if known != CHAR_UNK_SLOT:
                char2slot[character] = known
            else:
                char2slot[character] = next_extra
                extra_slot_codepoints[next_extra] = ord(character)
                next_extra += 1
        entry_slots = tuple(char2slot[character] for character in prefix)
        return entry_slots, char2slot, extra_slot_codepoints

    # ------------------------------------------------------------- scanning

    def begin_episode(
        self, prefix: str, *, value_rotation: int = 0, entry_rotation: int = 0
    ) -> SequenceCharWorkspaceState:
        """Build the per-episode workspace.  ``entry_rotation`` is the
        evaluation-only misbind lesion, byte-graph twin semantics: the entry
        byte (identity) of each row is cyclically shifted while keys, values
        and candidate slots stay put, so the copy pathway emits from
        misaligned positions.  ``value_rotation`` cyclically shifts the value
        rows against their keys (R2-D2 value-misbind twin).  Both must stay
        zero on every training path.
        """

        if len(prefix) > int(self.config.max_sequence_chars):
            raise ValueError("prefix exceeds max_sequence_chars")
        slots = self.vocab.encode(prefix)
        entry_slots, char2slot, extra_slot_codepoints = self._episode_slots(prefix)
        embedding = self._parameters["prefix_embedding"]
        hidden = torch.zeros(int(self.config.renderer_width), dtype=torch.float32)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        # v5 rule at the character unit: the scan state right before the
        # material marker ("背景" / "线索" / "已知") projects the answer start.
        marker_starts = [
            index
            for marker in ("背景", "线索", "已知")
            for index in _find_all(prefix, marker)
        ]
        question_position = min(marker_starts) if marker_starts else len(prefix)
        question_hidden = hidden
        for position, slot in enumerate(slots):
            row = embedding[slot]
            hidden = torch.tanh(
                row @ self._parameters["prefix_input"]
                + hidden @ self._parameters["prefix_recur"]
                + self._parameters["prefix_bias"]
            )
            keys.append(hidden @ self._parameters["evidence_key"])
            values.append(hidden @ self._parameters["evidence_value"])
            if position + 1 == question_position:
                question_hidden = hidden
        key_rows = torch.stack(keys, dim=0) if keys else torch.zeros(0, int(self.config.slot_width))
        value_rows = torch.stack(values, dim=0) if values else torch.zeros(
            0, int(self.config.slot_width)
        )
        # Fixed sinusoidal positional keys (v6 rule): added AFTER projection.
        if key_rows.shape[0]:
            key_rows = key_rows + _sinusoidal_positional_encoding(
                int(key_rows.shape[0]), int(self.config.slot_width)
            )
        renderer_start = torch.tanh(
            question_hidden @ self._parameters["answer_start_weight"]
            + self._parameters["answer_start_bias"]
        )
        entry_codepoints = tuple(ord(character) for character in prefix)
        if int(value_rotation) and value_rows.shape[0]:
            vshift = int(value_rotation) % int(value_rows.shape[0])
            value_rows = torch.cat((value_rows[vshift:], value_rows[:vshift]), dim=0)
        if entry_rotation:
            shift = int(entry_rotation) % max(1, len(entry_codepoints))
            entry_codepoints = entry_codepoints[shift:] + entry_codepoints[:shift]
            # Copy slots follow the rotated identities (byte-graph twin: the
            # entry byte aligned to each evidence row shifts, keys/values do
            # not); dynamic extras already exist for every material glyph.
            entry_slots = tuple(char2slot[chr(codepoint)] for codepoint in entry_codepoints)
        return SequenceCharWorkspaceState(
            workspace_key=key_rows,
            workspace_value=value_rows,
            renderer_state=renderer_start,
            entry_codepoints=entry_codepoints,
            entry_slots=entry_slots,
            char2slot=char2slot,
            extra_slot_codepoints=extra_slot_codepoints,
            candidate_size=int(self.vocab.size) + len(extra_slot_codepoints),
            last_emitted_codepoint=None,
        )

    # ------------------------------------------------------------ addressing

    def _head_weights(
        self, state: SequenceCharWorkspaceState, *, induce: bool = True
    ) -> torch.Tensor:
        key = state.workspace_key
        renderer = state.renderer_state
        queries = torch.einsum("r,hrw->hw", renderer, self._readout_parameters())
        scores = torch.einsum("nw,hw->hn", key, queries)
        scores = scores / math.sqrt(float(self.config.slot_width))
        if self.config.copy_induction and induce and state.last_emitted_codepoint is not None:
            previous = int(state.last_emitted_codepoint)
            match_index = [
                j
                for j in range(1, len(state.entry_codepoints))
                if state.entry_codepoints[j - 1] == previous
            ]
            if match_index:
                scores = scores.clone()
                scores[:, match_index] = scores[:, match_index] + self._parameters[
                    "copy_induce_bias"
                ]
        return torch.softmax(scores, dim=-1)

    def _read(self, state: SequenceCharWorkspaceState) -> torch.Tensor:
        weights = self._head_weights(state)
        return (weights @ state.workspace_value).mean(dim=0)

    # ------------------------------------------------------------- generation

    def step(
        self, state: SequenceCharWorkspaceState, previous_slot: int
    ) -> tuple[SequenceCharWorkspaceState, torch.Tensor]:
        """One renderer step at the character unit.

        Returns ``(new_state, logits_over_candidates)``; logits is dense over
        the episode candidate space (vocab slots + dynamic extras).
        """

        vocab_size = int(self.vocab.size)
        embedding_slot = (
            previous_slot if previous_slot < vocab_size else CHAR_UNK_SLOT
        )
        renderer_state = torch.tanh(
            self._parameters["renderer_embedding"][embedding_slot]
            @ self._parameters["renderer_input"]
            + state.renderer_state @ self._parameters["renderer_recur"]
            + self._parameters["renderer_bias"]
        )
        read = self._read(state)
        vocab_logits = (
            torch.cat((renderer_state, read), dim=0) @ self._parameters["decoder"]
            + self._parameters["decoder_bias"]
        )
        new_state = replace(state, renderer_state=renderer_state)
        return new_state, vocab_logits

    def step_distribution(
        self, state: SequenceCharWorkspaceState, previous_slot: int
    ) -> tuple[SequenceCharWorkspaceState, torch.Tensor, torch.Tensor]:
        """Mixture over candidates plus the raw copy-slot distribution."""

        new_state, vocab_logits = self.step(state, int(previous_slot))
        weights = self._head_weights(new_state)  # induction uses carried pointer
        stacked = weights if weights.ndim == 2 else weights.unsqueeze(0)
        copy_distribution = torch.zeros(int(new_state.candidate_size), dtype=torch.float32)
        slot_tensor = torch.tensor(new_state.entry_slots, dtype=torch.long)
        for head_weights in stacked:
            copy_distribution = copy_distribution.index_add(0, slot_tensor, head_weights)
        copy_distribution = copy_distribution / stacked.shape[0]
        gate = torch.sigmoid(
            new_state.renderer_state @ self._parameters["copy_gate_weight"]
            + self._parameters["copy_gate_bias"]
        )
        mixture = torch.zeros(int(new_state.candidate_size), dtype=torch.float32)
        mixture[: int(self.vocab.size)] = gate * torch.softmax(vocab_logits, dim=0)
        mixture = mixture + (1.0 - gate) * copy_distribution
        return new_state, mixture, copy_distribution

    def _advance_pointer(
        self, state: SequenceCharWorkspaceState, emitted_slot: int
    ) -> SequenceCharWorkspaceState:
        codepoint = self._slot_codepoint(state, emitted_slot)
        return replace(state, last_emitted_codepoint=codepoint)

    def _slot_codepoint(self, state: SequenceCharWorkspaceState, slot: int) -> int:
        if slot == CHAR_BOUNDARY_SLOT:
            return 0
        if slot == CHAR_UNK_SLOT:
            return 1
        if slot < int(self.vocab.size):
            return ord(self.vocab.glyph_of(slot))
        return int(state.extra_slot_codepoints[slot])

    def glyph_for_slot(self, state: SequenceCharWorkspaceState, slot: int) -> str:
        if slot == CHAR_BOUNDARY_SLOT:
            return ""
        if slot == CHAR_UNK_SLOT:
            return CHAR_UNK_GLYPH
        if slot < int(self.vocab.size):
            return self.vocab.glyph_of(slot)
        return chr(int(state.extra_slot_codepoints[slot]))

    @torch.no_grad()
    def generate_lesioned(
        self,
        prefix: str,
        *,
        value_rotation: int = 0,
        entry_rotation: int = 0,
        max_chars: int | None = None,
    ) -> CharGenerationResult:
        """Greedy generation under an evaluation lesion applied at episode start.

        Induction pointer transitions run inside the lesioned episode exactly
        as intact generation does; the answer glyphs' slot identities come from
        the INTACT mapping (a lesion may not change what the correct answer
        character is, only which row supplies it).
        """

        limit = int(max_chars if max_chars is not None else self.config.max_sequence_chars)
        state = self.begin_episode(
            prefix, value_rotation=int(value_rotation), entry_rotation=int(entry_rotation)
        )
        previous_slot = CHAR_BOUNDARY_SLOT
        produced: list[str] = []
        stopped = False
        steps = 0
        while steps < limit:
            state, mixture, _ = self.step_distribution(state, previous_slot)
            slot = int(mixture.argmax())
            steps += 1
            if slot == CHAR_BOUNDARY_SLOT:
                stopped = True
                break
            glyph = self.glyph_for_slot(state, slot)
            produced.append(glyph)
            state = replace(state, last_emitted_codepoint=ord(glyph) if glyph else None)
            previous_slot = slot
        return CharGenerationResult("".join(produced), stopped, steps)

    @torch.no_grad()
    def generate(self, prefix: str, *, max_chars: int | None = None) -> CharGenerationResult:
        limit = int(max_chars if max_chars is not None else self.config.max_sequence_chars)
        state = self.begin_episode(prefix)
        previous_slot = CHAR_BOUNDARY_SLOT
        produced: list[str] = []
        stopped = False
        steps = 0
        while steps < limit:
            state, mixture, _ = self.step_distribution(state, previous_slot)
            slot = int(mixture.argmax())
            steps += 1
            if slot == CHAR_BOUNDARY_SLOT:
                stopped = True
                break
            produced.append(self.vocab.glyph_of(slot) if slot < int(self.vocab.size) else chr(int(state.extra_slot_codepoints[slot])))
            state = self._advance_pointer(state, slot)
            previous_slot = slot
        return CharGenerationResult("".join(produced), stopped, steps)

    # --------------------------------------------------------- teacher forcing

    def teacher_forced_distributions(
        self, prefix: str, response: str, *, entry_rotation: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(mixture, copy) rows over candidates for positions 0..len(response).

        Teacher forcing feeds the TRUE previous glyph as the renderer input and
        carries the true previous codepoint as the induction pointer (the
        byte-graph rule: emissions before the first carry no bonus).
        ``entry_rotation`` is the evaluation-only misbind lesion (training and
        the D7 gates always use 0).
        """

        state = self.begin_episode(prefix, entry_rotation=int(entry_rotation))
        mixtures: list[torch.Tensor] = []
        copies: list[torch.Tensor] = []
        previous_slot = CHAR_BOUNDARY_SLOT
        previous_char: str | None = None
        for character in response:
            state = replace(
                state,
                last_emitted_codepoint=ord(previous_char) if previous_char is not None else None,
            )
            state, mixture, copy_distribution = self.step_distribution(state, previous_slot)
            mixtures.append(mixture)
            copies.append(copy_distribution)
            previous_slot = state.char2slot.get(character, self.vocab.slot_of(character))
            previous_char = character
        state = replace(state, last_emitted_codepoint=ord(previous_char or ""))
        state, mixture, copy_distribution = self.step_distribution(state, previous_slot)
        mixtures.append(mixture)
        copies.append(copy_distribution)
        return torch.stack(mixtures, dim=0), torch.stack(copies, dim=0)

    def sequence_loss(
        self,
        prefix: str,
        response: str,
        *,
        copy_value_weight: float = 0.0,
        value_mask: Sequence[bool] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Answer CE over the candidate space + optional D4 copy-value NLL.

        ``value_mask`` selects response characters (string-indexed, matching
        the contract's character-unit rule: fact/sof = whole answer, negation
        = minus the leading 不是, others = empty).
        """

        mixture, copies = self.teacher_forced_distributions(prefix, response)
        state0 = self.begin_episode(prefix)
        boundary = float(CHAR_BOUNDARY_SLOT)
        targets = torch.tensor(
            [state0.char2slot.get(character, self.vocab.slot_of(character)) for character in response]
            + [int(boundary)],
            dtype=torch.long,
        )
        target_probs = mixture.gather(1, targets.unsqueeze(1)).squeeze(1)
        per_position = -target_probs.clamp_min(1e-12).log()
        loss = per_position.mean()
        value_prob_mean = 0.0
        if copy_value_weight > 0.0:
            weight = float(copy_value_weight)
            if weight < 0.0:
                raise ValueError("copy_value_weight cannot be negative")
            if value_mask is None:
                raise ValueError("copy_value_weight requires value_mask")
            mask = torch.tensor([bool(flag) for flag in value_mask], dtype=torch.bool)
            if mask.numel() != len(response):
                raise ValueError("value_mask must align with the response characters")
            if mask.any():
                hits = copies[:-1][mask].gather(
                    1, targets[:-1][mask].unsqueeze(1)
                ).squeeze(1)
                loss = loss + weight * (-hits.clamp_min(1e-12).log().mean())
                value_prob_mean = float(hits.detach().mean())
        with torch.no_grad():
            predictions = mixture.argmax(dim=1)
            correct = int((predictions == targets).sum())
            metrics = {
                "positions": int(targets.numel()),
                "correct": correct,
                "accuracy": correct / max(1, int(targets.numel())),
                "mean_surprise": float(loss.detach()),
                "copy_value_prob_mean": value_prob_mean,
            }
        return loss, metrics


def _clone_optimizer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return dict(convert(dict(state)))


class SequenceCharTrainer:
    """Deterministic backprop trainer for the char graph (byte-trainer twin)."""

    def __init__(
        self,
        workspace: SequenceCharWorkspace,
        *,
        learning_rate: float = 0.01,
        code_revision: str = "",
    ) -> None:
        self.workspace = workspace
        self.learning_rate = _finite_positive(learning_rate, "learning_rate")
        self.code_revision = str(code_revision)
        self.optimizer = torch.optim.Adam(workspace.parameters(), lr=self.learning_rate)
        self.episodes: tuple[tuple[str, str], ...] = ()
        self.copy_value_weight = 0.0
        self.cursor = 0
        self.global_step = 0

    def enable_copy_value_supervision(self, weight: float) -> None:
        value = float(weight)
        if value < 0.0:
            raise ValueError("copy_value_weight cannot be negative")
        self.copy_value_weight = value

    def set_episodes(self, episodes: Sequence[tuple[str, str]]) -> None:
        self.episodes = tuple((str(prefix), str(response)) for prefix, response in episodes)
        self.cursor = 0

    def train_step(
        self,
        batch: Sequence[tuple[str, str]],
        *,
        value_masks: Sequence[Sequence[bool]] | None = None,
    ) -> dict[str, Any]:
        if not batch:
            raise ValueError("char workspace batch cannot be empty")
        if value_masks is not None and len(value_masks) != len(batch):
            raise ValueError("value_masks must align with the batch")
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for position, (prefix, response) in enumerate(batch):
            mask = None if value_masks is None else value_masks[position]
            loss, _ = self.workspace.sequence_loss(
                prefix, response, copy_value_weight=self.copy_value_weight, value_mask=mask
            )
            losses.append(loss)
        total = torch.stack(losses).mean()
        total.backward()
        self.optimizer.step()
        self.global_step += 1
        return {"loss": float(total.detach()), "global_step": self.global_step}

    # ------------------------------------------------------------- checkpoint

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": SEQUENCE_CHAR_WORKSPACE_TRAINER_FORMAT,
            "version": SEQUENCE_CHAR_WORKSPACE_VERSION,
            "serialization": "torch.save/atomic",
            "code_revision": self.code_revision,
            "config": self.workspace.config.to_payload(),
            "vocab_chars": "".join(
                self.workspace.vocab.slot_to_char[slot]
                for slot in sorted(self.workspace.vocab.slot_to_char)
            ),
            "parameters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.workspace.named_parameters()
            },
            "optimizer": _clone_optimizer_state(self.optimizer.state_dict()),
            "rng_state": torch.get_rng_state().clone(),
            "episodes": [{"prefix": p, "response": r} for p, r in self.episodes],
            "global_step": int(self.global_step),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        from .persistence import atomic_save

        return atomic_save(self.checkpoint(), path)

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> SequenceCharTrainer:
        if payload.get("format") != SEQUENCE_CHAR_WORKSPACE_TRAINER_FORMAT:
            raise ValueError("not a sequence char workspace trainer checkpoint")
        version = int(payload.get("version", -1))
        if version not in SEQUENCE_CHAR_WORKSPACE_SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported char workspace checkpoint version: {version}")
        digest = payload.get("checkpoint_digest")
        body = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        if digest is None or str(digest) != str(content_digest(body)):
            raise ValueError("char workspace checkpoint digest mismatch")
        vocab = CharVocab(str(payload["vocab_chars"]))
        workspace = SequenceCharWorkspace(vocab, SequenceCharConfig(**dict(payload["config"])))
        parameters = payload["parameters"]
        missing = [name for name in _CHAR_CANONICAL_ORDER if name not in parameters]
        if missing:
            raise ValueError(f"char workspace checkpoint missing parameters: {missing}")
        with torch.no_grad():
            for name in _CHAR_CANONICAL_ORDER:
                tensor = parameters[name].detach().to(dtype=torch.float32).clone()
                parameter = workspace._parameters[name]
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise ValueError(f"char workspace parameter shape mismatch: {name}")
                parameter.copy_(tensor)
        trainer = cls(
            workspace,
            learning_rate=float(payload.get("learning_rate", 0.01))
            if "learning_rate" in payload
            else 0.01,
            code_revision=str(payload.get("code_revision", "")),
        )
        try:
            trainer.optimizer.load_state_dict(dict(payload["optimizer"]))
        except (ValueError, KeyError):
            pass  # fresh optimizer state when param identity differs
        trainer.global_step = int(payload.get("global_step", 0))
        episodes = payload.get("episodes", [])
        trainer.episodes = tuple(
            (str(item["prefix"]), str(item["response"])) for item in episodes
        )
        state = payload.get("rng_state")
        if isinstance(state, torch.Tensor):
            torch.set_rng_state(state)
        return trainer
