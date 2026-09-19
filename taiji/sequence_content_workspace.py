"""R2 content-binding workspace: arms A (implicit) and B (object/relation slots).

Module: taiji/sequence_content_workspace.py
Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md (§2)

Independent module and format space from the byte/char graphs (v2-v8,
char-v1): the old modules keep their contracts and regressions untouched.

Shared front end (identical in both arms, initialised by stable parameter
NAME so arm choice can never change common initial values):

- material and question are separate explicit fields; each character becomes
  ``char_embedding[slot] + byte_feature_projection(char)`` (48-dim).  Byte
  features are added to EVERY character uniformly: up to 4 UTF-8 bytes, 8 bit
  dimensions per byte position, a 5-way length one-hot (padding slot distinct
  from byte 0), projected to 48.  Unseen characters therefore stay
  distinguishable before any semantic learning happens.
- recurrent scans (hidden 64) produce per-position evidence key/value rows
  (48) over the material and a question vector from the question.
- generation: 64-dim renderer consuming [previous char feature, content C,
  source read] each step; vocab softmax mixed with a C-conditioned copy
  distribution over the material's characters (dynamic codepoint candidates
  for unseen glyphs, duplicate-character mass merged, boundary slot 0).  TF
  and free generation share the same ``step``; an unseen output character
  keeps its true codepoint and its own byte-derived input feature next step.
- source access: all material rows stay available; the copy query and the
  mixture gate consume C; both arms share the learnable emission-successor
  bonus (initialised to exactly zero).

Arm-specific content modules (independently initialised):
- ``B``: 6 learnable 48-dim slots, 3 rounds of shared-GRU binding over the
  material evidence (per-position softmax over slots, then per-slot
  normalisation over positions); every ordered slot pair through one shared
  MLP (two slots + question vector, hidden 96, out 48); question-conditioned
  pooling of slots and relations projected to the 48-dim content C.
- ``A``: 6 independent question-conditioned evidence readouts (per-way
  position softmax, no cross-way competition), concatenated and projected to
  the same 48-dim C.  No slots, no pair computation.

The runtime contract is structural: ``begin_episode`` accepts only the
question and material strings -- never shapes, answers, pair labels or gold
roles.  Inputs over ``max_input_chars`` and free generation beyond
``max_output_chars`` are explicit range errors, never silent truncation.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .internalization import content_digest
from .persistence import atomic_save
from .sequence_char_workspace import CharVocab

SEQUENCE_CONTENT_WORKSPACE_FORMAT = "taiji-sequence-content-workspace-v1"
SEQUENCE_CONTENT_WORKSPACE_VERSION = 1
SEQUENCE_CONTENT_WORKSPACE_SUPPORTED_VERSIONS = frozenset({1})
SEQUENCE_CONTENT_WORKSPACE_TRAINER_FORMAT = "taiji-sequence-content-workspace-trainer-v1"

#: Identity of the dynamic candidate construction rule (contract section 2:
#: saved with every checkpoint so candidate behaviour is reproducible).
CANDIDATE_CONSTRUCTION_VERSION = "content-candidates-v1"
#: Selection rule recorded in checkpoints (contract section 4).
SELECTION_RULE_ID = "content-binding-selection-v1"
CONTRACT_PATH = "plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md"

CONTENT_BOUNDARY_SLOT = 0
CONTENT_UNK_SLOT = 1
CONTENT_UNK_GLYPH = "\ufffd"
ARMS = ("A", "B")

#: D6 emission-successor semantics: a bonus applies to the row AFTER a row
#: whose character equals the previously emitted character.
_EMISSION_SUCCESSOR = True


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _name_seed(seed: int, name: str) -> int:
    """Stable per-name seed: shared names init identically in both arms and
    parameter creation order can never influence initial values."""

    digest = hashlib.sha256(f"{int(seed)}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class SequenceContentConfig:
    """Geometry frozen by contract section 2 (prototype sizes, not ceilings)."""

    arm: str = "B"
    prefix_width: int = 48
    hidden_width: int = 64
    evidence_width: int = 48
    slot_count: int = 6
    binding_rounds: int = 3
    relation_hidden: int = 96
    content_width: int = 48
    renderer_width: int = 64
    byte_width: int = 8
    max_input_chars: int = 256
    max_output_chars: int = 32
    seed: int = 20260920

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}: {self.arm}")
        for name in (
            "prefix_width",
            "hidden_width",
            "evidence_width",
            "slot_count",
            "binding_rounds",
            "relation_hidden",
            "content_width",
            "renderer_width",
            "byte_width",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"content workspace {name} must be positive")
        if int(self.max_input_chars) <= 0 or int(self.max_output_chars) <= 0:
            raise ValueError("content workspace char limits must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "prefix_width": int(self.prefix_width),
            "hidden_width": int(self.hidden_width),
            "evidence_width": int(self.evidence_width),
            "slot_count": int(self.slot_count),
            "binding_rounds": int(self.binding_rounds),
            "relation_hidden": int(self.relation_hidden),
            "content_width": int(self.content_width),
            "renderer_width": int(self.renderer_width),
            "byte_width": int(self.byte_width),
            "max_input_chars": int(self.max_input_chars),
            "max_output_chars": int(self.max_output_chars),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class SequenceContentWorkspaceState:
    """Per-episode state; temporary W/R/C fields reset on every episode."""

    material_keys: torch.Tensor  # (L, evidence_width)
    material_values: torch.Tensor  # (L, evidence_width)
    renderer_state: torch.Tensor  # (renderer_width,)
    content: torch.Tensor  # (content_width,) -- C
    question_vec: torch.Tensor  # (hidden_width,)
    material_chars: tuple[str, ...]
    entry_slots: tuple[int, ...]  # candidate slot per material position
    char2slot: Mapping[str, int]  # episode candidates: vocab slots + extras
    extra_slot_codepoints: Mapping[int, int]
    candidate_size: int


@dataclass(frozen=True)
class ContentGenerationResult:
    text: str
    stopped_on_boundary: bool
    steps: int
    range_error: bool


# Canonical parameter order.  New names must be APPENDED at the tail
# (the D5 canonical-index lesson applies from module inception).
_COMMON_ORDER: tuple[str, ...] = (
    "char_embedding",
    "byte_proj",
    "byte_proj_bias",
    "material_input",
    "material_recur",
    "material_bias",
    "evidence_key",
    "evidence_value",
    "question_input",
    "question_recur",
    "question_bias",
    "start_weight",
    "start_bias",
    "renderer_input",
    "renderer_recur",
    "renderer_bias",
    "decoder",
    "decoder_bias",
    "copy_query_state",
    "copy_query_content",
    "copy_gate_weight",
    "copy_gate_bias",
    "copy_induce_bias",
)
_ARM_B_ORDER: tuple[str, ...] = (
    "slot_init",
    "slot_gru_weight_input",
    "slot_gru_weight_hidden",
    "slot_gru_bias_input",
    "slot_gru_bias_hidden",
    "relation_mlp_input",
    "relation_mlp_bias",
    "relation_mlp_output",
    "relation_mlp_output_bias",
    "slot_pool_query",
    "relation_pool_query",
    "content_proj",
    "content_proj_bias",
)


def arm_parameter_order(arm: str, slot_count: int = 6) -> tuple[str, ...]:
    """Full canonical order for one arm (common block + arm content block)."""

    if arm == "B":
        return _COMMON_ORDER + _ARM_B_ORDER
    heads = tuple(f"head_query_{index}" for index in range(1, slot_count + 1))
    return _COMMON_ORDER + heads + (
        "a_content_mlp_input",
        "a_content_mlp_bias",
        "a_content_mlp_output",
        "a_content_mlp_output_bias",
    )


class SequenceContentWorkspace:
    """Two-arm content workspace (contract section 2)."""

    def __init__(
        self,
        vocab: CharVocab,
        config: SequenceContentConfig | None = None,
    ) -> None:
        self.vocab = vocab
        self.config = config or SequenceContentConfig()
        arm = self.config.arm
        pw = int(self.config.prefix_width)
        hw = int(self.config.hidden_width)
        ew = int(self.config.evidence_width)
        sc = int(self.config.slot_count)
        rh = int(self.config.relation_hidden)
        cw = int(self.config.content_width)
        rw = int(self.config.renderer_width)
        bw = int(self.config.byte_width)
        vocab_size = int(vocab.size)
        self._parameters: dict[str, torch.nn.Parameter] = {}

        def make(name: str, shape: tuple[int, ...], scale: float) -> None:
            generator = torch.Generator().manual_seed(_name_seed(self.config.seed, name))
            tensor = (
                torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
            ) * scale
            self._parameters[name] = torch.nn.Parameter(tensor)

        # ---- shared front end -------------------------------------------------
        make("char_embedding", (vocab_size, pw), 0.5)
        byte_features = 4 * bw + 5
        make("byte_proj", (pw, byte_features), 1.0 / math.sqrt(byte_features))
        make("byte_proj_bias", (pw,), 0.0)
        make("material_input", (pw, hw), 1.0 / math.sqrt(pw))
        make("material_recur", (hw, hw), 1.0 / math.sqrt(hw))
        make("material_bias", (hw,), 0.1)
        make("evidence_key", (hw, ew), 1.0 / math.sqrt(hw))
        make("evidence_value", (hw, ew), 1.0 / math.sqrt(hw))
        make("question_input", (pw, hw), 1.0 / math.sqrt(pw))
        make("question_recur", (hw, hw), 1.0 / math.sqrt(hw))
        make("question_bias", (hw,), 0.1)
        make("start_weight", (hw, rw), 1.0 / math.sqrt(hw))
        make("start_bias", (rw,), 0.0)
        make("renderer_input", (pw + cw + ew, rw), 1.0 / math.sqrt(pw + cw + ew))
        make("renderer_recur", (rw, rw), 1.0 / math.sqrt(rw))
        make("renderer_bias", (rw,), 0.1)
        make("decoder", (rw + cw, vocab_size), 1.0 / math.sqrt(rw + cw))
        make("decoder_bias", (vocab_size,), 0.1)
        make("copy_query_state", (rw, ew), 1.0 / math.sqrt(rw))
        make("copy_query_content", (cw, ew), 1.0 / math.sqrt(cw))
        make("copy_gate_weight", (rw + cw,), 1.0 / math.sqrt(rw + cw))
        make("copy_gate_bias", (1,), 0.0)
        # Emission-successor bonus: fixed zero start in BOTH arms (contract).
        self._parameters["copy_induce_bias"] = torch.nn.Parameter(torch.zeros(1))

        # ---- arm-specific content module --------------------------------------
        if arm == "B":
            make("slot_init", (sc, ew), 0.5)
            gates = 3 * ew
            make("slot_gru_weight_input", (gates, ew), 1.0 / math.sqrt(ew))
            make("slot_gru_weight_hidden", (gates, ew), 1.0 / math.sqrt(ew))
            make("slot_gru_bias_input", (gates,), 0.0)
            make("slot_gru_bias_hidden", (gates,), 0.0)
            make("relation_mlp_input", (2 * ew + hw, rh), 1.0 / math.sqrt(2 * ew + hw))
            make("relation_mlp_bias", (rh,), 0.1)
            make("relation_mlp_output", (rh, ew), 1.0 / math.sqrt(rh))
            make("relation_mlp_output_bias", (ew,), 0.1)
            make("slot_pool_query", (hw, ew), 1.0 / math.sqrt(hw))
            make("relation_pool_query", (hw, ew), 1.0 / math.sqrt(hw))
            make("content_proj", (2 * ew, cw), 1.0 / math.sqrt(2 * ew))
            make("content_proj_bias", (cw,), 0.1)
        else:
            for index in range(1, sc + 1):
                make(f"head_query_{index}", (hw, ew), 1.0 / math.sqrt(hw))
            reads = sc * ew
            make("a_content_mlp_input", (reads, rh), 1.0 / math.sqrt(reads))
            make("a_content_mlp_bias", (rh,), 0.1)
            make("a_content_mlp_output", (rh, ew), 1.0 / math.sqrt(rh))
            make("a_content_mlp_output_bias", (ew,), 0.1)

        self._order = arm_parameter_order(arm, sc)
        missing = [name for name in self._order if name not in self._parameters]
        if missing:
            raise RuntimeError(f"content workspace init missed parameters: {missing}")

    # ------------------------------------------------------------------ setup

    @property
    def arm(self) -> str:
        return self.config.arm

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self._parameters.values()))

    def named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        return tuple((name, self._parameters[name]) for name in self._order)

    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self._parameters[name] for name in self._order)

    def compute_profile(self) -> dict[str, Any]:
        """Disclosed parameter/compute shape difference between the arms.

        Rough per-episode multiply-accumulate counts for an L-character
        material and a 32-step free generation; the contract requires the B/A
        difference to be disclosed rather than hidden (section 1).
        """

        pw, hw, ew = 48, 64, 48
        sc, rh, cw, rw = (
            int(self.config.slot_count),
            int(self.config.relation_hidden),
            int(self.config.content_width),
            int(self.config.renderer_width),
        )
        length = 32
        common_mac = length * (pw * hw) + 2 * length * (hw * ew) + 32 * (
            (pw + cw + ew) * rw + rw * rw + (rw + cw) * int(self.vocab.size)
        )
        if self.arm == "B":
            rounds = int(self.config.binding_rounds)
            binding = rounds * (length * ew * sc + sc * (3 * ew * ew * 2))
            relations = sc * sc * (2 * ew + hw) * rh * 2
            pooling = sc * (hw * ew) + sc * sc * (hw * ew) + 2 * ew * cw
            content_mac = binding + relations + pooling
        else:
            heads = sc * (hw * ew + length * ew)
            content_mac = heads + sc * ew * rh * 2 + rh * ew
        return {
            "arm": self.arm,
            "parameter_count": self.parameter_count(),
            "per_episode_mac_estimate": int(common_mac + content_mac),
            "content_module_mac_estimate": int(content_mac),
        }

    # ---------------------------------------------------------- byte identity

    def _byte_features(self, character: str | None) -> torch.Tensor:
        bw = int(self.config.byte_width)
        if character is None:
            # padding vector: zero bits with the dedicated padding length slot
            features = torch.zeros(4 * bw + 5, dtype=torch.float32)
            features[4 * bw + 4] = 1.0
            return features
        raw = character.encode("utf-8")
        if len(raw) > 4:
            raise ValueError(
                f"character {character!r} needs {len(raw)} UTF-8 bytes; the "
                "contract supports at most 4"
            )
        bits = torch.zeros(4, bw, dtype=torch.float32)
        for index, byte in enumerate(raw):
            for bit in range(bw):
                bits[index, bit] = float((byte >> (bw - 1 - bit)) & 1)
        length_onehot = torch.zeros(5, dtype=torch.float32)
        length_onehot[len(raw) - 1] = 1.0
        return torch.cat((bits.reshape(-1), length_onehot))

    def _char_input_feature(self, character: str, slot: int) -> torch.Tensor:
        """Shared front-end feature: embedding slot (unk for unseen) plus the
        uniform byte-feature projection -- unseen characters stay distinct."""

        byte_tensor = self._byte_features(character)
        projected = (
            byte_tensor @ self._parameters["byte_proj"].T
            + self._parameters["byte_proj_bias"]
        )
        return self._parameters["char_embedding"][slot] + projected

    def _char_output_feature(self, character: str | None) -> torch.Tensor:
        if character is None:
            return self._char_input_feature("", CONTENT_BOUNDARY_SLOT)
        slot = self.vocab.slot_of(character)
        return self._char_input_feature(character, slot)

    # ------------------------------------------------------------- tokenizing

    def _episode_candidates(
        self, material: str
    ) -> tuple[dict[str, int], dict[int, int]]:
        char2slot: dict[str, int] = {}
        extra_slot_codepoints: dict[int, int] = {}
        next_extra = int(self.vocab.size)
        for character in material:
            if character in char2slot:
                continue
            known = self.vocab.slot_of(character)
            if known != CONTENT_UNK_SLOT:
                char2slot[character] = known
            else:
                char2slot[character] = next_extra
                extra_slot_codepoints[next_extra] = ord(character)
                next_extra += 1
        return char2slot, extra_slot_codepoints

    def _scan(
        self, text: str, input_name: str, recur_name: str, bias_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent scan returning (per-step hiddens, final hidden)."""

        hidden = torch.zeros(int(self.config.hidden_width), dtype=torch.float32)
        steps: list[torch.Tensor] = []
        for character in text:
            slot = self.vocab.slot_of(character)
            feature = self._char_input_feature(character, slot)
            hidden = torch.tanh(
                feature @ self._parameters[input_name]
                + hidden @ self._parameters[recur_name]
                + self._parameters[bias_name]
            )
            steps.append(hidden)
        rows = (
            torch.stack(steps, dim=0)
            if steps
            else torch.zeros(0, int(self.config.hidden_width))
        )
        return rows, hidden

    # -------------------------------------------------------- content modules

    def _gru_cell(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        weight_input = self._parameters["slot_gru_weight_input"]
        weight_hidden = self._parameters["slot_gru_weight_hidden"]
        bias_input = self._parameters["slot_gru_bias_input"]
        bias_hidden = self._parameters["slot_gru_bias_hidden"]
        ew = int(self.config.evidence_width)
        gates = x @ weight_input.T + h @ weight_hidden.T + bias_input + bias_hidden
        r = torch.sigmoid(gates[:ew])
        z = torch.sigmoid(gates[ew : 2 * ew])
        n = torch.tanh(
            gates[2 * ew :]
            + r * (h @ weight_hidden[2 * ew :].T + bias_hidden[2 * ew :])
        )
        return (1.0 - z) * n + z * h

    def _content_b(
        self, values: torch.Tensor, question_vec: torch.Tensor
    ) -> torch.Tensor:
        sc = int(self.config.slot_count)
        ew = int(self.config.evidence_width)
        slots = self._parameters["slot_init"]
        length = int(values.shape[0])
        for _round in range(int(self.config.binding_rounds)):
            if length:
                scores = values @ slots.T / math.sqrt(float(ew))
                assignment = torch.softmax(scores, dim=1)  # softmax over slots
                column = assignment.sum(dim=0).clamp_min(1e-8)  # per-slot normaliser
                weights = assignment / column.unsqueeze(0)  # positions per slot
                slot_inputs = weights.T @ values
            else:
                slot_inputs = torch.zeros(sc, ew)
            slots = torch.stack(
                [self._gru_cell(slot_inputs[i], slots[i]) for i in range(sc)], dim=0
            )
        pairs = torch.stack(
            [
                torch.cat((slots[i], slots[j], question_vec))
                for i in range(sc)
                for j in range(sc)
            ],
            dim=0,
        )
        hidden = torch.tanh(pairs @ self._parameters["relation_mlp_input"] + self._parameters["relation_mlp_bias"])
        relations = hidden @ self._parameters["relation_mlp_output"] + self._parameters["relation_mlp_output_bias"]
        slot_query = question_vec @ self._parameters["slot_pool_query"]
        slot_weights = torch.softmax(slots @ slot_query / math.sqrt(float(ew)), dim=0)
        pooled_slots = slot_weights @ slots
        relation_query = question_vec @ self._parameters["relation_pool_query"]
        relation_weights = torch.softmax(
            relations @ relation_query / math.sqrt(float(ew)), dim=0
        )
        pooled_relations = relation_weights @ relations
        return torch.tanh(
            torch.cat((pooled_slots, pooled_relations)) @ self._parameters["content_proj"]
            + self._parameters["content_proj_bias"]
        )

    def _content_a(
        self, values: torch.Tensor, question_vec: torch.Tensor
    ) -> torch.Tensor:
        ew = int(self.config.evidence_width)
        length = int(values.shape[0])
        reads = []
        for index in range(1, int(self.config.slot_count) + 1):
            query = question_vec @ self._parameters[f"head_query_{index}"]
            if length:
                scores = values @ query / math.sqrt(float(ew))
                weights = torch.softmax(scores, dim=0)  # per-way, no cross-way competition
                reads.append(weights @ values)
            else:
                reads.append(torch.zeros(ew))
        hidden = torch.tanh(
            torch.cat(reads) @ self._parameters["a_content_mlp_input"]
            + self._parameters["a_content_mlp_bias"]
        )
        return torch.tanh(
            hidden @ self._parameters["a_content_mlp_output"]
            + self._parameters["a_content_mlp_output_bias"]
        )

    # ------------------------------------------------------------ episode init

    def begin_episode(self, question: str, material: str) -> SequenceContentWorkspaceState:
        """Build the per-episode state from ONLY the question and material."""

        if len(question) + len(material) > int(self.config.max_input_chars):
            raise ValueError(
                f"input exceeds max_input_chars={int(self.config.max_input_chars)}"
            )
        keys_rows, _ = self._scan(
            material, "material_input", "material_recur", "material_bias"
        )
        question_rows, question_vec = self._scan(
            question, "question_input", "question_recur", "question_bias"
        )
        del question_rows
        ew = int(self.config.evidence_width)
        material_values = (
            keys_rows @ self._parameters["evidence_value"]
            if keys_rows.shape[0]
            else torch.zeros(0, ew)
        )
        material_keys = (
            keys_rows @ self._parameters["evidence_key"]
            if keys_rows.shape[0]
            else torch.zeros(0, ew)
        )
        if self.arm == "B":
            content = self._content_b(material_values, question_vec)
        else:
            content = self._content_a(material_values, question_vec)
        renderer_state = torch.tanh(
            question_vec @ self._parameters["start_weight"] + self._parameters["start_bias"]
        )
        char2slot, extra_slot_codepoints = self._episode_candidates(material)
        entry_slots = tuple(char2slot[character] for character in material)
        return SequenceContentWorkspaceState(
            material_keys=material_keys,
            material_values=material_values,
            renderer_state=renderer_state,
            content=content,
            question_vec=question_vec,
            material_chars=tuple(material),
            entry_slots=entry_slots,
            char2slot=char2slot,
            extra_slot_codepoints=extra_slot_codepoints,
            candidate_size=int(self.vocab.size) + len(extra_slot_codepoints),
        )

    # ------------------------------------------------------------- addressing

    def _copy_weights(
        self, state: SequenceContentWorkspaceState, previous_char: str | None
    ) -> torch.Tensor | None:
        """C-conditioned source addressing over material rows (None if empty).

        The emission-successor bonus lands on rows whose PREVIOUS row carries
        ``previous_char`` (D6 semantics, shared by both arms, zero start)."""

        length = int(state.material_keys.shape[0])
        if not length:
            return None
        scores = (
            state.renderer_state @ self._parameters["copy_query_state"]
            + state.content @ self._parameters["copy_query_content"]
        ) @ state.material_keys.T / math.sqrt(float(self.config.evidence_width))
        if _EMISSION_SUCCESSOR and previous_char is not None:
            successors = [
                position
                for position in range(1, length)
                if state.material_chars[position - 1] == previous_char
            ]
            if successors:
                scores = scores.clone()
                scores[successors] = (
                    scores[successors] + self._parameters["copy_induce_bias"]
                )
        return torch.softmax(scores, dim=0)

    def _copy_distribution(
        self, state: SequenceContentWorkspaceState, weights: torch.Tensor | None
    ) -> torch.Tensor:
        distribution = torch.zeros(int(state.candidate_size), dtype=torch.float32)
        if weights is None:
            return distribution
        slot_tensor = torch.tensor(state.entry_slots, dtype=torch.long)
        return distribution.index_add(0, slot_tensor, weights)

    # ------------------------------------------------------------- generation

    def step(
        self, state: SequenceContentWorkspaceState, previous_char: str | None
    ) -> tuple[SequenceContentWorkspaceState, torch.Tensor, torch.Tensor]:
        """One shared TF/free step: returns (new_state, mixture, copy_dist)."""

        previous_feature = self._char_output_feature(previous_char)
        weights = self._copy_weights(state, previous_char)
        source_read = (
            weights @ state.material_values
            if weights is not None
            else torch.zeros(int(self.config.evidence_width))
        )
        renderer_state = torch.tanh(
            torch.cat((previous_feature, state.content, source_read))
            @ self._parameters["renderer_input"]
            + state.renderer_state @ self._parameters["renderer_recur"]
            + self._parameters["renderer_bias"]
        )
        new_state = replace(state, renderer_state=renderer_state)
        vocab_logits = (
            torch.cat((renderer_state, state.content)) @ self._parameters["decoder"]
            + self._parameters["decoder_bias"]
        )
        vocab_probs = torch.softmax(vocab_logits, dim=0)
        copy_distribution = self._copy_distribution(new_state, weights)
        if weights is None:
            # no material: copy channel off, distribution stays finite+normal
            mixture = torch.zeros(int(new_state.candidate_size), dtype=torch.float32)
            mixture[: int(self.vocab.size)] = vocab_probs
            return new_state, mixture, copy_distribution
        gate = torch.sigmoid(
            torch.cat((renderer_state, state.content)) @ self._parameters["copy_gate_weight"]
            + self._parameters["copy_gate_bias"]
        )
        mixture = torch.zeros(int(new_state.candidate_size), dtype=torch.float32)
        mixture[: int(self.vocab.size)] = gate * vocab_probs
        mixture = mixture + (1.0 - gate) * copy_distribution
        return new_state, mixture, copy_distribution

    def _target_slot(self, state: SequenceContentWorkspaceState, character: str) -> int:
        slot = state.char2slot.get(character)
        if slot is not None:
            return slot
        known = self.vocab.slot_of(character)
        if known != CONTENT_UNK_SLOT:
            return known
        raise ValueError(
            f"response character {character!r} is neither in the train vocabulary "
            "nor in this episode's material"
        )

    def _teacher_forced_from_state(
        self, state: SequenceContentWorkspaceState, response: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixtures: list[torch.Tensor] = []
        copies: list[torch.Tensor] = []
        previous_char: str | None = None
        for character in response:
            state, mixture, copy_distribution = self.step(state, previous_char)
            mixtures.append(mixture)
            copies.append(copy_distribution)
            previous_char = character
        state, mixture, copy_distribution = self.step(state, previous_char)
        mixtures.append(mixture)
        copies.append(copy_distribution)
        return torch.stack(mixtures, dim=0), torch.stack(copies, dim=0)

    def teacher_forced_distributions(
        self, question: str, material: str, response: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(mixture, copy) rows for positions 0..len(response); the last row is
        the EOS (boundary) position.  Shares ``step`` with free generation."""

        return self._teacher_forced_from_state(
            self.begin_episode(question, material), response
        )

    def loss_components(
        self,
        question: str,
        material: str,
        response: str,
        copy_mask: Sequence[bool] | None = None,
        *,
        copy_value_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Composite-loss parts with live gradients: (ce, copy_nll, metrics).

        ``ce`` is the answer-length-mean CE including the EOS position;
        ``copy_nll`` is the copy-channel NLL on masked value positions (a
        zero tensor when the item has none, so the composite
        ``ce + weight * copy_nll`` is exact for non-copy items too).
        """

        if copy_value_weight < 0.0:
            raise ValueError("copy_value_weight cannot be negative")
        if len(response) > int(self.config.max_output_chars):
            raise ValueError(
                f"response exceeds max_output_chars={int(self.config.max_output_chars)}"
            )
        state0 = self.begin_episode(question, material)
        mixtures, copies = self._teacher_forced_from_state(state0, response)
        targets = torch.tensor(
            [self._target_slot(state0, character) for character in response]
            + [CONTENT_BOUNDARY_SLOT],
            dtype=torch.long,
        )
        target_probs = mixtures.gather(1, targets.unsqueeze(1)).squeeze(1)
        per_position = -target_probs.clamp_min(1e-12).log()
        ce = per_position.mean()
        copy_nll = torch.zeros((), dtype=torch.float32)
        copy_prob_mean = 0.0
        if copy_value_weight > 0.0:
            if copy_mask is None:
                raise ValueError("copy_value_weight requires copy_mask")
            mask = torch.tensor([bool(flag) for flag in copy_mask], dtype=torch.bool)
            if mask.numel() != len(response):
                raise ValueError("copy_mask must align with the response characters")
            if mask.any():
                for character, flag in zip(response, copy_mask, strict=True):
                    if flag and character not in state0.material_chars:
                        raise ValueError(
                            f"copy-masked character {character!r} is not in the material"
                        )
                hits = copies[:-1][mask].gather(1, targets[:-1][mask].unsqueeze(1)).squeeze(1)
                copy_nll = (-hits.clamp_min(1e-12).log().mean())
                copy_prob_mean = float(hits.detach().mean())
        with torch.no_grad():
            predictions = mixtures.argmax(dim=1)
            correct = int((predictions == targets).sum())
            metrics = {
                "positions": int(targets.numel()),
                "correct": correct,
                "accuracy": correct / max(1, int(targets.numel())),
                "mean_surprise": float((ce + copy_value_weight * copy_nll).detach()),
                "ce": float(ce.detach()),
                "copy_nll": float(copy_nll.detach()),
                "copy_value_prob_mean": float(copy_prob_mean),
            }
        return ce, copy_nll, metrics

    def sequence_loglik(
        self, question: str, material: str, response: str
    ) -> tuple[torch.Tensor, int]:
        """Summed TF log-likelihood of the full answer (incl. EOS) with live
        gradients -- the pair-contrastive objective's building block."""

        state0 = self.begin_episode(question, material)
        mixtures, _ = self._teacher_forced_from_state(state0, response)
        targets = torch.tensor(
            [self._target_slot(state0, character) for character in response]
            + [CONTENT_BOUNDARY_SLOT],
            dtype=torch.long,
        )
        probs = mixtures.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        return probs.log().sum(), int(targets.numel())

    def episode_loss(
        self,
        question: str,
        material: str,
        response: str,
        copy_mask: Sequence[bool] | None = None,
        *,
        copy_value_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Composite per-item loss: CE + ``copy_value_weight`` x copy NLL."""

        ce, copy_nll, metrics = self.loss_components(
            question, material, response, copy_mask, copy_value_weight=copy_value_weight
        )
        return ce + copy_value_weight * copy_nll, metrics

    def glyph_for_slot(self, state: SequenceContentWorkspaceState, slot: int) -> str:
        if slot == CONTENT_BOUNDARY_SLOT:
            return ""
        if slot == CONTENT_UNK_SLOT:
            return CONTENT_UNK_GLYPH
        if slot < int(self.vocab.size):
            return self.vocab.glyph_of(slot)
        return chr(int(state.extra_slot_codepoints[slot]))

    @torch.no_grad()
    def generate(
        self, question: str, material: str, *, max_chars: int | None = None
    ) -> ContentGenerationResult:
        limit = int(
            max_chars if max_chars is not None else self.config.max_output_chars
        )
        state = self.begin_episode(question, material)
        previous_char: str | None = None
        produced: list[str] = []
        stopped = False
        while len(produced) < limit:
            state, mixture, _ = self.step(state, previous_char)
            slot = int(mixture.argmax())
            if slot == CONTENT_BOUNDARY_SLOT:
                stopped = True
                break
            produced.append(self.glyph_for_slot(state, slot))
            previous_char = produced[-1]
        return ContentGenerationResult(
            "".join(produced), stopped, len(produced), not stopped and len(produced) >= limit
        )

    @torch.no_grad()
    def generate_with_content(
        self, question: str, material: str, content: torch.Tensor, *, max_chars: int | None = None
    ) -> ContentGenerationResult:
        """Content-consumption diagnostic: generate with a SWAPPED C while the
        material/source mapping stays this episode's own (contract section 5)."""

        if content.shape != (int(self.config.content_width),):
            raise ValueError("swapped content must match content_width")
        limit = int(
            max_chars if max_chars is not None else self.config.max_output_chars
        )
        state = replace(self.begin_episode(question, material), content=content)
        previous_char: str | None = None
        produced: list[str] = []
        stopped = False
        while len(produced) < limit:
            state, mixture, _ = self.step(state, previous_char)
            slot = int(mixture.argmax())
            if slot == CONTENT_BOUNDARY_SLOT:
                stopped = True
                break
            produced.append(self.glyph_for_slot(state, slot))
            previous_char = produced[-1]
        return ContentGenerationResult(
            "".join(produced), stopped, len(produced), not stopped and len(produced) >= limit
        )


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


class SequenceContentTrainer:
    """AdamW trainer with the contract's two-recipe schedule (section 4)."""

    def __init__(
        self,
        workspace: SequenceContentWorkspace,
        *,
        learning_rate: float,
        total_updates: int,
        warmup_fraction: float = 0.05,
        final_lr_fraction: float = 0.1,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        code_revision: str = "",
        data_digest: str = "",
    ) -> None:
        self.workspace = workspace
        self.learning_rate = _finite_positive(learning_rate, "learning_rate")
        self.total_updates = int(total_updates)
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive")
        self.warmup_fraction = float(warmup_fraction)
        self.final_lr_fraction = float(final_lr_fraction)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.code_revision = str(code_revision)
        self.data_digest = str(data_digest)
        self.copy_value_weight = 1.0
        #: Pair-contrastive auxiliary (contract v2 section D1); off in v1.
        self.pair_contrastive_weight = 0.0
        self.pair_contrastive_margin = 1.0
        self.trainer_revision = "v1"
        self.global_step = 0
        self.optimizer = torch.optim.AdamW(
            self.workspace.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------- schedule

    def lr_at(self, step: int) -> float:
        """Peak lr with linear warmup for the first ``warmup_fraction`` of the
        run, then cosine decay to ``final_lr_fraction`` x peak (contract §4)."""

        step = max(0, int(step))
        warmup_steps = max(1, int(self.total_updates * self.warmup_fraction))
        if step <= warmup_steps:
            return self.learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / max(1, self.total_updates - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        final = self.learning_rate * self.final_lr_fraction
        return final + (self.learning_rate - final) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply_schedule(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_at(self.global_step + 1)

    # -------------------------------------------------------------- training

    def enable_pair_contrastive(self, weight: float = 1.0, margin: float = 1.0) -> None:
        """Contract v2 section D1: group-counterfactual contrastive auxiliary.

        Requires group-structured batches (consecutive items sharing
        ``group_id``, two members per group).  gamma/weight freeze before any
        v2 run; they are never tuned on calibration results.
        """

        if weight < 0.0 or margin < 0.0:
            raise ValueError("pair_contrastive weight and margin cannot be negative")
        self.pair_contrastive_weight = float(weight)
        self.pair_contrastive_margin = float(margin)
        self.trainer_revision = "v2-pair-contrastive" if weight > 0.0 else "v1"

    def _pair_contrastive_term(
        self, batch: Sequence[Mapping[str, Any]]
    ) -> tuple[torch.Tensor, int]:
        """softplus(s_crossed - s_own + gamma) per member, mean over groups.

        s_own comes from the CE pass (summed log-likelihood = -ce x positions,
        gradient-preserving); the two crossed sequences need one extra TF pass
        per member.  Equal-answer groups degenerate to a constant (no grad).
        """

        groups: dict[str, list[Mapping[str, Any]]] = OrderedDict()
        for item in batch:
            group_id = item.get("group_id")
            if group_id is None:
                raise ValueError(
                    "pair-contrastive batches must carry group_id on every item"
                )
            groups.setdefault(str(group_id), []).append(item)
        terms = []
        for group_id, members in groups.items():
            if len(members) != 2:
                raise ValueError(
                    f"group {group_id} has {len(members)} members; exactly 2 required"
                )
            (item_a, item_b) = members
            qa, ma, ra = (
                str(item_a["question"]),
                str(item_a["material"]),
                str(item_a["response"]),
            )
            qb, mb, rb = (
                str(item_b["question"]),
                str(item_b["material"]),
                str(item_b["response"]),
            )
            s_xx, _ = self.workspace.sequence_loglik(qa, ma, ra)
            s_yy, _ = self.workspace.sequence_loglik(qb, mb, rb)
            s_yx, _ = self.workspace.sequence_loglik(qa, ma, rb)
            s_xy, _ = self.workspace.sequence_loglik(qb, mb, ra)
            gamma = self.pair_contrastive_margin
            terms.append(
                torch.nn.functional.softplus(s_yx - s_xx + gamma)
                + torch.nn.functional.softplus(s_xy - s_yy + gamma)
            )
        return torch.stack(terms).mean(), len(terms)

    def train_step(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """One microbatch of items (question/material/response/copy_mask).

        Composite loss per item: answer-length-mean CE incl. EOS plus
        ``copy_value_weight`` x copy NLL on masked positions (0 when the item
        has none).  Non-finite loss/gradient aborts the step (contract stop
        rule); gradient norms are recorded.
        """

        if not batch:
            raise ValueError("content workspace batch cannot be empty")
        self._apply_schedule()
        self.optimizer.zero_grad(set_to_none=True)
        composites = []
        ce_values = []
        copy_values = []
        for item in batch:
            ce, copy_nll, metrics = self.workspace.loss_components(
                str(item["question"]),
                str(item["material"]),
                str(item["response"]),
                item.get("copy_mask"),
                copy_value_weight=float(self.copy_value_weight),
            )
            composites.append(ce + float(self.copy_value_weight) * copy_nll)
            ce_values.append(metrics["ce"])
            copy_values.append(metrics["copy_nll"])
        total = torch.stack(composites).mean()
        pair_term = 0.0
        if self.pair_contrastive_weight > 0.0:
            pair_term, pair_groups = self._pair_contrastive_term(batch)
            total = total + float(self.pair_contrastive_weight) * pair_term
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite loss at step {self.global_step}")
        total.backward()
        parameters = [p for p in self.workspace.parameters() if p.requires_grad]
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        )
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {self.global_step}")
        self.optimizer.step()
        self.global_step += 1
        return {
            "loss": float(total.detach()),
            "ce": sum(ce_values) / len(ce_values) if ce_values else 0.0,
            "copy_nll": sum(copy_values) / len(copy_values) if copy_values else 0.0,
            "pair_contrastive": (
                float(pair_term.detach()) if torch.is_tensor(pair_term) else float(pair_term)
            ),
            "grad_norm": grad_norm,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "global_step": self.global_step,
        }

    # ------------------------------------------------------------- checkpoint

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": SEQUENCE_CONTENT_WORKSPACE_TRAINER_FORMAT,
            "version": SEQUENCE_CONTENT_WORKSPACE_VERSION,
            "serialization": "torch.save/atomic",
            "code_revision": self.code_revision,
            "contract": CONTRACT_PATH,
            "candidate_construction": CANDIDATE_CONSTRUCTION_VERSION,
            "selection_rule": SELECTION_RULE_ID,
            "data_digest": self.data_digest,
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
            "global_step": int(self.global_step),
            "copy_value_weight": float(self.copy_value_weight),
            "trainer_revision": self.trainer_revision,
            "pair_contrastive_weight": float(self.pair_contrastive_weight),
            "pair_contrastive_margin": float(self.pair_contrastive_margin),
            "learning_rate": float(self.learning_rate),
            "total_updates": int(self.total_updates),
            "warmup_fraction": float(self.warmup_fraction),
            "final_lr_fraction": float(self.final_lr_fraction),
            "weight_decay": float(self.weight_decay),
            "grad_clip": float(self.grad_clip),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        return atomic_save(self.checkpoint(), path)

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> SequenceContentTrainer:
        if payload.get("format") != SEQUENCE_CONTENT_WORKSPACE_TRAINER_FORMAT:
            raise ValueError("not a sequence content workspace trainer checkpoint")
        version = int(payload.get("version", -1))
        if version not in SEQUENCE_CONTENT_WORKSPACE_SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported content workspace checkpoint version: {version}")
        digest = payload.get("checkpoint_digest")
        body = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        if digest is None or str(digest) != str(content_digest(body)):
            raise ValueError("content workspace checkpoint digest mismatch")
        vocab = CharVocab(str(payload["vocab_chars"]))
        config = SequenceContentConfig(**dict(payload["config"]))
        workspace = SequenceContentWorkspace(vocab, config)
        parameters = payload["parameters"]
        order = workspace._order
        missing = [name for name in order if name not in parameters]
        if missing:
            raise ValueError(f"content workspace checkpoint missing parameters: {missing}")
        with torch.no_grad():
            for name in order:
                tensor = parameters[name].detach().to(dtype=torch.float32).clone()
                parameter = workspace._parameters[name]
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise ValueError(f"content workspace parameter shape mismatch: {name}")
                parameter.copy_(tensor)
        trainer = cls(
            workspace,
            learning_rate=float(payload["learning_rate"]),
            total_updates=int(payload["total_updates"]),
            warmup_fraction=float(payload.get("warmup_fraction", 0.05)),
            final_lr_fraction=float(payload.get("final_lr_fraction", 0.1)),
            weight_decay=float(payload.get("weight_decay", 0.0)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
            code_revision=str(payload.get("code_revision", "")),
            data_digest=str(payload.get("data_digest", "")),
        )
        try:
            trainer.optimizer.load_state_dict(dict(payload["optimizer"]))
        except (ValueError, KeyError):
            pass  # fresh optimizer state when param identity differs
        trainer.global_step = int(payload.get("global_step", 0))
        trainer.copy_value_weight = float(payload.get("copy_value_weight", 1.0))
        trainer.pair_contrastive_weight = float(payload.get("pair_contrastive_weight", 0.0))
        trainer.pair_contrastive_margin = float(payload.get("pair_contrastive_margin", 1.0))
        trainer.trainer_revision = str(payload.get("trainer_revision", "v1"))
        state = payload.get("rng_state")
        if isinstance(state, torch.Tensor):
            torch.set_rng_state(state)
        return trainer
