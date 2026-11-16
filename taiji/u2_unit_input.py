"""Unit-core B event->unit input encoding contract (``unit-input-column-color-v1``).

This is the byte-world task interface for the unit-native core B (route B of
DECISION-TAIJI_NEURAL_CORE_PATH_20261114). It maps a frozen first-A public scene
(16x16 RGB raster + a natural-Chinese position request) onto the six legal input
channels of ``taiji.u2_neural_units`` (each ``[T, 8]``, ``T = 16`` pixel columns).

Design grounding (22.5.U.1): the A line closed because the pooled 8-dim rgb token
could not carry "where is the queried colour" and the byte path never read it.
This contract deliberately preserves the task-relevant sensory evidence -- the
per-column colour-presence mass and the column position axis -- so that U.1
isolates the *unit binding/read* question (can direct unit read bind query ->
colour -> position?) from the *encoder loss* question that killed the A line.

Information rights and declared dimensionality reduction:
  * rgb: 16 columns x 48 raw values (768) -> 16 x 8 (128). Each column pools its
    16 rows into six palette colour masses (the public colour-name prior of
    ``locate_public``), plus the column centre x and total mass. Lost: within-
    column row detail and exact pixel values. Preserved: colour identity per
    column (palette well separated) and the position axis exactly. Validated to
    reproduce the authoritative public oracle on all 96 train+fit scenes.
  * question: the request ``{entity}现在在哪里？`` -> a constant 6-way entity
    one-hot plus a question-type marker and bias. Lost: exact wording (every
    request shares the template, so no task-relevant loss).
  * goal: a fixed marker (the goal bytes are constant across scenes; carries no
    discriminative information -- declared, not hidden).
  * audio / proprio / feedback: exact zeros. The public-scene fixture carries no
    task-relevant audio/proprio for position reading and U.1 is read-only (no
    feedback). No information rights are claimed for these channels.

No parameters, no buffers, nothing enters any ``state_dict``; the contract is a
pure function plus a frozen description whose digest is pinned by the U.1 card.
Units are NOT pre-labelled with entity/colour/answer (SPEC section 7): the
encoding supplies sensory evidence only; the binding is the graph's computation.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from .u2_observation_columns import column_values
from .u2_task_episode import NAMES, PALETTE, position_word

IDENTITY = "unit-input-column-color-v1"
COLUMNS = 16
INPUT_DIM = 8
QUESTION_SUFFIX = "现在在哪里？"
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
_PALETTE = torch.tensor(PALETTE, dtype=torch.float64)
_X_CENTER = 2 * (torch.arange(COLUMNS, dtype=torch.float64) + 0.5) / COLUMNS - 1
_MASS_FLOOR = 0.25  # identical to locate_public's presence threshold


def scene_entity(request: bytes) -> str:
    """Parse the queried entity from a frozen first-A position request."""

    text = request.decode("utf-8")
    if not text.endswith(QUESTION_SUFFIX):
        raise ValueError("unsupported_unit_input_request")
    entity = text[: -len(QUESTION_SUFFIX)]
    if entity not in NAMES:
        raise ValueError("unknown_unit_input_entity")
    return entity


def color_mass(rgb: torch.Tensor) -> torch.Tensor:
    """[16,48] patch observation -> [16,6] per-column palette colour mass.

    Mirrors ``locate_public``'s public colour prior exactly: distance to each
    palette colour, nearest-colour assignment, weight ``max(0, 1 - d/0.35)``,
    summed over the 16 rows of each pixel column.
    """

    if rgb.shape != (COLUMNS, 48):
        raise ValueError(f"unit_input_rgb_expects_16x48_got_{tuple(rgb.shape)}")
    columns = column_values(rgb).to(torch.float64)  # [16, 48]
    pixels = columns.reshape(COLUMNS, COLUMNS, 3)  # [column, row, channel]
    distance = torch.norm(pixels[:, :, None, :] - _PALETTE[None, None, :, :], dim=-1)
    assign = distance.argmin(-1)  # [column, row]
    weight = torch.clamp(1 - distance / 0.35, min=0) * (
        assign[..., None] == torch.arange(len(NAMES))
    )
    return cast(torch.Tensor, weight.sum(1))  # [column, 6]


def derive_label(mass: torch.Tensor, entity: str) -> str:
    """Oracle-consistency readout from the encoding (verification only).

    This is the public oracle recomputed on the encoded colour mass; it is used
    to prove the encoding is information-faithful, NEVER as a training target or
    a model output. The unit graph must learn this binding itself.
    """

    index = NAMES.index(entity)
    column_mass = mass[:, index]
    if float(column_mass.sum()) < _MASS_FLOOR:
        return "未知"
    x = float((column_mass * _X_CENTER).sum() / column_mass.sum())
    return cast(str, position_word(x))


def encode_scene(
    public: dict[str, Any], dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Map a frozen first-A public scene to six legal ``[16, 8]`` channels."""

    entity = scene_entity(public["request"])
    mass = color_mass(public["observations"]["rgb"])  # [16, 6]
    total = mass.sum(-1, keepdim=True)  # [16, 1]
    rgb = torch.cat([mass, _X_CENTER[:, None], total], dim=-1)  # [16, 8]

    question = torch.zeros(COLUMNS, INPUT_DIM, dtype=torch.float64)
    question[:, NAMES.index(entity)] = 1.0
    question[:, len(NAMES)] = 1.0  # question-type marker: position-now
    question[:, len(NAMES) + 1] = 1.0  # bias

    goal = torch.zeros(COLUMNS, INPUT_DIM, dtype=torch.float64)
    goal[:, 0] = 1.0  # fixed goal marker (constant across scenes)

    zero = torch.zeros(COLUMNS, INPUT_DIM, dtype=torch.float64)
    channels = {
        "rgb": rgb,
        "audio": zero.clone(),
        "proprio": zero.clone(),
        "question": question,
        "goal": goal,
        "feedback": zero.clone(),
    }
    return {name: value.to(dtype) for name, value in channels.items()}


def encoding_contract() -> dict[str, Any]:
    """Frozen description of the encoding contract (digest pinned by the card)."""

    return {
        "identity": IDENTITY,
        "columns": COLUMNS,
        "input_dim": INPUT_DIM,
        "channels": list(CHANNELS),
        "rgb_layout": "[6 palette colour masses, column centre x, total mass]",
        "question_layout": "[6 entity one-hot, question-type marker, bias]",
        "goal_layout": "fixed marker at slot 0 (constant)",
        "zero_channels": ["audio", "proprio", "feedback"],
        "palette": [list(map(int, row)) for row in (PALETTE * 255).astype(int)],
        "entities": list(NAMES),
        "mass_floor": _MASS_FLOOR,
        "color_prior": "max(0, 1 - distance/0.35) masked to nearest palette colour",
        "position_axis": "x = 2*(column+0.5)/16 - 1; 左边 x<-1/3, 右边 x>1/3, else 中间",
        "dimensionality_reduction": {
            "rgb": "768 raw -> 128; row detail and exact pixels lost, colour identity and position preserved",
            "question": "template wording lost; entity identity and question type preserved",
            "goal": "constant; no discriminative information",
            "audio_proprio_feedback": "exact zeros; no information rights claimed",
        },
        "oracle_consistency": "derive_label reproduces the public oracle on 96/96 train+fit scenes",
        "unit_prelabeling": "none: sensory evidence only, binding is the graph computation",
    }
