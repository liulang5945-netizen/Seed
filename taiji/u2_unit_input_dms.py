"""unit-input-dms-v1: delayed match-to-sample encoding for 22.5.W.

One episode is a sequence of stages over the same six legal [16,8] channels:
sample (attribute one-hot in every column), D blank delay steps, an 8-slot test
row, then a same/different query. The graph carries its own z/a state across the
stages; nothing outside the legal channels carries the sample.

No parameters, no buffers; a pure function plus a frozen description (like
``unit-input-column-color-v1``).
"""

from __future__ import annotations

from typing import Any

import torch

IDENTITY = "unit-input-dms-v1"
COLUMNS = 16
INPUT_DIM = 8
ATTRIBUTES = 4
DELAY_STEPS = 4
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def _zero(dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(COLUMNS, INPUT_DIM, dtype=dtype)


def encode_stage(
    stage: str,
    sample: int | None,
    test: list[int | None],
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    rgb = _zero(dtype)
    question = _zero(dtype)
    if stage == "sample":
        if sample is None or not 0 <= sample < ATTRIBUTES:
            raise ValueError("sample_attribute_out_of_range")
        for column in range(COLUMNS):
            rgb[column, sample] = 1.0
            rgb[column, 4] = 1.0  # present
            rgb[column, 6] = 1.0  # sample marker
        question[:, 0] = 1.0  # sample stage
    elif stage == "delay":
        question[:, 1] = 1.0
    elif stage == "test":
        if len(test) != 8:
            raise ValueError("test_row_expects_8_slots")
        for column, attribute in enumerate(test):
            if attribute is None:
                continue
            if not 0 <= attribute < ATTRIBUTES:
                raise ValueError("test_attribute_out_of_range")
            rgb[column, attribute] = 1.0
            rgb[column, 4] = 1.0
            rgb[column, 5] = 1.0  # test marker
        question[:, 2] = 1.0
    elif stage == "query":
        question[:, 3] = 1.0
        question[:, 4] = 1.0  # same? query
    else:
        raise ValueError(f"unknown_stage:{stage}")
    question[:, 7] = 1.0  # bias
    goal = _zero(dtype)
    goal[:, 0] = 1.0
    zero = _zero(dtype)
    return {
        "rgb": rgb,
        "audio": zero.clone(),
        "proprio": zero.clone(),
        "question": question,
        "goal": goal,
        "feedback": zero.clone(),
    }


def encode_episode(
    episode: dict[str, Any], delay: int = DELAY_STEPS, dtype: torch.dtype = torch.float64
) -> list[dict[str, torch.Tensor]]:
    stages = [encode_stage("sample", episode["sample"], episode["test"], dtype)]
    stages.extend(encode_stage("delay", None, episode["test"], dtype) for _ in range(delay))
    stages.append(encode_stage("test", episode["sample"], episode["test"], dtype))
    stages.append(encode_stage("query", episode["sample"], episode["test"], dtype))
    return stages


def label(episode: dict[str, Any]) -> int:
    return int(any(attribute == episode["sample"] for attribute in episode["test"]))


def encoding_contract() -> dict[str, Any]:
    return {
        "identity": IDENTITY,
        "columns": COLUMNS,
        "channels": list(CHANNELS),
        "rgb_layout": "[4 attribute one-hot, present, test marker, sample marker, bias]",
        "question_layout": "[sample, delay, test, query, same-query marker, 0, 0, bias]",
        "goal_layout": "fixed marker at slot 0 (constant)",
        "attributes": ATTRIBUTES,
        "delay_steps": DELAY_STEPS,
        "label": "1 iff the sample attribute appears in the test row",
        "unit_prelabeling": "none: sensory evidence only, the graph carries the sample",
    }
