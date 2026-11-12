"""Counterfactual input credit; no additional head or inference answer table."""

from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from scripts.training.first_a_support_22_5 import task_loss
from taiji.u2_task_episode import NAMES, locate_public, position_word


def paired_example(example):
    public = example["public"]
    image = (
        (
            public["observations"]["rgb"]
            .reshape(4, 4, 4, 4, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(16, 16, 3)
            * 255
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    original = public["request"].decode().removesuffix("现在在哪里？")
    for name in NAMES:
        if name == original:
            continue
        x = locate_public(image, name)
        answer = "未知" if x is None else position_word(x)
        if answer.encode() != example["target"]:
            negative = copy.deepcopy(example)
            negative["public"]["request"] = f"{name}现在在哪里？".encode()
            return negative
    raise ValueError("counterfactual_pair_unavailable")


def objective(model, batch, mode):
    if mode == "token":
        return task_loss(model, batch)
    if mode != "conditional":
        raise ValueError("unknown_A_objective")
    terms = []
    for example in batch:
        positive = task_loss(model, [example])
        negative = task_loss(model, [paired_example(example)])
        terms.append(positive + 10 * F.softplus(0.1 + positive - negative))
    return torch.stack(terms).mean()
