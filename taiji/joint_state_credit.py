"""E2c task-credit candidate for the N177 E1 consumption slice.

One answered query = one hard selection among same-entity candidates, then a
linear value readout over the **live** byte-encoded feature. `credit_on` leaves
the answer CE gradient on that feature; `credit_stop` cuts it at the readout
input while the readout keeps updating. The selector scores by entity/attribute
identity, so the answer CE reaches it in neither arm — that is reported as a
fact about this candidate, not papered over.

Nothing here trains the producer, the V4 decoder or a state store: this is the
zero-step engineering bundle that E2c/E3 judge (credit path, forbidden-update
groups, optimizer/RNG/cursor round-trip).
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .internalization import content_digest
from .joint_state_consumption import (
    PersistentContentFeatureV1,
    QueryConditionedSelectorV1,
)
from .joint_state_readout import FeatureValueReadoutV1
from .joint_state_value_encoder import ValueByteEncoderV1

STATE_CREDIT_FORMAT = "taiji-n177-e2c-state-credit-v1"
STATE_CREDIT_COMPONENT = "joint_state_credit_v1"
CREDIT_MODES = ("credit_on", "credit_stop")
MODULE_GROUPS = ("encoder", "selector", "readout")


@dataclass(frozen=True)
class CreditCandidateV1:
    feature: PersistentContentFeatureV1
    value_bytes: torch.Tensor

    @classmethod
    def from_value(cls, *, feature: PersistentContentFeatureV1, value: str) -> CreditCandidateV1:
        raw = torch.tensor(list(value.encode("utf-8")), dtype=torch.long)
        return cls(feature=feature, value_bytes=raw)


@dataclass(frozen=True)
class CreditCaseV1:
    """One answered query: candidate features, the live value bytes, the label."""

    case_id: str
    entity: str
    attribute: str
    candidates: tuple[CreditCandidateV1, ...]
    label: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "feature_ids": [item.feature.feature_id for item in self.candidates],
            "label": self.label,
        }


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squares += float(parameter.grad.detach().square().sum())
    return squares**0.5


class StateCreditCandidateV1(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "credit_on",
        feature_dim: int = 64,
        byte_dim: int = 32,
        hidden_dim: int = 64,
        seed: int = 0,
        learning_rate: float = 1e-2,
        freeze: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if mode not in CREDIT_MODES:
            raise ValueError(f"unsupported state credit mode: {mode}")
        unknown = sorted(set(freeze) - set(MODULE_GROUPS))
        if unknown:
            raise ValueError(f"unknown freeze groups: {unknown}")
        torch.manual_seed(seed)
        random.seed(seed)
        self.mode = mode
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.feature_dim = int(feature_dim)
        self.byte_dim = int(byte_dim)
        self.hidden_dim = int(hidden_dim)
        self.freeze_groups = tuple(sorted(freeze))
        self.encoder = ValueByteEncoderV1(byte_dim=byte_dim, hidden_dim=feature_dim)
        self.selector = QueryConditionedSelectorV1()
        self.readout = FeatureValueReadoutV1(feature_dim=feature_dim)
        self.cursor = 0
        trainable = [
            parameter
            for group in MODULE_GROUPS
            if group not in self.freeze_groups
            for parameter in getattr(self, group).parameters()
        ]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=0.0)
        self.parent_digest = self.parameters_digest()

    def parameters_by_group(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {group: tuple(getattr(self, group).parameters()) for group in MODULE_GROUPS}

    def parameters_digest(self) -> dict[str, str]:
        return {group: content_digest(getattr(self, group).state_dict()) for group in MODULE_GROUPS}

    def forward(self, case: CreditCaseV1) -> dict[str, Any]:
        features = tuple(item.feature for item in case.candidates)
        scores = self.selector.score_candidates(
            entity=case.entity,
            attribute=case.attribute,
            candidates=features,
        )
        index = int(torch.argmax(scores))
        live = self.encoder(case.candidates[index].value_bytes)
        fed = live if self.mode == "credit_on" else live.detach()
        logits = self.readout(fed)
        target = torch.tensor([case.label], dtype=torch.long)
        loss = F.cross_entropy(logits.reshape(1, -1), target)
        return {
            "logits": logits,
            "loss": loss,
            "selection_index": index,
            "selection_scores": scores,
            "feature": live,
            "case": case.to_payload(),
        }

    def backward(self, loss: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for group in self.freeze_groups:
            for parameter in getattr(self, group).parameters():
                parameter.grad = None
        return {
            group: _grad_norm(group_parameters)
            for group, group_parameters in self.parameters_by_group().items()
        }

    def step(self) -> int:
        self.optimizer.step()
        self.cursor += 1
        return self.cursor

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": STATE_CREDIT_FORMAT,
            "mode": self.mode,
            "seed": self.seed,
            "cursor": self.cursor,
            "feature_dim": self.feature_dim,
            "byte_dim": self.byte_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "freeze_groups": list(self.freeze_groups),
            "parent_digest": self.parent_digest,
            "parameters": {
                group: copy.deepcopy(getattr(self, group).state_dict()) for group in MODULE_GROUPS
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "torch_rng_state": torch.random.get_rng_state().clone(),
            "python_rng_state": random.getstate(),
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], *, mode: str | None = None
    ) -> StateCreditCandidateV1:
        if payload.get("format") != STATE_CREDIT_FORMAT:
            raise ValueError("state credit checkpoint has an unexpected format")
        candidate = cls(
            mode=mode or payload["mode"],
            feature_dim=int(payload["feature_dim"]),
            byte_dim=int(payload["byte_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            seed=int(payload["seed"]),
            learning_rate=float(payload["learning_rate"]),
            freeze=tuple(payload["freeze_groups"]),
        )
        for group in MODULE_GROUPS:
            getattr(candidate, group).load_state_dict(copy.deepcopy(payload["parameters"][group]))
        candidate.optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
        candidate.cursor = int(payload["cursor"])
        torch.random.set_rng_state(payload["torch_rng_state"].clone())
        random.setstate(payload["python_rng_state"])
        return candidate


def checkpoint_holds_no_graph(payload: dict[str, Any]) -> bool:
    """A checkpoint must not persist autograd history: no saved tensor requires grad."""

    def walk(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return not value.requires_grad
        if isinstance(value, dict):
            return all(walk(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(walk(item) for item in value)
        return True

    return walk(payload)
